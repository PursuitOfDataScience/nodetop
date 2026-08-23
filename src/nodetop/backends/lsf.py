"""IBM Spectrum LSF.

LSF states the accept/start distinction more plainly than any other system:
a queue's status is literally two words, ``Open:Active`` or ``Closed:Inact``.
``Open`` means it takes submissions; ``Active`` means it dispatches them.  A
queue sitting at ``Open:Inact`` accepts everything and runs nothing, which is
the same trap PBS spells ``enabled/started`` and Slurm hides inside a single
partition state.

**LSF has no verify-only submission**, so entitlement here is declared (from
``bqueues -l`` USERS/HOSTS) and cannot be confirmed in advance.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
from collections.abc import Iterable
from datetime import datetime

from ..core.hardware import identify_accelerator
from ..core.model import Identity, Job, JobShape, Limits, Node, Queue, Verdict
from ..runner import Runner, SubprocessRunner, which
from .base import BackendCapabilities

__all__ = ["LsfBackend"]

#: bhosts STATUS -> neutral condition.
_STATUS_TO_CONDITION = {
    "unavail": "DOWN",
    "unreach": "DOWN",
    "unlicensed": "DOWN",
    "closed_adm": "DRAIN",       # administratively closed
    "closed_lock": "DRAIN",
    "closed_excl": "RESERVED",
    "closed_full": None,         # simply busy: NOT a blocking condition
    "closed_busy": None,         # load thresholds exceeded; still schedulable
    "closed_cu_excl": "RESERVED",
    "ok": None,
}


def _int(value: str | None) -> int:
    if not value:
        return 0
    m = re.match(r"^\s*(\d+)", str(value).replace("-", "0"))
    return int(m.group(1)) if m else 0


class LsfBackend:
    """Adapter for IBM Spectrum LSF."""

    name = "lsf"
    queue_term = "queue"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()
        self._nodes: list[Node] | None = None
        self._bqueues_cache: str | None = None

    @classmethod
    def detect(cls) -> bool:
        return which("bhosts") and which("bqueues")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=False,
            notes=(
                "no verify-only mode: entitlement is DECLARED, from bqueues USERS/HOSTS",
                "Open:Inact accepts jobs and dispatches none -- QUEUE_NOT_STARTED",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_nodes(self, bhosts: str, lshosts: str = "", gpu: str = "") -> list[Node]:
        """Parse ``bhosts -w``, enriched with ``lshosts -w`` and GPU info.

        ``bhosts`` has the occupancy and status; ``lshosts`` has the hardware
        (core count, memory, model string) and is where an accelerator model
        can be recovered from the resource list.
        """
        hardware: dict[str, dict[str, str]] = {}
        for line in lshosts.splitlines()[1:]:
            f = line.split()
            if len(f) >= 6:
                hardware[f[0]] = {
                    "type": f[1],
                    "model": f[2],
                    "ncpus": f[4],
                    "maxmem": f[5],
                    "resources": " ".join(f[6:]) if len(f) > 6 else "",
                }

        gpus: dict[str, tuple[int, int, str]] = {}
        for line in gpu.splitlines():
            f = line.split()
            # bhosts -gpu: HOST_NAME GPU_ID MODEL MUSED MRSV NJOBS RUN ...
            if len(f) >= 3 and not f[0].upper().startswith("HOST"):
                total, used, model = gpus.get(f[0], (0, 0, ""))
                busy = 1 if len(f) >= 7 and _int(f[6]) > 0 else 0
                gpus[f[0]] = (total + 1, used + busy, model or f[2])

        out: list[Node] = []
        for line in bhosts.splitlines():
            f = line.split()
            if len(f) < 6 or f[0].upper().startswith("HOST"):
                continue
            name, status = f[0], f[1].lower()
            condition = _STATUS_TO_CONDITION.get(status)
            hw = hardware.get(name, {})
            g_total, g_used, g_model = gpus.get(name, (0, 0, ""))
            max_slots = _int(f[3])
            resources = hw.get("resources", "")
            out.append(
                Node(
                    name=name,
                    state_raw=f[1],
                    conditions=frozenset({condition} if condition else set()),
                    cpus_total=_int(hw.get("ncpus")) or max_slots,
                    cpus_alloc=_int(f[5]),  # RUN
                    memory_mb=_mem_to_mb(hw.get("maxmem")),
                    gpus_total=g_total,
                    gpus_alloc=g_used,
                    accelerator=identify_accelerator(
                        None, ",".join(x for x in (g_model, hw.get("model", ""), resources) if x)
                    ),
                    labels=tuple(
                        x for x in (hw.get("type", ""), hw.get("model", "")) if x
                    ),
                    unreachable=status in {"unavail", "unreach"},
                    reason="" if condition is None else f"bhosts status {f[1]}",
                )
            )
        return out

    def load_nodes(self) -> list[Node]:
        # Cached deliberately. load_queues() needs the nodes too, and
        # re-deriving them there would query the control plane a second time --
        # so a single report could mix two different instants, which is exactly
        # what taking one snapshot is supposed to prevent.
        if self._nodes is not None:
            return self._nodes
        bhosts = self.runner.run(["bhosts", "-w"])
        lshosts = ""
        gpu = ""
        with contextlib.suppress(Exception):
            lshosts = self.runner.run(["lshosts", "-w"])
        with contextlib.suppress(Exception):
            gpu = self.runner.run(["bhosts", "-gpu", "-w"])
        self._nodes = self.parse_nodes(bhosts, lshosts, gpu)
        return self._nodes

    # -- queues -------------------------------------------------------------
    def parse_queues(self, text: str) -> list[Queue]:
        """Parse ``bqueues -l``, whose records are separated by ``QUEUE:``."""
        out: list[Queue] = []
        for block in re.split(r"^QUEUE:\s*", text, flags=re.MULTILINE)[1:]:
            lines = block.splitlines()
            name = lines[0].strip()
            body = "\n".join(lines[1:])

            status = _param_status(body)
            # "Open:Active" -> accepts and dispatches.
            accept, _, dispatch = status.partition(":")
            users = _after(body, "USERS:")
            hosts = _after(body, "HOSTS:")
            walltime = None
            # RUNLIMIT is on the line after the label, in minutes.
            m = re.search(r"RUNLIMIT\s*\n\s*([\d.]+)\s*min", body)
            if m:
                walltime = int(float(m.group(1)) * 60)

            out.append(
                Queue(
                    name=name,
                    state_raw=status or "unknown",
                    enabled=accept.lower() != "closed",
                    started=dispatch.lower() not in {"inact", "inactive"},
                    # "all" is LSF's wildcard; anything else is a real list.
                    allow_users=(
                        () if users.strip() in {"all", "all users", ""}
                        else tuple(_tokens(users))
                    ),
                    max_walltime_seconds=walltime,
                    limits_name=name,
                    node_names=(),
                    priority=_int(_first_word(_after(body, "PRIO:"))),
                )
            )
            # HOSTS is a host-group expression; "all" means every host.
            out[-1].labels_hosts = hosts  # type: ignore[attr-defined]
        return out

    def parse_host_groups(self, text: str) -> dict[str, list[str]]:
        """Parse ``bmgroup -w``: ``GROUP_NAME    host1 host2 ...``."""
        groups: dict[str, list[str]] = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[0].upper().startswith("GROUP"):
                continue
            # LSF may suffix a member with a slice count ("hostA+1") or mark a
            # group with a trailing slash. Strip only those: a blanket
            # strip("0123456789") also eats the digits in "gpu-01".
            groups[fields[0].strip("/")] = [
                re.sub(r"[/+]\d*$", "", h) for h in fields[1:]
            ]
        return groups

    def resolve_hosts(
        self, spec: str, all_names: tuple[str, ...], groups: dict[str, list[str]]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Expand a queue's HOSTS field into ``(members, unresolved_tokens)``.

        A token may be a hostname, a host group, or ``all``.  Anything that
        cannot be resolved is *reported*, never guessed at: the previous
        behaviour fell back to every host in the cluster, which handed a queue
        restricted to four machines the free capacity of all of them -- and
        manufacturing capacity is exactly the failure this tool exists to
        catch.
        """
        spec = spec.strip()
        if not spec or spec.lower().startswith("all"):
            return all_names, ()

        known = set(all_names)
        members: list[str] = []
        unresolved: list[str] = []
        for token in _tokens(spec):
            if token in known:
                members.append(token)
            elif token in groups:
                members.extend(h for h in groups[token] if h in known)
            else:
                unresolved.append(token)
        # Preserve cluster order and drop duplicates from overlapping groups.
        seen = dict.fromkeys(members)
        return tuple(n for n in all_names if n in seen), tuple(unresolved)

    def _bqueues(self) -> str:
        """``bqueues -l`` output, fetched once.

        Both the queue list and the limits are read out of it, and fetching it
        twice means a report could describe two different instants.
        """
        if self._bqueues_cache is None:
            self._bqueues_cache = self.runner.run(["bqueues", "-l"])
        return self._bqueues_cache

    def load_queues(self) -> list[Queue]:
        queues = self.parse_queues(self._bqueues())
        nodes = self.load_nodes()
        all_names = tuple(n.name for n in nodes)
        groups: dict[str, list[str]] = {}
        # Without group expansion, a group-scoped queue resolves to fewer nodes
        # than it claims. That is surfaced as unresolved rather than papered over.
        with contextlib.suppress(Exception):
            groups = self.parse_host_groups(self.runner.run(["bmgroup", "-w"]))

        for q in queues:
            spec = getattr(q, "labels_hosts", "")
            members, unresolved = self.resolve_hosts(spec, all_names, groups)
            q.node_names = members
            # declared_nodes drives Queue.unresolved_nodes, which the report
            # prints as "+N claimed but unresolved".
            q.declared_nodes = len(members) + len(unresolved)
        return queues

    # -- limits -------------------------------------------------------------
    def load_limits(self) -> dict[str, Limits]:
        # Raised rather than turned into "no limits": an empty result cannot be
        # told apart from a cluster whose queues declare no ceilings, which
        # disables the check at the moment it is most needed. Same rule as the
        # Slurm and Kubernetes backends; `Cluster.load` records it.
        text = self._bqueues()
        out: dict[str, Limits] = {}
        for block in re.split(r"^QUEUE:\s*", text, flags=re.MULTILINE)[1:]:
            lines = block.splitlines()
            name = lines[0].strip()
            body = "\n".join(lines[1:])
            per_user: dict[str, int] = {}
            per_job: dict[str, int] = {}
            m = re.search(r"MAX_JOBS_PER_USER[^\d]*(\d+)", body)
            max_jobs = int(m.group(1)) if m else None
            m = re.search(r"PJOB_LIMIT[^\d]*(\d+)", body)
            if m:
                per_job["cpu"] = int(m.group(1))
            m = re.search(r"UJOB_LIMIT[^\d]*(\d+)", body)
            if m:
                per_user["cpu"] = int(m.group(1))
            walltime = None
            m = re.search(r"RUNLIMIT\s*\n\s*([\d.]+)\s*min", body)
            if m:
                walltime = int(float(m.group(1)) * 60)
            out[name] = Limits(
                name=name,
                max_walltime_seconds=walltime,
                per_job=per_job,
                per_user=per_user,
                max_jobs=max_jobs,
                source="lsf queue limits",
            )
        return out

    def load_jobs(self) -> list[Job]:
        """Not implemented for this system.

        The protocol requires the method; an empty list here means "this adapter
        cannot list jobs", which the caller tells apart from "this node has no
        jobs" by asking whether the cluster returned any jobs at all. Reporting
        an idle node because the query does not exist would be phantom capacity
        in a new place.
        """
        return []

    # -- identity -----------------------------------------------------------
    def load_identity(self) -> Identity:
        user = os.environ.get("USER") or getpass.getuser()
        groups: set[str] = set()
        # Committed only if the whole lookup succeeded. A half-collected group
        # list is read downstream as authoritative and yields a false denial;
        # see `slurm._unix_groups` for the same reasoning. Unix groups are LSF's
        # only entitlement signal here, so an empty set means no group
        # restriction can be evaluated at all -- which is what `identity is
        # None` is for, and why this raises rather than returning nothing.
        found: set[str] = set()
        import grp
        import pwd

        for g in grp.getgrall():
            if user in g.gr_mem:
                found.add(g.gr_name)
        found.add(grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name)
        groups = found
        return Identity(user=user, groups=tuple(sorted(groups)))

    def load_node_free_times(self) -> dict[str, datetime]:
        # LSF reports a job's remaining time only with -l per job, which is far
        # too many calls on a busy cluster.  Returning nothing is honest: the
        # report then shows no self-computed estimate rather than a bad one.
        return {}

    # -- probe --------------------------------------------------------------
    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        args = ["-q", queue, "-n", str(shape.total_cpus)]
        if shape.nodes > 1:
            args += ["-R", f"span[ptile={shape.cpus_per_node}]"]
        if shape.gpus_per_node:
            args += ["-gpu", f"num={shape.gpus_per_node}:mode=shared"]
        if shape.memory_gb:
            args += ["-M", str(int(shape.memory_gb * 1024))]
        if shape.walltime_seconds:
            args += ["-W", str(shape.walltime_seconds // 60)]
        return args

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """LSF offers no verify-only submission, so there is nothing to ask."""
        return None

    def format_nodelist(self, names: Iterable[str]) -> str:
        return " ".join(sorted(names))


def _first_word(text: str) -> str:
    parts = text.split()
    return parts[0] if parts else ""


def _param_status(body: str) -> str:
    """Extract STATUS from ``bqueues -l``'s positional PARAM table.

    The value is not labelled -- it sits under a column header::

        PARAM: PRIO NICE STATUS          MAX JL/U ...
               50    20  Open:Active       -    8 ...

    so the header row is used to find the column index.  Searching for
    ``STATUS:`` finds nothing and silently yields "", which reads as a healthy
    queue -- turning ``Open:Inact`` (accepts everything, dispatches nothing)
    into an apparently available one.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if "STATUS" not in line:
            continue
        header = line.replace("PARAM:", " " * 6).split()
        if "STATUS" not in header:
            continue
        idx = header.index("STATUS")
        for follow in lines[i + 1:]:
            values = follow.split()
            if len(values) > idx and ":" in values[idx]:
                return values[idx]
            if values:
                break
    return ""


#: A new ``LABEL:`` in `bqueues -l` output: at least two capitals, then a colon.
#: What ends a wrapped value, and deliberately case-sensitive -- a lowercase
#: host name followed by a colon must not be mistaken for a section header.
_LSF_LABEL = re.compile(r"^[A-Z][A-Z_ ]+:")


def _after(body: str, label: str) -> str:
    """The text following a ``LABEL:`` marker, including wrapped continuations.

    ``bqueues -l`` wraps a long value onto indented following lines, and this
    used to stop at the end of the label's own line -- so a queue whose
    ``USERS:`` list ran past one line had the tail silently dropped. A truncated
    allowlist is then read as authoritative and produces a **false denial**: the
    tool reporting no access to a queue that would have taken the job. Same
    defect the PBS backend had in its own dialect; see `pbs._unwrap`.

    A continuation is an indented line that does not begin a new label. Values
    are joined with a space, because LSF breaks these lists at whitespace.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        m = re.search(re.escape(label) + r"(.*)", line)
        if not m:
            continue
        parts = [m.group(1).strip()]
        for nxt in lines[i + 1:]:
            if not nxt[:1].isspace() or _LSF_LABEL.match(nxt.strip()):
                break
            parts.append(nxt.strip())
        return " ".join(p for p in parts if p).strip()
    return ""


def _tokens(text: str) -> list[str]:
    return [t.strip("/") for t in re.split(r"[,\s]+", text) if t.strip() and t != "all"]


def _mem_to_mb(text: str | None) -> int:
    if not text:
        return 0
    m = re.match(r"^\s*([\d.]+)\s*([KMGTP]?)", str(text), re.IGNORECASE)
    if not m:
        return 0
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024**3}
    return int(float(m.group(1)) * scale.get(m.group(2).upper(), 1))
