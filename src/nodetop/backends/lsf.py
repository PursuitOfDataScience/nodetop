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

from ..core.hardware import identify_accelerator, name_accelerator
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
    # `bhosts -w` prints these two as well, and both mean the host runs nothing
    # right now: `closed_LIM` is a host whose sbatchd the master cannot reach
    # (so it is down for scheduling purposes however healthy its LIM looks),
    # and `closed_Wind` is one shut by its own run window.  Neither was in the
    # table, and see `_condition_for` for what the omission cost.
    "closed_lim": "DOWN",
    "closed_wind": "DRAIN",
    "ok": None,
}


def _condition_for(status: str) -> str | None:
    """One lowercased ``bhosts`` STATUS as a neutral condition.

    The lookup used to be ``_STATUS_TO_CONDITION.get(status)``, and ``None`` is
    the table's word for *nothing wrong here* -- the answer it gives ``ok``.
    So every status the table did not list read as a **healthy host with its
    whole complement free**, which is the wrong direction for the one thing
    this tool exists to catch.  ``closed_LIM`` and ``closed_Wind`` are real
    ``bhosts -w`` values and were both missing; so is anything a newer LSF or a
    site's wrapper prints, and a 40-slot host reported idle is 40 phantom slots
    each time.

    An unrecognised status is not evidence of health, so it degrades to
    ``UNKNOWN`` -- "state cannot be determined", which is exactly the claim
    being made -- and the raw string is carried through in ``state_raw`` and
    the node's reason so the reader can see what LSF actually said.  The two
    deliberate ``None`` entries (``closed_Full``, ``closed_Busy``: busy, not
    broken) are in the table and keep their answer.
    """
    if status in _STATUS_TO_CONDITION:
        return _STATUS_TO_CONDITION[status]
    return "UNKNOWN"


def _int(value: str | None) -> int:
    if not value:
        return 0
    m = re.match(r"^\s*(\d+)", str(value).replace("-", "0"))
    return int(m.group(1)) if m else 0


def _dispatched(fields: list[str], njobs: int, run: int) -> int:
    """Slots (or cards) held here, from ``bhosts``'s NJOBS column -- not RUN.

    ``bhosts`` prints ``HOST_NAME STATUS JL/U MAX NJOBS RUN SSUSP USUSP RSV``,
    and LSF documents **NJOBS as the slots used by every job DISPATCHED to the
    host** -- running, suspended and reserved -- while ``RUN`` counts only the
    ones currently executing.  The occupancy was read from ``RUN``, so a host's
    ``SSUSP``, ``USUSP`` and ``RSV`` slots were all reported as *free capacity*:

    * a suspended job has not given anything back.  LSF keeps its slots, its
      memory and its GPUs, and resumes it in place -- and LSF preemption
      SUSPENDS rather than requeues by default, so on any cluster with
      preemption configured this is the ordinary state of a busy host, not a
      corner case.  A 40-slot host with 40 preempted slots read as **empty**.
    * ``RSV`` slots are held by the scheduler for a pending job, usually a
      large parallel one that is being backfilled around.  Handing them out is
      how that job never starts.

    Both directions of the same failure this tool exists to catch: capacity on
    the screen that no job can have.

    ``max`` with RUN rather than NJOBS alone because :func:`_int` reads LSF's
    ``-`` as ``0``: a row whose NJOBS column is dashed or missing must not
    report a host running 40 jobs as idle.
    """
    return max(
        _int(fields[i]) if len(fields) > i else 0 for i in (njobs, run)
    )


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
        # Resolved once per parse, not once per host row: `lshosts`'s sizes are
        # all scaled by the same cluster-wide setting. Skipped altogether when
        # there is no `lshosts` output to interpret.
        bare_mb = _site_unit_mb() if lshosts.strip() else None
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
            # bhosts -gpu: HOST_NAME GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV
            if len(f) >= 3 and not f[0].upper().startswith("HOST"):
                total, used, model = gpus.get(f[0], (0, 0, ""))
                # NJOBS, not RUN: see `_dispatched`. A suspended job keeps the
                # card AND the memory on it -- `MUSED` stays non-zero, which is
                # the same reading from the other direction -- so counting only
                # RUN hands every preempted job's GPU back as free.
                busy = 1 if _dispatched(f, 5, 6) > 0 else 0
                gpus[f[0]] = (total + 1, used + busy, model or f[2])

        out: list[Node] = []
        for line in bhosts.splitlines():
            f = line.split()
            if len(f) < 6 or f[0].upper().startswith("HOST"):
                continue
            name, status = f[0], f[1].lower()
            condition = _condition_for(status)
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
                    # NJOBS, not RUN. See `_dispatched`: a suspended job keeps
                    # its slots, so RUN alone reported every SSUSP/USUSP slot as
                    # free capacity.
                    cpus_alloc=_dispatched(f, 4, 5),
                    memory_mb=_mem_to_mb(hw.get("maxmem"), bare_mb),
                    gpus_total=g_total,
                    gpus_alloc=g_used,
                    accelerator=identify_accelerator(
                        None, ",".join(x for x in (g_model, hw.get("model", ""), resources) if x)
                    ),
                    accelerator_label=name_accelerator(
                        None, ",".join(x for x in (g_model, hw.get("model", ""), resources) if x)
                    ) or "",
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
        # Read once per load, not once per queue -- the same reason
        # `parse_nodes` resolves it once per parse.
        site_unit = _site_unit_mb()
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
            # The memory ceiling, on the terms `_memlimit_is_per_job` sets out.
            # Absent: silent, or every queue on a cluster that sets no MEMLIMIT
            # gains a warning. Declared and job-scoped: the per-job axis, which
            # `Limits.blockers` already checks and which nothing filled before,
            # so a queue publishing a memory ceiling produced zero blockers for
            # a job asking many times it. Declared and NOT comparable -- a
            # per-process ceiling, an unreadable `lsf.conf`, a size this cannot
            # parse, or the combined table `_memlimit_text` will not guess at --
            # named in `unreadable` rather than converted anyway.
            #
            # The two not-comparable causes get DIFFERENT entries, because
            # `fit.py` spells this field into "could not read {entry} on ...".
            # A per-process MEMLIMIT was read perfectly well -- what could not be
            # read is a *job-wide* ceiling -- so a bare "MEMLIMIT" there makes
            # that sentence untrue, and untrue in the direction that sends an
            # admin looking for a parse bug instead of at `LSB_JOB_MEMLIMIT`.
            # Every other producer of this field (`pbs.resources_max.walltime`,
            # Slurm's `MaxWall`) names a genuine read failure, and the ones below
            # that really are unreadable still say just "MEMLIMIT".
            unreadable: tuple[str, ...] = ()
            memlimit = _memlimit_text(body)
            if memlimit is not None:
                mem_mb = _mem_to_mb(memlimit, bare_mb=site_unit) if memlimit else 0
                if mem_mb > 0 and _memlimit_is_per_job():
                    per_job["mem_mb"] = mem_mb
                elif mem_mb > 0:
                    unreadable = ("MEMLIMIT as a job-wide ceiling (LSB_JOB_MEMLIMIT is not set)",)
                else:
                    unreadable = ("MEMLIMIT",)
            out[name] = Limits(
                name=name,
                max_walltime_seconds=walltime,
                per_job=per_job,
                per_user=per_user,
                max_jobs=max_jobs,
                source="lsf queue limits",
                unreadable=unreadable,
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


#: MB per suffix letter.  LSF's suffixes are binary and case does not matter
#: (LSF prints them upper case), which is the ordinary convention -- unlike
#: `sge._mem_to_mb`, where case selects the base.
#:
#: There is deliberately no ``""`` row.  A bare size carries no unit in the
#: string at all, so what it means is `_site_unit_mb`'s question rather than
#: this table's -- and a ``1`` here is how that question got answered by
#: assumption.  There is no ``E`` row either: ``EB`` is a documented
#: ``LSF_UNIT_FOR_LIMITS`` value, so an unrecognised unit has to answer
#: "unknown" rather than guess at 1024**4.
_LSF_SCALE = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024**3}

#: An *active* ``LSF_UNIT_FOR_LIMITS`` assignment, i.e. one that is not
#: commented out.  ``lsf.conf`` ships most of its parameters
#: present-but-commented, so an unanchored search finds
#: ``#LSF_UNIT_FOR_LIMITS=GB`` and reports a setting the site never made.
_LSF_UNIT_RE = re.compile(
    r"^[ \t]*LSF_UNIT_FOR_LIMITS[ \t]*=[ \t]*[\"]?[ \t]*([A-Za-z]+)",
    re.MULTILINE,
)

#: An *active* ``LSB_JOB_MEMLIMIT`` assignment -- the parameter that decides
#: whether ``MEMLIMIT`` is a per-job or a per-process ceiling.  Same
#: line-anchored shape as `_LSF_UNIT_RE` and for the same reason: ``lsf.conf``
#: ships this one commented out, and ``#LSB_JOB_MEMLIMIT=Y`` is not a setting.
_LSB_JOB_MEMLIMIT_RE = re.compile(
    r"^[ \t]*LSB_JOB_MEMLIMIT[ \t]*=[ \t]*[\"]?[ \t]*([A-Za-z]+)",
    re.MULTILINE,
)

#: Where LSF looks for ``lsf.conf`` when ``LSF_ENVDIR`` is unset: the symlink
#: the installer leaves in ``/etc`` pointing into ``LSF_CONFDIR``.  Named
#: rather than inlined so the suite stays hermetic -- a test asserting "no
#: readable setting" must not pick up the real one on a host that has LSF
#: installed, the same reason `conftest._isolated_cache` exists.
_LSF_CONF_FALLBACK = "/etc/lsf.conf"


def _site_unit_mb() -> float | None:
    """MB per *bare* LSF size, read from ``lsf.conf`` -- or ``None``.

    A bare LSF size has no unit in it.  LSF takes the unit from
    ``LSF_UNIT_FOR_LIMITS`` in ``lsf.conf``, a cluster-wide setting that
    governs the display of sizes in ``lshosts``, ``bhosts``, ``bqueues`` and
    the rest.  It is not something the string can be asked: it has to be looked
    up or given up on, and the gap between the two plausible answers is 1024x.

    So it is looked up where LSF itself looks -- ``$LSF_ENVDIR/lsf.conf``, then
    the ``/etc/lsf.conf`` symlink the installer leaves pointing into
    ``LSF_CONFDIR``.  Any client that could run ``lshosts`` at all had to read
    one of those, so on a real LSF host this normally succeeds.

    ``None`` means the setting could not be read: no such file, no active
    assignment, or a unit `_LSF_SCALE` has no row for (``EB``).  That is
    deliberately *not* the same answer as MB -- see `_mem_to_mb`.
    """
    unit = _lsf_conf_setting(_LSF_UNIT_RE)
    return _LSF_SCALE.get(unit[:1].upper()) if unit is not None else None


def _lsf_conf_setting(pattern: re.Pattern[str]) -> str | None:
    """The first active assignment `pattern` finds in ``lsf.conf`` -- or ``None``.

    Looked up where LSF itself looks: ``$LSF_ENVDIR/lsf.conf``, then the
    ``/etc/lsf.conf`` symlink the installer leaves pointing into
    ``LSF_CONFDIR``.  Shared by the two callers rather than copied, because
    both want the same "the file, or give up" answer and only the parameter
    differs.

    ``None`` means the parameter could not be read at all -- no such file, or
    no *active* assignment in any of them.  Neither caller reads that as the
    product default, because the product defaults are exactly the expensive
    guesses: 1024x on the unit (`_site_unit_mb`) and the whole *scope* of a
    memory ceiling (`_memlimit_is_per_job`).
    """
    paths: list[str] = []
    envdir = os.environ.get("LSF_ENVDIR")
    if envdir:
        paths.append(os.path.join(envdir, "lsf.conf"))
    paths.append(_LSF_CONF_FALLBACK)
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            # Absent, unreadable, a directory: all of them "could not be read".
            continue
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def _memlimit_is_per_job() -> bool:
    """Whether a queue's ``MEMLIMIT`` bounds the whole job or one process.

    **LSF enforces ``MEMLIMIT`` per process by default.**  ``LSB_JOB_MEMLIMIT=Y``
    in ``lsf.conf`` is what makes it a job-wide ceiling -- LSF then compares the
    limit against the summed memory of every process in the job, which is the
    quantity `Limits.blockers` computes for ``mem_mb``
    (``memory_gb * 1024 * nodes``).  Without it the published number bounds one
    process out of an unknown number, and nothing in a queue definition says
    how many there will be.

    So the setting is not cosmetic and it is not a rounding difference: filling
    the job-wide axis from a per-process ceiling understates it by the process
    count, which invents a blocker for a job that would have run -- the false
    denial `resolve_hosts` and `slurm._unix_groups` are both written against,
    pointed at the ceiling check instead of at access.  ``False`` (including
    "could not read ``lsf.conf``") therefore means *this ceiling is not
    comparable*, and `LsfBackend.load_limits` records it in `Limits.unreadable`
    rather than either dropping it or converting it anyway.
    """
    value = _lsf_conf_setting(_LSB_JOB_MEMLIMIT_RE)
    return value is not None and value[:1].upper() == "Y"


#: A limit label in a ``bqueues -l`` table: ``MEMLIMIT``, ``RUNLIMIT``, ``CORELIMIT``.
_LSF_LIMIT_LABEL = re.compile(r"[A-Z][A-Z0-9_]+")


def _memlimit_text(body: str) -> str | None:
    """The ``MEMLIMIT`` cell from a ``bqueues -l`` queue block, or ``None``.

    ``None`` means the queue declares no memory ceiling.  The empty string means
    one is declared in a shape this cannot read, which the caller turns into a
    `Limits.unreadable` entry rather than a number -- see `load_limits`.

    ``bqueues -l`` prints a limit two ways.  Alone on its line, the value is on
    the next line and unambiguous::

        MEMLIMIT
         4194304 K

    In a *combined* table it shares a header row with its neighbours, and IBM's
    own example has five limits in one row::

        FILELIMIT   DATALIMIT   STACKLIMIT  CORELIMIT   MEMLIMIT
           8000 K    10000 K      2000 K     20000 K     5000 K

    Reading the next line's first size there returns ``8000 K`` -- ``FILELIMIT``'s
    value, off by whatever ratio separates two unrelated ceilings.  Recovering the
    right cell needs the header's character columns, and the values are
    right-aligned under labels of differing width, so the column rule cannot be
    checked without a recorded sample of that layout.  There is none: this
    package's ``bqueues -l`` fixture declares only ``RUNLIMIT``, and no LSF
    cluster is reachable from here to record one.

    So the combined form answers ``""``.  Guessing a cell is the one outcome the
    module rules out for this field: `_memlimit_is_per_job` already says that
    filling the job-wide axis from the wrong figure "invents a blocker for a job
    that would have run".  A ceiling named as unreadable keeps the gap visible;
    a wrong one is a false denial.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        labels = _LSF_LIMIT_LABEL.findall(line)
        if "MEMLIMIT" not in labels:
            continue
        if len(labels) > 1:
            return ""
        for follow in lines[i + 1:]:
            if follow.strip():
                return follow.strip()
        return ""
    return None


def _mem_to_mb(text: str | None, bare_mb: float | None = None) -> int:
    """LSF memory sizes: ``256G``, ``2016M``, ``1000000 K``, bare numbers.

    A suffixed size says what it means, and is read exactly as it was before.

    **A bare number does not, and it used to be assumed to be MB.** LSF takes
    the unit for a bare size from ``LSF_UNIT_FOR_LIMITS`` in ``lsf.conf``: MB
    by default in LSF 10.1, but ``KB`` was the default in older releases and
    sites set ``GB`` as well, and nothing in the string says which. Reading a
    ``KB`` site's number as MB overstates memory **1024x** -- the largest wrong
    answer available in this backend, and in the direction this tool exists to
    catch, since a node that looks 1024x bigger than it is accepts jobs that
    cannot run on it.

    The assumption was left standing as unexercised, on the grounds that the
    one caller -- ``lshosts``'s ``maxmem`` -- always prints a suffix. That is
    an inference from IBM's example output, not a documented guarantee: the
    ``lshosts`` reference says only that ``maxmem`` "is displayed in KB by
    default" and that ``LSF_UNIT_FOR_LIMITS`` selects "a larger unit", and the
    admin guide adds that in command output "the larger unit appears as T, G, P
    or E". Nothing documents a letter for the *base* unit. Betting 1024x on an
    undocumented "always" is the wrong side of that trade, so the bet is off.

    The unit is looked up instead (`_site_unit_mb`, resolved by
    `LsfBackend.parse_nodes` so the file is read once per parse rather than
    once per host), and when it cannot be read the answer is **0 = "not
    read"** -- this package's word for a value that was published and could not
    be interpreted, the same one `Limits.unreadable` carries for a ceiling and
    the one `capacity.hardware_ok` already handles by gating on
    ``memory_mb > 0``. A node whose RAM nobody could read is then reported as
    "would queue" rather than credited with 1024x the memory it has.

    The digit group takes at most one decimal point. ``[\\d.]+`` also matched
    ``1.2.3.4``, which `float` then rejected with an uncaught `ValueError`: one
    malformed ``maxmem`` field raised straight out of `LsfBackend.parse_nodes`
    and emptied the entire node listing, which this tool reports as "wrong
    backend, or the control plane is down" -- a misdiagnosis, not a gap. The
    match is anchored for the same reason PBS's is: an unreadable shape answers
    0 = "not read" instead of silently becoming its leading prefix.
    """
    if not text:
        return 0
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)\s*$", str(text), re.IGNORECASE)
    if not m:
        return 0
    unit = m.group(2).upper()
    # A suffix answers for itself; a bare number needs the site setting, and
    # `None` there means unknown -- which is 0, not MB.
    scale = _LSF_SCALE[unit] if unit else bare_mb
    if scale is None:
        return 0
    return int(float(m.group(1)) * scale)
