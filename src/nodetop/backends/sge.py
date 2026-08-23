"""Grid Engine (SGE / UGE / Son of Grid Engine / Altair Grid Engine).

Grid Engine *does* have a verify-only submission: ``qsub -w v`` runs the
scheduler's assignment logic and reports whether the job could ever run,
without queuing anything.  That makes it one of the three systems where nodetop
can confirm an entitlement rather than merely relay a declared one.

Its own trap is different from the others: work is scheduled onto *queue
instances* (``all.q@node001``), each with its own state, so a queue can be
perfectly healthy while every one of its instances is disabled.  The disabled
instances still appear in ``qhost`` output with their slot counts intact.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
from collections.abc import Iterable
from datetime import datetime

from ..core.hardware import identify_accelerator
from ..core.model import (
    Identity,
    Job,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)
from ..runner import Runner, SubprocessRunner, which
from .base import BackendCapabilities

__all__ = ["SgeBackend"]

#: Grid Engine queue-instance state letters that block new work.
_STATE_LETTERS = {
    "u": "DOWN",       # unreachable: sge_execd not responding
    "d": "DRAIN",      # disabled by an administrator
    "D": "DRAIN",      # disabled by calendar
    "E": "FAIL",       # error state
    "s": "DRAIN",      # suspended
    "S": "DRAIN",      # suspended by calendar
    "C": "DRAIN",      # suspended by calendar (threshold)
    "o": "UNKNOWN",    # orphaned
}


def _mem_to_mb(text: str | None) -> int:
    if not text or text in {"-", ""}:
        return 0
    m = re.match(r"^\s*([\d.]+)\s*([KMGTP]?)", str(text), re.IGNORECASE)
    if not m:
        return 0
    scale = {"": 1 / (1024 * 1024), "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024,
             "P": 1024**3}
    return int(float(m.group(1)) * scale.get(m.group(2).upper(), 1))


def _sge_seconds(text: str | None) -> int | None:
    """Grid Engine ``h_rt`` is ``HH:MM:SS`` or ``INFINITY``."""
    if not text or text.strip().upper() in {"INFINITY", "NONE", ""}:
        return None
    parts = text.strip().split(":")
    if not all(p.isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0)
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


class SgeBackend:
    """Adapter for Grid Engine."""

    name = "sge"
    queue_term = "queue"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()
        self._nodes: list[Node] | None = None
        self._skipped_hosts: list[str] = []

    @classmethod
    def detect(cls) -> bool:
        # qhost is Grid Engine specific; qstat alone is shared with PBS.
        return which("qhost") and which("qconf")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=which("qsub"),
            probe_supported=True,
            probe_command="qsub -w v (verify only)",
            free_times=False,
            notes=(
                "qsub -w v verifies assignment but not per-user RQS quotas",
                "work runs on queue INSTANCES (queue@host); all of them can be disabled",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_nodes(self, qhost: str, qstat: str = "") -> list[Node]:
        """Parse ``qhost -q -F`` output.

        Layout is a host header line, then indented queue-instance lines, then
        (with ``-F``) indented ``hl:``/``hc:`` resource lines.

        Columns are located by *name* from the header row rather than by index:
        Grid Engine versions ship different column sets (older builds have no
        NSOC/NCOR/NTHR), so a fixed offset silently reads the wrong field.
        """
        instance_state = _parse_instance_states(qstat)

        out: list[Node] = []
        columns: list[str] = []
        name: str | None = None
        header: list[str] = []
        queues: list[str] = []
        resources: dict[str, str] = {}
        states: set[str] = set()

        def col(field: str) -> str:
            """Value of a named qhost column for the current host."""
            if field in columns:
                idx = columns.index(field)
                if idx < len(header):
                    return header[idx]
            return ""

        def flush() -> None:
            if name is None or name == "global":
                return
            ncpu = int(float(col("NCPU"))) if _num(col("NCPU")) else 0
            memtot = _mem_to_mb(col("MEMTOT"))
            memuse = _mem_to_mb(col("MEMUSE"))
            # Accelerator accounting in Grid Engine needs care.  `qhost -F`
            # reports `hc:gpu` = the amount of the consumable still AVAILABLE;
            # the configured total lives in the host complex (`qconf -se`) and
            # is not in this output at all.
            #
            # So when only `hc:` is known, it is used as both the free count
            # and the total.  Free is then exact and the total is a floor,
            # which understates capacity -- the safe direction.  Assuming the
            # host is idle would overstate it and invent room that is not there.
            avail = resources.get("_avail_gpu")
            declared = resources.get("gpu") or resources.get("fixed_gpu")
            if declared is not None:
                gpus = int(float(declared))
                gpus_used = (
                    max(0, gpus - int(float(avail))) if avail is not None else 0
                )
            elif avail is not None:
                gpus = int(float(avail))
                gpus_used = 0
            else:
                gpus, gpus_used = 0, 0
            model = next(
                (v for k, v in resources.items()
                 if k.removeprefix("fixed_").removeprefix("_avail_")
                 in {"gpu_model", "gputype", "gpu_type", "accelerator"} and v),
                "",
            )
            letters = instance_state.get(name, "")
            for ch in letters:
                if ch in _STATE_LETTERS:
                    states.add(_STATE_LETTERS[ch])
            out.append(
                Node(
                    name=name,
                    state_raw=letters or "ok",
                    conditions=frozenset(states),
                    cpus_total=ncpu,
                    # qhost reports LOAD, not allocated slots; slot occupancy
                    # comes from the queue-instance line "used/resv/total".
                    cpus_alloc=int(resources.get("_slots_used", 0) or 0),
                    memory_mb=memtot,
                    memory_alloc_mb=memuse,
                    gpus_total=gpus,
                    gpus_alloc=gpus_used,
                    accelerator=identify_accelerator(
                        None, model or [f"{k}={v}" for k, v in resources.items()]
                    ),
                    labels=tuple(f"{k}={v}" for k, v in resources.items()),
                    queues=tuple(queues),
                    unreachable="u" in letters,
                    reason=(
                        "queue instance disabled by an administrator"
                        if "d" in letters
                        else "queue instance in error state"
                        if "E" in letters
                        else ""
                    ),
                )
            )

        for raw in qhost.splitlines():
            if raw.startswith("HOSTNAME"):
                columns = raw.split()
                continue
            if not raw.strip() or raw.startswith("-"):
                continue
            if not raw[0].isspace():
                flush()
                header = raw.split()
                name = header[0]
                queues, resources, states = [], {}, set()
                continue
            line = raw.strip()
            # Resource line: "hl:gpu=4.000000" / "hc:gpu=2.000000"
            m = re.match(r"^(?P<scope>h[lcf]):(?P<key>[^=]+)=(?P<val>\S+)$", line)
            if m:
                key = m.group("key").strip()
                value = m.group("val")
                # Trim Grid Engine's "4.000000" to "4" -- but only for numbers.
                # Applying rstrip("0") unconditionally turns the model name
                # "A100" into "A1", which then identifies as nothing.
                if _num(value):
                    value = value.rstrip("0").rstrip(".") or "0"
                # hl: is the host's total; hc: is the amount of a consumable
                # still AVAILABLE, not the amount used.
                prefix = {"hl": "", "hc": "_avail_", "hf": "fixed_"}[m.group("scope")]
                resources[f"{prefix}{key}"] = value
                continue
            # Queue-instance line: "all.q  BIP  0/2/32"
            parts = line.split()
            if parts and "/" in (parts[-1] if parts else ""):
                queues.append(parts[0])
                triple = parts[-1].split("/")
                # The triple is resv/used/total, so the *middle* field is the
                # occupancy; reading the first gives reservations instead.
                if len(triple) == 3 and triple[1].isdigit():
                    resources["_slots_used"] = triple[1]
        flush()
        return out

    #: Cap on per-host ``qconf -se`` calls. Grid Engine has no bulk equivalent,
    #: so a very large accelerator estate would cost one round trip per host.
    #: Whatever is skipped is reported rather than silently dropped.
    MAX_HOST_QUERIES = 96

    def parse_complex_values(self, text: str) -> dict[str, int]:
        """``complex_values  gpu=4,slots=32`` from one ``qconf -se`` record."""
        out: dict[str, int] = {}
        for raw in text.splitlines():
            m = re.match(r"^\s*complex_values\s+(?P<body>\S.*)$", raw)
            if not m or m.group("body").strip().upper() == "NONE":
                continue
            for item in m.group("body").split(","):
                key, _, value = item.partition("=")
                v = value.strip().rstrip("KMGT")
                if v.replace(".", "", 1).isdigit():
                    out[key.strip()] = int(float(v))
        return out

    def accelerator_totals(self, hosts: list[str]) -> tuple[dict[str, int], list[str]]:
        """Configured accelerator count per host, and the hosts left unqueried.

        ``qhost -F`` reports only how much of a consumable is still
        *available*; the configured total lives in the exec host definition.
        Without it a fully busy GPU host reads as ``0/0``, which makes it look
        like a CPU-only machine and quietly removes its accelerators from the
        inventory entirely.
        """
        totals: dict[str, int] = {}
        skipped: list[str] = []
        for i, host in enumerate(hosts):
            if i >= self.MAX_HOST_QUERIES:
                skipped.append(host)
                continue
            try:
                values = self.parse_complex_values(
                    self.runner.run(["qconf", "-se", host])
                )
            except Exception:
                skipped.append(host)
                continue
            if "gpu" in values:
                totals[host] = values["gpu"]
            else:
                skipped.append(host)
        return totals, skipped

    def load_nodes(self) -> list[Node]:
        # Cached: load_queues needs the nodes too, and re-deriving them would
        # double the per-host qconf round trips.
        if self._nodes is not None:
            return self._nodes
        qhost = self.runner.run(["qhost", "-q", "-F"])
        qstat = ""
        with contextlib.suppress(Exception):
            qstat = self.runner.run(["qstat", "-f"])
        nodes = self.parse_nodes(qhost, qstat)

        # Ask only about hosts that look like accelerator hosts: they either
        # advertise a gpu consumable or their labels name a model.
        candidates = [
            n.name for n in nodes
            if n.gpus_total or n.accelerator is not None
        ]
        if candidates:
            totals, skipped = self.accelerator_totals(candidates)
            for n in nodes:
                if n.name in totals:
                    available = n.gpus_total  # what hc: reported as free
                    n.gpus_total = totals[n.name]
                    n.gpus_alloc = max(0, totals[n.name] - available)
                elif n.name in skipped and n.accelerator is not None:
                    n.reason = n.reason or (
                        "accelerator count unknown: qconf -se gave no complex_values"
                    )
            self._skipped_hosts = skipped
        self._nodes = nodes
        return nodes

    # -- queues -------------------------------------------------------------
    def parse_queue(self, name: str, text: str) -> Queue:
        f: dict[str, str] = {}
        for raw in text.splitlines():
            m = re.match(r"^(?P<key>\S+)\s+(?P<val>.*)$", raw.strip())
            if m:
                f[m.group("key")] = m.group("val").strip()
        users = f.get("user_lists", "NONE")
        xusers = f.get("xuser_lists", "NONE")
        return Queue(
            name=name,
            state_raw=f.get("qtype", "BIP"),
            enabled=True,   # per-queue; instance states are on the nodes
            started=True,
            allow_users=() if users.upper() in {"NONE", ""} else tuple(_tokens(users)),
            deny_users=() if xusers.upper() in {"NONE", ""} else tuple(_tokens(xusers)),
            max_walltime_seconds=_sge_seconds(f.get("h_rt")),
            node_names=(),
            limits_name=name,
            priority=int(f.get("priority", "0") or 0),
        )

    def load_queues(self) -> list[Queue]:
        names = [n.strip() for n in self.runner.run(["qconf", "-sql"]).splitlines() if n.strip()]
        nodes = self.load_nodes()
        queues: list[Queue] = []
        for name in names:
            try:
                q = self.parse_queue(name, self.runner.run(["qconf", "-sq", name]))
            except Exception:
                q = Queue(name=name, limits_name=name)
            members = tuple(n.name for n in nodes if name in n.queues)
            q.node_names = members
            q.declared_nodes = len(members)
            queues.append(q)
        return queues

    # -- limits -------------------------------------------------------------
    def load_limits(self) -> dict[str, Limits]:
        """Read resource quota sets, Grid Engine's per-user ceiling mechanism."""
        out: dict[str, Limits] = {}
        try:
            names = [
                n.strip()
                for n in self.runner.run(["qconf", "-srqsl"]).splitlines()
                if n.strip()
            ]
        except Exception:
            return out
        for name in names:
            # No `continue` on failure. Skipping one resource quota set yields a
            # PARTIAL ceiling map that is then applied as though complete, so a
            # job over the ceiling nobody could read is reported as fitting. The
            # outer failure above is different and stays tolerated: `qconf
            # -srqsl` exits non-zero on a cluster with no RQS defined at all,
            # which is a real state rather than a failed query.
            text = self.runner.run(["qconf", "-srqs", name])
            per_user: dict[str, int] = {}
            # Anchor on the "limit ... to <resources>" line. Matching a bare
            # " to " would also fire on a description that happens to contain
            # the word, and parse prose as a quota.
            for m in re.finditer(r"^\s*limit\b.*?\bto\s+(?P<body>.+)$", text,
                                 re.MULTILINE):
                for item in m.group("body").split(","):
                    k, _, v = item.partition("=")
                    key = {"slots": "cpu", "gpu": "gpu"}.get(k.strip())
                    if key and v.strip().isdigit():
                        per_user[key] = int(v.strip())
            if per_user:
                out[name] = Limits(name=name, per_user=per_user, source="sge RQS")
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
        # Grid Engine userset lists are the entitlement mechanism; membership
        # in one is what a queue's user_lists refers to.
        # Built into a local and committed only when the whole sweep finished,
        # then raised on failure rather than returned half-done. Both halves
        # matter and they fail in opposite directions:
        #
        # * a PARTIAL list looks authoritative, so the tri-state membership
        #   check returns a verdict from it -- "you are in none of the permitted
        #   usersets" -- on the strength of a scan that did not complete. That is
        #   a false denial, and it hides a queue the caller can submit to.
        # * an EMPTY list reads as "cannot tell", so every userset restriction
        #   is silently ignored and queues are claimed that will refuse the job.
        #
        # Raising is the only answer that is neither: Cluster.load records it and
        # leaves `identity` as None, which the entitlement filter treats as
        # "cannot filter". Same rule as the Slurm and Kubernetes backends.
        found: set[str] = set()
        for uset in self.runner.run(["qconf", "-sul"]).splitlines():
            uset = uset.strip()
            if not uset:
                continue
            body = self.runner.run(["qconf", "-su", uset])
            if re.search(rf"\b{re.escape(user)}\b", body):
                found.add(uset)
        groups = found
        return Identity(user=user, groups=tuple(sorted(groups)))

    def load_node_free_times(self) -> dict[str, datetime]:
        return {}

    # -- probe --------------------------------------------------------------
    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        args = ["-q", queue]
        if shape.total_cpus > 1:
            args += ["-pe", "smp", str(shape.cpus_per_node)]
        resources = []
        if shape.walltime_seconds:
            h, rem = divmod(shape.walltime_seconds, 3600)
            m, s = divmod(rem, 60)
            resources.append(f"h_rt={h:02d}:{m:02d}:{s:02d}")
        if shape.memory_gb:
            resources.append(f"h_vmem={int(shape.memory_gb)}G")
        if shape.gpus_per_node:
            resources.append(f"gpu={shape.gpus_per_node}")
        if resources:
            args += ["-l", ",".join(resources)]
        return args

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """Read-only: ``-w v`` verifies assignment and queues nothing.

        The flag is hard-coded so a caller cannot omit it and submit for real.
        """
        if not self.capabilities().probe:
            # The declaration is gated on qsub existing. Answering anyway would
            # make the object contradict itself, and any caller that trusts
            # capabilities() over trying it would get a verdict the backend
            # said it could not produce.
            return None
        cmd = ["qsub", "-w", "v", *self.submit_flags(queue, shape), "-b", "y", "/bin/true"]
        try:
            rc, out, err = self.runner.run_full(cmd)
        except Exception as exc:
            return Verdict(
                queue=queue, account=account, allowed=False,
                category=VerdictCategory.CONTROL_PLANE_DOWN, reason=str(exc), raw=str(exc),
            )
        text = f"{out}\n{err}"
        low = text.lower()
        # "verification: found suitable queue(s)" is the success phrasing.
        if "found suitable queue" in low or "verification: ok" in low:
            return Verdict(queue=queue, account=account, allowed=True,
                           category=VerdictCategory.OK, raw=text.strip())
        category = VerdictCategory.UNKNOWN
        if "no suitable queue" in low or "cannot run in queue" in low:
            category = VerdictCategory.SHAPE_UNAVAILABLE
        elif "no permission" in low or "not allowed to submit" in low:
            category = VerdictCategory.NOT_ENTITLED
        elif "unknown queue" in low or "does not exist" in low:
            category = VerdictCategory.UNKNOWN_QUEUE
        elif "unable to contact" in low or "unable to send message" in low:
            category = VerdictCategory.CONTROL_PLANE_DOWN
        return Verdict(
            queue=queue, account=account, allowed=False, category=category,
            reason=text.strip().splitlines()[-1] if text.strip() else "no output",
            raw=text.strip(),
        )

    def format_nodelist(self, names: Iterable[str]) -> str:
        return ",".join(sorted(names))


#: The complete set of Grid Engine queue-instance state letters.  Validating
#: against it is essential: the states column is optional, so the field that
#: follows the load average is often the ARCH string -- and "lx-amd64" contains
#: both "d" and "u", which would otherwise read as disabled-and-unreachable on
#: every healthy host.
_STATE_ALPHABET = set("aAcdDsSuEC")


def _parse_instance_states(qstat: str) -> dict[str, str]:
    """Host -> queue-instance state letters, from ``qstat -f``.

    Line shape is ``queue@host qtype resv/used/tot load arch [states]``, with
    the trailing states field present only when non-empty.
    """
    out: dict[str, str] = {}
    for line in qstat.splitlines():
        if "@" not in line or line.startswith(("-", "queuename", " ")):
            continue
        fields = line.split()
        if len(fields) < 5 or "/" not in fields[2]:
            continue
        host = fields[0].split("@", 1)[1]
        letters = fields[5] if len(fields) >= 6 else ""
        if letters and set(letters) <= _STATE_ALPHABET:
            out[host] = letters
    return out


def _tokens(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]


def _num(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
