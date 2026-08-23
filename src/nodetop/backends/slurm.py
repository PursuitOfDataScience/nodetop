"""Slurm.

The reference backend, and the only one validated against a live cluster
during development.  Slurm is also the system with the richest dry-run, which
is why it is the one place nodetop can *confirm* an entitlement rather than
merely report a declared one.

Two Slurm-specific hazards are handled here rather than in the core:

* **A compound node state.**  ``DOWN*+DRAIN`` is a base state, a
  "not responding" marker and a flag fused into one token, so code comparing
  the whole string to ``"IDLE"`` gets every mixed case wrong.
* **A two-layer submit verdict.**  A site ``job_submit`` plugin prints its own
  PASSED/REJECTED block and Slurm's core then runs its own checks, and the two
  disagree.  Observed verbatim on a production cluster::

      sbatch: error: Verification: ***PASSED***
      allocation failure: Invalid account or account/partition combination specified

  Reading only the plugin line concludes the opposite of the truth.
"""

from __future__ import annotations

import getpass
import grp
import os
import re
from collections.abc import Iterable
from datetime import datetime
from typing import TypedDict

from ..core.duration import parse_timestamp
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
from ..hostlist import collapse, expand
from ..runner import Runner, SubprocessRunner, which
from .base import BackendCapabilities

__all__ = ["SlurmBackend", "parse_slurm_duration"]

# scontrol --oneliner emits "Key=Value Key=Value"; values may contain spaces
# (Reason=, OS=), so a field runs until the next "Key=" or end of line.
_NODE_FIELD = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_/]*)=(?P<val>.*?)(?=\s+[A-Za-z_][A-Za-z0-9_/]*=|$)"
)
_PART_FIELD = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)"
)

_ALL = frozenset({"all", "(null)", ""})

#: Slurm state flags meaning "will not accept new work", mapped onto the
#: neutral conditions in :mod:`nodetop.core.model`.
_FLAG_TO_CONDITION = {
    "DRAIN": "DRAIN", "DRAINING": "DRAIN", "DRAINED": "DRAIN",
    "FAIL": "FAIL", "FAILING": "FAIL",
    "MAINT": "MAINT", "PLANNED": "MAINT",
    "REBOOT": "MAINT", "REBOOT_REQUESTED": "MAINT",
    "POWER_DOWN": "POWERSAVE", "POWERING_DOWN": "POWERSAVE",
    "RESERVED": "RESERVED", "INVAL": "UNKNOWN",
}
_BASE_TO_CONDITION = {
    "DOWN": "DOWN", "UNKNOWN": "UNKNOWN", "FUTURE": "DOWN",
    "DRAIN": "DRAIN", "ERROR": "FAIL", "INVAL": "UNKNOWN",
}


def _fields(pattern: re.Pattern[str], record: str) -> dict[str, str]:
    """``key=value`` pairs from one record, **first occurrence wins**.

    No field repeats within a real ``scontrol`` record, so a second occurrence
    can only have come out of a free-text value -- and one field on every node
    is operator-authored prose. A dict comprehension keeps the *last* match, so
    a node drained with ``Reason=replacing NodeName=n2 per ticket`` was renamed
    to ``n2 per ticket``: the node vanished from the report under its own name
    and reappeared under a mangled one. The real header is at the record start,
    which is what makes first-wins the safe rule rather than merely a different
    one.
    """
    out: dict[str, str] = {}
    for m in pattern.finditer(record):
        out.setdefault(m.group("key"), m.group("val").strip())
    return out


def _records(output: str, key: str) -> list[str]:
    """Split ``scontrol`` output into one flat string per record.

    **Delimited by the record's own header keyword, not by layout**, so the same
    parser reads either form ``scontrol`` emits. The two forms are one flag
    apart -- this backend asks for ``--oneliner`` when listing nodes and not
    when listing partitions -- and each parser used to understand only the shape
    its own command happened to produce. Both failed silently on the other:

    * partitions were split on blank lines, so oneliner input became a *single*
      record whose field dictionary kept only the last value for each key. 2000
      partitions collapsed to 1, with nothing reported.
    * nodes were read one per line, so multi-line input yielded a record per
      line. Every node came back with 0 CPUs and no state -- a cluster that
      appears to own no resources.

    Neither is reachable while the argv here is fixed, which is exactly why it
    would go unnoticed: the way in is a **replayed snapshot** recorded where
    ``scontrol`` behaved differently, a site wrapper, or a version whose output
    changed shape. A parser that cannot tell "no records" from "one merged
    record" has no way to complain, so the fix is to stop depending on layout.

    The header is required at a line start, so the keyword appearing inside a
    free-text field (a node's ``Reason``) cannot split a record in two.
    """
    marker = re.compile(rf"(?:^|\n)\s*(?={re.escape(key)}=)")
    bounds = [m.end() for m in marker.finditer(output)]
    if not bounds:
        return []
    pieces = [output[a:b] for a, b in zip(bounds, bounds[1:] + [len(output)], strict=True)]
    return [" ".join(x.strip() for x in piece.splitlines()).strip() for piece in pieces]


def parse_slurm_duration(text: str | None) -> int | None:
    """Parse a duration exactly as Slurm writes it.

    Kept separate from :func:`nodetop.core.duration.parse_duration` because
    here the grammar is fixed by Slurm rather than chosen by us, and the
    two-field case is genuinely ambiguous without the day part: ``2:00`` is
    ``MM:SS`` (two minutes) but ``1-2:00`` is ``D-HH:MM``.
    """
    if text is None:
        return None
    t = text.strip()
    if t.lower() in {"unlimited", "infinite", "none", "n/a", "(null)", "", "-"}:
        return None
    m = re.match(r"^(?:(?P<days>\d+)-)?(?P<rest>\d+(?::\d+)*)$", t)
    if not m:
        return None
    days = int(m.group("days") or 0)
    fields = [int(x) for x in m.group("rest").split(":")]
    if m.group("days") is not None:
        units = [3600, 60, 1][: len(fields)]
    elif len(fields) == 1:
        units = [60]
    elif len(fields) == 2:
        units = [60, 1]
    elif len(fields) == 3:
        units = [3600, 60, 1]
    else:
        return None
    if len(fields) > len(units):
        return None
    return days * 86400 + sum(v * u for v, u in zip(fields, units, strict=True))


def _looks_present(value: str | None) -> bool:
    """Whether a field held something, as opposed to nothing or a sentinel."""
    if not value or not value.strip():
        return False
    return value.strip().lower() not in {
        "unlimited", "infinite", "none", "n/a", "(null)", "-",
    }


def _int(value: str | None, default: int = 0) -> int:
    """Leading integer from a field, never negative.

    A resource count below zero is meaningless, and letting one through is
    actively harmful: ``cpus_free`` is ``total - alloc``, so an allocation of
    -5 against a total of 0 reports five free CPUs that do not exist.
    """
    if not value:
        return default
    m = re.match(r"-?\d+", value.strip())
    if not m:
        return default
    return max(0, int(m.group()))


def _opt_int(value: str | None) -> int | None:
    if not value or value.strip().upper() in {"UNLIMITED", "INFINITE", "N/A", "(NULL)"}:
        return None
    m = re.match(r"\d+", value.strip())
    return int(m.group()) if m else None


def _gres_gpus(gres: str | None) -> int:
    """Total accelerators from a ``Gres=`` field: ``gpu:4`` or ``gpu:a30:4``."""
    if not gres or gres.strip().lower() in {"(null)", "none", ""}:
        return 0
    total = 0
    for entry in gres.split(","):
        parts = entry.strip().split(":")
        if not parts or parts[0].lower() != "gpu":
            continue
        total += _int(parts[-1].split("(")[0], 0)
    return total


def _gres_count(text: str | None) -> int:
    """Accelerators from ``squeue %b`` (``gres:gpu:4``), or 0 for ``N/A``."""
    if not text:
        return 0
    m = re.search(r"gpu:?(?::[^:]*)?:?(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _tres_gpus(tres: str | None) -> int:
    if not tres:
        return 0
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres)
    return int(m.group(1)) if m else 0


def _parse_tres_map(text: str | None) -> dict[str, int]:
    """``cpu=256,gres/gpu=32,node=8,mem=100G`` -> neutral limit keys."""
    if not text or text.strip().lower() in {"", "(null)", "none"}:
        return {}
    out: dict[str, int] = {}
    for m in re.finditer(r"(?P<key>[a-zA-Z/_]+(?::[a-zA-Z0-9_]+)?)=(?P<val>\d+)(?P<unit>[KMGT]?)",
                         text):
        key = m.group("key").lower()
        val = int(m.group("val"))
        unit = m.group("unit")
        if key.startswith("gres/gpu"):
            out["gpu"] = val
        elif key == "node":
            out["node"] = val
        elif key == "cpu":
            out["cpu"] = val
        elif key == "mem":
            # Slurm TRES memory defaults to MB, with an optional suffix. The
            # scale must be a float: an integer 1 // 1024 is 0, and writing
            # `1 // 1024 or 1` to dodge that silently makes K equal M -- so a
            # `mem=500K` ceiling reads as 500 MB, 1024x too large. Overstating
            # a ceiling means an over-limit job is never flagged.
            scale = {"": 1.0, "K": 1 / 1024, "M": 1.0,
                     "G": 1024.0, "T": 1024.0 ** 2, "P": 1024.0 ** 3}
            out["mem_mb"] = int(val * scale.get(unit, 1.0))
    return out


def _literal_csv(value: str | None) -> tuple[str, ...]:
    """Split an Allow* field, keeping a literal ``none``.

    Collapsing ``none`` to an empty tuple would make "nobody may submit"
    indistinguishable from "no restriction" -- the exact confusion that lets a
    closed partition look open.
    """
    if value is None or value.strip().lower() in _ALL:
        return ()
    return tuple(p.strip() for p in value.split(",") if p.strip())


class SlurmBackend:
    """Adapter for Slurm."""

    name = "slurm"
    queue_term = "partition"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()

    @classmethod
    def detect(cls) -> bool:
        return which("scontrol") and which("sinfo")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=which("sbatch"),
            probe_supported=True,
            probe_command="sbatch --test-only",
            notes=(
                "--test-only ignores QOS ceilings: an over-limit job returns PASSED, "
                "then pends forever",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_nodes(self, output: str) -> list[Node]:
        nodes: list[Node] = []
        for line in _records(output, "NodeName"):
            f = _fields(_NODE_FIELD, line)
            name = f.get("NodeName", "")
            if not name:
                continue
            state = f.get("State", "")
            if not state.strip():
                # A record with no State is a truncated or partial read, not a
                # healthy node. Left alone it carries no conditions at all,
                # which makes it look schedulable AND empty -- the most
                # attractive thing in the cluster to a placement search.
                nodes.append(
                    Node(
                        name=name,
                        state_raw="",
                        conditions=frozenset({"UNKNOWN"}),
                        reason="no state reported (truncated or partial record)",
                    )
                )
                continue
            # Decompose "DOWN*+DRAIN" into base, markers and flags.
            cleaned = state.replace("*", "").replace("~", "").replace("#", "")
            parts = cleaned.split("+")
            base = parts[0].upper()
            flags = [p.upper() for p in parts[1:] if p]
            conditions = set()
            if base in _BASE_TO_CONDITION:
                conditions.add(_BASE_TO_CONDITION[base])
            for flag in flags:
                if flag in _FLAG_TO_CONDITION:
                    conditions.add(_FLAG_TO_CONDITION[flag])

            labels = f.get("ActiveFeatures") or f.get("AvailableFeatures") or ""
            gres = f.get("Gres", "")
            reason = f.get("Reason", "")
            if reason.lower() in {"(null)", "none"}:
                reason = ""
            nodes.append(
                Node(
                    name=name,
                    state_raw=state,
                    conditions=frozenset(conditions),
                    cpus_total=_int(f.get("CPUTot")),
                    cpus_alloc=_int(f.get("CPUAlloc")),
                    memory_mb=_int(f.get("RealMemory")),
                    memory_alloc_mb=_int(f.get("AllocMem")),
                    gpus_total=_gres_gpus(gres),
                    gpus_alloc=_tres_gpus(f.get("AllocTRES")),
                    accelerator=identify_accelerator(gres, labels),
                    labels=tuple(t.strip() for t in labels.split(",") if t.strip()),
                    queues=tuple(expand(f.get("Partitions", ""))),
                    reason=reason,
                    unreachable="*" in state,
                )
            )
        return nodes

    def load_nodes(self) -> list[Node]:
        return self.parse_nodes(self.runner.run(["scontrol", "show", "node", "--oneliner"]))

    # -- partitions ---------------------------------------------------------
    def parse_queues(self, output: str) -> list[Queue]:
        out: list[Queue] = []
        for flat in _records(output, "PartitionName"):
            f = _fields(_PART_FIELD, flat)
            name = f.get("PartitionName", "")
            if not name:
                continue
            state = f.get("State", "UP").upper()
            out.append(
                Queue(
                    name=name,
                    state_raw=f.get("State", "UP"),
                    # Slurm has one switch where PBS has two; DRAIN accepts
                    # but never starts, which is exactly the "not started"
                    # case the neutral model separates out.
                    enabled=state in {"UP", "DRAIN"},
                    started=state == "UP",
                    hidden=f.get("Hidden", "NO").upper() == "YES",
                    allow_accounts=_literal_csv(f.get("AllowAccounts")),
                    deny_accounts=_literal_csv(f.get("DenyAccounts")),
                    allow_qos=_literal_csv(f.get("AllowQos")),
                    allow_groups=_literal_csv(f.get("AllowGroups")),
                    max_walltime_seconds=parse_slurm_duration(f.get("MaxTime")),
                    max_nodes=_opt_int(f.get("MaxNodes")),
                    min_nodes=_opt_int(f.get("MinNodes")) or 0,
                    declared_nodes=_opt_int(f.get("TotalNodes")) or 0,
                    node_names=tuple(expand(f.get("Nodes", ""))),
                    requires_reservation=f.get("ReqResv", "NO").upper() == "YES",
                    is_default=f.get("Default", "NO").upper() == "YES",
                    priority=_opt_int(f.get("PriorityTier")) or 1,
                    limits_name=f.get("QoS") if f.get("QoS") not in {None, "N/A"} else None,
                )
            )
        return out

    def load_queues(self) -> list[Queue]:
        # --all is essential: a hidden partition is exactly the kind that has
        # quietly been taken out of service, so omitting it hides the diagnosis.
        return self.parse_queues(self.runner.run(["scontrol", "show", "partition", "--all"]))

    # -- limits -------------------------------------------------------------
    def parse_limits(self, output: str) -> dict[str, Limits]:
        out: dict[str, Limits] = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            f = line.split("|")
            f += [""] * (8 - len(f))
            name = f[0].strip()
            if not name:
                continue
            wall = parse_slurm_duration(f[1])
            per_user = _parse_tres_map(f[2])
            per_job = _parse_tres_map(f[3])
            # A field that held something we could not read is not the same as
            # an absent one, and only the first silently disables a check.
            unreadable = []
            if wall is None and _looks_present(f[1]):
                unreadable.append("MaxWall")
            if not per_user and _looks_present(f[2]):
                unreadable.append("MaxTRESPerUser")
            if not per_job and _looks_present(f[3]):
                unreadable.append("MaxTRESPerJob")
            out[name] = Limits(
                name=name,
                max_walltime_seconds=wall,
                per_user=per_user,
                per_job=per_job,
                max_jobs=_opt_int(f[4]),
                max_submitted=_opt_int(f[5]),
                source="slurm QOS",
                unreadable=tuple(unreadable),
            )
        return out

    def load_limits(self) -> dict[str, Limits]:
        return self.parse_limits(
            self.runner.run(
                [
                    "sacctmgr", "-nP", "show", "qos",
                    "format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob,"
                    "MaxJobsPerUser,MaxSubmitJobsPerUser,Priority,Flags",
                ]
            )
        )

    # -- identity -----------------------------------------------------------
    def parse_identity(self, output: str, user: str) -> Identity:
        account_queues: dict[str, set[str]] = {}
        qos: set[str] = set()
        for line in output.splitlines():
            parts = line.split("|")
            if not parts or not parts[0].strip():
                continue
            acct = parts[0].strip()
            partition = parts[1].strip() if len(parts) > 1 else ""
            qos_field = parts[2].strip() if len(parts) > 2 else ""
            bucket = account_queues.setdefault(acct, set())
            if partition:
                bucket.add(partition)
            for q in qos_field.split(","):
                if q.strip():
                    qos.add(q.strip())
                    bucket.add(q.strip())
        return Identity.from_account_queues(user, account_queues, qos, _unix_groups(user))

    def load_identity(self) -> Identity:
        user = os.environ.get("USER") or getpass.getuser()
        try:
            # "where" and the condition must be SEPARATE argv elements.
            # Passing "where user=X" as one string makes sacctmgr answer
            # `Unknown condition`, which returned an empty identity -- and an
            # empty identity silently disables every account and QOS access
            # check downstream, since those are tri-state and treat "no values
            # to compare against" as "no verdict".
            out = self.runner.run(
                ["sacctmgr", "-nP", "show", "assoc", "where", f"user={user}",
                 "format=Account,Partition,QOS"]
            )
        except Exception:
            # Not swallowed into an empty identity. The comment above says why:
            # a zero-account identity silently disables every account and QOS
            # check downstream, because those are tri-state and read "nothing
            # to compare against" as "no verdict". Substituting "" here made
            # the failure indistinguishable from a user with no associations,
            # and the consequence was measured: with `sacctmgr` down, all 34
            # accounts vanished, every probe ran with no --account, and the
            # overview reported "0 open to you - 83 refused" -- total loss of
            # access, stated with confidence, during a database hiccup.
            #
            # Raising lets Cluster.load record it under "identity" and leave
            # `identity` as None, which the entitlement filter already treats
            # as "cannot filter" rather than as "entitled to nothing".
            raise
        return self.parse_identity(out, user)

    # -- jobs ---------------------------------------------------------------
    #: ``squeue`` fields, in the order :meth:`parse_jobs` expects them. Pipe
    #: separated because a job NAME may contain spaces and nothing else here is
    #: safe to split on.
    JOB_FORMAT = "%i|%u|%a|%P|%N|%C|%b|%M|%L|%T|%j"

    def parse_jobs(self, output: str) -> list[Job]:
        """One :class:`Job` per line of ``squeue`` output.

        The node field is a bracket-notation nodelist, not a single name -- a
        job holding forty nodes reports them collapsed -- so it goes through
        :func:`expand` like every other nodelist in this backend.

        The NAME field is last on the line on purpose: it is user-authored and
        may contain the separator, so it absorbs whatever is left rather than
        shifting every field after it.
        """
        out: list[Job] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", len(self.JOB_FORMAT.split("|")) - 1)
            if len(parts) < 10:
                continue
            jid, user, acct, part, nodes, cpus, gres, used, left, state = parts[:10]
            out.append(Job(
                id=jid.strip(),
                user=user.strip(),
                account=acct.strip(),
                queue=part.strip(),
                name=parts[10].strip() if len(parts) > 10 else "",
                state=state.strip(),
                nodes=tuple(expand(nodes.strip())),
                cpus=_opt_int(cpus) or 0,
                gpus=_tres_gpus(gres) or _gres_count(gres),
                elapsed=used.strip(),
                remaining=left.strip(),
            ))
        return out

    def load_jobs(self) -> list[Job]:
        return self.parse_jobs(
            self.runner.run(["squeue", "-h", "-t", "RUNNING",
                             "-o", self.JOB_FORMAT])
        )

    # -- free times ---------------------------------------------------------
    def load_node_free_times(self) -> dict[str, datetime]:
        try:
            out = self.runner.run(["squeue", "-h", "-t", "RUNNING", "-o", "%N|%e"])
        except Exception:
            return {}
        latest: dict[str, datetime] = {}
        for line in out.splitlines():
            nodelist, _, end = line.partition("|")
            when = parse_timestamp(end)
            if when is None:
                continue
            for nm in expand(nodelist.strip()):
                if nm not in latest or when > latest[nm]:
                    latest[nm] = when
        return latest

    # -- probe --------------------------------------------------------------
    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        args = [
            f"--partition={queue}",
            f"--nodes={shape.nodes}",
            f"--ntasks-per-node={shape.tasks_per_node}",
            f"--cpus-per-task={shape.cpus_per_task}",
            f"--time={shape.walltime}",
        ]
        if shape.gpus_per_node:
            args.append(f"--gres=gpu:{shape.gpus_per_node}")
        if shape.memory_gb:
            args.append(f"--mem={shape.memory_mb_per_node}M")
        if shape.account:
            args.append(f"--account={shape.account}")
        if shape.qos:
            args.append(f"--qos={shape.qos}")
        if shape.exclude:
            args.append(f"--exclude={collapse(shape.exclude)}")
        return args

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """Read-only: ``--test-only`` verifies, estimates, and discards.

        The flag is hard-coded so a caller cannot omit it and submit for real.
        """
        if not self.capabilities().probe:
            # Gated on sbatch existing. Trying anyway would report
            # CONTROL_PLANE_DOWN for a missing client, which blames the cluster
            # for a local problem.
            return None
        flags = [f for f in self.submit_flags(queue, shape) if not f.startswith("--account=")]
        cmd = ["sbatch", "--test-only", *flags]
        if account:
            cmd.append(f"--account={account}")
        cmd.extend(["--wrap", "true"])
        try:
            rc, out, err = self.runner.run_full(cmd)
        except Exception as exc:
            return Verdict(
                queue=queue, account=account, allowed=False,
                category=VerdictCategory.CONTROL_PLANE_DOWN, reason=str(exc), raw=str(exc),
            )
        return parse_probe(queue, account, rc, out, err)

    def format_nodelist(self, names: Iterable[str]) -> str:
        return collapse(names)


# ---------------------------------------------------------------------------
# probe output parsing
# ---------------------------------------------------------------------------
_PATTERNS: tuple[tuple[str, str], ...] = (
    # A controller that cannot write job scripts fails every submission
    # cluster-wide.  It must never be read as "you lack access here".
    (r"i/o error writing script|error writing script/environment",
     VerdictCategory.CONTROL_PLANE_DOWN),
    (r"unable to contact slurm controller|slurmctld.*not responding|connection timed out",
     VerdictCategory.CONTROL_PLANE_DOWN),
    (r"invalid membership to account", VerdictCategory.NOT_ENTITLED),
    (r"invalid account or account/partition combination", VerdictCategory.ACCOUNT_MISMATCH),
    (r"account is not specified|no account specified", VerdictCategory.NO_ACCOUNT),
    (r"invalid account", VerdictCategory.NOT_ENTITLED),
    (r"user'?s group not permitted|group not permitted to use this partition",
     VerdictCategory.GROUP_DENIED),
    (r"invalid partition name|partition name too long", VerdictCategory.UNKNOWN_QUEUE),
    (r"partition is down|required partition is down|partition .* is not available",
     VerdictCategory.QUEUE_CLOSED),
    (r"invalid qos", VerdictCategory.INVALID_QOS),
    (r"job violates accounting/qos policy", VerdictCategory.QUOTA_EXCEEDED),
    (r"requested time limit is invalid|time limit .*exceeds", VerdictCategory.TIME_LIMIT),
    (r"requested node configuration is not available|node count specification invalid"
     r"|more processors requested than permitted|memory required by task is not available",
     VerdictCategory.SHAPE_UNAVAILABLE),
    # An exhausted allocation, before the generic access pattern below.
    #
    # Sites reject this through the same channel as a permission failure -- the
    # observed pair is "Reason: No sufficient SU allocations for a shared
    # partition" on stdout and "allocation failure: Access/permission denied" on
    # stderr -- so it fell through to ACCESS_DENIED and read as "you are not
    # allowed here". It is a quota, and the remedy is the opposite kind of
    # request: ask for service units, not for access. First match wins in this
    # table, which is why the order matters.
    (r"no sufficient su allocation|insufficient su\b|out of service units"
     r"|allocation exhausted|no remaining allocation",
     VerdictCategory.QUOTA_EXCEEDED),
    (r"access/permission denied", VerdictCategory.ACCESS_DENIED),
)

_START = re.compile(
    r"Job\s+(?P<jobid>\d+)\s+to start at\s+(?P<when>\S+)"
    r"(?:\s+using\s+(?P<procs>\d+)\s+processors)?"
    r"(?:\s+on nodes?\s+(?P<nodes>\S+))?"
)
_VERIFICATION = re.compile(r"Verification:\s*\**\s*(?P<v>PASSED|REJECTED)", re.IGNORECASE)
_REASON = re.compile(r"^\s*Reason:\s*(?P<reason>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ALLOC_FAIL = re.compile(r"allocation failure:\s*(?P<msg>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SUBMIT_FAIL = re.compile(
    r"Batch job submission failed:\s*(?P<msg>.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_QOS_FLAG = re.compile(r"QOS-Flag:\s*(?P<qos>\S+)", re.IGNORECASE)
_ACCOUNT = re.compile(r"^\s*Account:\s*(?P<acct>\S+)", re.IGNORECASE | re.MULTILINE)


def classify(text: str) -> str:
    low = text.lower()
    for pattern, category in _PATTERNS:
        if re.search(pattern, low):
            return category
    return VerdictCategory.UNKNOWN


class _CommonFields(TypedDict):
    """The Verdict fields every branch of :func:`parse_probe` fills alike.

    Mirrors the corresponding annotations on :class:`Verdict`; a mismatch is a
    type error at the splat site, which is the whole point of writing it down.
    """

    queue: str
    account: str | None
    filter_verdict: str | None
    effective_qos: str | None
    effective_account: str | None
    raw: str


def parse_probe(
    queue: str, account: str | None, returncode: int, stdout: str, stderr: str
) -> Verdict:
    """Interpret one ``sbatch --test-only`` run, reading *both* verdict layers."""
    combined = f"{stdout}\n{stderr}"
    # Strip "sbatch: " / "sbatch: error: " so patterns match the message text
    # regardless of how the line was decorated.
    clean = re.sub(r"^\s*sbatch:\s*(error:\s*)?", "", combined, flags=re.MULTILINE)

    vm = _VERIFICATION.search(clean)
    filter_verdict = vm.group("v").upper() if vm else None
    qos = _QOS_FLAG.search(clean)
    acct = _ACCOUNT.search(clean)
    core_fail = _ALLOC_FAIL.search(clean) or _SUBMIT_FAIL.search(clean)
    core_msg = core_fail.group("msg").strip() if core_fail else ""
    rm = _REASON.search(clean)
    filter_reason = rm.group("reason").strip() if rm else ""
    start = _START.search(clean)

    # A TypedDict, not a plain one. Inference widens a plain dict of mixed
    # values to `str | Any | None`, and every `**common` splat below then checks
    # against nothing: a field renamed or retyped in Verdict would be caught by
    # no tool, and by no test that happens not to read that exact field.
    # Naming the shape is what makes the splat verifiable.
    common: _CommonFields = {
        "queue": queue,
        "account": account,
        "filter_verdict": filter_verdict,
        "effective_qos": qos.group("qos") if qos else None,
        "effective_account": acct.group("acct") if acct else None,
        "raw": combined.strip(),
    }

    # Slurm core's refusal outranks a site plugin's PASSED.
    if filter_verdict == "REJECTED" or core_msg:
        reason = filter_reason or core_msg
        if filter_verdict == "PASSED" and core_msg:
            reason = f"{core_msg} (site submit filter reported PASSED)"
        return Verdict(
            allowed=False,
            category=classify(f"{filter_reason} {core_msg}"),
            reason=reason,
            **common,
        )

    # A non-zero exit vetoes acceptance, whatever the text said.
    #
    # `filter_verdict == "PASSED"` used to be sufficient on its own, so a run
    # that exited 1 while its site plugin printed PASSED was reported as
    # allowed. That inverts the reliability ordering this whole module is built
    # on: the exit status is the one authoritative, wording-independent signal
    # -- `sbatch --test-only` exits 0 if and only if the job would be accepted
    # -- while the filter's PASSED is precisely the claim documented as generous
    # fiction. Reading the text over the status is how "verification passed"
    # becomes a job that never runs.
    #
    # It matters because the core-refusal branch above recognises only two
    # message prefixes (`allocation failure:` and `Batch job submission
    # failed:`). Any other refusal wording -- Slurm has many, and site plugins
    # add their own -- fell through to here and was granted.
    #
    # Ordered after the core-refusal branch so a recognised message still
    # supplies the better category and reason, and before the acceptance branch
    # so a predicted start time cannot outvote a failure: printing one and then
    # exiting non-zero is contradictory output, and the conservative reading is
    # the one this module commits to everywhere else.
    if returncode != 0:
        return Verdict(
            allowed=False,
            category=classify(clean),
            reason=(filter_reason or core_msg
                    or (clean.strip().splitlines() or ["no output"])[-1]),
            **common,
        )

    if start or returncode == 0 or filter_verdict == "PASSED":
        return Verdict(
            allowed=True,
            category=VerdictCategory.OK,
            predicted_start=parse_timestamp(start.group("when")) if start else None,
            predicted_nodes=(
                tuple(expand(start.group("nodes")))
                if start and start.group("nodes")
                else ()
            ),
            **common,
        )

    return Verdict(
        allowed=False,
        category=classify(clean),
        reason=clean.strip().splitlines()[-1] if clean.strip() else "no output",
        **common,
    )


def _unix_groups(user: str) -> tuple[str, ...]:
    """Local groups for ``user``, or nothing at all.

    **Atomic on purpose: a partial answer is worse than no answer here.** The
    set is committed only if the whole lookup succeeded, because the tri-state
    check downstream reads an empty set as "cannot tell" (no verdict) and a
    *non-empty* one as authoritative. So a lookup that collected the
    supplementary groups and then failed on the primary would have the tool
    assert "none of your groups are permitted here" on the strength of a list it
    knows to be incomplete -- a false denial, which hides a queue you can use.
    Empty degrades to no verdict, which is the honest reading of a failure.
    """
    names: set[str] = set()
    try:
        import pwd

        for g in grp.getgrall():
            if user in g.gr_mem:
                names.add(g.gr_name)
        names.add(grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name)
    except (KeyError, OSError):  # pragma: no cover - platform dependent
        return ()
    return tuple(sorted(names))
