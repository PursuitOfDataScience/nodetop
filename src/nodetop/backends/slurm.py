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

import dataclasses
import functools
import getpass
import grp
import os
import re
import threading
from collections.abc import Iterable
from datetime import datetime
from typing import TypedDict

from ..core.duration import parse_timestamp
from ..core.hardware import identify_accelerator, name_accelerator
from ..core.model import (
    Allocation,
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
from .base import BackendCapabilities, count

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
#:
#: ``PLANNED`` is deliberately absent, and the omission is load-bearing: it is
#: the backfill scheduler saying "I intend to use this node for a queued job",
#: which is a *plan*, not an outage. Slurm keeps running work there and keeps
#: placing more (sinfo writes it as the ``-`` suffix, alongside ``*`` for
#: unreachable and ``$`` for a maintenance reservation). Reading it as MAINT
#: cost two H100 nodes on a Slurm 25.11 cluster: ``MIXED+PLANNED``, 8 GPUs
#: allocated to other people's running jobs, reported as "all 2 nodes are
#: down, drained or unreachable", their partition marked
#: ALL_NODES_UNSCHEDULABLE and every GPU on the cluster erased from the
#: inventory -- `accelerators` said "no GPUs found in this cluster" one line
#: under a header counting 8 of them.
#: The rest of the table is read off `man sinfo`'s NODE STATE CODES rather than
#: off one cluster's output, because an unknown flag fails silently in the
#: expensive direction: it contributes no condition, the node stays
#: "schedulable", and its idle cores are reported as room. Every entry below
#: whose wording is "not usable"/"not capable of running any jobs" is therefore
#: mapped, even where no cluster to hand has one set.
_FLAG_TO_CONDITION = {
    "DRAIN": "DRAIN", "DRAINING": "DRAIN", "DRAINED": "DRAIN",
    "FAIL": "FAIL", "FAILING": "FAIL",
    "MAINT": "MAINT",
    "REBOOT": "MAINT", "REBOOT_REQUESTED": "MAINT", "REBOOT_ISSUED": "MAINT",
    "POWER_DOWN": "POWERSAVE", "POWERING_DOWN": "POWERSAVE",
    # "currently powered down and not capable of running any jobs" -- the
    # sibling of the two above, and the one that actually advertises a full
    # complement of idle CPUs and memory while doing it.
    "POWERED_DOWN": "POWERSAVE",
    # "blocked by exclusive topo job" and "Network Performance Counters ... in
    # use, rendering this node as not usable for any other jobs": someone
    # else's exclusive claim on a node that is otherwise up, which is what
    # RESERVED means here. `sinfo` prints the second as `PERFCTRS (NPC)`, so
    # both spellings are taken.
    "BLOCKED": "RESERVED", "PERFCTRS": "RESERVED", "NPC": "RESERVED",
    "RESERVED": "RESERVED", "INVAL": "UNKNOWN",
}
#: Flags worth carrying into the model without blocking anything: they change
#: what a reader should expect, not whether the node accepts work.
#:
#: ``POWERING_UP`` is absent from both tables on purpose. It is the one
#: powersave state Slurm places work on -- that is what powering up is *for* --
#: so calling it unschedulable would erase real capacity on any cluster that
#: cycles nodes, and the job it would have hidden simply waits for the boot.
_FLAG_TO_NOTE = {"PLANNED": "PLANNED"}
_BASE_TO_CONDITION = {
    "DOWN": "DOWN", "UNKNOWN": "UNKNOWN", "FUTURE": "DOWN",
    "DRAIN": "DRAIN", "ERROR": "FAIL", "INVAL": "UNKNOWN",
}


#: Whitespace that is a field separator but is not a plain space.
#:
#: `_fields_fast` cannot see these, so their presence sends the record back to
#: the regex. A `Reason` carrying a tab is rare and a tab-delimited record is
#: rarer still, but the failure mode if one slipped past would be the worst one
#: available: the whole record read as a single field, so a node with no state,
#: which :func:`parse_nodes` then reports as UNKNOWN.
_ODD_SPACE = re.compile(r"[\t\n\r\f\v]")

#: Identifier shapes for the two record types, anchored at the end so a token
#: like `a=b` inside a value cannot be mistaken for a key.
_IDENT = {
    "node": re.compile(r"[A-Za-z_][A-Za-z0-9_/]*\Z"),
    "part": re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z"),
}


@functools.lru_cache(maxsize=512)
def _is_key(kind: str, head: str) -> bool:
    """Whether ``head`` is a field name rather than part of a value.

    Cached because `scontrol`'s vocabulary is fixed and tiny while the tokens
    are many: 607 nodes asked this 217,383 times about **37 distinct** strings.
    That turned the regex from the second-hottest line in parsing into 37 calls
    -- 13.1 ms to 10.1 ms over those nodes.

    Bounded rather than a plain dict, which was faster still (8.7 ms) and
    unbounded: an operator `Reason` full of `ticket=12345`-shaped words would
    add an entry per node, and this process can outlive one parse by hours when
    somebody leaves the browse open.
    """
    return _IDENT[kind].match(head) is not None


def _fields_fast(kind: str, record: str) -> dict[str, str]:
    r""":func:`_fields` without the backtracking, for the ordinary record.

    The pattern `_fields` uses ends each value with a lazy match plus a
    lookahead -- `(?P<val>.*?)(?=\s+Ident=|$)` -- which re-evaluates the
    lookahead at every character of every value. Profiling `parse_nodes` over
    607 nodes put `_fields` at **35% of the total** with 460,680 calls to
    `re.Match.group` behind it.

    Splitting on spaces cannot backtrack: a token that is `Ident=` starts a
    field and everything else belongs to the value being collected. Same
    first-wins rule, same `strip`, same output -- verified field by field
    against the regex on every record of two live clusters (607 nodes and 87
    partitions) and on a set of adversarial records: values containing `=`, a
    key-shaped token mid-value, doubled spaces, empty values, leading and
    trailing space. 22.2 ms -> 12.9 ms for 607 records.
    """
    out: dict[str, str] = {}
    key: str | None = None
    parts: list[str] = []
    for token in record.split(" "):
        head, sep, rest = token.partition("=")
        if sep and _is_key(kind, head):
            if key is not None:
                out.setdefault(key, " ".join(parts).strip())
            key, parts = head, [rest]
        elif key is not None:
            parts.append(token)
    if key is not None:
        out.setdefault(key, " ".join(parts).strip())
    return out


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
    if not _ODD_SPACE.search(record):
        return _fields_fast("node" if pattern is _NODE_FIELD else "part", record)
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


#: Leading integer from a field, never negative.
#:
#: The shared implementation lives in :func:`nodetop.backends.base.count`, since
#: every adapter needs one and each having its own is how they drifted -- PBS's
#: raised on a non-numeric value and let a negative one through. Kept under this
#: name because fifteen call sites and a test reference it.
_int = count


def _opt_int(value: str | None) -> int | None:
    if not value or value.strip().upper() in {"UNLIMITED", "INFINITE", "N/A", "(NULL)"}:
        return None
    m = re.match(r"\d+", value.strip())
    return int(m.group()) if m else None


#: A parenthesised suffix on a GRES value.  Slurm appends the specific devices
#: -- ``gpu:2(IDX:0,3)`` from a job's allocation, ``gpu:v100:4(S:0-1)`` from a
#: node's socket affinity -- and the contents carry both colons and commas, the
#: two characters the field is otherwise split on.
_GRES_SUFFIX = re.compile(r"\([^)]*\)")


def _gres_gpus(gres: str | None) -> int:
    """Total accelerators from a ``Gres=`` field: ``gpu:4`` or ``gpu:a30:4``.

    **The parenthesised suffix is removed first, and that is the whole
    subtlety.** ``gpu:2(IDX:0,3)`` split on commas gives ``gpu:2(IDX:0`` and
    ``3)``; the first then split on colons ends in ``0``, which is the *device
    index* being read as the count. Measured on a real node, for the three jobs
    holding its four accelerators:

    ==============  ======================  =========  ========
    job             GRES                    reported   actual
    ==============  ======================  =========  ========
    53741225        ``gpu:2(IDX:0,3)``      0          2
    54466230_1      ``gpu:1(IDX:2)``        2          1
    54072325        ``gpu:1(IDX:1)``        1          1
    ==============  ======================  =========  ========

    Two of three wrong, and wrong in the direction that reads as another job's
    number -- the reason it looked like the rows had been shuffled rather than
    misparsed. The same field on a node, ``gpu:v100:4(S:0-1)``, read as **zero
    accelerators**: this cluster does not print the socket suffix, so that one
    was latent, and it would have made every GPU node on a cluster that does
    print it look like a CPU node.
    """
    if not gres or gres.strip().lower() in {"(null)", "none", ""}:
        return 0
    total = 0
    for entry in _GRES_SUFFIX.sub("", gres).split(","):
        parts = [x for x in entry.strip().split(":") if x]
        if not parts or parts[0].lower() != "gpu":
            continue
        total += _int(parts[-1], 0)
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


def _count_cpu_ids(text: str | None) -> int:
    """How many CPUs ``CPU_IDs=0-17,19-24`` names.

    Slurm reports the allocation as core *identifiers*, not a count, so the
    ranges have to be counted rather than read.
    """
    total = 0
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        lo, dash, hi = part.partition("-")
        if dash:
            if lo.isdigit() and hi.isdigit() and int(hi) >= int(lo):
                total += int(hi) - int(lo) + 1
        elif lo.isdigit():
            total += 1
    return total


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
        self._config_cache: str | None = None
        self._config_lock = threading.Lock()

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
                elif flag in _FLAG_TO_NOTE:
                    conditions.add(_FLAG_TO_NOTE[flag])

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
                    accelerator_label=name_accelerator(gres, labels) or "",
                    labels=tuple(t.strip() for t in labels.split(",") if t.strip()),
                    queues=tuple(expand(f.get("Partitions", ""))),
                    reason=reason,
                    # Two spellings of one fact, and a Slurm version decides
                    # which you get. `sinfo` writes the `*` suffix on every
                    # release; `scontrol show node` -- what this backend reads
                    # -- writes `+NOT_RESPONDING` on 25.11 and no `*` anywhere
                    # in the record. Reading only the marker made "the control
                    # plane has lost contact with this node" unsayable there:
                    # a node down for 23 days looked merely DOWN.
                    unreachable="*" in state or "NOT_RESPONDING" in flags,
                )
            )
        return nodes

    def default_memory_per_cpu_mb(self) -> int:
        """``DefMemPerCPU``: what one core of a job that names no ``--mem`` costs.

        The floor a node's free memory has to clear before anything can be
        placed there, and 0 where the site sets none -- Slurm then defaults a
        job to the whole node, which is a shortage this cannot quantify, so it
        claims nothing rather than erasing most of a cluster. Read from the
        config query :meth:`memory_is_consumable` already makes.

        A partition may override the cluster value; that is not modelled, so a
        partition with a *higher* floor is under-detected. The cluster default
        is what every partition inherits unless told otherwise.
        """
        for line in (self._config() or "").splitlines():
            if line.strip().startswith("DefMemPerCPU"):
                value = line.partition("=")[2].strip()
                return int(value) if value.isdigit() else 0
        return 0

    def load_nodes(self) -> list[Node]:
        nodes = self.parse_nodes(
            # `--all`, to match the partition query. Without it Slurm applies
            # the hidden-partition filter to nodes as well, so a node whose
            # only partition is hidden is absent from the answer -- and the
            # partition list, which DOES ask for --all, then names a member the
            # node list never mentions. Measured: 31 nodes without the flag, 32
            # with it, and the missing one was the sole member of a hidden
            # partition that reported "1 node claimed but unresolved" and no
            # capacity at all. `scontrol show node <name>` hides it too, so
            # there is no per-node way to recover it afterwards.
            self.runner.run(["scontrol", "show", "node", "--all", "--oneliner"]))
        floor = self.default_memory_per_cpu_mb()
        if self.memory_is_consumable():
            if not floor:
                return nodes
            return [dataclasses.replace(n, memory_floor_mb=floor) for n in nodes]
        # The floor is irrelevant where memory is not consumable: `AllocMem`
        # there is a record of requests, not a ceiling, so there is nothing to
        # compare it against.
        return [dataclasses.replace(n, memory_consumable=False) for n in nodes]

    def warm(self) -> None:
        """Fetch `scontrol show config` now, with the first wave.

        See :meth:`Backend.warm`. `load_nodes` and `load_identity` both need
        this file, and asking for it from inside `load_nodes` -- after the node
        query has already returned -- sent it ~70 ms behind the rest. It is
        cached and locked, so warming it here means the loaders find it waiting
        rather than fetching it themselves, and nothing is asked for twice.
        """
        self._config()

    def _config(self) -> str | None:
        """``scontrol show config``, fetched at most once, ``None`` if it failed.

        Cached because two questions are answered from it -- whether memory is
        consumable, and whether this cluster refuses a caller with no
        association -- and asking twice would put two instants in one report.
        The failure is cached too, so a controller that cannot answer is asked
        once rather than once per consumer.
        """
        # Locked, because `load_nodes` and `load_identity` both reach for this
        # and now run together: without it they would each fetch, which is two
        # readings of one file in one report and a query the discipline tests
        # forbid.
        with self._config_lock:
            if self._config_cache is None:
                try:
                    self._config_cache = self.runner.run(
                        ["scontrol", "show", "config"])
                except Exception:
                    self._config_cache = ""
            return self._config_cache or None

    def memory_is_consumable(self) -> bool:
        """Whether this cluster accounts for memory when it places work.

        `SelectTypeParameters=CR_CORE_MEMORY` (or `CR_CPU_MEMORY`) makes memory
        a consumable resource: a node whose `AllocMem` has reached its
        `RealMemory` can host nothing more, whatever its idle core count says.
        On the cluster this was written against that is the difference between
        `wide` advertising **2322 free cores** and the 287 it could actually
        hand out -- 2035 of them sat on 47 nodes whose memory was entirely
        allocated to a handful of four-core jobs, and `DefMemPerNode=UNLIMITED`
        there means a job that names no `--mem` asks for the whole node.

        Without the `_MEMORY` suffix Slurm does not decrement memory at all, so
        `AllocMem` is a record of what jobs requested rather than a ceiling the
        scheduler enforces, and reading it as one would report a whole cluster
        as full. Hence the question, rather than the assumption.

        Unreadable config answers yes, which claims less capacity -- the bias
        everywhere else here, and the alternative is recommending a node the
        scheduler will refuse.
        """
        out = self._config()
        if out is None:
            return True
        for line in out.splitlines():
            if "SelectTypeParameters" in line:
                return "MEMORY" in line.upper()
        return True

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
        defaults: set[str] = set()
        # Kept per ROW as well as unioned. The union answers "may I use this
        # QOS anywhere"; only the row answers "which QOS applies to me on this
        # partition", and that is the one a dry-run has to name. See
        # `Identity.qos_for`.
        per_assoc: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
        for line in output.splitlines():
            parts = line.split("|")
            if not parts or not parts[0].strip():
                continue
            acct = parts[0].strip()
            partition = parts[1].strip() if len(parts) > 1 else ""
            qos_field = parts[2].strip() if len(parts) > 2 else ""
            # Optional, and read positionally: an older `sacctmgr` that does not
            # know the column simply leaves the field off, which must not
            # change the three before it.
            default_field = parts[3].strip() if len(parts) > 3 else ""
            bucket = account_queues.setdefault(acct, set())
            if partition:
                bucket.add(partition)
            row_qos: list[str] = []
            for q in qos_field.split(","):
                if q.strip():
                    qos.add(q.strip())
                    bucket.add(q.strip())
                    row_qos.append(q.strip())
            if default_field and default_field != "(null)":
                defaults.add(default_field)
            row_default = "" if default_field == "(null)" else default_field
            if row_qos or row_default:
                # First row wins for a duplicated key: `sacctmgr` can list the
                # same association twice through a cluster column this query
                # does not ask for, and the two are the same grant.
                per_assoc.setdefault((acct, partition), (tuple(row_qos), row_default))
        return Identity.from_account_queues(
            user, account_queues, qos, _unix_groups(user),
            # Whether an empty account list is a fact about the site or about
            # the caller. `AccountingStorageEnforce=associations` means Slurm
            # refuses a submission from a user with no association row, so an
            # empty list there is not "this cluster does not use accounts" --
            # it is "nothing you submit will run". Read from the config query
            # `memory_is_consumable` already makes, so it costs nothing.
            accounts_required=(
                "associations" in (self._config() or "")
                .partition("AccountingStorageEnforce")[2].partition("\n")[0].lower()
            ),
            # One default across every association, or none: two accounts whose
            # jobs run under different ceilings have no single answer, and
            # picking one of them would invent a limit for the other.
            default_qos=defaults.pop() if len(defaults) == 1 else None,
            association_qos=per_assoc,
        )

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
                 "format=Account,Partition,QOS,DefaultQOS"]
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

    #: One allocation block from ``scontrol show job -d``::
    #:
    #:     Nodes=cn-[0521-0522] CPU_IDs=78-94 Mem=29750 GRES=gpu:1(IDX:1)
    #:
    #: ``Nodes`` is a nodelist, not a name: Slurm collapses consecutive nodes
    #: that got the same shape of allocation, and the CPU/memory figures then
    #: apply to *each* of them.
    _ALLOC = re.compile(
        r"Nodes=(?P<nodes>\S+)\s+CPU_IDs=(?P<cpus>\S+)\s+Mem=(?P<mem>\d+)"
        r"(?:\s+GRES=(?P<gres>\S*))?"
    )

    def parse_allocations(self, output: str) -> list[Allocation]:
        """Per-node shares, from ``scontrol show job -d``.

        **The two commands do not agree on what a job is called.** ``squeue``
        names a running array task ``54462542_132``; ``scontrol`` gives it a
        JobId of its own and records the array it came from separately::

            JobId=54465084 ArrayJobId=54462542 ArrayTaskId=65 JobName=...

        1864 of this cluster's 2928 jobs are array tasks, so keying on
        ``JobId`` alone would have found a share for none of them. Each
        allocation is therefore registered under both spellings, and the caller
        can look up whichever it holds.
        """
        out: list[Allocation] = []
        ids: list[str] = []
        for raw in output.splitlines():
            line = raw.strip()
            if line.startswith("JobId="):
                f = _fields(_NODE_FIELD, line)
                ids = [f.get("JobId", "")]
                array, task = f.get("ArrayJobId", ""), f.get("ArrayTaskId", "")
                # A single task, not a pending range like `1-20%10`.
                if array and task.isdigit():
                    ids.append(f"{array}_{task}")
                ids = [x for x in ids if x]
                continue
            if not ids or not line.startswith("Nodes="):
                continue
            m = self._ALLOC.match(line)
            if not m:
                continue
            cpus = _count_cpu_ids(m.group("cpus"))
            mem = _int(m.group("mem"))
            gpus = _gres_gpus(m.group("gres"))
            for node in expand(m.group("nodes")):
                out.extend(
                    Allocation(job=jid, node=node, cpus=cpus, memory_mb=mem,
                               gpus=gpus)
                    for jid in ids
                )
        return out

    def load_allocations(self) -> list[Allocation]:
        """Every job's per-node share, in one call.

        One whole-cluster query rather than one per job: measured on 2928 jobs
        it costs 0.6s and 4.7 MB, against 0.13s for a single job -- so asking
        about five jobs already pays for asking about all of them, and a node
        with 49 tasks on it would otherwise stall an interactive repaint for
        six seconds.
        """
        return self.parse_allocations(
            self.runner.run(["scontrol", "show", "job", "-d"]))

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

# The acceptance line carries several separate facts, so each is read by its own
# pattern applied to that one line -- NOT by one sentence-shaped regex. A single
# pattern makes every fact after the first hostage to the exact wording between
# them, and that wording moves. Slurm 25.11 prints
#
#     sbatch: Job 556812 to start at 2026-08-24T17:32:50 a using 1 processors \
#         on nodes mcn53 in partition standard
#
# -- a stray `a` after the timestamp, verified byte for byte with `cat -A`.
# One chained regex matched the time, then failed at `using`, and both optional
# tails fell off with it: `check` reported a confirmed start with
# `predicted_nodes: []`, so the one thing a dry-run knows and nothing else does
# -- which nodes the scheduler picked -- was dropped on every accepted probe on
# that cluster, silently, while the verdict beside it still looked complete.
#
# Two patterns, not three: the line's `using N processors` clause is deliberately
# not read. A processor count is derivable from the request nodetop just made, so
# capturing it would add a field nothing consumes -- where the node list is, as
# above, the one fact a dry-run knows and nothing else does. A third regex was
# carried here unused for exactly that reason; it is gone rather than left to
# read as a capability.
_START = re.compile(r"Job\s+(?P<jobid>\d+)\s+to start at\s+(?P<when>\S+)")
_START_NODES = re.compile(r"\bon nodes?\s+(?P<nodes>\S+)")
_VERIFICATION = re.compile(r"Verification:\s*\**\s*(?P<v>PASSED|REJECTED)", re.IGNORECASE)
_REASON = re.compile(r"^\s*Reason:\s*(?P<reason>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ALLOC_FAIL = re.compile(r"allocation failure:\s*(?P<msg>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SUBMIT_FAIL = re.compile(
    r"Batch job submission failed:\s*(?P<msg>.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
#: A site `job_submit` plugin's own sentence, as Slurm renders its `err_msg`.
#:
#: This is the most specific thing sbatch prints, and the core line beside it is
#: generic by construction -- the plugin is *why* the submission failed, so
#: Slurm has nothing of its own to report. Observed on a Slurm 24.11 cluster::
#:
#:     sbatch: error: Job submission rejected: Batch jobs cannot use the
#:         `interactive_*` partitions.
#:     allocation failure: Unspecified error
#:
#: Reading only the core line gave `UNKNOWN: Unspecified error` for two of six
#: partitions -- the tool discarding the one sentence that explained the
#: refusal, on a cluster where it was the whole answer.
_PLUGIN_REJECT = re.compile(
    r"Job submission rejected:\s*(?P<msg>.+?)\s*$", re.IGNORECASE | re.MULTILINE
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
    plugin = _PLUGIN_REJECT.search(clean)
    plugin_msg = plugin.group("msg").strip() if plugin else ""
    rm = _REASON.search(clean)
    filter_reason = rm.group("reason").strip() if rm else ""
    # The other two facts are read from the line the start verdict is on, not
    # from the whole message: a nodelist somewhere else in a multi-line answer
    # is not this prediction's.
    start = start_line = None
    for line in clean.splitlines():
        found = _START.search(line)
        if found:
            start, start_line = found, line
            break
    predicted_nodes: tuple[str, ...] = ()
    if start_line:
        on = _START_NODES.search(start_line)
        if on:
            predicted_nodes = tuple(expand(on.group("nodes")))

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
    if filter_verdict == "REJECTED" or core_msg or plugin_msg:
        reason = filter_reason or plugin_msg or core_msg
        if filter_verdict == "PASSED" and core_msg:
            reason = f"{core_msg} (site submit filter reported PASSED)"
        return Verdict(
            allowed=False,
            # All three layers, because the recognisable phrase can be in any
            # of them and the generic one is usually not it.
            category=classify(f"{filter_reason} {plugin_msg} {core_msg}"),
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
            predicted_nodes=predicted_nodes,
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
