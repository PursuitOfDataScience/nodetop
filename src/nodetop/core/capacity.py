"""Can this shape start now, and if not, when could it?

**Every inference here fails toward less.**  Where a fact is missing or
ambiguous, the answer is the one that claims *less* capacity and *less*
access, never more.  A truncated node record is unschedulable rather than
idle; an unreadable resource count is zero rather than negative; an
accelerator whose model cannot be identified does not count toward a stated
capability; a memory size that could be one of two values is assumed to be the
smaller.  The failure mode of that bias is a needless warning.  The failure
mode of the opposite bias is a job sent somewhere it cannot run, discovered
ninety minutes later.


Two questions with very different epistemic status, kept apart on purpose:

* **Fits now** is a fact -- arithmetic over the current node states.
* **When** is an estimate, and a *lower bound*.  Counting when nodes free up
  ignores every pending job ahead of you in priority order, so the real wait
  is never shorter and usually longer.  It is reported as "earliest possible",
  never as "your job will start at".

Where the backend can obtain a scheduler prediction, that supersedes this
estimate, because it accounts for the queue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..render import plural
from .model import JobShape, Node, capability_gap

__all__ = [
    "UNVERIFIED",
    "Capacity",
    "NodeFit",
    "assess_capacity",
    "hardware_ok",
    "node_fits",
]

#: Marker in a rejection reason meaning "the hardware might do this, but we
#: cannot tell". Distinct from a known incapability, because the two call for
#: different actions: one is the wrong cluster, the other is a labelling gap.
UNVERIFIED = "accelerator model unidentified"


def hardware_ok(node: Node, shape: JobShape) -> tuple[bool, tuple[str, ...]]:
    """Does this node have the right hardware, ignoring who is using it?

    Deliberately blind to state and occupancy.  "Wrong hardware" is a durable
    property of a queue; "drained" and "busy" are properties of today, and
    conflating them turns a temporary outage into a permanent-looking verdict.

    Sizes are checked against the node's *configured totals*, not its free
    space, for the same reason.
    """
    reasons: list[str] = []
    if shape.needs_gpu:
        if not node.is_gpu_node:
            reasons.append("no accelerator")
        elif node.gpus_total < shape.gpus_per_node:
            reasons.append(
                f"only {plural(node.gpus_total, 'accelerator')} installed, "
                f"need {shape.gpus_per_node}"
            )
        if node.accelerator is not None:
            spec = node.accelerator
            if shape.gpu_memory_gb and spec.memory_gb < shape.gpu_memory_gb:
                suffix = "" if spec.memory_certain else " (inferred from model)"
                reasons.append(
                    f"{spec.model} has {spec.memory_gb} GiB, "
                    f"need {shape.gpu_memory_gb:g}{suffix}"
                )
            for gap in capability_gap(spec, shape.requires):
                reasons.append(f"{spec.model} lacks {gap}")
        elif node.is_gpu_node and (shape.requires or shape.gpu_memory_gb):
            # The model could not be identified, and the job states a
            # requirement only the model can answer. Counting it as satisfied
            # would let the tool recommend a node that fails at run time -- so
            # it is set aside, and reported, rather than assumed capable.
            asked = list(shape.requires)
            if shape.gpu_memory_gb:
                asked.append(f">={shape.gpu_memory_gb:g} GiB")
            reasons.append(
                f"{UNVERIFIED}: cannot confirm {'+'.join(asked)}"
            )
    if node.cpus_total < shape.cpus_per_node:
        reasons.append(
            f"only {plural(node.cpus_total, 'CPU')} installed, "
            f"need {shape.cpus_per_node}"
        )
    # `memory_mb > 0`, because 0 is "the backend never mentioned memory", not
    # "this machine has none" -- the distinction `Node.memory_exhausted` states
    # outright, and the one place a size is read here. Reachable with ordinary
    # input: `SshPoolBackend.parse_host` on a host whose probe returns no
    # MEMTOTAL line yields 48 idle cores and `memory_mb=0`, and SGE and PBS
    # both leave it 0 where the resource is absent from the host report.
    #
    # Read as a size it is the smallest one possible, so it fails EVERY stated
    # `--mem`, and it fails it in the durable direction rather than the
    # cautious one: an empty `hardware_nodes` makes `ever_possible` False,
    # which `where` labels WRONG HW -- "no node of the right kind", legend "go
    # elsewhere; waiting will not help" -- over a machine whose RAM nobody
    # measured, and exits 1. That is the lie `memory_exhausted` names: a
    # shortage invented on a system that never mentioned memory. Claiming less
    # capacity is the bias throughout this module, but the claim has to rest on
    # a reading; here there was none.
    #
    # Deliberately only the DURABLE gate. The live check in `node_fits` still
    # keeps such a node out of `fitting_nodes`, so the answer becomes "would
    # queue" rather than "runs now" -- suppressing that one too would invent
    # room, which is the error this module is built to avoid.
    if (
        shape.memory_mb_per_node
        and node.memory_mb > 0
        and node.memory_mb < shape.memory_mb_per_node
    ):
        reasons.append(
            f"only {node.memory_mb // 1024} GiB RAM installed, "
            f"need {shape.memory_mb_per_node // 1024}"
        )
    # Node restrictions the job does not tolerate (k8s taints and equivalents).
    untolerated = [t for t in node.taints if t not in shape.tolerates]
    if untolerated:
        reasons.append(f"untolerated restriction: {', '.join(untolerated[:3])}")
    return (not reasons, tuple(reasons))


@dataclass
class NodeFit:
    """Whether one node can host one node's worth of the shape, and why not."""

    node: Node
    fits: bool
    reasons: tuple[str, ...] = ()


def node_fits(node: Node, shape: JobShape) -> NodeFit:
    """Check one node against the per-node slice of a shape, right now.

    Hardware capability is checked tri-state: a node whose accelerator cannot
    be identified is not excluded, because "unknown" is not "incapable" -- it
    is flagged upstream instead.
    """
    reasons: list[str] = []
    if not node.schedulable:
        reasons.append(f"state {node.state_raw}" if node.state_raw else "unschedulable")
    if node.name in shape.exclude:
        reasons.append("excluded")

    hw_ok, hw_why = hardware_ok(node, shape)
    if not hw_ok:
        reasons.extend(hw_why)

    if shape.needs_gpu and node.is_gpu_node and node.gpus_free < shape.gpus_per_node:
        reasons.append(f"{node.gpus_free}/{shape.gpus_per_node} accelerators free")
    if node.cpus_free < shape.cpus_per_node:
        reasons.append(f"{node.cpus_free}/{shape.cpus_per_node} CPUs free")
    # Checked whether or not the shape names a memory figure.
    #
    # It used to be conditional on `shape.memory_mb_per_node`, so a job that
    # asked only for cores was told a node with 44 of 48 cores idle and every
    # byte of its memory allocated would take it.  It would not: the scheduler
    # gives every job memory, so a job that names no figure gets the default
    # -- and there is none left to give.  See `Node.memory_exhausted`.
    if node.memory_exhausted:
        reasons.append("no memory free")
    elif shape.memory_mb_per_node and node.memory_free_mb < shape.memory_mb_per_node:
        reasons.append(
            f"{node.memory_free_mb // 1024}/{shape.memory_mb_per_node // 1024} GiB RAM free"
        )
    # Deduplicate while preserving order: hardware_ok and the live checks can
    # both flag the same shortage on a node that is small *and* busy.
    seen: dict[str, None] = {}
    for r in reasons:
        seen.setdefault(r, None)
    return NodeFit(node=node, fits=not seen, reasons=tuple(seen))


@dataclass
class Capacity:
    """Current room for a shape within one set of nodes."""

    fitting_nodes: tuple[str, ...] = ()
    #: Schedulable nodes that would fit if they were empty -- the hardware is
    #: right and only current occupancy is in the way.  Separates "wrong
    #: cluster" from "come back later".
    capable_nodes: tuple[str, ...] = ()
    #: Nodes with the right hardware regardless of state or occupancy.  Empty
    #: means this set can never run the shape; non-empty with an empty
    #: ``capable_nodes`` means the right machines exist but are all down.
    hardware_nodes: tuple[str, ...] = ()
    #: Hardware mismatch reason -> node count, so a "wrong hardware" verdict
    #: can always say *what* was wrong rather than just refusing.
    #: Why nodes lack the hardware, as reason -> node count.
    #:
    #: A node can contribute more than one reason, so **these values do not sum
    #: to a node count** -- use :attr:`considered` for the denominator.  Summing
    #: the histogram to get a total is the obvious mistake and it overcounts.
    hardware_reasons: dict[str, int] = field(default_factory=dict)
    #: How many nodes were assessed.  The denominator for every bucket above,
    #: recorded rather than derived because the reason histogram cannot be
    #: summed back into one.
    considered: int = 0
    #: How many nodes the shape needs, or 0 when the node list is known to be
    #: incomplete and a count-based verdict would therefore be unsafe.
    required_nodes: int = 0
    #: Nodes set aside only because their accelerator could not be identified,
    #: so a stated capability could not be confirmed.  Not "incapable" -- just
    #: unverifiable, which is a labelling problem rather than a hardware one.
    unverified_nodes: tuple[str, ...] = ()
    earliest_free: datetime | None = None
    blocked_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def nodes_available(self) -> int:
        return len(self.fitting_nodes)

    @property
    def ever_possible(self) -> bool:
        """Whether this queue could host the shape at all, however empty.

        Both halves matter: the right *kind* of node, and *enough* of them.
        Checking only the kind made a one-node queue asked for forty report
        possible, so it rendered as "would queue" -- an invitation to wait for
        capacity that cannot arrive, because the queue does not contain it.
        That is precisely the wrong-moment/wrong-place confusion the rest of
        this module exists to keep apart.

        ``required_nodes`` is 0 when the queue's node list is incomplete; there
        the count is not judged, since ruling a queue out on the strength of a
        resolution failure would be a worse error than leaving it in.

        **That suppression has to cover the empty case too, and it did not.**
        The short-circuit below reads an empty ``hardware_nodes`` as a verdict
        about hardware, which it only is when we believe we looked at the
        queue's nodes.  A queue declaring four nodes that all failed to resolve
        assessed ``considered=0`` of them and was reported ``WRONG HW`` -- whose
        legend is "go elsewhere; waiting will not help" -- directly above its
        own caveat saying "the queue claims 4 nodes but only 0 could be
        resolved".  One tree, asserting a fact about nodes and then admitting it
        had seen none of them, with the confident half in the verdict column.
        ``fit.py`` already found this short-circuit defeating
        ``count_is_complete`` and fixed it for the dry-run gate; the verdict
        itself was left behind.

        Narrow on purpose -- **only** where nothing at all was assessed *and*
        the caller flagged the list as incomplete.  A queue with zero declared
        nodes still answers False (there is genuinely no hardware there), and so
        does a partially-resolved queue whose resolved nodes are all the wrong
        kind, because that verdict rests on nodes actually examined.
        """
        if not self.hardware_nodes:
            return not self.considered and not self.required_nodes
        return len(self.hardware_nodes) >= self.required_nodes

    @property
    def too_few_nodes(self) -> bool:
        """The right kind of node, but never enough of them.

        Worth separating from "wrong hardware": both mean this queue cannot
        host the request, but one is answered by asking for fewer nodes and the
        other only by going somewhere else.  Labelling a queue with eleven
        perfectly suitable nodes "wrong hardware" because forty were asked for
        is accurate about the outcome and misleading about the cause.
        """
        return (
            bool(self.hardware_nodes)
            and len(self.hardware_nodes) < self.required_nodes
        )

    @property
    def capable_but_all_unavailable(self) -> bool:
        """Right hardware exists, but every such node is down or drained."""
        return bool(self.hardware_nodes) and not self.capable_nodes and not self.fitting_nodes

    def satisfies(self, shape: JobShape) -> bool:
        return self.nodes_available >= shape.nodes


#: The reason recorded for a node kept out ONLY by :attr:`JobShape.exclude`.
#: Exported so a consumer can tell "you excluded these" from a genuine
#: hardware mismatch without matching on a sentence.
EXCLUDED_REASON = "kept out by --exclude"


def assess_capacity(
    nodes: list[Node],
    shape: JobShape,
    free_times: dict[str, datetime] | None = None,
    count_is_complete: bool = True,
) -> Capacity:
    """Summarise how much of ``shape`` these nodes can host right now.

    ``count_is_complete`` should be False when the caller knows the node list
    is partial (a queue declaring more nodes than could be resolved).  It
    suppresses the "not enough nodes here, ever" verdict, which would otherwise
    rule out a queue because of a lookup failure.
    """
    fitting: list[str] = []
    capable: list[str] = []
    hardware: list[str] = []
    reasons: dict[str, int] = {}
    hw_reasons: dict[str, int] = {}

    unverified: list[str] = []
    for node in nodes:
        hw_ok, hw_why = hardware_ok(node, shape)
        excluded = node.name in shape.exclude
        if hw_ok and not excluded:
            hardware.append(node.name)
        else:
            if hw_ok:
                # Exclusion is the ONLY thing keeping this node out, so
                # `hardware_ok` returned no reasons and the loop below added
                # nothing: the histogram came out empty and the verdict was the
                # bare refusal this field exists to prevent -- "so a 'wrong
                # hardware' verdict can always say *what* was wrong rather than
                # just refusing". `--exclude` was the one path that reached it
                # with nothing to say, and the screen then read "no node here has
                # the right hardware" about nodes whose hardware was fine.
                #
                # Recorded only when the hardware is otherwise fine: a node that
                # genuinely does not fit already explains itself below, and
                # adding "also excluded" there would just crowd the histogram.
                hw_reasons[EXCLUDED_REASON] = hw_reasons.get(EXCLUDED_REASON, 0) + 1
            for r in hw_why:
                hw_reasons[r] = hw_reasons.get(r, 0) + 1
                if r.startswith(UNVERIFIED):
                    unverified.append(node.name)

        fit = node_fits(node, shape)
        if fit.fits:
            fitting.append(node.name)
            continue
        for r in fit.reasons:
            # Collapse numeric detail so the histogram stays readable: "3/4
            # free" and "1/4 free" are the same category.
            reasons[re.sub(r"\d+", "N", r)] = reasons.get(re.sub(r"\d+", "N", r), 0) + 1
        if hw_ok and node.schedulable and not excluded:
            capable.append(node.name)

    earliest: datetime | None = None
    if len(fitting) < shape.nodes and free_times:
        candidates = sorted(free_times[n] for n in capable if n in free_times)
        needed = shape.nodes - len(fitting)
        if len(candidates) >= needed:
            earliest = candidates[needed - 1]

    return Capacity(
        considered=len(nodes),
        required_nodes=shape.nodes if count_is_complete else 0,
        fitting_nodes=tuple(fitting),
        capable_nodes=tuple(capable),
        hardware_nodes=tuple(hardware),
        hardware_reasons=dict(sorted(hw_reasons.items(), key=lambda kv: -kv[1])),
        unverified_nodes=tuple(dict.fromkeys(unverified)),
        earliest_free=earliest,
        blocked_reasons=dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    )
