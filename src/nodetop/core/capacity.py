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
    if shape.memory_mb_per_node and node.memory_mb < shape.memory_mb_per_node:
        reasons.append(
            f"only {node.memory_mb // 1024} GB RAM installed, "
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
    if shape.memory_mb_per_node and node.memory_free_mb < shape.memory_mb_per_node:
        reasons.append(
            f"{node.memory_free_mb // 1024}/{shape.memory_mb_per_node // 1024} GB RAM free"
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
        """
        if not self.hardware_nodes:
            return False
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
