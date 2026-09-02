"""Scheduler-neutral model and reasoning.

Nothing in this package imports a backend, and nothing in it knows what a
scheduler is.  That separation is the point: the interesting logic -- what
blocks a job, which nodes are genuinely capable, how to rank the options, what
may honestly be claimed -- is identical on Slurm, PBS, LSF, SGE, Kubernetes and
a bare pool of machines.  Only *acquiring* the facts differs.
"""

from .capacity import Capacity, NodeFit, assess_capacity, hardware_ok, node_fits
from .cluster import Cluster
from .duration import format_duration, format_wait, parse_duration, parse_timestamp
from .fit import Placement, evaluate, rank
from .hardware import ACCELERATORS, AcceleratorSpec, identify_accelerator, supports
from .model import (
    BackendCapabilities,
    Blocker,
    Identity,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
    category_label,
)

__all__ = [
    "ACCELERATORS",
    "AcceleratorSpec",
    "BackendCapabilities",
    "Blocker",
    "Capacity",
    "Cluster",
    "Identity",
    "JobShape",
    "Limits",
    "Node",
    "NodeFit",
    "Placement",
    "Queue",
    "Verdict",
    "VerdictCategory",
    "assess_capacity",
    "category_label",
    "evaluate",
    "format_duration",
    "format_wait",
    "hardware_ok",
    "identify_accelerator",
    "node_fits",
    "parse_duration",
    "parse_timestamp",
    "rank",
    "supports",
]
