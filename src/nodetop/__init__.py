"""nodetop -- what a cluster actually has free, and what will accept your job.

The premise is that the obvious sources disagree with reality in specific,
repeatable ways, and that they do so on *every* batch system:

* a queue can advertise idle nodes while being unable to start anything;
* the entitlement table can claim access the submit filter will refuse;
* a dry-run can report success for a job that will then pend forever;
* no scheduler models the accelerator, so none of them can tell you the card
  lacks the dtype your job needs.

Each of those is handled by a module in :mod:`nodetop.core`, which knows nothing
about schedulers, and fed by an adapter in :mod:`nodetop.backends`, which knows
about exactly one.  A placement is reported as good only when every available
layer agrees -- and when a layer is unavailable, that is said out loud rather
than assumed benign.

Supported: Slurm, PBS Pro / OpenPBS / Torque, LSF, Grid Engine, Kubernetes, and
an unscheduled pool of machines reached over ssh.
"""

from ._version import VERSION as __version__  # noqa: N811
from .backends import Backend, BackendCapabilities
from .core import (
    ACCELERATORS,
    AcceleratorSpec,
    Blocker,
    Capacity,
    Cluster,
    Identity,
    JobShape,
    Limits,
    Node,
    NodeFit,
    Placement,
    Queue,
    Verdict,
    VerdictCategory,
    assess_capacity,
    evaluate,
    hardware_ok,
    identify_accelerator,
    node_fits,
    rank,
    supports,
)
from .exceptions import (
    CommandError,
    CommandTimeoutError,
    NoBackendError,
    NodetopError,
    SchedulerUnavailableError,
)
from .hostlist import collapse, expand

__all__ = [
    "ACCELERATORS",
    "AcceleratorSpec",
    "Backend",
    "BackendCapabilities",
    "Blocker",
    "Capacity",
    "Cluster",
    "CommandError",
    "CommandTimeoutError",
    "Identity",
    "JobShape",
    "NodetopError",
    "Limits",
    "NoBackendError",
    "Node",
    "NodeFit",
    "Placement",
    "Queue",
    "SchedulerUnavailableError",
    "Verdict",
    "VerdictCategory",
    "__version__",
    "assess_capacity",
    "collapse",
    "evaluate",
    "expand",
    "hardware_ok",
    "identify_accelerator",
    "node_fits",
    "rank",
    "supports",
]
