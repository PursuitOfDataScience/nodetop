"""The scheduler adapter interface.

Everything scheduler-specific lives behind this protocol.  A backend's job is
narrow: turn whatever its system reports into the neutral objects in
:mod:`nodetop.core.model`, and answer one question honestly --
:meth:`Backend.can_probe`.

That last method is the reason the interface is shaped this way.  Batch systems
differ sharply in whether they will tell you the truth *before* you commit:

=================  ======================================  ==================
system             dry-run facility                        entitlement check
=================  ======================================  ==================
Slurm              ``sbatch --test-only``                  confirmed
SGE / UGE          ``qsub -w v`` (verify only)             confirmed
Kubernetes         ``--dry-run=server`` + ``auth can-i``   confirmed
PBS Pro / Torque   none                                    declared only
LSF                none                                    declared only
ssh pool           no scheduler at all                      n/a
=================  ======================================  ==================

When a backend cannot confirm, nodetop must say "declared, unconfirmed" rather
than "allowed".  Silently presenting a declared entitlement as a verified one
is the exact failure this tool exists to prevent, so the absence of a probe is
reported, not papered over.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..core.model import (
    BackendCapabilities,
    Identity,
    Job,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
)

__all__ = ["Backend", "BackendCapabilities"]


@runtime_checkable
class Backend(Protocol):
    """Adapter for one batch system."""

    #: Short identifier, e.g. ``"slurm"``.
    name: str
    #: What this system calls a :class:`~nodetop.core.model.Queue`.
    queue_term: str

    @classmethod
    def detect(cls) -> bool:
        """Whether this system appears to be present and usable here."""
        ...

    def capabilities(self) -> BackendCapabilities:
        """What this backend can establish; see :class:`BackendCapabilities`."""
        ...

    def load_nodes(self) -> list[Node]:
        ...

    def load_queues(self) -> list[Queue]:
        ...

    def load_limits(self) -> dict[str, Limits]:
        """Resource ceilings, keyed by the name a queue would reference."""
        ...

    def load_identity(self) -> Identity:
        ...

    def load_jobs(self) -> list[Job]:
        """Running jobs, for explaining why a node is busy.

        Optional: a backend with no notion of a job list returns nothing, and
        the caller says so rather than showing an empty table that reads as "no
        jobs here".  Separate from the other loaders because it is fetched
        lazily -- most invocations never ask which jobs hold a node, and it is
        another control-plane round trip.
        """
        return []

    def load_node_free_times(self) -> dict[str, datetime]:
        """Latest end time per node, from currently running work."""
        ...

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """Ask the control plane whether this submission would be accepted.

        Must be **read-only**: no job may be created, queued or charged.
        Returns ``None`` when the system offers no way to ask.
        """
        ...

    def format_nodelist(self, names: Iterable[str]) -> str:
        """Render a node set in this system's own notation."""
        ...

    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        """The arguments that would request this shape here.

        Useful on its own: it is how a caller builds a submission that matches
        exactly what was checked.
        """
        ...
