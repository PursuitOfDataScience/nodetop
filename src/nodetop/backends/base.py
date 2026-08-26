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

import re
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

__all__ = ["Backend", "BackendCapabilities", "count"]


#: The leading integer in a field, as an adapter should read one.
_LEADING_INT = re.compile(r"-?\d+")


def count(value: object, default: int = 0) -> int:
    """A resource count from whatever a scheduler put in the field.

    Every adapter needs this and each one wrote its own, which is how they
    diverged.  Two properties, both learned from a defect:

    * **A value that is not a number is not a crash.**  PBS reports
      ``resources_available.ngpus = unlimited`` for an uncapped resource, and
      site scripts emit ``4x`` and ``8gb``.  ``int()`` on any of those raised
      straight through the node parser, so one odd field on one node emptied
      the entire node list -- and an empty node list is reported as "wrong
      backend, or the control plane is down", which is a misdiagnosis rather
      than a gap.
    * **A count below zero is meaningless, and letting one through is actively
      harmful.**  ``cpus_free`` is ``total - alloc``, so an allocation of -5
      against a total of 0 reports five free CPUs that do not exist.

    Accepts the JSON scalars a ``-F json`` or REST backend yields as well as
    the strings a text one does.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    m = _LEADING_INT.match(str(value).strip())
    if not m:
        return default
    return max(0, int(m.group()))


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

    def warm(self) -> None:
        """Fetch anything the loaders will ask for *inside* their own work.

        Optional, and empty for most backends. It exists because a query issued
        from within another query's handler goes out later than it needs to. On
        Slurm, `load_nodes` reached for `scontrol show config` partway through
        its own parse, so that query left ~70 ms after the other four -- and
        during one bad spell on a live controller it took **2.0s on two runs in
        five** while the four first-wave queries stayed normal. Asked on its own
        that query is 17 ms, twenty times out of twenty.

        Honest about what that proves: the stalls stopped when this changed, but
        an interleaved A/B of both versions a while later found **no stalls in
        either**, so the controller had calmed down and the 12-for-12 clean runs
        after the change are not proof it caused them. What stands without the
        measurement is the staging: one wave of queries rather than one wave and
        a straggler, for the same queries at the same instant.
        """
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
