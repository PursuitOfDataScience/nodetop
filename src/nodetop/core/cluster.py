"""One snapshot of a cluster, whatever schedules it.

:class:`Cluster` is the entry point for everything.  It asks its backend for
nodes, queues, limits and identity once, wires nodes to queues, and hands the
result to the analysis functions.  Taking the snapshot in one place means every
number in a report describes the same instant rather than drifting across a
dozen independent queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .duration import format_duration
from .model import (
    Allocation,
    BackendCapabilities,
    Identity,
    Job,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
)

if TYPE_CHECKING:  # only for annotations: core never imports a backend
    from ..backends.base import Backend
    from ..runner import CapturingRunner

__all__ = ["Cluster"]


@dataclass
class Cluster:
    """A point-in-time view of nodes, queues, ceilings and identity."""

    backend_name: str = "unknown"
    queue_term: str = "queue"
    nodes: list[Node] = field(default_factory=list)
    queues: dict[str, Queue] = field(default_factory=dict)
    limits: dict[str, Limits] = field(default_factory=dict)
    identity: Identity | None = None
    capabilities: BackendCapabilities | None = None
    node_free_times: dict[str, datetime] = field(default_factory=dict, repr=False)
    taken_at: datetime | None = None
    #: Queries that failed, so a partial snapshot is visibly partial.
    errors: dict[str, str] = field(default_factory=dict)
    #: True when rebuilt from a recorded snapshot rather than a live control
    #: plane.  A recording cannot be dry-run against, so entitlement must not
    #: be presented as confirmable.
    replayed: bool = False
    _backend: Backend | None = field(default=None, repr=False, compare=False)
    #: Set by the snapshot command so it can dump what was recorded. Typed
    #: under TYPE_CHECKING rather than as `object`: the annotation costs no
    #: import at runtime and `object` meant every read of `.captured` was
    #: unchecked, which is the sort of thing that goes wrong silently in the
    #: one code path nobody exercises until a cluster is already broken.
    capture: CapturingRunner | None = field(
        default=None, repr=False, compare=False)

    #: Running jobs, fetched on first use rather than with the snapshot.
    _jobs: list[Job] | None = field(default=None, repr=False, compare=False)
    _allocations: dict[tuple[str, str], Allocation] | None = field(
        default=None, repr=False, compare=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        backend: Backend | None = None,
        *,
        with_free_times: bool = True,
        replayed: bool = False,
        taken_at: datetime | None = None,
    ) -> Cluster:
        """Take a snapshot, autodetecting the batch system if none is given.

        Every query is individually guarded: a cluster whose accounting
        database is unreachable should still report its nodes, with the failure
        recorded rather than silently turned into an empty result.

        ``taken_at`` must be supplied when replaying a recording, and is the
        moment the data was *captured*, not the moment it is read.  Defaulting
        to ``now()`` for a replay dates a week-old post-mortem to today, and --
        worse -- makes every wait estimate nonsense, because node free times are
        absolute instants compared against the clock: a node recorded as free in
        three hours reads as "overdue" once the snapshot ages past that.
        """
        if backend is None:
            from .. import backends

            backend = backends.detect()

        errors: dict[str, str] = {}

        def _try(label, fn, default):
            try:
                return fn()
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                return default

        nodes = _try("nodes", backend.load_nodes, [])
        queues = _try("queues", backend.load_queues, [])
        # One record per node. A duplicate -- from a re-read, a merged capture,
        # or a control plane listing a node twice -- would otherwise inflate
        # every count and every gauge in the report.
        by_name: dict[str, Node] = {}
        for node in nodes:
            by_name.setdefault(node.name, node)
        nodes = list(by_name.values())
        for q in queues:
            q.nodes = [by_name[n] for n in q.node_names if n in by_name]

        return cls(
            backend_name=backend.name,
            queue_term=backend.queue_term,
            nodes=nodes,
            queues={q.name: q for q in queues},
            limits=_try("limits", backend.load_limits, {}),
            identity=_try("identity", backend.load_identity, None),
            capabilities=backend.capabilities(),
            node_free_times=(
                _try("free_times", backend.load_node_free_times, {})
                if with_free_times
                else {}
            ),
            taken_at=taken_at or datetime.now(),
            errors=errors,
            replayed=replayed,
            _backend=backend,
        )

    # -- delegation ---------------------------------------------------------
    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """Ask the control plane about one (queue, account) pair."""
        if self._backend is None or self.replayed or not self.can_probe:
            return None
        return self._backend.probe(queue, shape, account)

    def format_nodelist(self, names) -> str:
        """Render a node set in the backend's own notation.

        Without a backend -- a hand-built or replayed cluster -- fall back to
        bracket notation rather than a plain join.  Spelling out every name
        turns a few thousand nodes into tens of kilobytes, and the bracket form
        is understood well beyond the scheduler that invented it.
        """
        if self._backend is None:
            from ..hostlist import collapse

            return collapse(sorted(names))
        return self._backend.format_nodelist(names)

    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        if self._backend is None:
            return []
        return self._backend.submit_flags(queue, shape)

    @property
    def can_probe(self) -> bool:
        """Whether entitlement can be confirmed rather than merely declared.

        Always False for a replay: the recording holds the answers to the
        queries that were made, not to a dry-run nobody ran.
        """
        if self.replayed:
            return False
        return bool(self.capabilities and self.capabilities.probe)

    # -- selection ----------------------------------------------------------
    @property
    def gpu_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.is_gpu_node]

    @property
    def cpu_nodes(self) -> list[Node]:
        """Nodes with no accelerator, decided by resource count, not hostname."""
        return [n for n in self.nodes if not n.is_gpu_node]

    @property
    def degraded_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.degraded]

    @property
    def unschedulable_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.schedulable]

    def usable_queues(self) -> list[Queue]:
        return [q for q in self.queues.values() if q.usable]

    def jobs(self) -> list[Job]:
        """Every running job, fetched once on first use.

        **Lazily, and that is a deliberate exception to the one-snapshot rule.**
        Every other number in a report comes from the same instant, because a
        report that mixes two instants can contradict itself. Job lists are
        different in two ways: almost no invocation asks for them, so fetching
        one with the snapshot would add a control-plane round trip to every run;
        and they are read while *browsing*, minutes after the overview was
        drawn, where the newest answer is the useful one rather than the
        consistent one.

        Cached after the first call so one browsing session does not re-query
        per node, which would be both slow and inconsistent between rows.
        """
        if self._jobs is None:
            if self._backend is None:
                self._jobs = []
            else:
                try:
                    self._jobs = list(self._backend.load_jobs())
                except Exception as exc:
                    # Recorded, not raised: the jobs view is an extra, and
                    # losing it must not take down the report it hangs off.
                    self.errors["jobs"] = f"{type(exc).__name__}: {exc}"
                    self._jobs = []
        return self._jobs

    def allocations(self) -> dict[tuple[str, str], Allocation]:
        """``(job id, node) -> that job's share of that node``, fetched once.

        Lazy for the same reasons as :meth:`jobs`, and cached for a stronger
        one: it is one whole-cluster query, so the first per-node view pays for
        every later one. A backend with no answer returns nothing and the caller
        says so, rather than showing a total in a column that means a share.
        """
        if self._allocations is None:
            got: list[Allocation] = []
            loader = getattr(self._backend, "load_allocations", None)
            if loader is not None:
                try:
                    got = list(loader())
                except Exception as exc:
                    # Recorded, not raised: this refines a view that works
                    # without it.
                    self.errors["allocations"] = f"{type(exc).__name__}: {exc}"
            self._allocations = {(a.job, a.node): a for a in got}
        return self._allocations

    def share_of(self, job: Job, node: str) -> Allocation | None:
        """What ``job`` holds on ``node``, or ``None`` if nothing can say.

        A job on one node needs no query -- its totals *are* its share -- which
        is most of them, and the whole answer on a backend that models a job as
        living on a single machine. Memory still comes from the allocation where
        one is available, because a job list reports what was requested rather
        than what was given.
        """
        got = self.allocations().get((job.id, node))
        if got is not None:
            return got
        if len(job.nodes) <= 1 and (not job.nodes or job.nodes[0] == node):
            return Allocation(job=job.id, node=node, cpus=job.cpus,
                              memory_mb=0, gpus=job.gpus)
        return None

    def jobs_on(self, node: str) -> list[Job]:
        """Running jobs occupying ``node``, biggest first.

        Ordered by what they hold rather than by id, because the question a
        node's job list answers is "what is in my way", and the answer is
        usually the largest one.
        """
        held = [j for j in self.jobs() if node in j.nodes]
        return sorted(held, key=lambda j: (-j.gpus, -j.cpus, j.id))

    def reachable_nodes(self) -> list[Node]:
        """Nodes owned by at least one queue that can actually start work.

        Free capacity on a node reachable only through a dead queue is not
        capacity, and :attr:`Queue.effective_free_gpus` has always known that
        -- it returns 0 for an unusable queue. :meth:`summary` did not: it
        counted every schedulable node's free resources, so an idle four-GPU
        node whose only partition was DOWN was reported as four free
        accelerators. Phantom capacity in the summary of a tool written to
        catch phantom capacity.

        A node in no queue at all counts as unreachable for the same reason:
        nothing can be submitted to it.
        """
        live = {n for q in self.queues.values() if q.usable for n in q.node_names}
        return [n for n in self.nodes if n.name in live]

    def unusable_queues(self) -> list[Queue]:
        return [q for q in self.queues.values() if not q.usable]

    def accelerator_exclude_list(self) -> list[str]:
        """Every accelerator node, for keeping CPU-only work off them."""
        return sorted(n.name for n in self.gpu_nodes)

    def limits_for(self, queue: str) -> Limits | None:
        """Best-guess ceilings for a queue.

        Systems overwhelmingly name a queue's limit set after the queue, so
        that is the fallback; a dry-run's ``effective_qos`` is more reliable
        and takes precedence wherever one is available.
        """
        q = self.queues.get(queue)
        if q is not None and q.limits_name and q.limits_name in self.limits:
            return self.limits[q.limits_name]
        return self.limits.get(queue)

    def effective_max_walltime(self, queue: str) -> int | None:
        """The wall limit that will actually bite, in seconds.

        A queue frequently reports no limit while its limit set caps the real
        ceiling -- so reading only the queue tells you there is none, and the
        job then pends forever on the one there is.  The binding value is the
        tighter of the two.
        """
        q = self.queues.get(queue)
        limits = self.limits_for(queue)
        candidates = [
            v
            for v in (
                q.max_walltime_seconds if q else None,
                limits.max_walltime_seconds if limits else None,
            )
            if v is not None
        ]
        return min(candidates) if candidates else None

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """A small JSON-safe digest, for logs and dashboards."""
        return {
            "backend": self.backend_name,
            "queue_term": self.queue_term,
            "identity": None if self.identity is None else {
                "user": self.identity.user,
                "accounts": len(self.identity.accounts),
                "qos": len(self.identity.qos),
                "groups": len(self.identity.groups),
                # A cluster-level property: when every account claims the same
                # list, the claim carries no per-account information at all.
                "entitlements_look_templated": (
                    self.identity.entitlements_look_templated
                ),
            },
            "taken_at": self.taken_at.isoformat() if self.taken_at else None,
            "can_confirm_entitlement": self.can_probe,
            "nodes": len(self.nodes),
            "accelerator_nodes": len(self.gpu_nodes),
            "accelerators_total": sum(n.gpus_total for n in self.nodes),
            # `effective_free_gpus`, so this agrees with every other free
            # figure in the tool: an accelerator on a node whose memory is
            # fully allocated cannot be given to anyone.
            "accelerators_free": sum(
                n.effective_free_gpus for n in self.reachable_nodes()
                if n.schedulable
            ),
            "unschedulable_nodes": len(self.unschedulable_nodes),
            "degraded_nodes": [n.name for n in self.degraded_nodes],
            "queues": len(self.queues),
            "unusable_queues": [q.name for q in self.unusable_queues()],
            "phantom_capacity": {
                q.name: len(q.idle_nodes)
                for q in self.unusable_queues()
                if q.idle_nodes
            },
            "max_walltime": {
                q: format_duration(self.effective_max_walltime(q)) for q in self.queues
            },
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary(), indent=indent, default=str)
