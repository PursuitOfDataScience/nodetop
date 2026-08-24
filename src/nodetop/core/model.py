"""The scheduler-neutral domain model.

Nothing in this module knows what a scheduler is.  It defines the six things
every batch system on a multi-node cluster has, whatever it calls them:

===================  ====================================================
nodetop               what each system calls it
===================  ====================================================
:class:`Node`        node (Slurm) / vnode (PBS) / host (LSF, SGE) /
                     Node (Kubernetes) / machine (ssh pool)
:class:`Queue`       partition (Slurm) / queue (PBS, LSF, SGE) /
                     namespace + quota (Kubernetes) / the pool itself
:class:`Limits`      QOS and association (Slurm) / queue limits (PBS, LSF) /
                     resource quota set (SGE) / ResourceQuota (Kubernetes)
:class:`Identity`    account and QOS (Slurm) / ACL and group (PBS) /
                     USERS and GROUPS (LSF) / userset (SGE) /
                     ServiceAccount and RBAC (Kubernetes)
:class:`JobShape`    job (Slurm, PBS, LSF, SGE) / Pod or Job spec (Kubernetes)
:class:`Verdict`     ``sbatch --test-only`` (Slurm) / ``qsub -w v`` (SGE) /
                     ``kubectl --dry-run=server`` (Kubernetes) /
                     nothing at all (PBS, LSF, ssh pool)
===================  ====================================================

Each backend declares its own word for a queue in ``Backend.queue_term``, and
the reports use it -- so a Slurm user reads "partition" and a Kubernetes user
reads "namespace" out of the same code.

The reasoning built on top -- what blocks a job, which nodes are capable, how
to rank the options -- is therefore shared by every backend.  Only *acquiring*
these objects is scheduler-specific, and that lives behind
:class:`nodetop.backends.base.Backend`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from ..render import sanitize
from .hardware import AcceleratorSpec, supports

__all__ = [
    "Allocation",
    "BackendCapabilities",
    "Blocker",
    "Identity",
    "Job",
    "JobShape",
    "Limits",
    "Node",
    "Queue",
    "Verdict",
    "VerdictCategory",
    "capability_gap",
    "split_reason",
]


# ---------------------------------------------------------------------------
# what a backend can establish
# ---------------------------------------------------------------------------
@dataclass
class BackendCapabilities:
    """What a backend can and cannot establish about its cluster.

    Lives in the core rather than in the backends because the *reasoning*
    depends on it: a report has to say "declared, unconfirmed" when no
    dry-run exists, and that decision belongs with the logic that renders the
    verdict, not with the adapter that happens to lack the feature.

    Stated explicitly, never inferred, so a backend has to own its limitations
    instead of having them assumed away.
    """

    #: True when the backend can obtain a real control-plane verdict **here**:
    #: the batch system offers a dry-run *and* its client is installed on this
    #: host.  Every decision about whether entitlement is confirmed or merely
    #: declared reads this one.
    probe: bool = False
    #: True when the batch system HAS a dry-run at all -- a property of the
    #: scheduler, independent of this machine.
    #:
    #: Split from :attr:`probe` because one field was answering two questions
    #: and got both wrong in the reference table: SGE reported "no dry-run"
    #: purely because ``qsub`` was absent from a Slurm login node, while
    #: Kubernetes hardcoded ``probe=True`` and so advertised confirmability on
    #: a host with no ``kubectl`` at all.  The ``●``/``○`` column already says
    #: what is usable here; the capability column must say what the system can
    #: do.
    probe_supported: bool = False
    #: The command used, so a report can cite it.
    probe_command: str = ""
    #: True when the system exposes resource ceilings separately from queues.
    limits: bool = True
    #: True when the caller's entitlements can be enumerated.
    identity: bool = True
    #: True when per-node free times can be estimated from running work.
    free_times: bool = True
    #: Caveats worth printing next to this backend's results.
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# blockers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Blocker:
    """One reason a job cannot run somewhere.

    ``fatal`` separates "no job of any shape can run here" from "this
    particular request does not fit", which is the difference between
    abandoning a queue and resizing a job.
    """

    code: str
    detail: str
    fatal: bool = True
    #: A few words for an overview line.  ``detail`` explains; this labels.
    short: str = ""

    def __post_init__(self) -> None:
        # `detail` is assembled from scheduler text by several backends.
        # object.__setattr__ because this dataclass is frozen.
        object.__setattr__(self, "detail", sanitize(self.detail))
        object.__setattr__(self, "short", sanitize(self.short))

    @property
    def label(self) -> str:
        """Terse phrase for a summary line, falling back to the code."""
        return self.short or self.code.lower().replace("_", " ")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.code}: {self.detail}"


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
#: Node conditions that mean "will not accept new work", normalised across
#: schedulers.  Backends translate their own vocabulary into these.
BLOCKING_CONDITIONS = frozenset(
    {
        "DOWN",          # slurm DOWN / pbs down / lsf unavail / k8s NotReady
        "DRAIN",         # slurm DRAIN / pbs offline / k8s cordoned (unschedulable)
        "MAINT",         # reserved for maintenance
        "FAIL",          # hardware or slurmd/pbs_mom failure
        "UNKNOWN",       # state cannot be determined
        "RESERVED",      # held by a reservation this job is not part of
        "POWERSAVE",     # powered down; may or may not come back in time
    }
)


@dataclass
class Node:
    """One machine, as the scheduler sees it plus what the scheduler omits.

    Backends fill this in; the fields are chosen to be the intersection of
    what every batch system actually reports, not the union.
    """

    name: str
    #: The scheduler's own state string, kept verbatim for display.
    state_raw: str = ""
    #: Normalised conditions drawn from :data:`BLOCKING_CONDITIONS`, plus any
    #: informational ones the backend wants to pass through.
    conditions: frozenset[str] = frozenset()
    cpus_total: int = 0
    cpus_alloc: int = 0
    memory_mb: int = 0
    memory_alloc_mb: int = 0
    gpus_total: int = 0
    gpus_alloc: int = 0
    accelerator: AcceleratorSpec | None = None
    #: Free-form labels: Slurm features, PBS resources, k8s labels.
    labels: tuple[str, ...] = ()
    #: Queues/partitions/namespaces this node serves.
    queues: tuple[str, ...] = ()
    #: Why the node is unavailable, if the scheduler records a reason.
    reason: str = ""
    #: True when the control plane has lost contact with the node agent.
    unreachable: bool = False
    #: Scheduling restrictions the node imposes (k8s taints, and any
    #: equivalent).  A job must tolerate these to land here.
    taints: tuple[str, ...] = ()
    #: Whether the scheduler treats memory as a consumable resource here, so
    #: that a node with none unallocated can host nothing.  True everywhere
    #: except a Slurm cluster configured without `_MEMORY` in
    #: `SelectTypeParameters`, where `memory_alloc_mb` records what jobs asked
    #: for rather than a ceiling the scheduler enforces.  See
    #: :attr:`memory_exhausted`.
    memory_consumable: bool = True

    def __post_init__(self) -> None:
        # Scheduler-supplied free text, scrubbed of control characters once, at
        # the boundary where it enters the model -- so every backend is covered
        # by one rule instead of six, and `--replay` of someone else's snapshot
        # cannot repaint the reader's terminal. See render.sanitize.
        self.name = sanitize(self.name)
        self.state_raw = sanitize(self.state_raw)
        self.reason = sanitize(self.reason)
        self.labels = tuple(sanitize(x) for x in self.labels)
        self.taints = tuple(sanitize(x) for x in self.taints)

    # -- availability -------------------------------------------------------
    @property
    def schedulable(self) -> bool:
        """Whether the scheduler would place new work here.

        This is about *acceptance*, not emptiness: a fully allocated node is
        schedulable, it just has nothing free right now.
        """
        return not (self.conditions & BLOCKING_CONDITIONS)

    @property
    def degraded(self) -> bool:
        """Schedulable, but with a reason suggesting impaired hardware.

        Deliberately narrow: a node that is already unschedulable is not
        "degraded", it is simply out.  The value is catching the node the
        scheduler still hands out while it runs at a fraction of its speed.
        """
        if not self.schedulable or not self.reason:
            return False
        # The reason TEXT only -- the trailing [who@when] would let an
        # operator's username match a hint (see :func:`split_reason`) -- and
        # matched on word boundaries rather than as bare substrings. Stripping
        # the stamp is not enough on its own: a name written into the prose
        # still collides, and every short hint had a plausible one.
        # "drained by Fang" matched `fan`, which is the same false positive the
        # stamp-stripping was introduced to fix; "Rebecca" matches `ecc` and
        # "Xidong" matches `xid`.
        return _hints_match(split_reason(self.reason)[0], DEGRADED_HINTS)

    # -- capacity -----------------------------------------------------------
    @property
    def is_gpu_node(self) -> bool:
        """Accelerator presence, decided by the resource count -- never the name.

        Clusters routinely have a ``gpu``-prefixed node with no GPU and an
        unremarkably-named node with four.  Filtering on the hostname is how
        CPU work ends up occupying an accelerator.
        """
        return self.gpus_total > 0

    @property
    def gpus_free(self) -> int:
        return max(0, self.gpus_total - self.gpus_alloc)

    @property
    def cpus_free(self) -> int:
        return max(0, self.cpus_total - self.cpus_alloc)

    @property
    def memory_free_mb(self) -> int:
        """Memory the scheduler considers unallocated (not the OS's free RAM)."""
        return max(0, self.memory_mb - self.memory_alloc_mb)

    @property
    def memory_exhausted(self) -> bool:
        """Every byte the scheduler can hand out here is already handed out.

        A node in this state can start nothing, however many cores are idle.
        Slurm allocates memory to every job -- ``DefMemPerCPU`` if the site
        sets one, the whole node if it does not -- so there is no such thing
        as a job that needs cores and no memory.  Measured on the cluster this
        was written against: ``caslake`` advertised **2322 free cores** across
        190 nodes, of which 2035 sat on 47 nodes whose memory was fully
        allocated to a handful of small jobs.  A four-core job sent at the
        biggest number on the screen would pend indefinitely.

        ``memory_mb <= 0`` means the backend does not report memory at all,
        which is not the same as reporting none: the answer there is "cannot
        tell", so the constraint is not applied.  Claiming less about capacity
        is the bias everywhere in this file, but inventing a shortage on a
        system that never mentioned memory would be its own kind of lie.

        :attr:`memory_consumable` is the same distinction one level up: a
        scheduler that does not account for memory when it places work has an
        ``AllocMem`` that is a record of requests, not a ceiling, and reading
        it as one would report a whole cluster as full.
        """
        return (self.memory_consumable and self.memory_mb > 0
                and self.memory_free_mb <= 0)

    @property
    def effective_free_cpus(self) -> int:
        """Free cores that still have memory behind them.

        The counterpart of :attr:`Queue.effective_free_cpus` one level down,
        and for the same reason: a count that cannot be acted on does not
        belong in an answer to "where is there room".
        """
        return 0 if self.memory_exhausted else self.cpus_free

    @property
    def effective_free_gpus(self) -> int:
        """Free accelerators that still have memory behind them.

        A GPU job needs host memory too, so an accelerator on a node with
        none allocatable is as unreachable as an idle core there.
        """
        return 0 if self.memory_exhausted else self.gpus_free

    @property
    def has_room(self) -> bool:
        """Something here could actually be allocated right now.

        The predicate every "with room" count and every ``--free`` filter
        shares, so they cannot drift apart -- and so that adding a reason a
        node is unreachable is one edit rather than six.
        """
        return self.schedulable and bool(self.effective_free_cpus
                                         or self.effective_free_gpus)

    @property
    def idle(self) -> bool:
        return self.schedulable and self.cpus_alloc == 0 and self.gpus_alloc == 0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        gpu = ""
        if self.is_gpu_node:
            model = self.accelerator.model if self.accelerator else "?"
            gpu = f" {self.gpus_free}/{self.gpus_total}x{model}"
        return f"{self.name} [{self.state_raw}]{gpu}"


#: Substrings in a node's reason field that suggest impaired-but-running
#: hardware.  Keyword matching is a floor, not a ceiling: it only catches what
#: an administrator wrote down.  See the README on what this cannot detect.
#: ``reason [who@when]`` -- Slurm stamps every drain reason with the operator
#: who set it and the second they set it.  Tolerant on purpose: the who may be
#: empty and the when may be any format the site uses.
_REASON_STAMP = re.compile(r"\s*\[(?P<who>[^\[\]@]*)@(?P<when>[^\[\]]*)\]\s*$")


def split_reason(reason: str | None) -> tuple[str, str, str]:
    """Separate a scheduler reason from its "who set it, and when" suffix.

    Returns ``(text, who, when)``, with empty strings for whatever is absent.

    Worth doing rather than reading the raw string two ways:

    * **Grouping.** Keying nodes on the raw reason splits one maintenance
      window into a row per *second* an administrator spent typing -- fifty
      nodes drained for the same cause rendered as five findings that differed
      only in a timestamp nobody asked about.
    * **Keyword matching.** :data:`DEGRADED_HINTS` contains short words like
      ``fan``, ``slow`` and ``clock``.  Run against the whole string they also
      match the *operator's username*, so an admin called ``fanl`` marked every
      node they touched as thermally impaired.

    The suffix is Slurm's convention, but the function is tolerant rather than
    Slurm-specific: a reason with no stamp comes back unchanged.
    """
    if not reason:
        return "", "", ""
    m = _REASON_STAMP.search(reason)
    if not m:
        return reason.strip(), "", ""
    return reason[: m.start()].strip(), m.group("who").strip(), m.group("when").strip()


#: Hints of four or more characters are matched as PREFIXES, so `throttl`
#: catches "throttled" and "throttling"; shorter ones are matched as whole
#: words, because three letters collide with ordinary names. Compiled once.
_HINT_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


#: A lowercase-to-uppercase transition, which is a word boundary in every
#: convention that matters here even though ``\b`` does not think so.
_CAMEL_SEAM = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _hints_match(text: str, hints: tuple[str, ...]) -> bool:
    """Does ``text`` contain any hint, at a word boundary?

    **CamelCase humps count as boundaries.** Requiring ``\b`` alone was right
    about names and wrong about the strings schedulers actually emit: Kubernetes
    writes conditions as ``MemoryPressure`` and NVIDIA writes flags as
    ``HwPowerBrake``, and in both the hint begins mid-token, preceded by a word
    character. Seams are split before matching, so ``MemoryPressure`` is read as
    two words while ``Fang`` and ``Rebecca`` remain one.
    """
    if not text:
        return False
    pattern = _HINT_CACHE.get(hints)
    if pattern is None:
        parts = [
            rf"\b{re.escape(h)}\b" if len(h) < 4 else rf"\b{re.escape(h)}"
            for h in hints
        ]
        pattern = re.compile("|".join(parts), re.IGNORECASE)
        _HINT_CACHE[hints] = pattern
    # Both spellings, because the seam cuts both ways: splitting is what lets
    # `pressure` match "MemoryPressure", and NOT splitting is what lets
    # `infiniband` match "InfiniBand" -- which the split turns into
    # "Infini Band". Searching each is cheaper than deciding which a given hint
    # needs, and neither form can match a name that the other rejects.
    return (
        pattern.search(text) is not None
        or pattern.search(_CAMEL_SEAM.sub(" ", text)) is not None
    )


#: Substrings that mark a still-schedulable node as impaired.
#:
#: Two groups, because the table started as a thermal/ECC list and that is not
#: where a GPU cluster actually loses nodes. The additions are the failure modes
#: an operator most often leaves *schedulable* -- an accelerator that has
#: dropped off the PCIe bus, a dead NVLink, a fabric link down -- which is
#: precisely the case this property exists to catch, and none of them were
#: matched. A node with a dead InfiniBand link still accepts work and fails
#: every multi-node job placed on it.
#:
#: **Every addition is six or more characters, or contains a space.** The short
#: entries below (``fan``, ``ecc``, ``xid``) predate the rule and are matched
#: only against the reason TEXT for that reason: run against the whole field
#: they also hit the operator's username, and an admin called ``fanl`` once
#: marked every node they touched as thermally impaired. A new hint short
#: enough to collide with a surname would reintroduce that.
DEGRADED_HINTS = (
    # thermal, power and clocks
    "powerbrake", "power brake", "throttl", "thermal", "overheat", "clock",
    "slow", "degraded", "performance", "unhealthy", "pressure", "fan",
    # accelerator memory faults
    "ecc", "retired page", "remapping", "row remap", "uncorrectable",
    "double bit",
    # the accelerator itself
    "xid", "fell off the bus", "nvidia-smi", "nvlink", "dcgm",
    # the fabric: fatal for multi-node work, and the node stays schedulable
    "link down", "infiniband", "ib link",
    # host
    "out of memory",
)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
@dataclass
class Job:
    """One running job, as far as it explains a node's occupancy.

    Deliberately thin. This exists to answer "what is using this machine", not
    to be a job-monitoring model: the fields are the ones that tell you whether
    the thing in your way is yours, how much of the node it holds, and how long
    it has left. Anything more is `squeue`'s job.
    """

    id: str
    user: str = ""
    account: str = ""
    queue: str = ""
    name: str = ""
    state: str = ""
    #: Nodes this job occupies. One job spans many; one node hosts many.
    nodes: tuple[str, ...] = ()
    cpus: int = 0
    gpus: int = 0
    #: Time used so far, and time left, as the scheduler reports them.
    elapsed: str = ""
    remaining: str = ""

    def __post_init__(self) -> None:
        # Same boundary as Node: a job NAME is user-authored free text and goes
        # straight into a table cell. See render.sanitize.
        for attr in ("id", "user", "account", "queue", "name", "state",
                     "elapsed", "remaining"):
            setattr(self, attr, sanitize(getattr(self, attr)))
        self.nodes = tuple(sanitize(n) for n in self.nodes)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.id} {self.user} on {len(self.nodes)} node(s)"


@dataclass
class Allocation:
    """One job's share of ONE node, as the scheduler actually assigned it.

    A job list reports totals over every node a job holds, and in a per-node
    view that is not merely imprecise -- it is a number the reader knows to be
    impossible.  A 42-node job appeared as **512 cores on a 48-core machine**,
    with a `x42` marker nobody could decode: "the cpu column doesn't make any
    sense.  what do the column entries mean?"  Its actual share of that node
    was seven cores and seven gigabytes.

    Only the scheduler knows the split -- a job need not be allocated uniformly
    -- so this is fetched rather than derived.  Dividing the total by the node
    count would be a guess dressed as a fact.
    """

    job: str
    node: str
    cpus: int = 0
    memory_mb: int = 0
    gpus: int = 0

    def __post_init__(self) -> None:
        self.job = sanitize(self.job)
        self.node = sanitize(self.node)


# ---------------------------------------------------------------------------
# queues
# ---------------------------------------------------------------------------
@dataclass
class Queue:
    """A named submission target with its own policy.

    Slurm calls this a partition, PBS/LSF/SGE call it a queue, Kubernetes has
    no single equivalent so the backend maps a namespace plus its quota onto
    one.  ``partition`` is available as an alias for Slurm users.
    """

    name: str
    #: Verbatim scheduler state, for display.
    state_raw: str = "UP"
    #: True when the queue accepts new submissions at all.
    enabled: bool = True
    #: True when the queue will actually *start* what it accepts.  These are
    #: two separate switches in PBS (``enabled``/``started``) and LSF, and
    #: conflating them hides the case where jobs queue up forever by design.
    started: bool = True
    hidden: bool = False
    #: Entitlement lists, as the scheduler declares them.  Empty means
    #: unrestricted; a single ``"none"`` entry means nobody.
    allow_accounts: tuple[str, ...] = ()
    deny_accounts: tuple[str, ...] = ()
    allow_users: tuple[str, ...] = ()
    deny_users: tuple[str, ...] = ()
    allow_groups: tuple[str, ...] = ()
    allow_qos: tuple[str, ...] = ()
    max_walltime_seconds: int | None = None
    max_nodes: int | None = None
    min_nodes: int = 0
    #: Declared node total, which can exceed the nodes we could resolve.
    declared_nodes: int = 0
    node_names: tuple[str, ...] = ()
    requires_reservation: bool = False
    is_default: bool = False
    priority: int = 0
    #: Name of the limit set that applies here, if it differs from the queue.
    limits_name: str | None = None
    #: Destinations this queue forwards to, for a routing queue.  Such a queue
    #: is submittable but owns no nodes -- its capacity belongs to the
    #: destinations -- so it is not a placement target in its own right.
    forwards_to: tuple[str, ...] = ()
    nodes: list[Node] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Same boundary as Node: the state string and every ACL entry are
        # scheduler text and all of them reach a table cell. See render.sanitize.
        self.name = sanitize(self.name)
        self.state_raw = sanitize(self.state_raw)
        if self.limits_name is not None:
            self.limits_name = sanitize(self.limits_name)
        for attr in ("allow_accounts", "deny_accounts", "allow_users",
                     "deny_users", "allow_groups", "allow_qos", "forwards_to"):
            setattr(self, attr, tuple(sanitize(x) for x in getattr(self, attr)))

    @property
    def partition(self) -> str:
        """Alias for :attr:`name`, for Slurm users."""
        return self.name

    # -- structural verdicts ------------------------------------------------
    def structural_blockers(self) -> list[Blocker]:
        """Blockers applying to *any* job, independent of who is asking.

        These are the ones that make a queue's advertised idle nodes a mirage.
        """
        out: list[Blocker] = []
        if not self.enabled:
            out.append(Blocker(
                "QUEUE_DISABLED", f"state={self.state_raw} (accepts nothing)",
                short=self.state_raw.split()[0] if self.state_raw else "disabled",
            ))
        elif not self.started:
            # Only interesting when the queue *does* accept work: that is the
            # surprising case, where submissions succeed and pile up forever.
            # A disabled queue is already covered above, and reporting both
            # would be two blockers for one switch.
            out.append(
                Blocker(
                    "QUEUE_NOT_STARTED",
                    f"state={self.state_raw}: accepts jobs but will not start them",
                    short="accepts but never starts",
                )
            )
        if _is_none(self.allow_accounts):
            out.append(Blocker(
                "NO_ACCOUNTS", "account allowlist is empty (nobody may submit)",
                short="no accounts",
            ))
        if _is_none(self.allow_qos):
            out.append(Blocker(
                "NO_QOS", "QOS allowlist is empty (no QOS is permitted)",
                short="no QOS",
            ))
        if _is_none(self.allow_users):
            out.append(Blocker(
                "NO_USERS", "user allowlist is empty (nobody may submit)",
                short="no users",
            ))
        if self.nodes and not self.routes and not any(n.schedulable for n in self.nodes):
            out.append(
                Blocker(
                    "ALL_NODES_UNSCHEDULABLE",
                    f"all {len(self.nodes)} nodes are down, drained or unreachable",
                    short=f"all {len(self.nodes)} nodes down",
                )
            )
        if self.requires_reservation:
            out.append(
                Blocker("REQUIRES_RESERVATION", "needs an explicit reservation",
                        fatal=False, short="needs a reservation")
            )
        return out

    def access_blockers(
        self,
        accounts: set[str] | None = None,
        qos: set[str] | None = None,
        groups: set[str] | None = None,
        user: str | None = None,
    ) -> list[Blocker]:
        """Blockers arising from this identity's declared entitlements.

        Every verdict here comes from what the scheduler *claims*.  On systems
        with a submit filter or admission webhook that claim can be generous
        fiction, which is what :class:`Verdict` is for.
        """
        out: list[Blocker] = []
        if _denied(self.allow_accounts, self.deny_accounts, accounts) is False:
            shown = ",".join(self.allow_accounts[:6]) or "none"
            out.append(
                Blocker("ACCOUNT_NOT_ALLOWED", f"none of your accounts are in [{shown}]")
            )
        if _denied(self.allow_qos, (), qos) is False:
            shown = ",".join(self.allow_qos[:6]) or "none"
            out.append(Blocker("QOS_NOT_ALLOWED", f"allowed QOS: [{shown}]"))
        if _denied(self.allow_groups, (), groups) is False:
            shown = ",".join(self.allow_groups[:6])
            out.append(Blocker("GROUP_NOT_ALLOWED", f"allowed groups: [{shown}]"))
        if user is not None:
            allowed = _denied(self.allow_users, self.deny_users, {user})
            if allowed is False:
                out.append(Blocker("USER_NOT_ALLOWED", f"{user} is not permitted here"))
        return out

    @property
    def routes(self) -> bool:
        """Whether this queue forwards work elsewhere instead of running it."""
        return bool(self.forwards_to)

    @property
    def usable(self) -> bool:
        """True when *some* job could start here, ignoring who is asking."""
        return not any(b.fatal for b in self.structural_blockers())

    # -- capacity -----------------------------------------------------------
    @property
    def schedulable_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.schedulable]

    @property
    def idle_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.idle]

    @property
    def gpus_total(self) -> int:
        return sum(n.gpus_total for n in self.nodes)

    @property
    def gpus_free(self) -> int:
        return sum(n.gpus_free for n in self.schedulable_nodes)

    @property
    def cpus_total(self) -> int:
        return sum(n.cpus_total for n in self.nodes)

    @property
    def cpus_free(self) -> int:
        return sum(n.cpus_free for n in self.schedulable_nodes)

    @property
    def effective_free_nodes(self) -> int:
        """Free nodes you can actually use -- zero when the queue cannot.

        The plain count stays available via ``len(q.idle_nodes)``; this is
        what any summary should show, so a dead queue never advertises
        phantom capacity.

        :attr:`Node.idle` means "nothing is running here", which is not the
        same as "something could".  A node with every core free and all of its
        memory allocated is idle and unusable at once -- reachable on
        Kubernetes, where a pod may request memory with no CPU request at all,
        so the node reports zero allocated CPU and no allocatable memory.  It
        would have been counted here, in the one column that claims a node is
        wholly free.
        """
        if not self.usable:
            return 0
        return sum(1 for n in self.idle_nodes if n.has_room)

    @property
    def effective_free_cpus(self) -> int:
        """Free cores you can actually use -- the honest measure of room.

        This exists because :attr:`effective_free_nodes` counts only *wholly*
        idle nodes, and on a busy cluster almost nothing is wholly idle. Every
        partially-used node then reads as zero room, which is not merely
        imprecise -- it inverts the ranking. Measured on the cluster this was
        written against:

        ==================  ==============  ========================
        partition           idle nodes      idle cores
        ==================  ==============  ========================
        ``amd``             1 of 40 (2%)    2825 of 5120 (55%)
        ``build``           0 of 1 (0%)     42 of 48 (88%)
        ``beagle3``         0 of 44 (0%)    200 of 1408, + 27 GPUs
        ``beagle3-bigmem``  4 of 4 (100%)   128 of 128 (100%)
        ==================  ==============  ========================

        So the summary put ``beagle3-bigmem`` top with a full meter and drew
        ``amd`` as an empty one, while ``amd`` had **22x** more free capacity.
        A node is not the unit of room; a core is. ``cmd_nodes`` already
        meters ``cpus_free / cpus_total`` per node -- this is the same
        arithmetic, one level up.

        Cores on a node whose memory is fully allocated are not counted; see
        :attr:`Node.memory_exhausted` for why, and for what it was worth on a
        real cluster.
        """
        if not self.usable:
            return 0
        return sum(n.effective_free_cpus for n in self.schedulable_nodes)

    @property
    def effective_free_gpus(self) -> int:
        if not self.usable:
            return 0
        return sum(n.effective_free_gpus for n in self.schedulable_nodes)

    #: An allowlist no wider than this is treated as a group's own hardware.
    #: Two rather than one because a PI partition routinely names the group
    #: account plus a collaborator.
    DEDICATED_ACCOUNT_LIMIT = 2

    @property
    def is_dedicated(self) -> bool:
        """Whether this queue is one group's private hardware.

        A *structural* reading, taken from the queue's own allowlist: a
        partition naming one or two accounts is somebody's cluster share, not a
        shared resource, whatever the accounting database says you are
        associated with.

        That last part is why this exists.  On the cluster this was written
        against, ``sacctmgr`` reports the user as associated with 34 accounts
        and gives every one of them an identical QOS list -- see
        :attr:`Identity.entitlements_look_templated` -- so the declared
        entitlement cannot distinguish a partition you may use from one that
        will reject you with "Invalid membership". The allowlist width can:
        ``pi-depablo`` alone on ``depablo-gpu`` says what that hardware is for.

        A heuristic, and labelled as one in the output.  It is not a
        substitute for :meth:`Cluster.probe`; it is what can be said honestly
        without one.
        """
        return 0 < len(self.allow_accounts) <= self.DEDICATED_ACCOUNT_LIMIT

    @property
    def unresolved_nodes(self) -> int:
        """Nodes the queue claims that we could not find in the node list.

        Non-zero means the two views disagree, which is worth showing rather
        than silently reporting the smaller number.
        """
        return max(0, self.declared_nodes - len(self.nodes))

    @property
    def accelerator_models(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            if n.is_gpu_node:
                key = n.accelerator.model if n.accelerator else "UNKNOWN"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} [{self.state_raw}]{' HIDDEN' if self.hidden else ''}"


def _is_none(values: tuple[str, ...]) -> bool:
    """Whether an allowlist explicitly names nobody."""
    return bool(values) and values[0].strip().lower() in {"none", "(none)"}


def _denied(
    allow: tuple[str, ...], deny: tuple[str, ...], values: set[str] | None
) -> bool | None:
    """Tri-state membership test against an allow/deny pair.

    ``None`` means "cannot tell" -- no values were supplied to check against,
    so claiming a verdict would be a fabrication.
    """
    if _is_none(allow):
        return False
    if not values:
        return None
    lowered = {v.lower() for v in values}
    if deny and lowered & {d.lower() for d in deny}:
        return False
    if allow:
        return bool(lowered & {a.lower() for a in allow})
    return True


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------
@dataclass
class Limits:
    """Resource ceilings that admit a job and then refuse to run it.

    Present in every scheduler under a different name -- Slurm QOS
    ``MaxTRES``, PBS ``max_run_res``, LSF queue limits, Kubernetes
    ``ResourceQuota`` -- and uniformly *not* checked by whatever dry-run the
    system offers.  That combination is why an over-limit job is accepted with
    a plausible start estimate and then sits forever.
    """

    name: str
    max_walltime_seconds: int | None = None
    #: Per-job ceilings, keyed by resource: ``cpu``, ``mem_mb``, ``gpu``, ``node``.
    per_job: dict[str, int] = field(default_factory=dict)
    #: Per-user aggregate ceilings, same keys.  These bound a single job too:
    #: one job cannot exceed what the user may hold in total.
    per_user: dict[str, int] = field(default_factory=dict)
    max_jobs: int | None = None
    max_submitted: int | None = None
    #: Human-readable note on where these came from, for the report.
    source: str = ""
    #: Fields that were present but could not be parsed.  An unreadable
    #: ceiling is indistinguishable from no ceiling once it becomes ``None``,
    #: so the check silently stops running -- naming it keeps the gap visible
    #: without inventing a limit nobody published.
    unreadable: tuple[str, ...] = ()

    def blockers(self, shape: JobShape) -> list[Blocker]:
        """Ceilings this shape would exceed.

        All non-fatal: the queue is fine, the *request* is too big, and a
        smaller one gets in.  Marking them fatal would wrongly rule out a
        queue the job could reach.
        """
        out: list[Blocker] = []
        want = shape.walltime_seconds
        if (
            self.max_walltime_seconds is not None
            and want is not None
            and want > self.max_walltime_seconds
        ):
            from .duration import format_duration

            out.append(
                Blocker(
                    "MAX_WALLTIME",
                    f"walltime {format_duration(want)} exceeds {self.name} limit "
                    f"{format_duration(self.max_walltime_seconds)} -- typically "
                    f"accepted at submit time and then queued indefinitely",
                    fatal=False,
                )
            )

        asked = {
            "gpu": shape.total_gpus,
            "node": shape.nodes,
            "cpu": shape.total_cpus,
            "mem_mb": int(shape.memory_gb * 1024) * shape.nodes,
        }
        # The internal keys are terse on purpose; the message should not be.
        # "4 gpu exceeds ... limit of 2" reads like a typo in the tool.
        wording = {
            "gpu": ("accelerator", "accelerators"),
            "node": ("node", "nodes"),
            "cpu": ("CPU", "CPUs"),
            "mem_mb": ("MB of memory", "MB of memory"),
        }
        for scope, table, label in (
            ("per job", self.per_job, "JOB"),
            ("per user", self.per_user, "USER"),
        ):
            for key, cap in table.items():
                if key in asked and asked[key] > cap:
                    one, many = wording.get(key, (key, key))
                    noun = one if asked[key] == 1 else many
                    out.append(
                        Blocker(
                            f"MAX_{key.upper()}_{label}",
                            f"{asked[key]} {noun} exceeds the {scope} limit of "
                            f"{cap} on {self.name}",
                            fatal=False,
                        )
                    )
        return out


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
@dataclass
class Identity:
    """Who the cluster thinks you are, as *declared*.

    Kept separate from any confirmed entitlement on purpose.  On systems where
    the declaration is generated from a template it carries no per-account
    information at all, and :attr:`entitlements_look_templated` says so.
    """

    user: str
    accounts: tuple[str, ...] = ()
    qos: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    #: account -> queues the scheduler associates with it.
    account_queues: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    #: True when every account claims an identical queue list -- a strong hint
    #: the entitlement table is a template rather than a grant.
    entitlements_look_templated: bool = False

    @property
    def account_set(self) -> set[str]:
        return set(self.accounts)

    @property
    def qos_set(self) -> set[str]:
        return set(self.qos)

    @property
    def group_set(self) -> set[str]:
        return set(self.groups)

    @classmethod
    def from_account_queues(
        cls,
        user: str,
        account_queues: dict[str, set[str]],
        qos: Iterable[str] = (),
        groups: Iterable[str] = (),
    ) -> Identity:
        """Build an identity and run the templated-entitlements check."""
        lists = [frozenset(v) for v in account_queues.values() if v]
        return cls(
            user=user,
            accounts=tuple(account_queues),
            qos=tuple(sorted(set(qos))),
            groups=tuple(sorted(set(groups))),
            account_queues={k: tuple(sorted(v)) for k, v in account_queues.items()},
            entitlements_look_templated=len(lists) > 1 and len(set(lists)) == 1,
        )


# ---------------------------------------------------------------------------
# job shape
# ---------------------------------------------------------------------------
@dataclass
class JobShape:
    """A resource request, in terms both the scheduler and the hardware care about.

    ``gpu_memory_gb`` and ``requires`` have no scheduler equivalent anywhere --
    no batch system can express "I need 40 GiB of HBM" or "I need bf16" --
    which is exactly why they belong in the model rather than in a backend.
    """

    nodes: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int = 1
    tasks_per_node: int = 1
    memory_gb: float = 0.0
    walltime: str = "01:00:00"
    #: Minimum per-accelerator memory in GiB; 0 disables the check.
    gpu_memory_gb: float = 0.0
    #: Named capabilities, e.g. ``("bf16",)`` or ``("fp8",)``.
    requires: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    account: str | None = None
    qos: str | None = None
    #: Node restrictions the job tolerates (k8s taints and equivalents).
    tolerates: tuple[str, ...] = ()

    @property
    def walltime_seconds(self) -> int | None:
        from .duration import parse_duration

        return parse_duration(self.walltime)

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def total_cpus(self) -> int:
        return self.nodes * self.tasks_per_node * self.cpus_per_task

    @property
    def cpus_per_node(self) -> int:
        return self.tasks_per_node * self.cpus_per_task

    @property
    def needs_gpu(self) -> bool:
        return self.gpus_per_node > 0

    @property
    def memory_mb_per_node(self) -> int:
        return int(self.memory_gb * 1024)

    def describe(self) -> str:
        from .duration import format_duration

        bits = [f"{self.nodes} node{'s' if self.nodes != 1 else ''}"]
        if self.gpus_per_node:
            bits.append(f"{self.gpus_per_node} GPU/node ({self.total_gpus} total)")
        if self.gpu_memory_gb:
            bits.append(f">={self.gpu_memory_gb:g} GiB HBM")
        bits.append(f"{self.cpus_per_node} CPU/node")
        if self.memory_gb:
            bits.append(f"{self.memory_gb:g} GB RAM/node")
        bits.append(format_duration(self.walltime_seconds))
        if self.requires:
            bits.append("needs " + "+".join(self.requires))
        return ", ".join(bits)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
class VerdictCategory:
    """Why a submission would be refused, normalised across schedulers.

    Separating "you are not permitted here" from "this shape does not fit"
    from "the control plane is sick" matters: only the first is a durable fact
    about your access.
    """

    OK = "OK"
    NOT_ENTITLED = "NOT_ENTITLED"
    ACCOUNT_MISMATCH = "ACCOUNT_MISMATCH"
    NO_ACCOUNT = "NO_ACCOUNT"
    ACCESS_DENIED = "ACCESS_DENIED"
    GROUP_DENIED = "GROUP_DENIED"
    UNKNOWN_QUEUE = "UNKNOWN_QUEUE"
    QUEUE_CLOSED = "QUEUE_CLOSED"
    SHAPE_UNAVAILABLE = "SHAPE_UNAVAILABLE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    INVALID_QOS = "INVALID_QOS"
    TIME_LIMIT = "TIME_LIMIT"
    CONTROL_PLANE_DOWN = "CONTROL_PLANE_DOWN"
    #: The backend has no dry-run facility, so nothing could be confirmed.
    NOT_SUPPORTED = "NOT_SUPPORTED"
    #: Every account we asked about was refused, but we did not ask about all
    #: of them -- so nothing durable is known about the *queue*.  Distinct from
    #: a refusal, and the distinction is load-bearing: reporting one as the
    #: other hid three partitions this user can submit to.
    ACCOUNTS_UNTRIED = "ACCOUNTS_UNTRIED"
    UNKNOWN = "UNKNOWN"


#: Categories that say nothing durable about your access.
TRANSIENT_CATEGORIES = frozenset(
    {
        VerdictCategory.CONTROL_PLANE_DOWN,
        VerdictCategory.SHAPE_UNAVAILABLE,
        VerdictCategory.NOT_SUPPORTED,
        VerdictCategory.ACCOUNTS_UNTRIED,
        VerdictCategory.UNKNOWN,
    }
)


@dataclass
class Verdict:
    """The control plane's own answer about one (queue, account) pair.

    Obtained from whatever dry-run the scheduler offers, and ``None`` from
    backends that have none -- an absence the report must state rather than
    paper over, because "we could not check" and "you are allowed" are very
    different claims.
    """

    queue: str
    account: str | None = None
    allowed: bool = False
    category: str = VerdictCategory.UNKNOWN
    reason: str = ""
    #: A site filter's own verdict, where one exists and is distinguishable
    #: from the scheduler core's.  These two can disagree.
    filter_verdict: str | None = None
    #: The QOS / priority class / queue the control plane actually selected,
    #: which is frequently not the one requested.
    effective_qos: str | None = None
    effective_account: str | None = None
    predicted_start: datetime | None = None
    predicted_nodes: tuple[str, ...] = ()
    raw: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        # `reason` and `raw` are the dry-run's own stderr, which is the least
        # trustworthy text in the program and goes straight into a cell.
        self.reason = sanitize(self.reason)
        self.category = sanitize(self.category)
        for attr in ("effective_qos", "effective_account", "filter_verdict"):
            got = getattr(self, attr)
            if got is not None:
                setattr(self, attr, sanitize(got))

    @property
    def durable(self) -> bool:
        """Whether this verdict says something lasting about your access."""
        return self.category not in TRANSIENT_CATEGORIES

    @property
    def confirmed(self) -> bool:
        """True only when the control plane positively accepted the request."""
        return self.allowed and self.category == VerdictCategory.OK

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.queue}/{self.account or '-'}: " + (
            "allowed" if self.allowed else self.category
        )


def capability_gap(spec: AcceleratorSpec | None, requires: Iterable[str]) -> list[str]:
    """Capabilities the accelerator is *known* to lack.

    Unknown hardware yields an empty list on purpose: "we cannot identify this
    card" is not "this card cannot do the job", and only the latter justifies
    excluding a node.
    """
    gaps: list[str] = []
    for req in requires:
        if supports(spec, req) is False:
            gaps.append(req)
    return gaps
