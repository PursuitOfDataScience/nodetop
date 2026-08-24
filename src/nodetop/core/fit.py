"""Where can this job actually land?

The ranking combines four independent sources, in increasing order of
authority:

1. **Queue configuration** -- structural blockers (disabled, stopped, empty
   allowlist, every node down).
2. **Hardware** -- capability and memory, which no scheduler models.
3. **Resource ceilings** -- the limits a dry-run does not evaluate.
4. **The control plane's own verdict** -- where the system offers one.

A queue is reported as usable only when every available layer agrees, and when
a layer is *unavailable* that is stated rather than assumed benign.  When two
layers disagree, the disagreement is the finding: a queue that passes a
dry-run but violates a resource ceiling will accept your job and never run it,
which is the worst available outcome and exactly where a single-source check
lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from .capacity import Capacity, assess_capacity
from .duration import format_duration
from .model import Blocker, JobShape, Limits, Queue, Verdict, VerdictCategory

__all__ = ["Placement", "ProbeBudget", "evaluate", "probe_accounts",
           "probe_queue", "rank", "unsettled"]


@dataclass
class Placement:
    """The verdict for one queue."""

    queue: str
    shape: JobShape
    blockers: list[Blocker] = field(default_factory=list)
    capacity: Capacity | None = None
    verdict: Verdict | None = None
    accelerator_models: dict[str, int] = field(default_factory=dict)
    #: Facts a decision here rests on that are inferences, not measurements.
    caveats: list[str] = field(default_factory=list)
    #: True when the backend has no dry-run, so entitlement is declared only.
    entitlement_unconfirmed: bool = False
    #: The instant the snapshot describes, against which a predicted start is
    #: judged near or far.  ``None`` falls back to the wall clock, which is the
    #: same thing outside a replay.
    as_of: datetime | None = None

    #: How far past :attr:`as_of` a predicted start can be and still count as
    #: immediate.  Slurm answers a dry-run with a start time two or three
    #: seconds out when it means "right now", so an exact comparison would call
    #: every placement a queue.
    START_SLACK_SECONDS = 120

    @property
    def partition(self) -> str:
        """Alias for :attr:`queue`, for Slurm users."""
        return self.queue

    @property
    def fatal_blockers(self) -> list[Blocker]:
        return [b for b in self.blockers if b.fatal]

    @property
    def soft_blockers(self) -> list[Blocker]:
        """Blockers a smaller or shorter request would clear."""
        return [b for b in self.blockers if not b.fatal]

    @property
    def reachable(self) -> bool:
        """You appear permitted to submit here and the shape is legal.

        "Appear" is load-bearing when :attr:`entitlement_unconfirmed` is set.
        """
        if self.fatal_blockers or self.soft_blockers:
            return False
        # Only a DURABLE refusal makes a placement unreachable. A transient
        # verdict -- control plane unreachable, client missing, an answer we
        # could not parse -- means the question went unanswered, and reading
        # that as "no" turns a scheduler hiccup into "nothing fits anywhere",
        # exit code and all, at the moment the caller most needs their options.
        return not (
            self.verdict is not None
            and not self.verdict.allowed
            and self.verdict.durable
        )

    @property
    def confirmed(self) -> bool:
        """Reachable *and* the control plane said so itself."""
        return self.reachable and bool(self.verdict and self.verdict.confirmed)

    @property
    def runnable_now(self) -> bool:
        """Reachable and there is room right now."""
        return bool(self.reachable and self.capacity and self.capacity.satisfies(self.shape))

    @property
    def starts_now(self) -> bool:
        """Room here *and* nothing ahead of you in the queue.

        :attr:`runnable_now` is a statement about hardware: nodes matching this
        shape are free this instant. That is not the same as the job starting,
        because the scheduler runs the queue in priority order -- and where it
        offers its own estimate, that estimate outranks our arithmetic, exactly
        as :attr:`earliest_start` already says.

        Measured on the cluster this was written against, for a four-core
        ten-minute job with every partition reporting free cores:

        ==================  =======================
        partition           the scheduler's answer
        ==================  =======================
        ``beagle3``         now
        ``bigmem``          now
        ``amd``             in 4h 24m
        ``build``           in 8h
        ``caslake``         in 18h
        ==================  =======================

        All five said RUN NOW before this existed, and ``amd`` -- the largest
        pool of free cores on the cluster and so the top row of every listing
        -- was four and a half hours from starting anything.
        """
        if not self.runnable_now:
            return False
        predicted = self.verdict.predicted_start if self.verdict else None
        if predicted is None:
            return True
        base = self.as_of or datetime.now()
        return (predicted - base).total_seconds() <= self.START_SLACK_SECONDS

    @property
    def hardware_incompatible(self) -> bool:
        """No node here could *ever* host this shape, however empty.

        A much stronger result than "nothing free right now": the queue is the
        wrong hardware, not the wrong moment.
        """
        return False if self.capacity is None else not self.capacity.ever_possible

    @property
    def earliest_start(self) -> datetime | None:
        """Best start estimate, or ``None`` if you cannot go here.

        A scheduler prediction wins when present -- it accounts for the queue
        ahead of you, which the node-free calculation deliberately does not.
        Note that schedulers return a plausible start time alongside a
        *refusal*, so an unreachable placement reports no estimate at all
        rather than a number that would read as encouragement.
        """
        if not self.reachable:
            return None
        if self.verdict and self.verdict.predicted_start:
            return self.verdict.predicted_start
        return self.capacity.earliest_free if self.capacity else None

    @property
    def start_estimate_from_scheduler(self) -> bool:
        """True when the estimate came from the scheduler, not our arithmetic."""
        return bool(self.reachable and self.verdict and self.verdict.predicted_start)

    @property
    def nodes_available(self) -> int:
        return self.capacity.nodes_available if self.capacity else 0

    def score(self) -> tuple:
        """Sort key, best first.

        Lexicographic over hard facts before soft preferences, so a queue you
        can use beats a bigger one you cannot, and a *confirmed* placement
        beats one that merely looks fine.
        """
        return (
            0 if self.runnable_now else 1 if self.reachable else 2,
            0 if self.confirmed else 1,
            1 if self.hardware_incompatible else 0,
            0 if not self.fatal_blockers else 1,
            -self.nodes_available,
            self.earliest_start or datetime.max,
            self.queue,
        )


def evaluate(
    cluster,  # nodetop.core.cluster.Cluster; untyped to avoid a circular import
    shape: JobShape,
    queue: Queue,
    *,
    use_probe: bool = False,
    accounts: list[str] | None = None,
    budget: ProbeBudget | None = None,
) -> Placement:
    """Assess one queue against one shape."""
    blockers: list[Blocker] = list(queue.structural_blockers())
    caveats: list[str] = []

    ident = cluster.identity
    if ident is not None:
        blockers.extend(
            queue.access_blockers(
                ident.account_set, ident.qos_set, ident.group_set, ident.user
            )
        )
        if ident.entitlements_look_templated and not blockers:
            caveats.append(
                "the scheduler reports an identical queue list for every account, so "
                "its access claim carries no information -- only a dry-run settles it"
            )

    if queue.max_nodes is not None and shape.nodes > queue.max_nodes:
        blockers.append(
            Blocker(
                "QUEUE_MAX_NODES",
                f"{shape.nodes} nodes exceeds the queue maximum of {queue.max_nodes}",
                fatal=False,
            )
        )
    want = shape.walltime_seconds
    if (
        queue.max_walltime_seconds is not None
        and want is not None
        and want > queue.max_walltime_seconds
    ):
        blockers.append(
            Blocker(
                "QUEUE_MAX_WALLTIME",
                f"walltime {format_duration(want)} exceeds the queue maximum of "
                f"{format_duration(queue.max_walltime_seconds)}",
                fatal=False,
            )
        )

    verdict: Verdict | None = None
    caps = cluster.capabilities
    if use_probe and caps is not None and caps.probe:
        budget = budget if budget is not None else ProbeBudget()
        verdict, tried, of = probe_queue(
            cluster, queue, queue.name, shape, accounts, budget
        )
        settled = unsettled(verdict, tried, of)
        if settled is not verdict:
            verdict = settled
            caveats.append(
                f"refused by {tried} of {of} accounts tried; the rest were not "
                f"asked -- name one with -A to settle it"
            )
        elif (
            verdict is not None
            and not verdict.allowed
            and verdict.durable
            and not accounts
            and cluster.errors.get("identity")
        ):
            # Refused when we could not tell the scheduler who you are.
            #
            # Keyed on the identity query having *failed*, not on there being no
            # identity: a backend with no notion of accounts at all (an ssh pool)
            # legitimately probes without one, and its refusals are real. What is
            # not real is a refusal obtained after `sacctmgr` died, where the
            # probe ran with no --account and the control plane fell back to a
            # default. That refusal is a fact about the default account, not
            # about you, and reporting it as a refusal of the queue turned one
            # failed association query into "you have access to nothing".
            verdict = replace(
                verdict,
                category=VerdictCategory.ACCOUNTS_UNTRIED,
                reason=f"{verdict.category}: {verdict.reason}".strip(": "),
            )
            caveats.append(
                "refused with no account named, because your associations "
                "could not be read -- name one with -A to settle it"
            )

    # Ceilings, against the limit set the control plane would actually apply.
    limits = _limits_for(cluster, queue, shape, verdict)
    if limits is not None:
        blockers.extend(limits.blockers(shape))
        if limits.unreadable:
            # No invented blocker: a ceiling nobody published must not become
            # one. But a check that did not run should not look like a check
            # that passed.
            caveats.append(
                f"could not read {', '.join(limits.unreadable)} on "
                f"{limits.source or 'the limit set'} {limits.name}, so "
                f"{'that ceiling was' if len(limits.unreadable) == 1 else 'those ceilings were'} "
                f"not checked at all"
            )

    capacity = assess_capacity(
        queue.nodes, shape, cluster.node_free_times,
        # Judge "not enough nodes here, ever" only when we believe we have seen
        # the queue's nodes.
        count_is_complete=not queue.unresolved_nodes,
    )

    if shape.gpu_memory_gb and any(
        n.accelerator is not None and not n.accelerator.memory_certain
        for n in queue.nodes
    ):
        # Deliberately model-agnostic. Naming the models made each queue emit a
        # different string, so the "applies to every queue" footer printed the
        # same fact once per model instead of once.
        caveats.append(
            "accelerator memory is inferred from the model name for parts that "
            "ship in several sizes; the smaller was assumed"
        )
    unverified = capacity.unverified_nodes if capacity else ()
    if unverified:
        asked = list(shape.requires)
        if shape.gpu_memory_gb:
            asked.append(f">={shape.gpu_memory_gb:g} GiB")
        caveats.append(
            f"{len(unverified)} node(s) were set aside because their accelerator "
            f"model could not be identified, so {'+'.join(asked)} cannot be "
            f"confirmed -- they may well be capable; check the node labels"
        )
    if queue.unresolved_nodes:
        caveats.append(
            f"the queue claims {queue.declared_nodes} nodes but only "
            f"{len(queue.nodes)} could be resolved; capacity is computed over "
            f"the {len(queue.nodes)} we can see"
        )

    unconfirmed = not (caps and caps.probe)
    return Placement(
        queue=queue.name,
        shape=shape,
        blockers=blockers,
        capacity=capacity,
        verdict=verdict,
        accelerator_models=queue.accelerator_models,
        caveats=caveats,
        entitlement_unconfirmed=unconfirmed,
        as_of=cluster.taken_at,
    )


#: Ceiling on dry-runs per queue.  Each one is a control-plane round trip, and
#: a user with dozens of associations against dozens of queues would otherwise
#: fire hundreds of them for a single question.
#:
#: It used to be 4, applied by truncating the candidate list, and that was a
#: **wrong-answer** bug rather than a slow one.  On a cluster where this user
#: holds 34 associations and the general partitions set ``AllowAccounts=ALL``,
#: the intersection is all 34 and the truncation tried the first 4.  `caslake`,
#: `gpu` and `bigmem` are all accepted with `rcc-staff` -- 32nd in that list --
#: and all three were reported as *refused* and hidden.  Three partitions the
#: user can genuinely submit to, including the 190-node CPU partition and a
#: 44-GPU one, missing from the answer to "where can I run this".
MAX_PROBES_PER_QUEUE = 12

#: Ceiling on dry-runs for one whole question, across every queue.
#:
#: This is the limit that actually protects the control plane, and it is the
#: right place for it: the per-queue loop stops at the first *accept*, so the
#: expensive case is not a user with many accounts, it is a queue that refuses
#: all of them.  A global budget bounds the total either way, where a per-queue
#: cap bounds nothing (50 queues x 4 is still 200) and silently corrupts the
#: verdict on the queues it truncates.
MAX_PROBES_TOTAL = 150


class ProbeBudget:
    """Shared probe accounting for one question.

    Two jobs, and the second is why this is an object rather than an integer:

    * **Spend a bounded number of dry-runs.**  Decrementing counter, checked
      before each round trip.
    * **Try the accounts that work first.**  An account accepted by one queue
      is overwhelmingly likely to be accepted by the next -- on the cluster
      this was written against, the single account that clears the SU check
      clears it for every shared partition -- so the order is re-learned as we
      go.  Without it, finding that account costs 32 probes on *every* queue
      instead of 32 once; with it the second queue onwards accepts on the
      first try, which is what makes raising the per-queue cap affordable.
    """

    def __init__(self, total: int = MAX_PROBES_TOTAL,
                 queues: int | None = None) -> None:
        self.left = total
        # Spend the budget where it was asked for. A fixed per-queue ceiling is
        # wrong at both ends: too small when the caller named two queues and
        # wants them settled, too large when it is sweeping ninety. Naming two
        # queues out of a 150-probe budget buys 75 tries each, which is what
        # makes `check -q caslake` able to reach the one account in thirty-four
        # that the control plane accepts -- the fixed ceiling of 12 stopped
        # twenty short of it and reported ACCOUNTS_UNTRIED for a queue the user
        # had explicitly asked about.
        self.per_queue = (
            max(MAX_PROBES_PER_QUEUE, total // queues) if queues
            else MAX_PROBES_PER_QUEUE
        )
        self._accepted: list[str] = []

    def spend(self) -> bool:
        """Claim one probe.  False when the budget is gone."""
        if self.left <= 0:
            return False
        self.left -= 1
        return True

    def accepted(self, account: str | None) -> None:
        if account and account not in self._accepted:
            self._accepted.append(account)

    def order(self, candidates: list[str | None]) -> list[str | None]:
        """``candidates`` with known-good accounts moved to the front."""
        if not self._accepted:
            return candidates
        rank = {a: i for i, a in enumerate(self._accepted)}
        return sorted(candidates, key=lambda a: rank.get(a or "", len(rank)))


def probe_accounts(
    queue: Queue, accounts: list[str] | None, shape: JobShape
) -> list[str | None]:
    """Which accounts are worth dry-running against this queue.

    A queue that names an account allowlist has already answered most of the
    question: an account it does not list cannot possibly be accepted, so
    trying it spends a round trip to learn nothing.  Narrowing to the
    intersection turns "every account against every queue" into a couple of
    calls.

    The result is **not** truncated here.  Whoever probes has to know how many
    candidates there were, because "refused by the 4 we tried" and "refused by
    all 34" are different answers and only one of them justifies hiding the
    queue.  See :data:`MAX_PROBES_PER_QUEUE` for the bug that came of
    conflating them.
    """
    if shape.account:
        return [shape.account]
    if not accounts:
        return [None]
    allowed = {a.lower() for a in queue.allow_accounts}
    if allowed and "all" not in allowed:
        narrowed = [a for a in accounts if a.lower() in allowed]
        # An empty intersection is itself informative: probe once anyway so the
        # control plane, not our reading of the allowlist, has the last word.
        return list(narrowed) or [accounts[0]]
    return list(accounts)


def _verdict_rank(v: Verdict) -> int:
    """Most informative first: yes, then unknown, then no.

    A loop over accounts used to keep whichever verdict came last, which is
    arbitrary when accounts disagree.  Order matters because the question is
    about the *queue*: one account accepted means yes, and one account
    unanswered means the queue's status is unknown even if another was durably
    refused, since the unanswered one might have worked.  Keeping the refusal
    there would claim more than was established.
    """
    if v.allowed:
        return 0
    return 2 if v.durable else 1


def probe_queue(
    cluster,
    queue: Queue | None,
    name: str,
    shape: JobShape,
    accounts: list[str] | None,
    budget: ProbeBudget,
) -> tuple[Verdict | None, int, int]:
    """Ask the control plane about one queue.  Returns ``(verdict, tried, of)``.

    **One loop, two callers.**  ``evaluate`` and the ``check`` command both walk
    a queue's candidate accounts, and keeping two copies of that walk is what
    let them drift: when the candidate list stopped being truncated inside
    :func:`probe_accounts`, ``evaluate`` picked up the new per-queue ceiling and
    ``check`` -- which never had one -- was left able to fire an unbounded
    number of dry-runs, 34 accounts against 84 queues for a bare invocation.
    A shared helper cannot drift.

    ``tried`` and ``of`` are returned rather than kept private because the
    difference between them is the whole difference between "refused" and "not
    established", and only the caller can decide how to say that.
    """
    candidates: list[str | None]
    if queue is not None:
        candidates = probe_accounts(queue, accounts, shape)
    elif accounts:
        # No queue object to read an allowlist from, so nothing can be narrowed
        # away. Reachable only if a caller asks about a name the cluster does
        # not have, which `_reject_unknown_queues` now refuses up front.
        candidates = list(accounts)
    else:
        candidates = [None]
    verdict: Verdict | None = None
    tried = 0
    for acct in budget.order(candidates):
        if tried >= budget.per_queue or not budget.spend():
            break
        tried += 1
        got = cluster.probe(name, shape, acct)
        if got is None:
            continue
        if verdict is None or _verdict_rank(got) < _verdict_rank(verdict):
            verdict = got
        if verdict.allowed:
            budget.accepted(acct)
            break
    return verdict, tried, len(candidates)


def unsettled(verdict: Verdict | None, tried: int, of: int) -> Verdict | None:
    """Downgrade a refusal that did not ask every account.

    Refused by every account we asked about, but we did not ask about them all,
    is not a refusal of the *queue* -- and reporting it as one hid three
    partitions this user can submit to.  Re-categorising is the honest edit:
    ``Placement.reachable`` and ``.confirmed`` both key off ``durable``, so the
    queue stops being dropped and starts being reported as unsettled, which is
    what it is.
    """
    if verdict is None or verdict.allowed or not verdict.durable or tried >= of:
        return verdict
    return replace(
        verdict,
        category=VerdictCategory.ACCOUNTS_UNTRIED,
        reason=f"{verdict.category}: {verdict.reason}".strip(": "),
    )


def _limits_for(cluster, queue: Queue, shape: JobShape, verdict: Verdict | None) -> Limits | None:
    """Pick the limit set that will actually apply.

    Preference order matters: the control plane's *chosen* QOS beats the one
    we asked for, because sites routinely auto-promote and checking the
    requested name checks the wrong ceilings.
    """
    for key in (
        verdict.effective_qos if verdict else None,
        shape.qos,
        queue.limits_name,
        queue.name,
    ):
        if key and key in cluster.limits:
            return cluster.limits[key]
    return None


def rank(
    cluster,
    shape: JobShape,
    *,
    queues: list[str] | None = None,
    use_probe: bool = False,
    accounts: list[str] | None = None,
    include_unusable: bool = False,
) -> list[Placement]:
    """Evaluate every candidate queue and order them best-first.

    By default queues with no chance are dropped; ``include_unusable`` keeps
    them with their blockers attached, which is what you want when the
    question is "why can nothing run anywhere?" rather than "where do I go?".
    """
    names = queues or list(cluster.queues)
    out: list[Placement] = []
    # One budget for the whole question, so the accounts that work are learned
    # on the first queue and tried first on every queue after it.
    budget = ProbeBudget(queues=len(names)) if use_probe else None
    if budget is not None:
        # Fewest candidate accounts first, and this is what makes the learning
        # work rather than merely exist.
        #
        # A queue whose allowlist admits exactly one of your accounts costs one
        # probe and teaches the most, because that account is definitely
        # admitted somewhere. A queue with `AllowAccounts=ALL` admits all 34 and
        # teaches nothing until one of them lands. Evaluated in the scheduler's
        # own order, the 34-candidate queues came first, spent their per-queue
        # cap without reaching the account that works, and were written off --
        # then `beagle3` accepted that very account one queue later. Cheapest
        # first means it is known before the expensive queues are asked.
        shape_for_order = shape
        names = sorted(
            names,
            key=lambda n: (
                len(probe_accounts(q, accounts, shape_for_order))
                if (q := cluster.queues.get(n)) is not None else 0,
                n,
            ),
        )
    for name in names:
        queue = cluster.queues.get(name)
        if queue is None:
            continue
        # A routing queue forwards to other queues and owns no nodes, so it
        # has no capacity of its own to rank. Its destinations are ranked
        # instead, which is where the job would actually land.
        if queue.routes and not include_unusable:
            continue
        # Skip queues with no relevant hardware before spending a probe on
        # them; a probe costs a control-plane round trip.
        no_accel = (
            shape.needs_gpu and queue.nodes and not any(n.is_gpu_node for n in queue.nodes)
        )
        if no_accel and not include_unusable:
            continue
        placement = evaluate(
            cluster, shape, queue, use_probe=use_probe, accounts=accounts,
            budget=budget,
        )
        if not include_unusable and placement.fatal_blockers:
            continue
        out.append(placement)
    return sorted(out, key=lambda p: p.score())
