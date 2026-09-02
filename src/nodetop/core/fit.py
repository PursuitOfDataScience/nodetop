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

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime

from .capacity import Capacity, assess_capacity
from .duration import format_duration
from .model import Blocker, JobShape, Limits, Queue, Verdict, VerdictCategory

__all__ = ["PROBE_WORKERS", "Placement", "ProbeBudget", "evaluate",
           "probe_accounts", "probe_queue", "rank", "unsettled"]


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
    #: True when the control plane did not settle entitlement here -- for ANY
    #: reason: the backend has no dry-run, this is a replay, the probe was not
    #: run, the budget ran out, or it ran and answered something transient.
    #:
    #: It used to mean only the first of those, read straight off the recorded
    #: capabilities, and a replay restores `probe=True` from the live run it was
    #: recorded on.  So on a snapshot taken seconds earlier, 21 partitions the
    #: live run had MEASURED as refusing this user came back `verdict: null`,
    #: `confirmed: false`, `entitlement_unconfirmed: false`, `blockers: []`,
    #: `caveats: []` -- with a ready-to-paste `submit_flags` list.  Three states
    #: are representable (confirmed yes, confirmed no, unconfirmed) and the
    #: replay used none of them; the one field whose job is to carry the hedge
    #: said there was nothing to hedge.
    entitlement_unconfirmed: bool = False
    #: ``(dry-runs fired for this queue, candidates there were)``.  ``(0, n)``
    #: with ``n`` non-zero means probing WAS asked for and the global budget was
    #: gone before this queue's turn -- a different fact from "no dry-run was
    #: requested", and one the reader can act on by naming fewer queues.
    probes: tuple[int, int] = (0, 0)
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
        ``gn``         now
        ``bigmem``          now
        ``amd``             in 4h 24m
        ``build``           in 8h
        ``wide``         in 18h
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

    # Assessed BEFORE the probe, because it decides whether to spend one. Pure
    # arithmetic over nodes already in the snapshot -- no round trip -- so the
    # only thing moving it up costs is this comment.
    capacity = assess_capacity(
        queue.nodes, shape, cluster.node_free_times,
        # Judge "not enough nodes here, ever" only when we believe we have seen
        # the queue's nodes.
        count_is_complete=not queue.unresolved_nodes,
    )

    verdict: Verdict | None = None
    # A queue that accepts nothing from anybody cannot be talked round by a
    # dry-run, and asking costs a round trip -- the most expensive thing this
    # tool does. `reachable` is already False the moment a fatal blocker
    # exists, whatever the control plane would say, so the answer could not
    # change the verdict on screen.
    #
    # Only the OPERATIONAL blockers count here: disabled, never-starts, an
    # empty allowlist, no nodes, every node down. Access is deliberately NOT in
    # that list -- a declared ACL disagreeing with the control plane is the
    # thing this tool exists to catch, so a queue whose allowlist appears to
    # exclude you is still asked. Nor a soft blocker: "your job is too big" is
    # about the request, and the entitlement answer beside it is still worth
    # having.
    #
    # Measured on a 607-node cluster, and the honest figure is small: **3 of 89
    # probes on `where --all`**, and none at all on the default views -- there
    # the entitlement filter and `usable_queues` have already dropped the dead
    # partitions before ranking sees them. It pays where those filters step
    # aside: `--all`, and a queue named explicitly. Free either way, since the
    # blockers are computed above regardless.
    probes = (0, 0)
    dead = [b for b in blockers if b.fatal and b.code in _OPERATIONAL_BLOCKERS]
    # ...and neither can a queue whose hardware could never host this shape,
    # which is the same argument one step further out: `Capacity.ever_possible`
    # is False when no node here has the right kind of hardware, or when the
    # queue does not contain enough of them. No entitlement answer changes that,
    # and the row already says WRONG HW or TOO FEW.
    #
    # This is where the round trips actually were. Measured on an 87-partition
    # cluster, `where -c 64 --mem 200`: **52 of 87 queues are
    # hardware_incompatible and a probe had been spent on every one**, 130
    # probes and 11.5s for a question whose answer was arithmetic. `-N 8 -c 32
    # --mem 100` took 24s. `rank` already declines to probe a queue with no
    # accelerator when one is asked for, and this is that rule stated in terms
    # of the shape rather than of one resource.
    # ...but only where the hardware is KNOWN not to fit.
    #
    # `ever_possible` is False the moment `hardware_nodes` is empty, and
    # `assess_capacity` routes a node whose accelerator model is out of
    # vocabulary into `unverified_nodes` rather than into `hardware`. So an
    # unidentifiable GPU read as an incapable one, and this gate then withheld
    # the dry-run as well: on the cluster the findings file is from, where **232
    # of 384 GPUs are UNKNOWN**, `where --gpus 1 --gpu-mem 40` would have
    # reported every GPU partition as wrong-hardware having asked the controller
    # nothing. Invisible on the cluster this was written on, where all 230
    # resolve -- which is exactly why it is checked here rather than trusted.
    #
    # That contradicts a rule this codebase states twice, in `node_fits`
    # ("unknown" is not "incapable") and on `Capacity.unverified_nodes` (not
    # incapable, just unverifiable), and the second clause honours the third
    # statement of it: `count_is_complete=False` exists so that a queue whose
    # node list did not resolve is never ruled out on the strength of a
    # resolution failure, and a short-circuit on empty `hardware_nodes` defeated
    # it.
    #
    # Costs nothing where the hardware IS known: measured over all 87 partitions
    # here, the skipped count is identical with and without these two clauses for
    # every shape tried (52, 70, 0 and 70 respectively).
    impossible = (
        capacity is not None
        and not capacity.ever_possible
        and not capacity.unverified_nodes
        and not queue.unresolved_nodes
    )
    # `cluster.can_probe`, not `caps.probe`: the capabilities are part of the
    # recording, so on a replay they say a dry-run is available while
    # `Cluster.probe` correctly refuses to invent one. Asking anyway spent a
    # probe budget on calls that returned None before they left the process.
    if use_probe and cluster.can_probe and not dead and not impossible:
        budget = budget if budget is not None else ProbeBudget()
        verdict, tried, of = probe_queue(
            cluster, queue, queue.name, shape, accounts, budget
        )
        probes = (tried, of)
        if of and not tried:
            # The budget ran out before this queue's turn. Recorded and said,
            # not silently indistinguishable from a queue nobody asked about:
            # on an 84-partition cluster with 6 accounts an exhaustive sweep is
            # 504 dry-runs against a 150 ceiling, so this is the ordinary case
            # there rather than an edge one -- and the remedy is a flag away.
            caveats.append(
                f"the dry-run budget ({MAX_PROBES_TOTAL} for the whole "
                f"question) ran out before this queue was asked -- name fewer "
                f"queues with -q to spend it here"
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
        elif (
            verdict is not None
            and not verdict.allowed
            and verdict.category in _SETTLEABLE_BY_FLAG
        ):
            # Two categories a reader can act on, and both used to be silent.
            #
            # `ACCOUNTS_UNTRIED` has said "name one with -A to settle it" for
            # some time, and it is the reason three partitions stopped being
            # hidden. `INVALID_QOS` and `ACCOUNT_MISMATCH` are the same shape
            # of answer -- the control plane refused the ENVELOPE, not the job
            # -- and carried no caveat, no hint and, until now, no flag to
            # point at. On one cluster `INVALID_QOS` was the single largest
            # group of unsettled answers, 16 of 25, on partitions that started
            # a job immediately once the right QOS was named.
            #
            # LAST in this chain, deliberately. It first sat above the
            # identity-error branch, and two of the categories it names --
            # `ACCOUNT_MISMATCH` and `NO_ACCOUNT` -- are exactly the ones that
            # branch exists to reclassify. So a refusal obtained after
            # `sacctmgr` had died stopped being downgraded to
            # `ACCOUNTS_UNTRIED`, stayed durable, and the queue was dropped
            # from `where`: one failed association query turned back into "you
            # have access to nothing", which is the finding that branch was
            # written for. A hint is worth less than a correct verdict, so the
            # hint goes last.
            flag, what = _SETTLEABLE_BY_FLAG[verdict.category]
            caveats.append(
                f"the control plane refused the {what}, not the job -- "
                f"name one with {flag} to settle it"
                + (f" (tried {verdict.effective_qos})"
                   if verdict.effective_qos else "")
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

    # Settled only by a DURABLE answer from the control plane. Everything else
    # -- no verdict at all, or a transient one -- is "we do not know", which is
    # what this flag is for.
    unconfirmed = verdict is None or not verdict.durable
    if verdict is None and not (probes[1] and not probes[0]):
        # A caveat, not just a boolean, because `submit_flags` is the part a
        # reader copies and `caveats` was empty beside it. Only for the
        # no-answer case: a transient verdict has already appended a caveat of
        # its own naming what the control plane actually said.
        # Deliberately NOT naming the queue: an identical sentence on every
        # row is hoisted into one footnote, and interpolating the name would
        # make each copy unique and print it once per queue.
        caveats.append(
            "entitlement is DECLARED here, read from the queue's own access "
            "lists -- no dry-run answer was obtained, so a submission may "
            "still be refused"
        )
    return Placement(
        queue=queue.name,
        shape=shape,
        blockers=blockers,
        capacity=capacity,
        verdict=verdict,
        accelerator_models=queue.accelerator_models,
        caveats=caveats,
        entitlement_unconfirmed=unconfirmed,
        probes=probes,
        as_of=cluster.taken_at,
    )


#: Ceiling on dry-runs per queue.  Each one is a control-plane round trip, and
#: a user with dozens of associations against dozens of queues would otherwise
#: fire hundreds of them for a single question.
#:
#: It used to be 4, applied by truncating the candidate list, and that was a
#: **wrong-answer** bug rather than a slow one.  On a cluster where this user
#: holds 34 associations and the general partitions set ``AllowAccounts=ALL``,
#: the intersection is all 34 and the truncation tried the first 4.  `wide`,
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


#: How many dry-runs may be in flight at once.
#:
#: They are independent round trips to the control plane and the tool used to
#: make them one after another: on a 607-node cluster a bare `nodetop` spent
#: **6.40s of its 10.26s in 21 sequential probes**, two of which took 2.2s each.
#: Run together they cost about as much as the slowest one.
#:
#: Three, and measured rather than picked: ten probes against a live controller,
#: five interleaved rounds, medians -- 0.91s sequential, 0.73s at three
#: concurrent, 0.73s at six. It saturates at three because the controller takes
#: a lock for the submit path, so all concurrency can overlap is the process
#: spawn and the RPC. Going wider buys nothing and asks more of somebody else's
#: controller, which a login node is already busy with.
PROBE_WORKERS = 3


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
        # Guards `left` and the learned account order: with probes in flight
        # together, two queues can reach for the last of the budget at once,
        # and a lost decrement is a probe fired past the ceiling.
        self._lock = threading.Lock()
        self.left = total
        # Spend the budget where it was asked for. A fixed per-queue ceiling is
        # wrong at both ends: too small when the caller named two queues and
        # wants them settled, too large when it is sweeping ninety. Naming two
        # queues out of a 150-probe budget buys 75 tries each, which is what
        # makes `check -q wide` able to reach the one account in thirty-four
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
        with self._lock:
            if self.left <= 0:
                return False
            self.left -= 1
            return True

    def accepted(self, account: str | None) -> None:
        with self._lock:
            if account and account not in self._accepted:
                self._accepted.append(account)

    def order(self, candidates: list[str | None]) -> list[str | None]:
        """``candidates`` with known-good accounts moved to the front."""
        with self._lock:
            known = list(self._accepted)
        if not known:
            return candidates
        rank = {a: i for i, a in enumerate(known)}
        return sorted(candidates, key=lambda a: rank.get(a or "", len(rank)))


#: category -> (the flag that settles it, what was refused).
#:
#: A refusal of the submission's ENVELOPE rather than of the job: naming a
#: different account or QOS may well be accepted, so this is not a fact about
#: the queue and the reader has something to do about it.
_SETTLEABLE_BY_FLAG: dict[str, tuple[str, str]] = {
    VerdictCategory.INVALID_QOS: ("--qos", "QOS"),
    VerdictCategory.ACCOUNT_MISMATCH: ("-A", "account"),
    VerdictCategory.NO_ACCOUNT: ("-A", "account"),
}


#: Blocker codes that mean "this queue accepts nothing from anyone".
#:
#: Distinguished from an ACCESS blocker (which the control plane may contradict,
#: and that contradiction is the point of probing at all) and from a soft one
#: (which is about the size of the request, not the queue). See `evaluate`.
#: Every code here is one a `Blocker` in this codebase actually carries, and
#: each was checked against where it is raised -- a set with a code nobody
#: emits silently stops saving anything, and a missing one silently keeps paying.
#: `REQUIRES_RESERVATION` is deliberately absent: that queue does accept work,
#: from a job that names a reservation, so the dry-run's answer is real.
_OPERATIONAL_BLOCKERS = frozenset({
    "QUEUE_DISABLED",
    "QUEUE_NOT_STARTED",
    "NO_ACCOUNTS",
    "NO_QOS",
    "NO_USERS",
    "ALL_NODES_UNSCHEDULABLE",
})


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
    # `None` is the "name no account" sentinel, and callers reach for it when
    # the caller HAS none: on a cluster that enforces associations, a user with
    # no association row has no account to name, and the honest probe is one
    # with `--account` omitted. Narrowing that against an allowlist crashed --
    # `AttributeError: 'NoneType' object has no attribute 'lower'` -- and it
    # took both halves to trigger: an identity with no accounts AND a queue
    # naming specific ones. Every cluster tested before had accounts, so
    # `nodetop check` died on the first cluster that did not.
    if not accounts:                       # None, or an empty list
        return [None]
    named = [a for a in accounts if a]
    if not named:                          # the sentinel, or a list of blanks
        return [None]
    allowed = {a.lower() for a in queue.allow_accounts}
    if allowed and "all" not in allowed:
        narrowed = [a for a in named if a.lower() in allowed]
        # An empty intersection is itself informative: probe once anyway so the
        # control plane, not our reading of the allowlist, has the last word.
        return list(narrowed) or [named[0]]
    return list(named)


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

    The caller's own set comes last and matters most where nothing above it
    hits: a queue with `QoS=N/A` on a cluster whose limits hang off the
    association has no ceiling under any of the four names above, and reporting
    none is a false "runs now" -- see :meth:`Cluster.caller_limits`.
    """
    for key in (
        verdict.effective_qos if verdict else None,
        shape.qos,
        queue.limits_name,
        queue.name,
    ):
        if key and key in cluster.limits:
            return cluster.limits[key]
    # Not when the job named a QOS that this cluster does not publish: the
    # request said which ceilings to check, and substituting the caller's
    # default would answer about a different one.
    return None if shape.qos else cluster.caller_limits()


def rank(
    cluster,
    shape: JobShape,
    *,
    queues: list[str] | None = None,
    use_probe: bool = False,
    accounts: list[str] | None = None,
    include_unusable: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Placement]:
    """Evaluate every candidate queue and order them best-first.

    By default queues with no chance are dropped; ``include_unusable`` keeps
    them with their blockers attached, which is what you want when the
    question is "why can nothing run anywhere?" rather than "where do I go?".

    ``on_progress(done, total)`` is called as each queue settles, from whichever
    worker thread settled it. It exists because this is the slow part of a run
    by a wide margin -- 1.60s of a 1.93s `status` on a 607-node cluster, all of
    it somebody else's controller -- and a caller that can say so is the
    difference between a wait and a hang. Optional, and this module does not
    know what a terminal is: it hands over two integers.
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
        # then `gn` accepted that very account one queue later. Cheapest
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
    candidates: list[Queue] = []
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
        candidates.append(queue)

    def assess(queue: Queue) -> Placement:
        return evaluate(
            cluster, shape, queue, use_probe=use_probe, accounts=accounts,
            budget=budget,
        )

    # Together where the work is a round trip, one at a time where it is
    # arithmetic. Each queue's own probe sequence stays sequential -- it learns
    # from what the last account did -- but the queues do not have to wait for
    # each other: 21 probes took 6.40s of a 10.26s run on a 607-node cluster,
    # and they are 21 independent questions.
    #
    # Without probes this is pure computation, and threads would buy contention
    # instead of speed, so that path is left exactly as it was. Submitted in the
    # cheapest-first order above, so the account learning still happens roughly
    # in the order it was designed to.
    if budget is not None and len(candidates) > 1:
        import concurrent.futures as _cf

        with _cf.ThreadPoolExecutor(
                max_workers=min(PROBE_WORKERS, len(candidates))) as pool:
            if on_progress is None:
                placements = list(pool.map(assess, candidates))
            else:
                # `map` yields in submission order, so the last queue to finish
                # would be reported as the last to start -- which is exactly
                # backwards for a progress count. Futures keyed by position
                # keep the order of the RESULTS while reporting completions as
                # they land.
                futures = {pool.submit(assess, q): i
                           for i, q in enumerate(candidates)}
                done: dict[int, Placement] = {}
                for future in _cf.as_completed(futures):
                    done[futures[future]] = future.result()
                    on_progress(len(done), len(candidates))
                placements = [done[i] for i in range(len(candidates))]
    else:
        placements = []
        for queue in candidates:
            placements.append(assess(queue))
            if on_progress is not None:
                on_progress(len(placements), len(candidates))

    for placement in placements:
        if not include_unusable and placement.fatal_blockers:
            continue
        out.append(placement)
    return sorted(out, key=lambda p: p.score())
