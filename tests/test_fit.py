"""Placement ranking, and the honesty rules it enforces."""

from __future__ import annotations

from datetime import datetime

import pytest

from nodetop.core.cluster import Cluster
from nodetop.core.fit import evaluate, rank
from nodetop.core.hardware import ACCELERATORS
from nodetop.core.model import (
    BackendCapabilities,
    Blocker,
    Identity,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)


def _node(name, model="A100", gpus=4, alloc=0, conditions=()):
    return Node(
        name=name, cpus_total=32, memory_mb=256 * 1024,
        gpus_total=gpus, gpus_alloc=alloc,
        accelerator=ACCELERATORS[model] if model else None,
        conditions=frozenset(conditions),
        state_raw="down" if conditions else "up",
    )


def _cluster(queues, *, limits=None, probe=True, identity=None, free_times=None):
    cl = Cluster(
        backend_name="test",
        queues={q.name: q for q in queues},
        nodes=[n for q in queues for n in q.nodes],
        limits=limits or {},
        identity=identity,
        capabilities=BackendCapabilities(probe=probe, probe_command="dry-run"),
        node_free_times=free_times or {},
    )
    return cl


class TestReachability:
    def test_a_clean_queue_is_reachable(self):
        q = Queue(name="q", nodes=[_node("n1")], node_names=("n1",))
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=4), q)
        assert p.reachable is True
        assert p.runnable_now is True

    def test_a_fatal_blocker_makes_it_unreachable(self):
        q = Queue(name="q", enabled=False, nodes=[_node("n1")])
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=4), q)
        assert p.reachable is False

    def test_a_soft_blocker_also_blocks_this_request(self):
        # The queue is fine; this shape is not.
        q = Queue(name="q", nodes=[_node("n1")], max_nodes=1)
        p = evaluate(_cluster([q]), JobShape(nodes=4, gpus_per_node=4), q)
        assert p.reachable is False
        assert p.fatal_blockers == []
        assert [b.code for b in p.soft_blockers] == ["QUEUE_MAX_NODES"]


class TestUnconfirmedEntitlement:
    """A backend with no dry-run must not have its claim read as verified."""

    def test_no_probe_means_unconfirmed(self):
        q = Queue(name="q", nodes=[_node("n1")])
        p = evaluate(_cluster([q], probe=False), JobShape(gpus_per_node=4), q)
        assert p.entitlement_unconfirmed is True
        assert p.confirmed is False
        # Still reachable: unconfirmed is not the same as refused.
        assert p.reachable is True

    def test_a_probe_confirms(self):
        q = Queue(name="q", nodes=[_node("n1")])
        cl = _cluster([q])
        cl._backend = _FakeBackend(
            Verdict(queue="q", allowed=True, category=VerdictCategory.OK)
        )
        p = evaluate(cl, JobShape(gpus_per_node=4), q, use_probe=True)
        assert p.confirmed is True
        assert p.entitlement_unconfirmed is False

    def test_a_probe_refusal_overrides_a_clean_static_read(self):
        # The whole point: the declared config looked fine.
        q = Queue(name="q", nodes=[_node("n1")])
        cl = _cluster([q])
        cl._backend = _FakeBackend(
            Verdict(queue="q", allowed=False, category=VerdictCategory.NOT_ENTITLED)
        )
        p = evaluate(cl, JobShape(gpus_per_node=4), q, use_probe=True)
        assert p.reachable is False

    def test_confirmed_placements_sort_above_merely_plausible_ones(self):
        assert _score(confirmed=True) < _score(confirmed=False)


def _score(confirmed):
    q = Queue(name="q", nodes=[_node("n1")])
    cl = _cluster([q])
    cl._backend = _FakeBackend(
        Verdict(queue="q", allowed=True, category=VerdictCategory.OK)
        if confirmed else None
    )
    p = evaluate(cl, JobShape(gpus_per_node=4), q, use_probe=confirmed)
    return p.score()


class _FakeBackend:
    name = "fake"
    queue_term = "queue"

    def __init__(self, verdict):
        self._verdict = verdict

    def probe(self, queue, shape, account=None):
        return self._verdict

    def format_nodelist(self, names):
        return ",".join(sorted(names))

    def submit_flags(self, queue, shape):
        return ["--queue", queue]


class TestStartEstimates:
    def test_an_unreachable_placement_reports_no_start_time(self):
        # Schedulers return a plausible start time alongside a refusal;
        # showing it reads as encouragement to wait for something that will
        # never run.
        q = Queue(name="q", enabled=False, nodes=[_node("n1", alloc=4)])
        cl = _cluster([q], free_times={"n1": datetime(2026, 8, 21, 12)})
        p = evaluate(cl, JobShape(gpus_per_node=4), q)
        assert p.reachable is False
        assert p.earliest_start is None

    def test_a_scheduler_prediction_is_marked_authoritative(self):
        q = Queue(name="q", nodes=[_node("n1", alloc=4)])
        cl = _cluster([q])
        cl._backend = _FakeBackend(Verdict(
            queue="q", allowed=True, category=VerdictCategory.OK,
            predicted_start=datetime(2026, 8, 21, 12),
        ))
        p = evaluate(cl, JobShape(gpus_per_node=4), q, use_probe=True)
        assert p.start_estimate_from_scheduler is True

    def test_our_own_estimate_is_not_marked_authoritative(self):
        # It ignores the queue ahead of you, so it is a lower bound.
        q = Queue(name="q", nodes=[_node("n1", alloc=4)], node_names=("n1",))
        cl = _cluster([q], probe=False, free_times={"n1": datetime(2026, 8, 21, 12)})
        p = evaluate(cl, JobShape(gpus_per_node=4), q)
        assert p.earliest_start == datetime(2026, 8, 21, 12)
        assert p.start_estimate_from_scheduler is False


class TestHardwareVerdicts:
    def test_wrong_hardware_everywhere(self):
        q = Queue(name="q", nodes=[_node("n1", model="V100")])
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=1, requires=("fp8",)), q)
        assert p.hardware_incompatible is True

    def test_right_hardware_all_down_is_a_different_verdict(self):
        q = Queue(name="q", nodes=[_node("n1", model="H100", conditions=("DRAIN",))])
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=4, requires=("fp8",)), q)
        assert p.hardware_incompatible is False
        assert p.capacity.capable_but_all_unavailable is True

    def test_inferred_memory_is_disclosed(self):
        q = Queue(name="q", nodes=[_node("n1", model="A100")])
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=4, gpu_memory_gb=40), q)
        assert any("inferred from" in c for c in p.caveats)

    def test_unidentifiable_accelerators_are_set_aside_and_disclosed(self):
        q = Queue(name="q", nodes=[_node("n1", model=None)])
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=1, requires=("fp8",)), q)
        # The old caveat said they were "neither included nor excluded", which
        # was untrue -- they were included, and counted as capable.
        assert p.capacity.unverified_nodes == ("n1",)
        assert any("could not be identified" in c for c in p.caveats)
        assert any("may well be capable" in c for c in p.caveats)

    def test_unresolved_nodes_are_disclosed(self):
        q = Queue(name="q", nodes=[_node("n1")], node_names=("n1",), declared_nodes=610)
        p = evaluate(_cluster([q]), JobShape(gpus_per_node=4), q)
        assert any("could be resolved" in c for c in p.caveats)


class TestLimitSelection:
    def test_the_probes_effective_qos_wins(self):
        # Sites auto-promote; checking the requested name checks the wrong
        # ceilings.
        q = Queue(name="q", nodes=[_node("n1")])
        limits = {
            "q": Limits(name="q", per_job={"gpu": 99}),
            "q-prio": Limits(name="q-prio", per_job={"gpu": 1}),
        }
        cl = _cluster([q], limits=limits)
        cl._backend = _FakeBackend(Verdict(
            queue="q", allowed=True, category=VerdictCategory.OK,
            effective_qos="q-prio",
        ))
        p = evaluate(cl, JobShape(gpus_per_node=4), q, use_probe=True)
        assert "MAX_GPU_JOB" in {b.code for b in p.blockers}

    def test_falls_back_to_the_queue_name(self):
        q = Queue(name="q", nodes=[_node("n1")])
        cl = _cluster([q], limits={"q": Limits(name="q", per_job={"gpu": 1})})
        p = evaluate(cl, JobShape(gpus_per_node=4), q)
        assert "MAX_GPU_JOB" in {b.code for b in p.blockers}


class TestTemplatedEntitlements:
    def test_the_caveat_appears_when_the_claim_is_worthless(self):
        q = Queue(name="q", nodes=[_node("n1")])
        ident = Identity.from_account_queues("u", {"a": {"x"}, "b": {"x"}})
        cl = _cluster([q], identity=ident)
        p = evaluate(cl, JobShape(gpus_per_node=4), q)
        assert any("carries no information" in c for c in p.caveats)


class TestRanking:
    def _mixed(self):
        return [
            Queue(name="good", nodes=[_node("g1")], node_names=("g1",)),
            Queue(name="busy", nodes=[_node("b1", alloc=4)], node_names=("b1",)),
            Queue(name="dead", enabled=False, nodes=[_node("d1")], node_names=("d1",)),
            Queue(name="wrongstuff", nodes=[_node("w1", model="V100")],
                  node_names=("w1",)),
        ]

    def test_unusable_queues_are_dropped_by_default(self):
        cl = _cluster(self._mixed(), probe=False)
        names = [p.queue for p in rank(cl, JobShape(gpus_per_node=4))]
        assert "dead" not in names

    def test_include_unusable_keeps_them_with_blockers(self):
        cl = _cluster(self._mixed(), probe=False)
        places = {p.queue: p for p in
                  rank(cl, JobShape(gpus_per_node=4), include_unusable=True)}
        assert places["dead"].fatal_blockers

    def test_runnable_first_then_reachable_then_blocked(self):
        cl = _cluster(self._mixed(), probe=False)
        places = rank(cl, JobShape(gpus_per_node=4), include_unusable=True)
        assert [p.score()[0] for p in places] == sorted(p.score()[0] for p in places)
        assert places[0].queue == "good"

    def test_a_gpu_job_skips_cpu_only_queues(self):
        cl = _cluster([
            Queue(name="cpu", nodes=[_node("c1", model=None, gpus=0)],
                  node_names=("c1",)),
            Queue(name="gpu", nodes=[_node("g1")], node_names=("g1",)),
        ], probe=False)
        assert [p.queue for p in rank(cl, JobShape(gpus_per_node=1))] == ["gpu"]

    def test_a_missing_queue_name_is_ignored(self):
        cl = _cluster(self._mixed(), probe=False)
        assert rank(cl, JobShape(gpus_per_node=4), queues=["nope"]) == []


class TestClusterHelpers:
    def test_effective_walltime_takes_the_tighter_of_two(self):
        # A queue reporting no limit while its limit set caps the real ceiling
        # is how a job pends forever on a limit nothing mentioned.
        q = Queue(name="q", max_walltime_seconds=None)
        cl = _cluster([q], limits={"q": Limits(name="q", max_walltime_seconds=3600)})
        assert cl.effective_max_walltime("q") == 3600

    def test_queue_limit_wins_when_tighter(self):
        q = Queue(name="q", max_walltime_seconds=600)
        cl = _cluster([q], limits={"q": Limits(name="q", max_walltime_seconds=3600)})
        assert cl.effective_max_walltime("q") == 600

    def test_no_limits_at_all(self):
        assert _cluster([Queue(name="q")]).effective_max_walltime("q") is None

    def test_phantom_capacity_appears_in_the_summary(self):
        q = Queue(name="dead", enabled=False,
                  nodes=[_node("n1"), _node("n2")], node_names=("n1", "n2"))
        summary = _cluster([q]).summary()
        assert summary["unusable_queues"] == ["dead"]
        assert summary["phantom_capacity"] == {"dead": 2}

    @pytest.mark.parametrize("gpus,expected", [(4, True), (0, False)])
    def test_accelerator_ness_from_count(self, gpus, expected):
        cl = _cluster([Queue(name="q", nodes=[_node("n1", gpus=gpus, model=None)])])
        assert bool(cl.gpu_nodes) is expected


class TestProbeAccounts:
    """Which accounts are worth a control-plane round trip."""

    def test_an_explicit_shape_account_wins(self):
        from nodetop.core.fit import probe_accounts

        q = Queue(name="q", allow_accounts=("a", "b"))
        assert probe_accounts(q, ["a", "b"], JobShape(account="chosen")) == ["chosen"]

    def test_no_accounts_means_one_probe_with_none(self):
        from nodetop.core.fit import probe_accounts

        assert probe_accounts(Queue(name="q"), None, JobShape()) == [None]

    def test_an_allowlist_narrows_the_candidates(self):
        # An account a queue does not list cannot be accepted, so trying it
        # spends a round trip to learn nothing.
        from nodetop.core.fit import probe_accounts

        q = Queue(name="q", allow_accounts=("keep",))
        got = probe_accounts(q, ["drop", "keep", "also-drop"], JobShape())
        assert got == ["keep"]

    def test_matching_is_case_insensitive(self):
        from nodetop.core.fit import probe_accounts

        q = Queue(name="q", allow_accounts=("Keep",))
        assert probe_accounts(q, ["keep"], JobShape()) == ["keep"]

    def test_an_open_queue_keeps_every_account(self):
        from nodetop.core.fit import probe_accounts

        got = probe_accounts(Queue(name="q"), ["a", "b", "c"], JobShape())
        assert got == ["a", "b", "c"]

    def test_an_empty_intersection_still_asks_once(self):
        # Our reading of the allowlist is not the authority; the control plane
        # is. One probe confirms rather than assumes.
        from nodetop.core.fit import probe_accounts

        q = Queue(name="q", allow_accounts=("nobody-here",))
        assert probe_accounts(q, ["mine"], JobShape()) == ["mine"]

    def test_the_candidates_are_not_truncated_here(self):
        # This function used to truncate to MAX_PROBES_PER_QUEUE, and that was
        # a wrong-answer bug rather than a slow-answer one: the caller could no
        # longer tell "refused by the 4 we tried" from "refused by all 50", so
        # it reported the first as the second and hid the queue. On the cluster
        # this was written against that lost `caslake`, `gpu` and `bigmem` --
        # all three accepted by an account lying 32nd in the list.
        #
        # The ceiling still exists; it is applied where the probing happens,
        # which is the only place that can also record that it was applied.
        from nodetop.core.fit import probe_accounts

        many = [f"acct{i}" for i in range(50)]
        assert probe_accounts(Queue(name="q"), many, JobShape()) == many

    def test_narrowing_beats_the_cap(self):
        # The allowlist filter runs first, so the right account is not lost to
        # truncation just because it sorts late.
        from nodetop.core.fit import probe_accounts

        many = [f"acct{i}" for i in range(50)] + ["winner"]
        q = Queue(name="q", allow_accounts=("winner",))
        assert probe_accounts(q, many, JobShape()) == ["winner"]


class TestVerdictLabel:
    """Each label implies a different next move, so they must not collapse.

    ``Placement.reachable`` is deliberately *both* "permitted" and "the shape is
    legal", and the renderer used to test it directly. So a queue whose only
    problem was a per-user accelerator ceiling rendered as BLOCKED / "not
    permitted" -- telling the reader to go request access they already had,
    instead of to ask for fewer accelerators. On a live cluster a 40-node
    request made all five candidate partitions read "not permitted".
    """

    @staticmethod
    def _placement(**kw):
        from nodetop.core.fit import Placement

        return Placement(queue="q", shape=JobShape(nodes=1), **kw)

    def _label(self, **kw):
        from nodetop.cli import _verdict_label

        return _verdict_label(self._placement(**kw))

    def _cap(self, hardware=1, required=1, fitting=0):
        from nodetop.core.capacity import Capacity

        return Capacity(
            considered=max(hardware, required),
            required_nodes=required,
            hardware_nodes=tuple(f"n{i}" for i in range(hardware)),
            capable_nodes=tuple(f"n{i}" for i in range(hardware)),
            fitting_nodes=tuple(f"n{i}" for i in range(fitting)),
        )

    def test_room_now_runs_now(self):
        assert self._label(capacity=self._cap(hardware=1, fitting=1)) == "RUN NOW"

    def test_a_fatal_blocker_is_blocked(self):
        assert self._label(
            blockers=[Blocker("ACCOUNT_NOT_ALLOWED", "no", fatal=True)],
            capacity=self._cap(),
        ) == "BLOCKED"

    def test_a_refusing_verdict_is_blocked(self):
        assert self._label(
            verdict=Verdict(queue="q", allowed=False,
                            category=VerdictCategory.NOT_ENTITLED, reason="no"),
            capacity=self._cap(),
        ) == "BLOCKED"

    def test_a_soft_blocker_is_a_limit_not_denied_access(self):
        # THE regression. A ceiling is cleared by asking for less; being
        # unpermitted is not.
        assert self._label(
            blockers=[Blocker("MAX_GPU_USER", "4 exceeds 2", fatal=False)],
            capacity=self._cap(),
        ) == "LIMIT"

    def test_wrong_kind_of_node_beats_a_ceiling(self):
        # Waiting or resizing will not conjure the right hardware here.
        assert self._label(
            blockers=[Blocker("MAX_GPU_USER", "4 exceeds 2", fatal=False)],
            capacity=self._cap(hardware=0, required=1),
        ) == "WRONG HW"

    def test_not_enough_nodes_is_distinguished_from_the_wrong_kind(self):
        assert self._label(capacity=self._cap(hardware=1, required=40)) == "TOO FEW"

    def test_nothing_wrong_but_no_room_is_a_queue(self):
        assert self._label(capacity=self._cap(hardware=4, required=1)) == "QUEUE"

    def test_access_outranks_everything(self):
        # If you cannot get in, the hardware and the ceilings are moot.
        assert self._label(
            blockers=[Blocker("QUEUE_DISABLED", "down", fatal=True),
                      Blocker("MAX_GPU_USER", "4 exceeds 2", fatal=False)],
            capacity=self._cap(hardware=0, required=40),
        ) == "BLOCKED"

    def test_every_label_has_a_legend_entry(self):
        from nodetop.cli import _VERDICT_LEGEND, _verdict_label
        from nodetop.core.fit import Placement

        # Any label the logic can produce must be renderable.
        produced = {
            _verdict_label(Placement(queue="q", shape=JobShape(nodes=1), **kw))
            for kw in (
                {"capacity": self._cap(hardware=1, fitting=1)},
                {"blockers": [Blocker("X", "x", fatal=True)]},
                {"blockers": [Blocker("X", "x", fatal=False)],
                 "capacity": self._cap()},
                {"capacity": self._cap(hardware=0, required=1)},
                {"capacity": self._cap(hardware=1, required=40)},
                {"capacity": self._cap(hardware=4, required=1)},
                # A verdict we could not obtain: not a refusal.
                {"verdict": Verdict(queue="q", allowed=False,
                                    category=VerdictCategory.CONTROL_PLANE_DOWN,
                                    reason="down"),
                 "capacity": self._cap()},
            )
        }
        # Not a magic number: every documented state must be reachable, and
        # nothing outside the legend may be produced.
        assert produced == set(_VERDICT_LEGEND)


class TestUnresolvedNodesSuppressTheCountVerdict:
    """A queue we could not fully resolve must not be ruled out on its size.

    `evaluate` has to pass the incompleteness through to `assess_capacity`;
    testing `assess_capacity` alone leaves that wiring uncovered, and a mutation
    forcing `count_is_complete=True` survived until this existed.
    """

    @staticmethod
    def _cluster(declared):
        node = Node(name="n0", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                    gpus_total=4, queues=("q",))
        queue = Queue(name="q", node_names=("n0",), declared_nodes=declared,
                      nodes=[node])
        return Cluster(backend_name="synthetic", nodes=[node],
                       queues={"q": queue})

    def _capacity(self, declared, wanted):
        cluster = self._cluster(declared)
        place = evaluate(cluster, JobShape(nodes=wanted), cluster.queues["q"])
        return place.capacity

    def test_a_fully_resolved_queue_is_judged_on_its_size(self):
        cap = self._capacity(declared=1, wanted=40)
        assert cap.required_nodes == 40
        assert cap.ever_possible is False

    def test_an_unresolved_queue_is_not(self):
        # It claims 44 nodes and we found 1; declaring the 40-node shape
        # impossible would rule the queue out for a lookup failure.
        cap = self._capacity(declared=44, wanted=40)
        assert cap.required_nodes == 0
        assert cap.ever_possible is True

    def test_the_disagreement_is_still_disclosed(self):
        cluster = self._cluster(declared=44)
        place = evaluate(cluster, JobShape(nodes=40), cluster.queues["q"])
        assert any("could be resolved" in c for c in place.caveats)


class TestATransientVerdictIsNotARefusal:
    """"We could not ask" must never read as "no".

    A probe that fails -- control plane unreachable, client missing, an answer
    we could not parse -- returns a verdict with `allowed=False` and a category
    in TRANSIENT_CATEGORIES. Three places read that as a denial: `reachable`
    made the placement unreachable (flipping the exit code to "nothing fits"),
    `where` filtered the row away, and the label said "not permitted" for a
    control-plane outage. All three now require the refusal to be *durable*.
    """

    @staticmethod
    def _place(category, allowed=False):
        from nodetop.core.fit import Placement

        return Placement(
            queue="q", shape=JobShape(nodes=1),
            capacity=None,
            verdict=Verdict(queue="q", allowed=allowed, category=category,
                            reason="x"),
        )

    TRANSIENT = VerdictCategory.CONTROL_PLANE_DOWN
    DURABLE = VerdictCategory.NOT_ENTITLED

    def test_a_transient_refusal_leaves_the_placement_reachable(self):
        assert self._place(self.TRANSIENT).reachable is True

    def test_a_durable_refusal_does_not(self):
        assert self._place(self.DURABLE).reachable is False

    def test_the_label_says_no_answer_not_not_permitted(self):
        from nodetop.cli import _verdict_label

        assert _verdict_label(self._place(self.TRANSIENT)) == "NO ANSWER"

    def test_a_durable_refusal_is_still_blocked(self):
        from nodetop.cli import _verdict_label

        assert _verdict_label(self._place(self.DURABLE)) == "BLOCKED"

    def test_no_answer_has_a_legend_entry(self):
        from nodetop.cli import _VERDICT_LEGEND

        assert "NO ANSWER" in _VERDICT_LEGEND
        assert "did not answer" in _VERDICT_LEGEND["NO ANSWER"][2]

    def test_the_two_readings_never_disagree(self):
        # reachable and the label are derived separately, and they drifted
        # apart once: reachable stopped counting transient refusals and the
        # label kept calling them "not permitted" on the same row.
        from nodetop.cli import _verdict_label

        for category in (self.TRANSIENT, self.DURABLE):
            place = self._place(category)
            says_blocked = _verdict_label(place) == "BLOCKED"
            assert says_blocked is (not place.reachable), category


class TestTheProbeKeepsTheMostInformativeVerdict:
    """Accounts disagree, and the loop used to keep whichever came last.

    The question is about the *queue*: one account accepted means yes; one
    account unanswered means unknown, even if another was durably refused,
    because the unanswered one might have worked. Keeping the refusal claims
    more than was established, and which one survived depended on the order the
    accounts happened to be tried in.
    """

    @staticmethod
    def _cluster(answers):
        import dataclasses

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import (
            BackendCapabilities,
            Identity,
            Node,
            Queue,
        )

        node = Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                    queues=("q",))
        queue = Queue(name="q", node_names=("n1",), declared_nodes=1,
                      nodes=[node])

        class _Backend:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub")

            def probe(self, q, shape, account=None):
                return answers[account]

            def submit_flags(self, q, shape):
                return []

        return dataclasses.replace(
            Cluster(backend_name="synthetic", nodes=[node], queues={"q": queue},
                    identity=Identity(user="me", accounts=tuple(answers),
                                      qos=("x",))),
            capabilities=_Backend().capabilities(), _backend=_Backend())

    @staticmethod
    def _v(allowed, category):
        return Verdict(queue="q", allowed=allowed, category=category, reason="r")

    OK = (True, VerdictCategory.OK)
    NO = (False, VerdictCategory.NOT_ENTITLED)
    HUH = (False, VerdictCategory.CONTROL_PLANE_DOWN)

    def _verdict(self, order):
        answers = {f"a{i}": self._v(*spec) for i, spec in enumerate(order)}
        cluster = self._cluster(answers)
        place = evaluate(cluster, JobShape(nodes=1), cluster.queues["q"],
                         use_probe=True, accounts=list(answers))
        return place.verdict

    def test_an_acceptance_wins_wherever_it_appears(self):
        for order in ((self.OK, self.NO), (self.NO, self.OK),
                      (self.HUH, self.OK), (self.OK, self.HUH)):
            assert self._verdict(order).allowed is True, order

    def test_unknown_beats_a_durable_refusal_in_either_order(self):
        # The unanswered account might have worked, so the queue is unknown.
        for order in ((self.NO, self.HUH), (self.HUH, self.NO)):
            v = self._verdict(order)
            assert v.allowed is False
            assert v.durable is False, order

    def test_all_refusals_stay_a_refusal(self):
        v = self._verdict((self.NO, self.NO))
        assert v.allowed is False and v.durable is True

    def test_the_result_does_not_depend_on_account_order(self):
        # The whole defect: it did.
        a = self._verdict((self.NO, self.HUH))
        b = self._verdict((self.HUH, self.NO))
        assert (a.allowed, a.durable) == (b.allowed, b.durable)

class TestTheWorkingAccountIsFoundHoweverLateItSorts:
    """The bug this class exists for hid three usable partitions.

    `probe_accounts` used to truncate its candidate list to 4. On a cluster
    where the user holds 34 associations and the general partitions set
    `AllowAccounts=ALL`, the intersection is all 34 and only the first 4 were
    ever tried. `caslake` (190 nodes), `gpu` (44 accelerators) and `bigmem` are
    all accepted with `rcc-staff`, which sorts 32nd -- so all three were
    reported as refused and dropped from the overview. Verified against the
    real control plane: `sbatch --test-only` accepts every one of them.

    Two mechanisms fix it, and both are tested here because either alone leaves
    a hole: the cap moved to the probe loop (which can record that it applied),
    and the accounts that work are learned and retried first.
    """

    @staticmethod
    def _cluster(*, accepts, allowlists=None, order=None):
        """A cluster whose control plane accepts only `accepts` (queue, acct)."""
        import dataclasses

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import (
            BackendCapabilities,
            Identity,
            Node,
            Queue,
            Verdict,
            VerdictCategory,
        )

        allowlists = allowlists or {}
        held = order or [f"acct{i}" for i in range(34)] + ["winner"]
        nodes, queues = [], {}
        for name in ("open-a", "open-b", "narrow"):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=16000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node],
                                 allow_accounts=allowlists.get(name, ()))
        calls: list[tuple[str, str | None]] = []

        class _Backend:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub --test-only")

            def probe(self, queue, shape, account=None):
                calls.append((queue, account))
                ok = (queue, account) in accepts
                return Verdict(
                    queue=queue, account=account, allowed=ok,
                    category=VerdictCategory.OK if ok
                    else VerdictCategory.NOT_ENTITLED,
                    reason="ok" if ok else "no")

            def submit_flags(self, queue, shape):
                return []

        cluster = dataclasses.replace(
            Cluster(backend_name="synthetic", queue_term="partition",
                    nodes=nodes, queues=queues,
                    identity=Identity(user="me", accounts=tuple(held), qos=("q",))),
            capabilities=_Backend().capabilities(), _backend=_Backend())
        return cluster, held, calls

    def test_an_account_past_the_per_queue_cap_is_still_found(self):
        # `narrow` admits only `winner`, so it costs one probe and proves the
        # account works; `open-a` then tries `winner` first instead of grinding
        # through 34 in declaration order and giving up.
        from nodetop.core.fit import rank

        cluster, held, calls = self._cluster(
            accepts={("narrow", "winner"), ("open-a", "winner")},
            allowlists={"narrow": ("winner",)},
        )
        places = {p.queue: p for p in rank(
            cluster, JobShape(nodes=1, cpus_per_task=1),
            queues=["open-a", "open-b", "narrow"], use_probe=True,
            accounts=held, include_unusable=True)}
        assert places["open-a"].confirmed, [c for c in calls if c[0] == "open-a"]
        assert places["narrow"].confirmed

    def test_the_cheap_queue_is_probed_before_the_open_ones(self):
        # The ordering IS the fix. In the scheduler's own order the 34-candidate
        # queues went first, spent the cap, and were written off -- then the
        # one-candidate queue accepted that very account a moment later.
        from nodetop.core.fit import rank

        cluster, held, calls = self._cluster(
            accepts={("narrow", "winner")},
            allowlists={"narrow": ("winner",)},
        )
        rank(cluster, JobShape(nodes=1, cpus_per_task=1),
             queues=["open-a", "open-b", "narrow"], use_probe=True,
             accounts=held, include_unusable=True)
        assert calls[0][0] == "narrow"

    def test_all_refused_but_not_all_asked_is_not_a_refusal(self):
        # The honesty half. When the ceiling is reached with candidates left
        # over, the queue must not be reported as having refused.
        from nodetop.core.fit import ProbeBudget, probe_queue, unsettled
        from nodetop.core.model import VerdictCategory

        cluster, held, _ = self._cluster(accepts=set())
        budget = ProbeBudget(total=3)           # smaller than the 35 candidates
        verdict, tried, of = probe_queue(
            cluster, cluster.queues["open-a"], "open-a",
            JobShape(nodes=1, cpus_per_task=1), held, budget)
        assert tried < of
        settled = unsettled(verdict, tried, of)
        assert settled is not None
        assert settled.category == VerdictCategory.ACCOUNTS_UNTRIED
        assert not settled.durable            # so it is not dropped as refused
        assert not settled.confirmed          # but nor is it claimed as usable

    def test_the_caveat_says_the_rest_were_not_asked(self):
        # The reader has to be able to tell the two apart on screen too.
        from nodetop.core.fit import ProbeBudget, evaluate

        cluster, held, _ = self._cluster(accepts=set())
        place = evaluate(
            cluster, JobShape(nodes=1, cpus_per_task=1),
            cluster.queues["open-a"], use_probe=True, accounts=held,
            budget=ProbeBudget(total=3))
        assert any("not asked" in c for c in place.caveats)

    def test_naming_one_queue_spends_the_budget_settling_it(self):
        # The other end of the same policy. A fixed per-queue ceiling of 12 left
        # `check -q caslake` reporting ACCOUNTS_UNTRIED for a queue the user had
        # explicitly asked about, twenty probes short of the account that works.
        from nodetop.core.fit import rank

        cluster, held, _ = self._cluster(accepts={("open-a", "winner")})
        place = next(p for p in rank(
            cluster, JobShape(nodes=1, cpus_per_task=1), queues=["open-a"],
            use_probe=True, accounts=held, include_unusable=True))
        assert place.confirmed, "a single named queue should be settled"

    def test_exhausting_the_candidates_is_a_real_refusal(self):
        # The other side of it: when every admitted account WAS asked, the
        # refusal is durable and the queue should be dropped.
        from nodetop.core.fit import rank

        cluster, held, _ = self._cluster(
            accepts=set(), allowlists={"narrow": ("winner",)})
        place = next(p for p in rank(
            cluster, JobShape(nodes=1, cpus_per_task=1), queues=["narrow"],
            use_probe=True, accounts=held, include_unusable=True))
        assert place.verdict is not None and place.verdict.durable
        assert not place.verdict.allowed

    def test_a_wide_sweep_keeps_the_per_queue_ceiling_low(self):
        # Budget per queue, so ninety queues do not each get seventy-five
        # tries. The global total is the real protection either way.
        from nodetop.core.fit import MAX_PROBES_PER_QUEUE, ProbeBudget

        assert ProbeBudget(queues=90).per_queue == MAX_PROBES_PER_QUEUE
        assert ProbeBudget(queues=2).per_queue > MAX_PROBES_PER_QUEUE
        assert ProbeBudget().per_queue == MAX_PROBES_PER_QUEUE

    def test_the_total_probe_count_is_bounded(self):
        # The ceiling that actually protects the control plane is global. A
        # per-queue cap bounds nothing -- 50 queues x 4 is still 200 round trips
        # -- which is why it could be raised once the budget existed.
        from nodetop.core.fit import MAX_PROBES_TOTAL, rank

        cluster, held, calls = self._cluster(accepts=set())
        rank(cluster, JobShape(nodes=1, cpus_per_task=1),
             queues=["open-a", "open-b", "narrow"], use_probe=True,
             accounts=held, include_unusable=True)
        assert len(calls) <= MAX_PROBES_TOTAL


class TestProbeBudget:
    def test_it_puts_known_good_accounts_first(self):
        from nodetop.core.fit import ProbeBudget

        b = ProbeBudget()
        assert b.order(["a", "b", "c"]) == ["a", "b", "c"]   # nothing learned
        b.accepted("c")
        assert b.order(["a", "b", "c"]) == ["c", "a", "b"]

    def test_the_order_is_otherwise_stable(self):
        # Reordering beyond moving the winners forward would make the probe
        # sequence depend on dict iteration, which is not reproducible.
        from nodetop.core.fit import ProbeBudget

        b = ProbeBudget()
        b.accepted("b")
        assert b.order(["a", "b", "c", "d"]) == ["b", "a", "c", "d"]

    def test_it_stops_spending_when_empty(self):
        from nodetop.core.fit import ProbeBudget

        b = ProbeBudget(total=2)
        assert b.spend() and b.spend()
        assert not b.spend()

    def test_a_none_account_is_not_remembered(self):
        # `None` means "no account named", which is not a fact about any
        # account and must not be promoted to the front of the next queue.
        from nodetop.core.fit import ProbeBudget

        b = ProbeBudget()
        b.accepted(None)
        assert b.order(["a", "b"]) == ["a", "b"]
