"""The scheduler-neutral model: node state, queue gates, ceilings, identity."""

from __future__ import annotations

import pathlib

import pytest

from nodetop.core.hardware import ACCELERATORS
from nodetop.core.model import (
    Blocker,
    Identity,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
    capability_gap,
    split_reason,
)


class TestNodeAvailability:
    @pytest.mark.parametrize("conditions", [
        {"DOWN"}, {"DRAIN"}, {"MAINT"}, {"FAIL"}, {"UNKNOWN"},
        {"RESERVED"}, {"POWERSAVE"}, {"DOWN", "DRAIN"},
    ])
    def test_blocking_conditions(self, conditions):
        assert Node(name="n", conditions=frozenset(conditions)).schedulable is False

    def test_no_conditions_is_schedulable(self):
        assert Node(name="n").schedulable is True

    def test_allocated_is_schedulable_but_not_idle(self):
        n = Node(name="n", cpus_total=32, cpus_alloc=32)
        assert n.schedulable is True
        assert n.idle is False

    def test_informational_conditions_do_not_block(self):
        # A backend may pass through its own labels; only the known blocking
        # set stops scheduling.
        assert Node(name="n", conditions=frozenset({"BUSY", "HOT"})).schedulable is True


class TestNodeCapacity:
    def test_accelerator_presence_is_decided_by_count_not_name(self):
        # A "gpu"-prefixed node with no accelerator, and an unremarkable one
        # with four, both occur; filtering on the name is how CPU work ends up
        # occupying an accelerator.
        assert Node(name="gpu-front-01").is_gpu_node is False
        assert Node(name="bigmem1", gpus_total=4).is_gpu_node is True

    def test_free_counts_never_go_negative(self):
        n = Node(name="n", cpus_total=2, cpus_alloc=8, gpus_total=1, gpus_alloc=4,
                 memory_mb=100, memory_alloc_mb=500)
        assert (n.cpus_free, n.gpus_free, n.memory_free_mb) == (0, 0, 0)


class TestNodeDegraded:
    @pytest.mark.parametrize("reason", [
        "HwPowerBrake: clocks pinned", "thermal throttling detected",
        "GPU ECC errors, row remapping pending", "XID 48 reported",
        "fan failure", "MemoryPressure reported by kubelet",
    ])
    def test_impaired_but_running_is_flagged(self, reason):
        # The nastiest category: the scheduler reports it healthy and hands it
        # out while it runs several times slower than its siblings.
        n = Node(name="n", reason=reason)
        assert n.schedulable is True
        assert n.degraded is True

    def test_an_unschedulable_node_is_not_called_degraded(self):
        # It is not impaired-but-usable, it is simply out.
        n = Node(name="n", conditions=frozenset({"DRAIN"}), reason="thermal")
        assert n.degraded is False

    def test_no_reason_means_not_degraded(self):
        assert Node(name="n").degraded is False

    def test_an_ordinary_reason_is_not_degradation(self):
        assert Node(name="n", reason="reserved for the workshop").degraded is False


class TestANodeIsOneReadingAndIsNotEdited:
    """The invariant that lets the hot answers be memoised.

    The same node is asked the same question once per queue it belongs to:
    `memory_exhausted` was called **4,830 times for 607 nodes** on a cluster
    with 84 partitions. Memoising it turns the aggregate capacity pass from
    53.9 ms into 13.1 ms (-76%) on a worst case of 607 nodes in 84 queues, and
    costs 1.4% on the one path that visits each node exactly once -- building
    a 10,624-row table, where the cost is string work either way.

    That is only correct while a Node is written once and never edited, so the
    tests here pin the invariant rather than the timing: a fresh object per
    correction, and no assignment to a Node's fields anywhere in the source.
    """

    def _node(self, **kw):
        from nodetop.core.model import Node

        base = {"name": "n1", "state_raw": "MIXED", "cpus_total": 8,
                "cpus_alloc": 4, "memory_mb": 1000, "memory_alloc_mb": 400}
        return Node(**{**base, **kw})

    def test_the_answer_is_remembered_on_the_instance(self):
        node = self._node()
        assert node.memory_exhausted is False
        # The cache lives in the instance dict, so two nodes cannot share one.
        assert "memory_exhausted" in node.__dict__
        assert "memory_exhausted" not in self._node().__dict__

    def test_replacing_a_field_gives_an_object_with_no_stale_answer(self):
        # This is what Grid Engine relies on: it has to correct accelerator
        # counts after `qhost` has been parsed, and it rebuilds rather than
        # patching precisely so a memoised answer cannot outlive its input.
        import dataclasses

        node = self._node(memory_alloc_mb=400)
        assert node.memory_exhausted is False
        fixed = dataclasses.replace(node, memory_alloc_mb=1000)
        assert "memory_exhausted" not in fixed.__dict__
        assert fixed.memory_exhausted is True

    def test_nothing_in_the_source_writes_a_nodes_fields(self):
        """A source scan, because this cannot be tested from behaviour.

        A backend that assigns `n.cpus_alloc = ...` after something has read
        `n.effective_free_cpus` gets the old answer, silently and only on that
        backend. Grid Engine did exactly that (accelerator counts from
        `qconf -se`, patched in place) and is why this test exists.
        """
        import ast
        import dataclasses

        from nodetop.core.model import Node

        fields = {f.name for f in dataclasses.fields(Node)}
        root = pathlib.Path(__file__).resolve().parent.parent / "src" / "nodetop"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr in fields:
                        # `self.x = ...` inside the model's own dataclasses is
                        # construction, not editing.
                        if path.name == "model.py" and isinstance(
                                target.value, ast.Name) and target.value.id == "self":
                            continue
                        offenders.append(
                            f"{path.relative_to(root)}:{node.lineno} "
                            f"-> .{target.attr}")
        assert not offenders, "a Node field is written after construction: " + \
            "; ".join(offenders)


class TestQueueStructuralGates:
    def test_disabled(self):
        q = Queue(name="q", enabled=False)
        assert q.usable is False
        assert "QUEUE_DISABLED" in {b.code for b in q.structural_blockers()}

    def test_accepts_but_never_starts(self):
        # PBS enabled/started, LSF Open:Inact, Slurm DRAIN: the queue swallows
        # submissions that will never run.
        q = Queue(name="q", enabled=True, started=False)
        assert q.usable is False
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}

    def test_disabled_does_not_also_report_not_started(self):
        # Two blockers for one switch is noise, not two facts.
        codes = {b.code for b in Queue(name="q", enabled=False, started=False)
                 .structural_blockers()}
        assert codes == {"QUEUE_DISABLED"}

    @pytest.mark.parametrize("field,code", [
        ("allow_accounts", "NO_ACCOUNTS"),
        ("allow_qos", "NO_QOS"),
        ("allow_users", "NO_USERS"),
    ])
    def test_an_explicit_empty_allowlist_permits_nobody(self, field, code):
        # A literal "none" must stay distinguishable from "no restriction".
        q = Queue(name="q", **{field: ("none",)})
        assert q.usable is False
        assert code in {b.code for b in q.structural_blockers()}

    def test_every_node_down(self):
        q = Queue(name="q", nodes=[
            Node(name="a", conditions=frozenset({"DOWN"})),
            Node(name="b", conditions=frozenset({"DRAIN"})),
        ])
        assert q.usable is False
        assert "ALL_NODES_UNSCHEDULABLE" in {b.code for b in q.structural_blockers()}

    def test_reservation_requirement_is_soft(self):
        q = Queue(name="q", requires_reservation=True)
        blockers = q.structural_blockers()
        assert [b.fatal for b in blockers] == [False]
        assert q.usable is True

    def test_several_gates_are_reported_independently(self):
        # Fixing one of four changes nothing, so all four must be visible.
        q = Queue(name="q", enabled=False, allow_accounts=("none",),
                  allow_qos=("none",), hidden=True)
        assert {b.code for b in q.structural_blockers()} == {
            "QUEUE_DISABLED", "NO_ACCOUNTS", "NO_QOS"
        }


class TestPhantomCapacity:
    """The headline behaviour, in neutral terms."""

    def _queue(self, **kw):
        idle = [Node(name=f"n{i}", cpus_total=8, gpus_total=4) for i in range(3)]
        return Queue(name="q", nodes=idle, **kw)

    def test_a_dead_queue_reports_zero_usable_capacity(self):
        q = self._queue(enabled=False)
        # The raw count stays available for diagnosis...
        assert len(q.idle_nodes) == 3
        assert q.gpus_free == 12
        # ...but what a summary shows is zero.
        assert q.effective_free_nodes == 0
        assert q.effective_free_gpus == 0

    def test_a_live_queue_reports_real_capacity(self):
        q = self._queue()
        assert q.effective_free_nodes == 3
        assert q.effective_free_gpus == 12

    def test_unresolved_nodes_are_surfaced_not_hidden(self):
        q = Queue(name="q", declared_nodes=610,
                  nodes=[Node(name="n1")], node_names=("n1",))
        assert q.unresolved_nodes == 609


class TestCoresWithNoMemoryBehindThem:
    """Idle cores on a node whose memory is spoken for are not room.

    Real and large: `wide` advertised 2322 free cores, 2035 of them on 47
    nodes with every byte of memory allocated to a handful of four-core jobs.
    The scheduler gives every job memory -- the site default if the job names
    no figure -- so those cores can start nothing, and they were the biggest
    number on the screen.
    """

    def _node(self, alloc_mb: int, **kw):
        return Node(name="n", cpus_total=48, cpus_alloc=4, memory_mb=184320,
                    memory_alloc_mb=alloc_mb, **kw)

    def test_a_node_with_memory_left_offers_its_free_cores(self):
        n = self._node(4096)
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 44
        assert n.has_room is True

    def test_a_node_with_no_memory_left_offers_nothing(self):
        n = self._node(184320)
        assert n.memory_exhausted is True
        # The claimed count is still there for diagnosis...
        assert n.cpus_free == 44
        # ...and what any summary shows is zero.
        assert n.effective_free_cpus == 0
        assert n.has_room is False

    def test_its_accelerators_are_unreachable_too(self):
        # A GPU job needs host memory like any other.
        n = self._node(184320, gpus_total=4, gpus_alloc=1)
        assert n.gpus_free == 3
        assert n.effective_free_gpus == 0
        assert n.has_room is False

    def test_a_backend_that_reports_no_memory_is_not_penalised(self):
        # "Cannot tell" is not "none": inventing a shortage on a system that
        # never mentioned memory would be its own kind of lie.
        n = Node(name="n", cpus_total=48, cpus_alloc=4, memory_mb=0)
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 44
        assert n.has_room is True

    def test_a_scheduler_that_does_not_enforce_memory_is_not_penalised_either(self):
        # Slurm without `_MEMORY` in `SelectTypeParameters` never decrements
        # memory, so `AllocMem` records what jobs asked for rather than a
        # ceiling. Reading it as one would report a whole cluster as full.
        n = self._node(184320, memory_consumable=False)
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 44
        assert n.has_room is True

    def test_the_floor_is_the_schedulers_own_not_zero(self):
        # The case that got away, by 250 megabytes. A 128-core node with 31
        # cores allocated, RealMemory=250000, AllocMem=249750, on a cluster
        # whose DefMemPerCPU is 3810 -- so one core of a default job needs
        # fifteen times what is left, and the row read "97/128" cores free with
        # a three-quarters-full meter beside "0/244G", sorted to the top of the
        # listing as the roomiest thing on the screen. 250 MB rounds down to
        # 0 GiB, which is why the memory column looked right while the CPU
        # column did not.
        n = Node(name="n", cpus_total=128, cpus_alloc=31, memory_mb=250000,
                 memory_alloc_mb=249750, memory_floor_mb=3810)
        assert n.cpus_free == 97            # still reported, for diagnosis
        assert n.memory_free_mb == 250      # above zero, below the floor
        assert n.memory_exhausted is True
        assert n.effective_free_cpus == 0
        assert n.has_room is False

    def test_memory_above_the_floor_is_still_room(self):
        # Not a blanket "nearly full is full": a node that can seat one default
        # core seats one, and the meter says so.
        n = Node(name="n", cpus_total=128, cpus_alloc=31, memory_mb=250000,
                 memory_alloc_mb=250000 - 4000, memory_floor_mb=3810)
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 97

    def test_a_site_with_no_published_floor_keeps_the_zero_test(self):
        # Slurm defaults a job to the WHOLE node where DefMemPerCPU is unset,
        # which is a shortage this cannot quantify -- so it claims nothing
        # rather than erasing most of a cluster.
        n = Node(name="n", cpus_total=128, cpus_alloc=31, memory_mb=250000,
                 memory_alloc_mb=249750)
        assert n.memory_floor_mb == 0
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 97

    def test_the_floor_is_irrelevant_where_memory_is_not_consumable(self):
        n = Node(name="n", cpus_total=128, cpus_alloc=31, memory_mb=250000,
                 memory_alloc_mb=250000, memory_floor_mb=3810,
                 memory_consumable=False)
        assert n.memory_exhausted is False
        assert n.effective_free_cpus == 97

    def test_an_unschedulable_node_has_no_room_whatever_it_reports(self):
        n = Node(name="n", cpus_total=48, memory_mb=1000,
                 conditions=frozenset({"DRAIN"}))
        assert n.effective_free_cpus == 48   # the claim
        assert n.has_room is False           # the answer

    def test_a_queue_counts_only_cores_it_can_hand_out(self):
        starved = self._node(184320)
        spare = Node(name="m", cpus_total=48, cpus_alloc=44, memory_mb=184320,
                     memory_alloc_mb=4096)
        q = Queue(name="q", nodes=[starved, spare])
        assert q.cpus_free == 48             # 44 + 4, as the scheduler says
        assert q.effective_free_cpus == 4    # what can actually be allocated

    def test_a_queue_of_starved_nodes_reports_zero(self):
        q = Queue(name="q", nodes=[self._node(184320) for _ in range(47)])
        assert q.cpus_free == 47 * 44
        assert q.effective_free_cpus == 0

    def test_a_node_with_nothing_running_and_no_memory_is_not_a_free_node(self):
        """"Nothing is running here" is not "something could run here".

        Reachable on Kubernetes, where a pod may request memory with no CPU
        request at all: the node reports zero allocated CPU -- so `idle` is
        true -- and no allocatable memory. It was counted in the one column
        that claims a node is wholly free.
        """
        n = Node(name="n", cpus_total=48, cpus_alloc=0, memory_mb=184320,
                 memory_alloc_mb=184320)
        assert n.idle is True            # nothing is running
        assert n.has_room is False       # and nothing can
        q = Queue(name="q", nodes=[n])
        assert len(q.idle_nodes) == 1    # the raw claim survives
        assert q.effective_free_nodes == 0

    def test_a_genuinely_idle_node_still_counts(self):
        n = Node(name="n", cpus_total=48, memory_mb=184320)
        q = Queue(name="q", nodes=[n])
        assert q.effective_free_nodes == 1


class TestQueueAccessGates:
    def test_account_not_in_allowlist(self):
        q = Queue(name="q", allow_accounts=("alice", "bob"))
        codes = {b.code for b in q.access_blockers(accounts={"carol"})}
        assert "ACCOUNT_NOT_ALLOWED" in codes

    def test_account_in_allowlist_passes(self):
        q = Queue(name="q", allow_accounts=("alice", "bob"))
        assert q.access_blockers(accounts={"alice"}) == []

    def test_deny_beats_allow(self):
        q = Queue(name="q", allow_accounts=("alice",), deny_accounts=("alice",))
        assert q.access_blockers(accounts={"alice"}) != []

    def test_no_values_supplied_yields_no_verdict(self):
        # With nothing to compare against, claiming a verdict is a fabrication.
        q = Queue(name="q", allow_accounts=("alice",))
        assert q.access_blockers() == []

    def test_open_queue_allows_anyone(self):
        assert Queue(name="q").access_blockers(accounts={"whoever"}) == []

    def test_user_allowlist(self):
        q = Queue(name="q", allow_users=("alice",))
        assert q.access_blockers(user="bob") != []
        assert q.access_blockers(user="alice") == []


class TestLimits:
    def test_walltime_ceiling(self):
        limits = Limits(name="l", max_walltime_seconds=172800)
        got = limits.blockers(JobShape(walltime="8-00:00:00"))
        assert [b.code for b in got] == ["MAX_WALLTIME"]

    def test_exactly_at_the_ceiling_is_allowed(self):
        limits = Limits(name="l", max_walltime_seconds=172800)
        assert limits.blockers(JobShape(walltime="2-00:00:00")) == []

    def test_the_message_names_the_silent_failure(self):
        limits = Limits(name="l", max_walltime_seconds=3600)
        detail = limits.blockers(JobShape(walltime="2h"))[0].detail
        assert "queued indefinitely" in detail

    def test_per_job_accelerator_ceiling(self):
        limits = Limits(name="l", per_job={"gpu": 4})
        got = limits.blockers(JobShape(nodes=3, gpus_per_node=4))
        assert [b.code for b in got] == ["MAX_GPU_JOB"]

    def test_per_user_ceiling_also_bounds_one_job(self):
        # One job cannot exceed what a user may hold in aggregate.
        limits = Limits(name="l", per_user={"node": 8})
        assert [b.code for b in limits.blockers(JobShape(nodes=40))] == ["MAX_NODE_USER"]

    def test_ceilings_are_soft_not_fatal(self):
        # The queue is fine; the request is too big, and shrinking it gets in.
        limits = Limits(name="l", per_job={"gpu": 4}, max_walltime_seconds=60)
        got = limits.blockers(JobShape(nodes=40, gpus_per_node=8, walltime="8h"))
        assert got and all(not b.fatal for b in got)

    def test_unlimited_request_against_a_ceiling(self):
        limits = Limits(name="l", max_walltime_seconds=3600)
        assert limits.blockers(JobShape(walltime="UNLIMITED")) == []


class TestIdentity:
    def test_identical_menus_are_flagged_as_templated(self):
        # The claim "you may use these queues" carries no information when it
        # is the same claim for every account.
        ident = Identity.from_account_queues("u", {
            "a": {"x", "y"}, "b": {"x", "y"}, "c": {"x", "y"},
        })
        assert ident.entitlements_look_templated is True

    def test_differing_menus_are_not_flagged(self):
        ident = Identity.from_account_queues("u", {"a": {"x"}, "b": {"y"}})
        assert ident.entitlements_look_templated is False

    def test_a_single_account_cannot_look_templated(self):
        # Nothing to compare against, so asserting the pattern would be a guess.
        ident = Identity.from_account_queues("u", {"a": {"x", "y"}})
        assert ident.entitlements_look_templated is False


class TestVerdict:
    @pytest.mark.parametrize("category", [
        VerdictCategory.CONTROL_PLANE_DOWN,
        VerdictCategory.SHAPE_UNAVAILABLE,
        VerdictCategory.NOT_SUPPORTED,
        VerdictCategory.UNKNOWN,
    ])
    def test_transient_categories_are_not_durable(self, category):
        # A sick control plane or a too-large shape says nothing lasting about
        # your access.
        assert Verdict(queue="q", category=category).durable is False

    def test_entitlement_refusal_is_durable(self):
        assert Verdict(queue="q", category=VerdictCategory.NOT_ENTITLED).durable is True

    def test_confirmed_requires_both_allowed_and_ok(self):
        assert Verdict(queue="q", allowed=True,
                       category=VerdictCategory.OK).confirmed is True
        assert Verdict(queue="q", allowed=True,
                       category=VerdictCategory.UNKNOWN).confirmed is False


class TestCapabilityGap:
    def test_known_gaps_are_listed(self):
        gaps = capability_gap(ACCELERATORS["V100"], ("bf16", "fp8"))
        assert set(gaps) == {"bf16", "fp8"}

    def test_unknown_hardware_yields_no_gaps(self):
        # "We cannot identify this card" is not "this card cannot do the job".
        assert capability_gap(None, ("bf16", "fp8")) == []

    def test_capable_hardware_yields_no_gaps(self):
        assert capability_gap(ACCELERATORS["H100"], ("bf16", "fp8")) == []


class TestJobShape:
    def test_totals(self):
        s = JobShape(nodes=4, gpus_per_node=8, cpus_per_task=4, tasks_per_node=2)
        assert (s.total_gpus, s.total_cpus, s.cpus_per_node) == (32, 32, 8)

    def test_describe_mentions_the_unschedulable_requirements(self):
        text = JobShape(nodes=2, gpus_per_node=4, gpu_memory_gb=40,
                        requires=("bf16",)).describe()
        assert "2 nodes" in text and "8 total" in text and "bf16" in text
        assert "40 GiB" in text


class TestBlocker:
    def test_str_is_readable(self):
        assert str(Blocker("CODE", "detail")) == "CODE: detail"


class TestRoutingQueues:
    def test_forwards_to_makes_a_queue_a_router(self):
        assert Queue(name="r", forwards_to=("a", "b")).routes is True

    def test_an_ordinary_queue_does_not_route(self):
        assert Queue(name="q").routes is False

    def test_owning_no_nodes_is_not_all_nodes_down(self):
        # "every node is down" is a finding; "owns no nodes by design" is not.
        r = Queue(name="r", forwards_to=("a",),
                  nodes=[Node(name="n", conditions=frozenset({"DOWN"}))])
        assert "ALL_NODES_UNSCHEDULABLE" not in {
            b.code for b in r.structural_blockers()
        }

    def test_an_execution_queue_with_every_node_down_still_reports_it(self):
        q = Queue(name="q", nodes=[Node(name="n", conditions=frozenset({"DOWN"}))])
        assert "ALL_NODES_UNSCHEDULABLE" in {b.code for b in q.structural_blockers()}

    def test_a_router_can_still_be_disabled(self):
        r = Queue(name="r", forwards_to=("a",), enabled=False)
        assert r.usable is False


class TestLimitWording:
    """The internal keys are terse; the message should not be."""

    def test_the_resource_is_named_in_english(self):
        limits = Limits(name="gpuq", per_job={"gpu": 2})
        detail = limits.blockers(JobShape(nodes=4, gpus_per_node=1))[0].detail
        # "4 gpu exceeds ... limit of 2" reads like a typo in the tool.
        assert "4 accelerators" in detail
        assert " gpu " not in detail

    @pytest.mark.parametrize("key,asked,expected", [
        ("cpu", {"cpus_per_task": 128}, "CPUs"),
        ("node", {"nodes": 8}, "nodes"),
        ("mem_mb", {"memory_gb": 8}, "MB of memory"),
    ])
    def test_each_resource_reads_naturally(self, key, asked, expected):
        limits = Limits(name="q", per_job={key: 1})
        detail = limits.blockers(JobShape(**asked))[0].detail
        assert expected in detail

    def test_it_agrees_in_number(self):
        limits = Limits(name="q", per_job={"node": 0})
        detail = limits.blockers(JobShape(nodes=1))[0].detail
        assert "1 node exceeds" in detail
        assert "1 nodes" not in detail

    def test_the_limit_set_is_named_at_the_end_not_mid_sentence(self):
        limits = Limits(name="gn", per_job={"gpu": 2})
        detail = limits.blockers(JobShape(nodes=4, gpus_per_node=1))[0].detail
        assert detail.endswith("on gn")


class TestBlockerLabel:
    """``detail`` explains; ``label`` labels. The overview needs the short one."""

    def test_it_prefers_the_short_form(self):
        b = Blocker("QUEUE_DISABLED", "the queue is DOWN and will not schedule",
                    short="DOWN")
        assert b.label == "DOWN"

    def test_it_falls_back_to_a_readable_code(self):
        # No short form is not a reason to print a sentence, nor a raw
        # SHOUTING_CODE, in a summary line.
        assert Blocker("NO_ACCOUNTS", "a long explanation").label == "no accounts"

    def test_the_fallback_never_leaks_underscores(self):
        assert "_" not in Blocker("ALL_NODES_UNSCHEDULABLE", "x").label

    def test_the_label_is_shorter_than_the_detail(self):
        # The whole point: it has to be usable on a shared line.
        b = Blocker("QUEUE_NOT_STARTED",
                    "the queue accepts jobs but has never started one",
                    short="accepts but never starts")
        assert len(b.label) < len(b.detail)

    def test_detail_is_still_what_str_shows(self):
        # Adding a label must not change the full rendering.
        b = Blocker("CODE", "detail", short="tag")
        assert str(b) == "CODE: detail"


class TestSplitReason:
    """``reason [who@when]``: the stamp has to come off before anything reads it."""

    def test_it_separates_the_three_parts(self):
        assert split_reason("maintenance [root@2026-07-16T08:51:31]") == (
            "maintenance", "root", "2026-07-16T08:51:31")

    def test_a_reason_with_no_stamp_is_returned_whole(self):
        assert split_reason("Not responding") == ("Not responding", "", "")

    def test_none_and_empty_are_all_empty(self):
        assert split_reason(None) == ("", "", "")
        assert split_reason("") == ("", "", "")

    def test_a_colon_in_the_reason_survives(self):
        text, who, _ = split_reason("maintenance: hardware issue [root@2026-08-20T18:55:05]")
        assert text == "maintenance: hardware issue"
        assert who == "root"

    def test_only_a_trailing_stamp_counts(self):
        # A bracketed aside mid-sentence is part of the reason.
        assert split_reason("drained [see ticket] pending")[0] == (
            "drained [see ticket] pending")

    def test_a_stamp_with_no_reason_yields_empty_text(self):
        assert split_reason("[root@now]") == ("", "root", "now")

    def test_whitespace_is_trimmed(self):
        assert split_reason("  maintenance   [root@x]  ")[0] == "maintenance"


class TestDegradedIgnoresTheOperator:
    """DEGRADED_HINTS holds short words like ``fan``, ``slow`` and ``clock``.

    Matched against the whole reason string they also match the *username* of
    whoever set it, so an administrator called ``fanl`` silently marked every
    node they drained as thermally impaired.
    """

    def _node(self, reason):
        return Node(name="n1", state_raw="IDLE", cpus_total=8, reason=reason)

    @pytest.mark.parametrize("who", ["fanl", "slowik", "clocke", "eccles", "xiding"])
    def test_a_hint_word_in_the_operator_name_is_not_impairment(self, who):
        assert self._node(f"rebooting [{who}@2026-01-01T00:00:00]").degraded is False

    @pytest.mark.parametrize("text", [
        "fan failure", "clock throttled", "ECC errors", "running slow",
        "HwPowerBrake", "thermal event",
    ])
    def test_a_hint_word_in_the_reason_still_is(self, text):
        assert self._node(f"{text} [root@2026-01-01T00:00:00]").degraded is True

    def test_an_unstamped_reason_is_still_matched(self):
        # Not every backend stamps; the match must not depend on the suffix.
        assert self._node("thermal throttling").degraded is True


class TestQueueIsDedicated:
    """Whether a queue is one group's private hardware, read structurally.

    The accounting database cannot answer this on a cluster whose associations
    are templated -- it reported the user as associated with 34 accounts and
    gave every one an identical QOS list -- so the queue's own allowlist is
    read instead. A partition naming `pi-okafor` alone says what that hardware
    is for.
    """

    def test_a_single_account_allowlist_is_dedicated(self):
        assert Queue(name="okafor-gpu", allow_accounts=("pi-okafor",)).is_dedicated

    def test_two_accounts_is_still_dedicated(self):
        # A PI partition routinely names the group plus a collaborator.
        assert Queue(name="q", allow_accounts=("lawson", "pi-varga")).is_dedicated

    def test_a_wide_allowlist_is_shared(self):
        assert not Queue(
            name="gn",
            allow_accounts=tuple(f"a{i}" for i in range(28)),
        ).is_dedicated

    def test_no_allowlist_at_all_is_shared(self):
        # Unrestricted: open to everyone with an account on the cluster.
        assert not Queue(name="wide").is_dedicated

    def test_the_threshold_is_a_named_constant(self):
        # It is quoted in the output, so it must be one value.
        assert Queue.DEDICATED_ACCOUNT_LIMIT >= 1
        wide = tuple(f"a{i}" for i in range(Queue.DEDICATED_ACCOUNT_LIMIT + 1))
        assert not Queue(name="q", allow_accounts=wide).is_dedicated

    def test_it_is_independent_of_capacity_and_state(self):
        # A property of who may submit, not of what is free or whether it is up.
        q = Queue(name="q", allow_accounts=("pi-x",), enabled=False,
                  declared_nodes=100)
        assert q.is_dedicated


class TestDegradedCatchesTheFailuresAGpuClusterActuallyHas:
    """The hint table began as a thermal/ECC list, which is not where a GPU
    cluster loses nodes.

    The additions are the failure modes an operator most often leaves
    *schedulable* -- an accelerator that has dropped off the PCIe bus, a dead
    NVLink, a fabric link down. A node with a dead InfiniBand link accepts work
    and then fails every multi-node job placed on it, which is exactly the case
    this property exists to catch, and none of them matched.
    """

    @staticmethod
    def _degraded(reason: str) -> bool:
        return Node(name="n", state_raw="IDLE", cpus_total=8, memory_mb=1000,
                    reason=reason).degraded

    @pytest.mark.parametrize("reason", [
        "gpu fell off the bus",        # Xid 79 in plain English
        "nvidia-smi timeout",
        "NVLink error on link 3",
        "DCGM health check failed",
        "uncorrectable ECC error",     # says nothing about "ecc" alone
        "double bit error",
        "IB link down",
        "InfiniBand port inactive",
        "out of memory",
    ])
    def test_it_is_flagged(self, reason):
        assert self._degraded(reason)

    @pytest.mark.parametrize("reason", [
        "drained by Fang",         # `fan`  -- the original false positive, again
        "reported by Rebecca",     # `ecc`
        "drained by Xidong",       # `xid`
        "node moved by Fanny",
        "maintenance",
        "planned",
        "reboot scheduled",
        "replaced disk",
    ])
    def test_an_operators_name_is_not_a_hardware_fault(self, reason):
        # Stripping the [who@when] stamp was not enough on its own: a name
        # written into the prose collides just as well, and every short hint had
        # a plausible one. Hints under four characters are matched as whole
        # words for this reason.
        assert not self._degraded(reason)

    @pytest.mark.parametrize("reason", [
        "MemoryPressure reported by kubelet",   # Kubernetes condition
        "HwPowerBrake",                         # NVIDIA flag
        "GpuFellOffTheBus",
    ])
    def test_camel_case_still_matches(self, reason):
        # Word boundaries alone were right about names and wrong about the
        # strings schedulers emit: the hint begins mid-token, preceded by a word
        # character. Case seams are split before matching.
        assert self._degraded(reason)

    def test_an_unschedulable_node_is_out_not_degraded(self):
        # The value is catching the node still being handed out while impaired.
        node = Node(name="n", state_raw="DOWN", cpus_total=8, memory_mb=1000,
                    reason="gpu fell off the bus",
                    conditions=frozenset({"DOWN"}))
        assert not node.schedulable
        assert not node.degraded
