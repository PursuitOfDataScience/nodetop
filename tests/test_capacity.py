"""Hardware gating and capacity arithmetic."""

from __future__ import annotations

from nodetop.core.capacity import (
    EXCLUDED_REASON,
    assess_capacity,
    hardware_ok,
    node_fits,
)
from nodetop.core.hardware import ACCELERATORS
from nodetop.core.model import JobShape, Node


def _node(name="n", model=None, gpus=4, gpus_alloc=0, cpus=32, cpus_alloc=0,
          mem=256 * 1024, mem_alloc=0, conditions=(), taints=()):
    return Node(
        name=name,
        state_raw="down" if conditions else "up",
        conditions=frozenset(conditions),
        cpus_total=cpus, cpus_alloc=cpus_alloc,
        memory_mb=mem, memory_alloc_mb=mem_alloc,
        gpus_total=gpus, gpus_alloc=gpus_alloc,
        accelerator=ACCELERATORS[model] if model else None,
        taints=tuple(taints),
    )


class TestHardwareGate:
    def test_bf16_excludes_a_v100(self):
        # No scheduler models this, so all of them will place the job here and
        # let it die at the first autocast.
        ok, why = hardware_ok(_node(model="V100"),
                              JobShape(gpus_per_node=1, requires=("bf16",)))
        assert ok is False
        assert any("lacks bf16" in w for w in why)

    def test_fp8_excludes_an_a100(self):
        ok, why = hardware_ok(_node(model="A100"),
                              JobShape(gpus_per_node=1, requires=("fp8",)))
        assert ok is False
        assert any("lacks fp8" in w for w in why)

    def test_fp8_accepts_an_h100_and_an_mi300x(self):
        for model in ("H100", "MI300X"):
            ok, _ = hardware_ok(_node(model=model),
                                JobShape(gpus_per_node=1, requires=("fp8",)))
            assert ok is True, model

    def test_memory_floor_says_it_is_inferred(self):
        ok, why = hardware_ok(_node(model="A100"),
                              JobShape(gpus_per_node=1, gpu_memory_gb=80))
        assert ok is False
        assert any("inferred from model" in w for w in why)

    def test_certain_memory_is_not_flagged_as_inferred(self):
        ok, why = hardware_ok(_node(model="A40"),
                              JobShape(gpus_per_node=1, gpu_memory_gb=80))
        assert ok is False
        assert not any("inferred" in w for w in why)

    def test_the_gate_ignores_occupancy_and_state(self):
        # "Wrong hardware" is durable; "busy" and "drained" are about today.
        # Conflating them turns an outage into a permanent-looking verdict.
        busy = _node(model="A100", gpus_alloc=4, cpus_alloc=32)
        drained = _node(model="A100", conditions=("DRAIN",))
        assert hardware_ok(busy, JobShape(gpus_per_node=4))[0] is True
        assert hardware_ok(drained, JobShape(gpus_per_node=4))[0] is True

    def test_more_accelerators_than_installed(self):
        ok, why = hardware_ok(_node(model="A100"), JobShape(gpus_per_node=8))
        assert ok is False
        assert any("only 4 accelerators installed" in w for w in why)

    def test_untolerated_taint_excludes(self):
        shape = JobShape(cpus_per_task=1)
        node = _node(taints=("dedicated=inference:NoSchedule",))
        assert hardware_ok(node, shape)[0] is False
        tolerant = JobShape(cpus_per_task=1,
                            tolerates=("dedicated=inference:NoSchedule",))
        assert hardware_ok(node, tolerant)[0] is True

    def test_cpu_job_on_an_accelerator_node_is_fine(self):
        assert hardware_ok(_node(model="A100"), JobShape(cpus_per_task=4))[0] is True

    def test_a_backend_that_never_mentioned_memory_is_not_wrong_hardware(self):
        # `memory_mb == 0` is "the backend does not report memory", not "this
        # machine has none" -- the rule `Node.memory_exhausted` states in so
        # many words ("inventing a shortage on a system that never mentioned
        # memory would be its own kind of lie"). Reachable through
        # `SshPoolBackend.parse_host`, where a host whose probe returns no
        # MEMTOTAL line lands here with 48 idle cores and memory_mb=0.
        #
        # Read as a size, it makes the durable verdict: an empty
        # `hardware_nodes` is `ever_possible is False`, which `where` renders
        # as WRONG HW -- "no node of the right kind ... go elsewhere; waiting
        # will not help" -- and exits 1.
        silent = _node(mem=0, gpus=0)
        shape = JobShape(nodes=1, cpus_per_task=4, memory_gb=8)
        ok, why = hardware_ok(silent, shape)
        assert ok is True, why
        assert not any("RAM installed" in w for w in why)
        cap = assess_capacity([silent], shape)
        assert cap.hardware_nodes == ("n",)
        assert cap.ever_possible is True

    def test_a_reported_memory_size_that_is_too_small_still_excludes(self):
        # CONTROL: the check itself has to keep working. A node that DID report
        # its memory and reported less than the job needs is genuinely the
        # wrong machine, and the suppression above must not reach it. Passes
        # both before and after that change.
        small = _node(mem=4096, gpus=0)
        shape = JobShape(nodes=1, cpus_per_task=4, memory_gb=8)
        ok, why = hardware_ok(small, shape)
        assert ok is False
        assert any("only 4 GiB RAM installed, need 8" in w for w in why)
        cap = assess_capacity([small], shape)
        assert cap.hardware_nodes == ()
        assert cap.ever_possible is False

    def test_the_unreported_size_is_still_kept_out_of_the_room_count(self):
        # CONTROL: suppressing the durable verdict must not invent capacity.
        # "Cannot tell how much RAM is here" is not "there is room here", so
        # the node stays out of `fitting_nodes` and the shape does not run now
        # -- it queues. Passes both before and after the change.
        silent = _node(mem=0, gpus=0)
        shape = JobShape(nodes=1, cpus_per_task=4, memory_gb=8)
        cap = assess_capacity([silent], shape)
        assert cap.fitting_nodes == ()
        assert cap.satisfies(shape) is False


class TestNodeFits:
    def test_idle_node_fits(self):
        assert node_fits(_node(model="A100"),
                         JobShape(gpus_per_node=4, cpus_per_task=8)).fits

    def test_busy_node_does_not(self):
        fit = node_fits(_node(model="A100", gpus_alloc=4), JobShape(gpus_per_node=1))
        assert fit.fits is False
        assert any("accelerators free" in r for r in fit.reasons)

    def test_drained_node_does_not(self):
        fit = node_fits(_node(model="A100", conditions=("DRAIN",)),
                        JobShape(gpus_per_node=1))
        assert fit.fits is False

    def test_explicit_exclusion(self):
        fit = node_fits(_node(name="n1", model="A100"),
                        JobShape(gpus_per_node=1, exclude=("n1",)))
        assert fit.fits is False
        assert "excluded" in fit.reasons

    def test_an_unknown_model_cannot_satisfy_a_stated_capability(self):
        # This assertion used to be the opposite, on the reasoning that
        # "unknown" is not "incapable". That is true, but it made the wrong
        # trade: counting it as satisfied let the tool recommend a node that
        # fails at run time. Unverifiable is now set aside and reported, which
        # fails toward less capacity -- see test_fail_safe.py.
        node = _node(model=None)
        assert node.accelerator is None
        assert not node_fits(node, JobShape(gpus_per_node=1, requires=("fp8",))).fits

    def test_an_unknown_model_still_fits_when_no_capability_is_asked_for(self):
        # With nothing depending on the model, an unidentified accelerator is
        # simply an accelerator.
        node = _node(model=None)
        assert node_fits(node, JobShape(gpus_per_node=1)).fits

    def test_reasons_are_deduplicated(self):
        # A node that is both too small and busy must not report the same
        # shortage twice.
        node = _node(model="A100", gpus=1, gpus_alloc=1)
        fit = node_fits(node, JobShape(gpus_per_node=4))
        assert len(fit.reasons) == len(set(fit.reasons))

    def test_a_node_with_no_allocatable_memory_fits_nothing(self):
        # The check used to be conditional on the shape naming a memory
        # figure, so a cores-only job was told a node with 28 of 32 cores idle
        # and all of its memory allocated would take it. The scheduler gives
        # every job memory -- the site default when the job names none -- so
        # it would not.
        node = _node(cpus_alloc=4, mem_alloc=256 * 1024, gpus=0)
        assert node.cpus_free == 28
        fit = node_fits(node, JobShape(cpus_per_task=4))
        assert fit.fits is False
        assert "no memory free" in fit.reasons

    def test_the_shortage_is_reported_once_not_twice(self):
        # Both memory branches must not fire on the same node.
        node = _node(mem_alloc=256 * 1024, gpus=0)
        fit = node_fits(node, JobShape(memory_gb=64))
        assert len(fit.reasons) == len(set(fit.reasons))
        assert sum("memory" in r or "RAM" in r for r in fit.reasons) == 1

    def test_a_node_that_reports_no_memory_at_all_still_fits(self):
        # "Cannot tell" is not "none" -- a backend that never mentions memory
        # must not have a shortage invented for it.
        node = _node(mem=0, gpus=0)
        assert node_fits(node, JobShape(cpus_per_task=4)).fits


class TestCapacityBuckets:
    def _mixed(self):
        return [
            _node(name="free", model="A100"),
            _node(name="busy", model="A100", gpus_alloc=4, cpus_alloc=32),
            _node(name="drained", model="H100", conditions=("DRAIN",)),
            _node(name="wrong", model="V100"),
        ]

    def test_three_buckets_are_distinguished(self):
        cap = assess_capacity(self._mixed(),
                              JobShape(gpus_per_node=4, requires=("bf16",)))
        assert cap.fitting_nodes == ("free",)                 # free right now
        assert set(cap.capable_nodes) == {"busy"}             # right hw, up, busy
        assert set(cap.hardware_nodes) == {"free", "busy", "drained"}
        assert "wrong" not in cap.hardware_nodes             # V100 has no bf16

    def test_capable_but_all_unavailable(self):
        # The right machines exist and every one is drained -- materially
        # different from "wrong cluster".
        cap = assess_capacity([_node(model="H100", conditions=("DRAIN",))],
                              JobShape(gpus_per_node=4, requires=("fp8",)))
        assert cap.ever_possible is True
        assert cap.capable_but_all_unavailable is True

    def test_wrong_hardware_everywhere(self):
        cap = assess_capacity([_node(model="V100")],
                              JobShape(gpus_per_node=1, requires=("fp8",)))
        assert cap.ever_possible is False

    def test_hardware_reasons_are_counted(self):
        nodes = [_node(name="a", model="V100"), _node(name="b", model="V100")]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=1, requires=("bf16",)))
        assert cap.hardware_reasons["V100 lacks bf16"] == 2

    def test_blocked_reasons_collapse_numeric_detail(self):
        # "3/4 free" and "1/4 free" are one category, not two.
        nodes = [_node(name="a", model="A100", gpus_alloc=1),
                 _node(name="b", model="A100", gpus_alloc=3)]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=4))
        assert any("N/N accelerators free" in k for k in cap.blocked_reasons)

    def test_satisfies_requires_enough_nodes(self):
        cap = assess_capacity([_node(model="A100")], JobShape(nodes=2, gpus_per_node=4))
        assert cap.nodes_available == 1
        assert cap.satisfies(JobShape(nodes=2, gpus_per_node=4)) is False

    def test_excluded_nodes_leave_every_bucket(self):
        cap = assess_capacity([_node(name="n1", model="A100")],
                              JobShape(gpus_per_node=4, exclude=("n1",)))
        assert cap.fitting_nodes == ()
        assert cap.hardware_nodes == ()


class TestEarliestFree:
    def test_the_kth_soonest_node_is_the_binding_one(self):
        from datetime import datetime

        nodes = [
            _node(name=f"n{i}", model="A100", gpus_alloc=4, cpus_alloc=32)
            for i in range(3)
        ]
        free = {
            "n0": datetime(2026, 8, 21, 10),
            "n1": datetime(2026, 8, 21, 12),
            "n2": datetime(2026, 8, 21, 14),
        }
        cap = assess_capacity(nodes, JobShape(nodes=2, gpus_per_node=4), free)
        # Two nodes are needed, so the second-soonest is when it can start.
        assert cap.earliest_free == datetime(2026, 8, 21, 12)

    def test_no_estimate_without_free_times(self):
        nodes = [_node(model="A100", gpus_alloc=4)]
        assert assess_capacity(nodes, JobShape(gpus_per_node=4)).earliest_free is None


class TestEverPossibleCountsNodes:
    """"Could this queue ever host the shape" needs the kind AND the count.

    Checking only the kind made a one-node queue asked for forty report
    possible, so `where` rendered it "would queue" -- inviting a wait for
    capacity the queue does not contain. That is the wrong-moment /
    wrong-place confusion this module exists to prevent.
    """

    @staticmethod
    def _nodes(n, gpus=4):
        return [
            Node(name=f"n{i}", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                 gpus_total=gpus, accelerator=ACCELERATORS["A100"])
            for i in range(n)
        ]

    def test_enough_suitable_nodes_is_possible(self):
        cap = assess_capacity(self._nodes(4), JobShape(nodes=4, gpus_per_node=1))
        assert cap.ever_possible is True
        assert cap.too_few_nodes is False

    def test_too_few_suitable_nodes_is_not_possible(self):
        cap = assess_capacity(self._nodes(1), JobShape(nodes=40, gpus_per_node=1))
        assert cap.ever_possible is False
        assert cap.too_few_nodes is True

    def test_exactly_enough_is_possible(self):
        cap = assess_capacity(self._nodes(40), JobShape(nodes=40, gpus_per_node=1))
        assert cap.ever_possible is True

    def test_no_suitable_node_is_wrong_kind_not_too_few(self):
        # The two must not collapse: one is answered by asking for fewer nodes,
        # the other only by going elsewhere.
        cap = assess_capacity(self._nodes(4, gpus=0), JobShape(nodes=1, gpus_per_node=1))
        assert cap.ever_possible is False
        assert cap.too_few_nodes is False

    def test_an_incomplete_node_list_suppresses_the_count_verdict(self):
        # Ruling a queue out because we failed to resolve its nodes would be a
        # worse error than leaving it in.
        cap = assess_capacity(
            self._nodes(1), JobShape(nodes=40, gpus_per_node=1),
            count_is_complete=False,
        )
        assert cap.required_nodes == 0
        assert cap.ever_possible is True
        assert cap.too_few_nodes is False

    def test_it_suppresses_the_hardware_verdict_too_when_nothing_resolved(self):
        """A queue whose nodes ALL failed to resolve is not "wrong hardware".

        The suppression above only covered the count.  With nothing resolved,
        `hardware_nodes` is empty for want of a node list rather than for want
        of the right node, and the short-circuit read that as a verdict: a
        partition declaring four nodes, none of them found, rendered as
        `WRONG HW` -- "go elsewhere; waiting will not help" -- immediately above
        its own caveat, "the queue claims 4 nodes but only 0 could be resolved".
        One report, asserting a fact about nodes and then admitting it had seen
        none of them.
        """
        cap = assess_capacity(
            [], JobShape(nodes=1, gpus_per_node=1), count_is_complete=False)
        assert cap.considered == 0
        assert cap.hardware_nodes == ()
        assert cap.ever_possible is True
        assert cap.too_few_nodes is False

    def test_considered_is_the_denominator(self):
        cap = assess_capacity(self._nodes(11), JobShape(nodes=1, gpus_per_node=1))
        assert cap.considered == 11

    def test_the_reason_histogram_is_not_a_node_count(self):
        # One node can contribute several reasons, so summing the histogram
        # overcounts -- which is why `considered` is recorded, not derived.
        nodes = self._nodes(1, gpus=0)
        cap = assess_capacity(nodes, JobShape(nodes=1, gpus_per_node=8,
                                              gpu_memory_gb=999))
        assert cap.considered == 1
        assert sum(cap.hardware_reasons.values()) >= 1


class TestTheEmptyCaseIsStillAVerdictWhereItIsEarned:
    """Controls on the suppression above: it must not swallow a real refusal.

    "We saw nothing" is not the same claim as "we saw nothing suitable", and
    only the first is exempt.  Both of these queues have been *looked at*, so
    an empty capable set is evidence rather than a lookup failure.
    """

    @staticmethod
    def _nodes(n, gpus=4):
        return [
            Node(name=f"n{i}", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                 gpus_total=gpus, accelerator=ACCELERATORS["A100"])
            for i in range(n)
        ]

    def test_a_queue_that_really_owns_no_nodes_is_still_impossible(self):
        # Complete list, and it is empty: there is no hardware here to wait for.
        cap = assess_capacity([], JobShape(nodes=1, gpus_per_node=1),
                              count_is_complete=True)
        assert cap.considered == 0
        assert cap.required_nodes == 1
        assert cap.ever_possible is False

    def test_a_partly_resolved_queue_of_the_wrong_kind_is_still_impossible(self):
        # Incomplete, but two nodes WERE examined and neither can host the
        # shape. That verdict rests on nodes, not on a failure to find any.
        cap = assess_capacity(
            self._nodes(2, gpus=0), JobShape(nodes=1, gpus_per_node=1),
            count_is_complete=False,
        )
        assert cap.considered == 2
        assert cap.required_nodes == 0
        assert cap.hardware_nodes == ()
        assert cap.ever_possible is False

    def test_a_complete_list_of_the_wrong_kind_is_still_impossible(self):
        cap = assess_capacity(self._nodes(4, gpus=0),
                              JobShape(nodes=1, gpus_per_node=1))
        assert cap.ever_possible is False


class TestAnExcludedNodeSaysSoInsteadOfSayingNothing:
    """`--exclude` was the one path to a refusal with an empty explanation.

    `Capacity.hardware_reasons` states its own contract: *"Hardware mismatch
    reason -> node count, so a 'wrong hardware' verdict can always say **what**
    was wrong rather than just refusing."* A node that is `hw_ok` but named in
    `shape.exclude` took the `else` branch with an empty `hw_why`, so the loop
    recorded nothing: exclude every node and `hardware_nodes` emptied,
    `ever_possible` went False, and the histogram stayed `{}`.

    The screen then read `✗ WRONG HW` / "no node here has the right hardware"
    about nodes whose hardware was fine, with the legend "go elsewhere; waiting
    will not help" -- when what would help is dropping `--exclude`.
    """

    @staticmethod
    def _nodes(n=3, cpus=16):
        return [
            Node(name=f"n{i}", state_raw="IDLE", cpus_total=cpus, memory_mb=64000,
                 gpus_total=4, queues=("q",))
            for i in range(n)
        ]

    def test_excluding_every_node_still_explains_itself(self):
        cap = assess_capacity(
            self._nodes(), JobShape(nodes=1, cpus_per_task=8,
                                    exclude=("n0", "n1", "n2")))
        assert cap.hardware_nodes == ()
        assert cap.hardware_reasons == {EXCLUDED_REASON: 3}

    def test_a_partial_exclusion_is_counted_without_changing_the_verdict(self):
        # One of three excluded: two remain, so the job is still possible. The
        # reason is recorded anyway -- it explains the 2/3, which is the number
        # the reader sees.
        cap = assess_capacity(
            self._nodes(), JobShape(nodes=1, cpus_per_task=8, exclude=("n0",)))
        assert len(cap.hardware_nodes) == 2
        assert cap.ever_possible is True
        assert cap.hardware_reasons == {EXCLUDED_REASON: 1}

    def test_a_genuine_hardware_mismatch_reads_exactly_as_before(self):
        """CONTROL -- passes in both states."""
        cap = assess_capacity(self._nodes(), JobShape(nodes=1, cpus_per_task=32))
        assert cap.hardware_nodes == ()
        assert cap.hardware_reasons == {"only 16 CPUs installed, need 32": 3}
        assert EXCLUDED_REASON not in cap.hardware_reasons

    def test_a_node_that_is_both_wrong_and_excluded_is_not_double_reported(self):
        """CONTROL -- and a deliberate scope line.

        A node that genuinely does not fit already explains itself, so the
        exclusion is not added on top; the histogram stays about the hardware.
        Recording both would be defensible (the field documents that its values
        do not sum to a node count) but it would crowd the one sentence the
        reader gets.
        """
        cap = assess_capacity(
            self._nodes(), JobShape(nodes=1, cpus_per_task=32,
                                    exclude=("n0", "n1", "n2")))
        assert cap.hardware_reasons == {"only 16 CPUs installed, need 32": 3}

    def test_no_exclusion_records_no_reason(self):
        """CONTROL -- the ordinary path must stay empty, not gain a 0 entry."""
        cap = assess_capacity(self._nodes(), JobShape(nodes=1, cpus_per_task=8))
        assert cap.hardware_reasons == {}
        assert cap.hardware_nodes == ("n0", "n1", "n2")
