"""Hardware gating and capacity arithmetic."""

from __future__ import annotations

from nodetop.core.capacity import assess_capacity, hardware_ok, node_fits
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
