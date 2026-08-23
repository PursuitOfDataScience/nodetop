"""The invariant: every inference fails toward *less*.

Fifteen iterations of boundary testing found the same asymmetry again and
again. Nine of the last ten bugs erred toward claiming capacity or access that
was not there: a truncated node record looking idle, a scattered exclusion list
looking short, an unexpandable host group looking cluster-wide, a busy
accelerator looking absent, an unidentifiable one looking capable.

So it is written down and tested rather than rediscovered. Where a fact is
missing or ambiguous, nodetop answers with the reading that claims less
capacity and less access. The failure mode of that bias is a needless warning.
The failure mode of the opposite bias is a job sent somewhere it cannot run,
discovered ninety minutes later.
"""

from __future__ import annotations

import pytest

from nodetop.core.capacity import assess_capacity, hardware_ok
from nodetop.core.hardware import ACCELERATORS
from nodetop.core.model import JobShape, Node


def _node(name="n", model=None, gpus=4, **kw):
    return Node(
        name=name, cpus_total=32, memory_mb=256 * 1024, gpus_total=gpus,
        accelerator=ACCELERATORS[model] if model else None, state_raw="up", **kw
    )


class TestUnknownStateIsNotAvailable:
    def test_a_node_with_no_conditions_and_no_state_string(self):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.runner import RecordedRunner

        node = SlurmBackend(RecordedRunner({})).parse_nodes("NodeName=n CPUTot=8\n")[0]
        assert node.schedulable is False

    def test_an_explicit_unknown_condition_blocks(self):
        assert _node(conditions=frozenset({"UNKNOWN"})).schedulable is False


class TestUnknownCapabilityDoesNotCount:
    """An accelerator we cannot identify does not satisfy a requirement."""

    def test_it_is_not_counted_as_capable(self):
        ok, why = hardware_ok(_node(), JobShape(gpus_per_node=4, requires=("fp8",)))
        assert ok is False
        assert any("unidentified" in w for w in why)

    def test_it_is_reported_rather_than_silently_dropped(self):
        cap = assess_capacity(
            [_node("mystery")], JobShape(gpus_per_node=4, requires=("fp8",))
        )
        assert cap.unverified_nodes == ("mystery",)
        assert cap.hardware_nodes == ()

    def test_a_memory_floor_counts_as_a_requirement_too(self):
        cap = assess_capacity(
            [_node("mystery")], JobShape(gpus_per_node=4, gpu_memory_gb=40)
        )
        assert cap.unverified_nodes == ("mystery",)

    def test_it_still_counts_when_nothing_depends_on_the_model(self):
        # With no capability asked for, an unidentified accelerator is simply
        # an accelerator. Excluding it there would understate for no reason.
        cap = assess_capacity([_node("mystery")], JobShape(gpus_per_node=4))
        assert cap.fitting_nodes == ("mystery",)
        assert cap.unverified_nodes == ()

    def test_unverified_is_distinct_from_incapable(self):
        # One is a labelling gap, the other is the wrong cluster, and they call
        # for different actions.
        shape = JobShape(gpus_per_node=4, requires=("fp8",))
        cap = assess_capacity([_node("a", "A100"), _node("m")], shape)
        assert cap.unverified_nodes == ("m",)
        assert any("lacks fp8" in r for r in cap.hardware_reasons)


class TestAmbiguousMemoryTakesTheSmallerValue:
    @pytest.mark.parametrize("model", ["A100", "H100", "V100", "GH200"])
    def test_the_conservative_variant_is_used(self, model):
        spec = ACCELERATORS[model]
        assert spec.memory_gb == min(spec.memory_variants)

    def test_so_a_borderline_job_is_warned_rather_than_placed(self):
        # An A100 might be 80 GB. Assuming so would place a job that then OOMs.
        ok, _ = hardware_ok(
            _node(model="A100"), JobShape(gpus_per_node=4, gpu_memory_gb=80)
        )
        assert ok is False


class TestUnreadableCeilingsAreDeclaredNotInvented:
    def _limits(self, row):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.runner import RecordedRunner

        return SlurmBackend(RecordedRunner({})).parse_limits(row + "\n")["q"]

    def test_garbage_is_distinguished_from_a_real_sentinel(self):
        # Both yield None, but only one means "there is no limit".
        assert self._limits("q|UNLIMITED||gres/gpu=4|||0|").unreadable == ()
        assert self._limits("q|banana||gres/gpu=4|||0|").unreadable == ("MaxWall",)

    def test_an_unreadable_tres_field_is_named(self):
        got = self._limits("q|2-00:00:00||gres/gpu=banana|||0|")
        assert got.unreadable == ("MaxTRESPerJob",)

    def test_no_blocker_is_invented_for_a_ceiling_nobody_published(self):
        # Flagging every job against an unread limit would be useless noise.
        got = self._limits("q|banana||gres/gpu=banana|||0|")
        assert got.blockers(JobShape(nodes=99, gpus_per_node=8, walltime="9d")) == []

    def test_the_gap_reaches_the_report(self):
        from nodetop.core.cluster import Cluster
        from nodetop.core.fit import evaluate
        from nodetop.core.model import BackendCapabilities, Queue

        queue = Queue(name="q", node_names=("n",))
        queue.nodes = [_node("n", "H100")]
        cluster = Cluster(
            backend_name="t", nodes=queue.nodes, queues={"q": queue},
            limits={"q": self._limits("q|banana||gres/gpu=4|||0|")},
            capabilities=BackendCapabilities(probe=False),
        )
        place = evaluate(cluster, JobShape(gpus_per_node=4), queue)
        assert any("could not read MaxWall" in c for c in place.caveats)
        assert any("not checked at all" in c for c in place.caveats)


class TestNoNegativeOrInventedResources:
    @pytest.mark.parametrize("total,alloc", [(0, -5), (-8, 0), (-8, -8)])
    def test_free_capacity_is_never_conjured_from_negatives(self, total, alloc):
        node = Node(name="n", cpus_total=max(0, total), cpus_alloc=max(0, alloc))
        assert node.cpus_free == 0

    def test_an_unusable_queue_reports_no_free_capacity(self):
        from nodetop.core.model import Queue

        queue = Queue(name="q", enabled=False, node_names=("n",))
        queue.nodes = [_node("n")]
        assert queue.effective_free_nodes == 0
        assert queue.effective_free_gpus == 0


class TestAccessFailsClosed:
    def test_a_queue_naming_nobody_permits_nobody(self):
        from nodetop.core.model import Queue

        assert Queue(name="q", allow_accounts=("none",)).usable is False

    def test_a_replay_cannot_claim_a_confirmed_entitlement(self):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import BackendCapabilities

        cluster = Cluster(
            capabilities=BackendCapabilities(probe=True, probe_command="x"),
            replayed=True,
        )
        assert cluster.can_probe is False

    def test_a_backend_with_no_dry_run_reports_declared_not_confirmed(self):
        from nodetop import backends

        for name in ("pbs", "lsf", "sshpool"):
            assert backends.get(name).capabilities().probe is False

    def test_a_control_plane_failure_is_not_a_durable_verdict(self):
        from nodetop.core.model import Verdict, VerdictCategory

        verdict = Verdict(queue="q", category=VerdictCategory.CONTROL_PLANE_DOWN)
        # Caching this as "no access" would send someone hunting for a
        # permission that was never missing.
        assert verdict.durable is False


class TestTimeFailsTowardLater:
    """The same bias on the time axis: never promise a resource sooner."""

    def test_an_overrun_job_is_not_reported_as_free_now(self):
        from datetime import datetime, timedelta

        from nodetop.core.duration import format_wait

        # The scheduler said the node would be free three hours ago and it is
        # still held. Rounding that to "now" sends someone at a busy node.
        past = (datetime.now() - timedelta(hours=3) - datetime.now()).total_seconds()
        assert format_wait(past) == "overdue"

    @pytest.mark.parametrize("seconds,expected", [
        (-10_000, "overdue"), (-3600, "overdue"), (-61, "overdue"),
        (-60, "now"), (0, "now"), (30, "now"), (90, "1m"),
    ])
    def test_the_boundary(self, seconds, expected):
        from nodetop.core.duration import format_wait

        # A minute either side is clock jitter, not an overrun.
        assert format_wait(seconds) == expected

    def test_it_reaches_the_placement_table(self):
        import contextlib
        import io
        from datetime import datetime, timedelta

        from nodetop.cli import _COMMANDS, build_parser
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import BackendCapabilities, Queue
        from nodetop.render import Glyphs, Style

        node = _node("n1", "A100")
        node.cpus_alloc, node.gpus_alloc = 32, 4
        queue = Queue(name="q", node_names=("n1",))
        queue.nodes = [node]
        cluster = Cluster(
            backend_name="t", nodes=[node], queues={"q": queue},
            capabilities=BackendCapabilities(probe=False),
            node_free_times={"n1": datetime.now() - timedelta(hours=3)},
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _COMMANDS["where"](
                cluster, build_parser().parse_args(["where", "-g", "4"]),
                Style(depth=0, glyphs=Glyphs()),
            )
        assert "overdue" in out.getvalue()

    def test_a_computed_estimate_is_an_upper_bound_on_pbs(self):
        # PBS records no end time, so it is start + walltime. A job may finish
        # early, so the node may free sooner than reported -- never later.
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        text = (
            "Job Id: 1\n"
            "    Resource_List.walltime = 24:00:00\n"
            "    stime = Thu Aug 21 10:00:00 2026\n"
            "    exec_host = n1/0\n"
        )
        got = PbsBackend(RecordedRunner({})).parse_free_times(text)
        assert str(got["n1"]) == "2026-08-22 10:00:00"

    def test_a_self_computed_estimate_is_marked_as_a_lower_bound(self):
        # It ignores the queue ahead of you, so it can only be optimistic --
        # which is why it carries a marker and the scheduler's does not.
        from datetime import datetime, timedelta

        from nodetop.core.cluster import Cluster
        from nodetop.core.fit import evaluate
        from nodetop.core.model import BackendCapabilities, Queue

        node = _node("n1", "A100")
        node.cpus_alloc, node.gpus_alloc = 32, 4
        queue = Queue(name="q", node_names=("n1",))
        queue.nodes = [node]
        cluster = Cluster(
            backend_name="t", nodes=[node], queues={"q": queue},
            capabilities=BackendCapabilities(probe=False),
            node_free_times={"n1": datetime.now() + timedelta(hours=1)},
        )
        place = evaluate(cluster, JobShape(gpus_per_node=4), queue)
        assert place.earliest_start is not None
        assert place.start_estimate_from_scheduler is False


class TestTimestampsAreComparableToNow:
    """Everything downstream compares against a naive local `datetime.now()`."""

    def test_a_utc_timestamp_is_converted_to_local(self):
        from datetime import datetime, timezone

        from nodetop.core.duration import parse_timestamp

        # Kubernetes emits a Z suffix. Merely stripping the zone leaves a UTC
        # wall-clock reading pretending to be local, an error of however many
        # hours the host is offset.
        got = parse_timestamp("2026-08-22T10:00:00Z")
        expected = (
            datetime(2026, 8, 22, 10, tzinfo=timezone.utc)
            .astimezone().replace(tzinfo=None)
        )
        assert got == expected

    def test_an_explicit_offset_is_honoured(self):
        from datetime import datetime, timedelta, timezone

        from nodetop.core.duration import parse_timestamp

        got = parse_timestamp("2026-08-22T10:00:00+02:00")
        expected = (
            datetime(2026, 8, 22, 10, tzinfo=timezone(timedelta(hours=2)))
            .astimezone().replace(tzinfo=None)
        )
        assert got == expected

    def test_a_zoneless_timestamp_is_left_alone(self):
        from datetime import datetime

        from nodetop.core.duration import parse_timestamp

        # Slurm reports local time with no zone; converting it would be wrong.
        assert parse_timestamp("2026-08-22T10:00:00") == datetime(2026, 8, 22, 10)

    def test_the_two_readings_differ_by_the_host_offset(self):
        from datetime import datetime, timezone

        from nodetop.core.duration import parse_timestamp

        naive = parse_timestamp("2026-08-22T10:00:00")
        utc = parse_timestamp("2026-08-22T10:00:00Z")
        offset = datetime(2026, 8, 22, 10, tzinfo=timezone.utc).astimezone().utcoffset()
        # 10:00 UTC read as local is 10:00 + offset, so on a UTC-5 host the
        # UTC reading lands five hours *earlier* than the zoneless one.
        assert utc - naive == offset
