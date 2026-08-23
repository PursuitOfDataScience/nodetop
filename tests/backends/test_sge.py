"""Grid Engine (SGE / UGE)."""

from __future__ import annotations

import pytest

from nodetop.core.model import JobShape, VerdictCategory
from nodetop.runner import RecordedRunner


def _nodes(sge_backend):
    return {n.name: n for n in sge_backend.load_nodes()}


class TestNodeColumns:
    def test_columns_are_located_by_name(self, sge_backend):
        # Grid Engine builds ship different column sets, so a fixed offset
        # reads the wrong field: header[1] is ARCH, not NCPU.
        n = _nodes(sge_backend)["node001"]
        assert n.cpus_total == 32
        assert n.memory_mb > 250_000

    def test_global_pseudo_host_is_skipped(self, sge_backend):
        assert "global" not in _nodes(sge_backend)

    def test_slot_occupancy_comes_from_the_middle_of_the_triple(self, sge_backend):
        # The queue-instance triple is resv/used/total, so reading the first
        # field gives reservations instead of usage.
        assert _nodes(sge_backend)["node001"].cpus_alloc == 2
        assert _nodes(sge_backend)["node002"].cpus_alloc == 30

    def test_cpu_only_host(self, sge_backend):
        n = _nodes(sge_backend)["node004"]
        assert n.cpus_total == 128
        assert n.is_gpu_node is False


class TestStateLetters:
    def test_no_state_column_means_healthy(self, sge_backend):
        assert _nodes(sge_backend)["node001"].schedulable is True

    def test_the_arch_string_is_not_mistaken_for_state_letters(self, sge_backend):
        # The states column is optional, so the field after the load average
        # is often ARCH -- and "lx-amd64" contains both "d" and "u", which
        # would read as disabled-and-unreachable on every healthy host.
        n = _nodes(sge_backend)["node001"]
        assert n.state_raw == "ok"
        assert n.schedulable is True
        assert "disabled" not in n.reason

    def test_au_is_alarm_and_unreachable(self, sge_backend):
        n = _nodes(sge_backend)["node003"]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_d_is_administratively_disabled(self, sge_backend):
        n = _nodes(sge_backend)["node004"]
        assert n.schedulable is False
        assert "disabled by an administrator" in n.reason


class TestConsumables:
    def test_model_name_survives_numeric_cleanup(self, sge_backend):
        # Grid Engine writes "4.000000"; trimming trailing zeros
        # unconditionally turns the model "A100" into "A1", which identifies
        # as nothing.
        assert _nodes(sge_backend)["node001"].accelerator.model == "A100"
        assert _nodes(sge_backend)["node002"].accelerator.model == "V100"

    def test_hc_is_available_not_used(self, sge_backend):
        # `hc:gpu` is the amount of the consumable still AVAILABLE; the
        # configured total is not in qhost output at all.
        assert _nodes(sge_backend)["node001"].gpus_free == 4

    def test_zero_available_understates_rather_than_inventing_capacity(
        self, sge_backend
    ):
        # node002 reports hc:gpu=0. With no visible total, reporting 0/0 is the
        # safe direction: claiming 4 installed would invent room that may be
        # in use.
        n = _nodes(sge_backend)["node002"]
        assert n.gpus_free == 0


class TestQueues:
    def _queues(self, sge_backend):
        return {q.name: q for q in sge_backend.load_queues()}

    def test_queue_config_is_read(self, sge_backend):
        q = self._queues(sge_backend)["all.q"]
        assert q.max_walltime_seconds == 48 * 3600
        assert q.allow_users == ("gpu_users",)

    def test_none_userlist_is_not_a_restriction(self, sge_backend):
        assert self._queues(sge_backend)["cpu.q"].allow_users == ()

    def test_membership_from_queue_instances(self, sge_backend):
        assert set(self._queues(sge_backend)["all.q"].node_names) == {
            "node001", "node002", "node003"
        }


class TestProbe:
    """Grid Engine has a genuine verify-only mode."""

    @pytest.fixture(autouse=True)
    def _qsub_exists(self, monkeypatch):
        # The probe honours its own capability declaration, which is gated on
        # qsub being present. Force it so the test exercises the parsing rather
        # than the machine it happens to run on.
        monkeypatch.setattr("nodetop.backends.sge.which", lambda _: True)

    def _backend(self, rc, out, err=""):
        from nodetop.backends.sge import SgeBackend

        return SgeBackend(RecordedRunner({"qsub": (rc, out, err)}))

    def test_verify_flag_is_hard_coded(self):
        from nodetop.backends.sge import SgeBackend

        runner = RecordedRunner({"qsub": (0, "verification: found suitable queue(s)", "")})
        SgeBackend(runner).probe("all.q", JobShape())
        cmd = runner.calls[0]
        # -w v verifies and queues nothing; a caller must not be able to omit it.
        assert cmd[:3] == ["qsub", "-w", "v"]

    def test_suitable_queue_is_allowed(self):
        v = self._backend(0, "verification: found suitable queue(s)").probe(
            "all.q", JobShape()
        )
        assert v.allowed is True
        assert v.category == VerdictCategory.OK

    def test_no_suitable_queue_is_a_shape_problem(self):
        v = self._backend(1, "", "verification: no suitable queues").probe(
            "all.q", JobShape()
        )
        assert v.allowed is False
        assert v.category == VerdictCategory.SHAPE_UNAVAILABLE
        # A shape that does not fit says nothing durable about access.
        assert v.durable is False

    def test_permission_refusal_is_durable(self):
        v = self._backend(1, "", "error: no permission to submit").probe(
            "all.q", JobShape()
        )
        assert v.category == VerdictCategory.NOT_ENTITLED
        assert v.durable is True

    def test_unreachable_qmaster(self):
        v = self._backend(1, "", "error: unable to contact qmaster").probe(
            "all.q", JobShape()
        )
        assert v.category == VerdictCategory.CONTROL_PLANE_DOWN
        assert v.durable is False


class TestSubmitFlags:
    def test_resource_list(self, sge_backend):
        flags = " ".join(
            sge_backend.submit_flags("all.q", JobShape(cpus_per_task=8, memory_gb=64,
                                                       gpus_per_node=2, walltime="4h"))
        )
        assert "-q all.q" in flags
        assert "h_rt=04:00:00" in flags
        assert "gpu=2" in flags
        assert "h_vmem=64G" in flags


RQS = """{
   name         max_gpus
   description  we want to limit gpus to four per user
   enabled      TRUE
   limit        users {*} to gpu=4,slots=64
}
"""


class TestResourceQuotaSets:
    def _limits(self, text):
        from nodetop.backends.sge import SgeBackend

        return SgeBackend(RecordedRunner({
            "qconf -srqsl": (0, "max_gpus\n", ""),
            "qconf -srqs": (0, text, ""),
        })).load_limits()

    def test_quota_is_read(self):
        assert self._limits(RQS)["max_gpus"].per_user == {"gpu": 4, "cpu": 64}

    def test_a_description_containing_to_is_not_parsed_as_a_rule(self):
        # "we want to limit gpus to four per user" is prose. Matching a bare
        # " to " would turn it into a quota.
        assert self._limits(RQS)["max_gpus"].per_user == {"gpu": 4, "cpu": 64}

    def test_no_quota_sets_configured(self):
        from nodetop.backends.sge import SgeBackend

        backend = SgeBackend(RecordedRunner({"qconf -srqsl": (1, "", "none")}))
        assert backend.load_limits() == {}

    def test_a_quota_naming_no_known_resource_is_dropped(self):
        text = "{\n   name x\n   limit users {*} to hypothetical=4\n}\n"
        assert self._limits(text) == {}


SE_WITH_GPU = """hostname              node001
load_scaling          NONE
complex_values        gpu=4,slots=32
user_lists            NONE
"""


def _sge(qconf_se: tuple[int, str, str]):
    from conftest import read

    from nodetop.backends.sge import SgeBackend
    from nodetop.core.cluster import Cluster

    return Cluster.load(
        SgeBackend(RecordedRunner({
            "qhost": (0, read("sge", "qhost.txt"), ""),
            "qstat -f": (0, read("sge", "qstat_f.txt"), ""),
            "qconf -sql": (0, "all.q\ncpu.q\n", ""),
            "qconf -sq all.q": (0, read("sge", "qconf_sq_allq.txt"), ""),
            "qconf -sq cpu.q": (0, read("sge", "qconf_sq_cpuq.txt"), ""),
            "qconf -srqsl": (1, "", "none"),
            "qconf -sul": (0, "", ""),
            "qconf -se": qconf_se,
        })),
        with_free_times=False,
    )


class TestComplexValues:
    def test_parsing(self, sge_backend):
        assert sge_backend.parse_complex_values(SE_WITH_GPU) == {"gpu": 4, "slots": 32}

    def test_none_is_not_a_value(self, sge_backend):
        assert sge_backend.parse_complex_values("complex_values  NONE\n") == {}

    def test_non_numeric_entries_are_skipped(self, sge_backend):
        got = sge_backend.parse_complex_values("complex_values  gpu=4,arch=lx-amd64\n")
        assert got == {"gpu": 4}

    def test_a_record_without_complex_values(self, sge_backend):
        assert sge_backend.parse_complex_values("hostname  n1\n") == {}


class TestAcceleratorTotals:
    """hc: reports what is free; the configured total is elsewhere."""

    def test_a_fully_busy_host_keeps_its_accelerators(self):
        # node002 reports hc:gpu=0 -- every V100 in use. Reading that as the
        # total makes it look like a CPU-only machine and removes four V100s
        # from the inventory.
        cluster = _sge((0, SE_WITH_GPU, ""))
        node = {n.name: n for n in cluster.nodes}["node002"]
        assert node.is_gpu_node is True
        assert (node.gpus_total, node.gpus_free) == (4, 0)

    def test_occupancy_is_total_minus_available(self):
        cluster = _sge((0, SE_WITH_GPU, ""))
        node = {n.name: n for n in cluster.nodes}["node002"]
        assert node.gpus_alloc == 4

    def test_an_idle_host_is_unchanged(self):
        cluster = _sge((0, SE_WITH_GPU, ""))
        node = {n.name: n for n in cluster.nodes}["node001"]
        assert (node.gpus_total, node.gpus_free) == (4, 4)

    def test_the_inventory_counts_busy_hardware(self):
        assert _sge((0, SE_WITH_GPU, "")).summary()["accelerators_total"] == 12

    def test_a_cpu_host_is_never_queried_into_being_a_gpu_host(self):
        cluster = _sge((0, SE_WITH_GPU, ""))
        node = {n.name: n for n in cluster.nodes}["node004"]
        assert node.is_gpu_node is False


class TestTotalsUnavailable:
    """Falling back is fine; falling back silently is not."""

    def test_it_stays_conservative(self):
        # Understating never invents room that is not there.
        assert _sge((1, "", "nope")).summary()["accelerators_total"] == 8

    def test_it_says_the_count_is_unknown(self):
        cluster = _sge((1, "", "nope"))
        affected = [n for n in cluster.nodes if n.accelerator is not None]
        assert affected
        assert all("count unknown" in n.reason for n in affected)

    def test_a_host_whose_record_has_no_gpu_entry_is_reported_too(self):
        cluster = _sge((0, "hostname n1\ncomplex_values slots=32\n", ""))
        affected = [n for n in cluster.nodes if n.accelerator is not None]
        assert all("count unknown" in n.reason for n in affected)


class TestQueryCaching:
    def test_nodes_are_derived_once(self):
        from conftest import read

        from nodetop.backends.sge import SgeBackend

        runner = RecordedRunner({
            "qhost": (0, read("sge", "qhost.txt"), ""),
            "qstat -f": (0, read("sge", "qstat_f.txt"), ""),
            "qconf -sql": (0, "all.q\n", ""),
            "qconf -sq": (0, read("sge", "qconf_sq_allq.txt"), ""),
            "qconf -se": (0, SE_WITH_GPU, ""),
        })
        backend = SgeBackend(runner)
        backend.load_nodes()
        backend.load_queues()   # needs the nodes too
        # One qhost sweep, not two: re-deriving would double the per-host
        # qconf round trips, which is one call per accelerator host.
        assert sum(1 for c in runner.calls if c[0] == "qhost") == 1


class TestProbeHonoursItsDeclaration:
    def test_no_qsub_means_no_verdict(self, monkeypatch):
        # Answering anyway would make the object contradict itself: a caller
        # trusting capabilities() would get a verdict the backend said it could
        # not produce, and a missing client would be reported as a sick cluster.
        from nodetop.backends.sge import SgeBackend

        monkeypatch.setattr("nodetop.backends.sge.which", lambda _: False)
        backend = SgeBackend(RecordedRunner({"qsub": (0, "found suitable queue", "")}))
        assert backend.capabilities().probe is False
        assert backend.probe("all.q", JobShape()) is None


class TestUsersetFailureIsNeitherEmptyNorPartial:
    """Grid Engine usersets are the entitlement mechanism, so both ends bite.

    The sweep used to sit inside a bare `except Exception: pass`, which fails in
    two opposite directions depending on when it broke:

    * a PARTIAL list looks authoritative, so the tri-state membership check
      returns a verdict from it -- "you are in none of the permitted usersets" --
      on the strength of a scan that did not finish. A false denial, which hides
      a queue the caller can actually submit to.
    * an EMPTY list reads as "cannot tell", so every userset restriction is
      silently ignored and queues are claimed that will refuse the job.
    """

    def test_a_failure_listing_usersets_raises(self):
        from nodetop.backends.sge import SgeBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({"qconf -sul": (1, "", "error: no usersets")})
        with pytest.raises(CommandError):
            SgeBackend(runner).load_identity()

    def test_a_failure_partway_through_raises_rather_than_returning_half(self):
        # `good` matched, `bad` blows up. Returning just {"good"} would have the
        # tool assert a verdict from an incomplete scan.
        from nodetop.backends.sge import SgeBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({
            "qconf -sul": (0, "good\nbad\n", ""),
            "qconf -su good": (0, "name good\nentries testuser\n", ""),
            "qconf -su bad": (1, "", "error: cannot read"),
        })
        with pytest.raises(CommandError):
            SgeBackend(runner).load_identity()

    def test_a_complete_sweep_still_returns_the_usersets(self, monkeypatch):
        from nodetop.backends.sge import SgeBackend

        monkeypatch.setenv("USER", "testuser")
        runner = RecordedRunner({
            "qconf -sul": (0, "mine\ntheirs\n", ""),
            "qconf -su mine": (0, "name mine\nentries testuser\n", ""),
            "qconf -su theirs": (0, "name theirs\nentries someoneelse\n", ""),
        })
        ident = SgeBackend(runner).load_identity()
        assert ident.groups == ("mine",)


class TestPartialQuotaSetsAreRefused:
    """Skipping one RQS yields a ceiling map applied as though complete.

    A job over a ceiling nobody could read is then reported as fitting -- the
    "admitted, then pends forever" lie, produced by the tool itself. The outer
    listing failure stays tolerated on purpose: `qconf -srqsl` exits non-zero on
    a cluster with no RQS defined at all, which is a real state rather than a
    failed query.
    """

    def test_a_failure_reading_one_set_raises(self):
        from nodetop.backends.sge import SgeBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({
            "qconf -srqsl": (0, "good\nbad\n", ""),
            "qconf -srqs good": (0, "{\n name good\n limit users * to slots=4\n}\n", ""),
            "qconf -srqs bad": (1, "", "error: cannot read"),
        })
        with pytest.raises(CommandError):
            SgeBackend(runner).load_limits()

    def test_no_quota_sets_configured_is_not_an_error(self):
        from nodetop.backends.sge import SgeBackend

        runner = RecordedRunner({
            "qconf -srqsl": (1, "", "no resource quota set list defined"),
        })
        assert SgeBackend(runner).load_limits() == {}
