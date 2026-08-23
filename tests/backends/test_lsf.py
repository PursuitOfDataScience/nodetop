"""IBM Spectrum LSF."""

from __future__ import annotations

import pytest

from nodetop.backends.lsf import LsfBackend
from nodetop.core.model import JobShape
from nodetop.runner import RecordedRunner


def _nodes(lsf_backend):
    return {n.name: n for n in lsf_backend.load_nodes()}


def _queues(lsf_backend):
    return {q.name: q for q in lsf_backend.load_queues()}


class TestNodes:
    def test_all_hosts(self, lsf_backend):
        assert len(lsf_backend.load_nodes()) == 6

    def test_hardware_comes_from_lshosts(self, lsf_backend):
        n = _nodes(lsf_backend)["gpu-01"]
        assert n.cpus_total == 40
        assert n.memory_mb == 512 * 1024

    def test_accelerator_model_from_the_gpu_table(self, lsf_backend):
        assert _nodes(lsf_backend)["gpu-01"].accelerator.model == "A100"
        assert _nodes(lsf_backend)["gpu-02"].accelerator.model == "V100"

    def test_gpu_occupancy(self, lsf_backend):
        # gpu-02's two cards each hold 30G, so neither is free.
        n = _nodes(lsf_backend)["gpu-02"]
        assert (n.gpus_total, n.gpus_free) == (2, 0)

    def test_closed_adm_is_unschedulable(self, lsf_backend):
        assert _nodes(lsf_backend)["gpu-03"].schedulable is False

    def test_unavail_is_unreachable(self, lsf_backend):
        n = _nodes(lsf_backend)["gpu-04"]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_closed_full_is_busy_not_broken(self, lsf_backend):
        # A host closed because it is full is still perfectly schedulable; it
        # simply has nothing free. Treating it as down would hide real capacity
        # that appears the moment a job ends.
        n = _nodes(lsf_backend)["cpu-01"]
        assert n.schedulable is True
        assert n.cpus_free == 0


class TestQueueStatus:
    """LSF states the accept/start distinction more plainly than anyone."""

    def test_open_active(self, lsf_backend):
        q = _queues(lsf_backend)["gpu"]
        assert (q.enabled, q.started, q.usable) == (True, True, True)

    def test_open_inact_accepts_everything_and_runs_nothing(self, lsf_backend):
        q = _queues(lsf_backend)["frozen"]
        assert (q.enabled, q.started) == (True, False)
        assert q.usable is False
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}

    def test_closed_active(self, lsf_backend):
        q = _queues(lsf_backend)["shut"]
        assert (q.enabled, q.started) == (False, True)
        assert q.usable is False
        assert "QUEUE_DISABLED" in {b.code for b in q.structural_blockers()}

    def test_status_is_read_from_the_positional_table(self, lsf_backend):
        # The value is not labelled "STATUS:" -- it sits under a column
        # header. Searching for a label finds nothing and silently yields a
        # healthy-looking queue.
        assert _queues(lsf_backend)["frozen"].state_raw == "Open:Inact"


class TestQueueEntitlement:
    def test_named_users(self, lsf_backend):
        assert set(_queues(lsf_backend)["gpu"].allow_users) == {"alice", "bob", "carol"}

    def test_all_users_is_not_a_restriction(self, lsf_backend):
        assert _queues(lsf_backend)["normal"].allow_users == ()

    def test_host_list_limits_membership(self, lsf_backend):
        assert set(_queues(lsf_backend)["gpu"].node_names) == {
            "gpu-01", "gpu-02", "gpu-03", "gpu-04"
        }

    def test_hosts_all_means_every_host(self, lsf_backend):
        assert len(_queues(lsf_backend)["normal"].node_names) == 6


class TestLimits:
    def test_runlimit_is_minutes(self, lsf_backend):
        # LSF RUNLIMIT is expressed in minutes: 2880 -> 48 hours.
        assert _queues(lsf_backend)["gpu"].max_walltime_seconds == 2880 * 60

    def test_max_jobs_per_user(self, lsf_backend):
        assert lsf_backend.load_limits()["gpu"].max_jobs == 12


class TestNoProbe:
    def test_probe_returns_none(self, lsf_backend):
        assert lsf_backend.probe("gpu", JobShape()) is None

    def test_capabilities_explain_the_gap(self, lsf_backend):
        caps = lsf_backend.capabilities()
        assert caps.probe is False
        assert any("Open:Inact" in n for n in caps.notes)

    def test_no_free_time_estimate_is_offered(self, lsf_backend):
        # Getting per-job remaining time needs one call per job; returning
        # nothing is more honest than a bad estimate.
        assert lsf_backend.load_node_free_times() == {}


class TestSubmitFlags:
    def test_flags(self, lsf_backend):
        flags = " ".join(
            lsf_backend.submit_flags("gpu", JobShape(nodes=2, cpus_per_task=8,
                                                     gpus_per_node=4, memory_gb=64,
                                                     walltime="4h"))
        )
        assert "-q gpu" in flags
        assert "-n 16" in flags
        assert "ptile=8" in flags
        assert "num=4" in flags
        assert "-W 240" in flags


BMGROUP = """GROUP_NAME   HOSTS
gpu_hosts    gpu-01 gpu-02+1 gpu-03/ gpu-04
cpu_hosts    cpu-01 cpu-02
"""


def _lsf(bqueues: str, bmgroup: tuple[int, str, str]):
    from conftest import read

    from nodetop.backends.lsf import LsfBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    return Cluster.load(
        LsfBackend(RecordedRunner({
            "bhosts -gpu": (0, read("lsf", "bhosts_gpu.txt"), ""),
            "bhosts -w": (0, read("lsf", "bhosts.txt"), ""),
            "lshosts": (0, read("lsf", "lshosts.txt"), ""),
            "bqueues": (0, bqueues, ""),
            "bmgroup": bmgroup,
        })),
        with_free_times=False,
    )


def _grouped_bqueues() -> str:
    from conftest import read

    return read("lsf", "bqueues_l.txt").replace(
        "HOSTS:  gpu-01 gpu-02 gpu-03 gpu-04", "HOSTS:  gpu_hosts/"
    )


class TestHostGroups:
    def test_group_members_are_parsed(self, lsf_backend):
        got = lsf_backend.parse_host_groups(BMGROUP)
        assert got["gpu_hosts"] == ["gpu-01", "gpu-02", "gpu-03", "gpu-04"]

    def test_a_slice_suffix_is_stripped_but_the_hostname_survives(self, lsf_backend):
        # "gpu-02+1" -> "gpu-02". A blanket strip of digits would give "gpu-",
        # which matches no host and silently empties the queue.
        got = lsf_backend.parse_host_groups(BMGROUP)
        assert "gpu-02" in got["gpu_hosts"]
        assert "gpu-" not in got["gpu_hosts"]

    def test_the_header_row_is_skipped(self, lsf_backend):
        assert "GROUP_NAME" not in lsf_backend.parse_host_groups(BMGROUP)

    def test_a_group_scoped_queue_expands(self):
        q = _lsf(_grouped_bqueues(), (0, BMGROUP, "")).queues["gpu"]
        assert len(q.node_names) == 4
        assert q.unresolved_nodes == 0


class TestUnresolvableHosts:
    """An unexpandable group is reported, never guessed at."""

    def test_capacity_is_not_invented(self):
        # The previous behaviour fell back to every host in the cluster, handing
        # a queue restricted to four machines the free capacity of all six --
        # manufacturing exactly the phantom capacity this tool exists to catch.
        q = _lsf(_grouped_bqueues(), (1, "", "bmgroup: not found")).queues["gpu"]
        assert q.node_names == ()
        assert q.effective_free_nodes == 0
        assert q.effective_free_gpus == 0

    def test_the_gap_is_declared(self):
        q = _lsf(_grouped_bqueues(), (1, "", "bmgroup: not found")).queues["gpu"]
        # Surfaces in the report as "+N claimed but unresolved".
        assert q.declared_nodes == 1
        assert q.unresolved_nodes == 1

    def test_an_explicit_host_list_still_resolves_without_bmgroup(self):
        from conftest import read

        q = _lsf(read("lsf", "bqueues_l.txt"), (1, "", "nope")).queues["gpu"]
        assert len(q.node_names) == 4
        assert q.unresolved_nodes == 0

    def test_hosts_all_still_means_every_host(self):
        q = _lsf(_grouped_bqueues(), (1, "", "nope")).queues["normal"]
        assert len(q.node_names) == 6
        assert q.unresolved_nodes == 0


class TestResolveHosts:
    def test_a_mix_of_hostnames_and_groups(self, lsf_backend):
        groups = {"gpu_hosts": ["gpu-01", "gpu-02"]}
        members, unresolved = lsf_backend.resolve_hosts(
            "cpu-01 gpu_hosts/ mystery", ("gpu-01", "gpu-02", "cpu-01"), groups
        )
        assert set(members) == {"gpu-01", "gpu-02", "cpu-01"}
        assert unresolved == ("mystery",)

    def test_cluster_order_is_preserved_and_duplicates_dropped(self, lsf_backend):
        groups = {"a": ["h2", "h1"], "b": ["h1"]}
        members, _ = lsf_backend.resolve_hosts("a b", ("h1", "h2", "h3"), groups)
        assert members == ("h1", "h2")

    def test_a_group_member_not_in_the_cluster_is_ignored(self, lsf_backend):
        groups = {"a": ["h1", "retired-host"]}
        members, unresolved = lsf_backend.resolve_hosts("a", ("h1",), groups)
        assert members == ("h1",)
        assert unresolved == ()


class TestGroupLookupFailureIsReported:
    """Unix groups are LSF's only entitlement signal, so losing them matters.

    An empty group set reads as "cannot tell" downstream, so every
    `USER_ADVOCATES`/group restriction is silently ignored; a partial one reads
    as authoritative and produces a false denial. Neither is acceptable, so the
    failure is raised and `Cluster.load` records it -- leaving `identity` as
    None, which the entitlement filter treats as "cannot filter".
    """

    def test_a_broken_group_database_raises(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.lsf import LsfBackend

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: (_ for _ in ()).throw(OSError("nss unavailable")))
        with pytest.raises(OSError):
            LsfBackend(RecordedRunner({})).load_identity()

    def test_the_cluster_records_it_rather_than_crashing(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.lsf import LsfBackend
        from nodetop.core.cluster import Cluster

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: (_ for _ in ()).throw(OSError("nss unavailable")))
        cluster = Cluster.load(LsfBackend(RecordedRunner({})))
        assert cluster.identity is None
        assert "identity" in cluster.errors


class TestWrappedQueueValuesAreNotTruncated:
    """`bqueues -l` wraps a long value onto indented following lines.

    `_after` stopped at the end of the label's own line, so a queue whose
    `USERS:` list ran past one line had the tail silently dropped. A truncated
    allowlist is read as authoritative and produces a FALSE DENIAL -- the tool
    reporting no access to a queue that would have taken the job. Same defect
    the PBS backend had in its own dialect.
    """

    BODY = ("QUEUE: gpuq\n"
            "PRIO: 30  STATUS: Open:Active\n"
            "USERS: alice bob carol dave erin frank grace heidi ivan judy\n"
            "    karl liam mona nina\n"
            "HOSTS:  gpu-01 gpu-02\n"
            "    gpu-03 gpu-04\n")

    def test_the_user_list_keeps_every_name(self):
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert len(queue.allow_users) == 14
        assert "nina" in queue.allow_users

    def test_the_host_list_keeps_every_host(self):
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert "gpu-04" in getattr(queue, "labels_hosts", "")

    def test_a_following_label_stops_the_value(self):
        # The continuation must not swallow the next section.
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert "gpu-01" not in queue.allow_users
        assert queue.priority == 30

    def test_the_real_fixture_parses_unchanged(self, lsf_backend):
        names = {q.name: q for q in lsf_backend.load_queues()}
        assert names["gpu"].allow_users == ("alice", "bob", "carol")
        assert names["normal"].allow_users == ()
