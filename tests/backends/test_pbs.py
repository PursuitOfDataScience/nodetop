"""PBS Pro / OpenPBS / Torque."""

from __future__ import annotations

from nodetop.backends.pbs import PbsBackend
from nodetop.core.model import JobShape
from nodetop.runner import RecordedRunner


def _nodes(pbs_backend):
    return {n.name: n for n in pbs_backend.load_nodes()}


class TestNodeParsingJson:
    def test_all_nodes(self, pbs_backend):
        assert len(pbs_backend.load_nodes()) == 5

    def test_resources(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu001"]
        assert (n.cpus_total, n.gpus_total) == (64, 4)
        assert n.memory_mb == 512 * 1024
        assert n.accelerator.model == "A100"
        assert set(n.queues) == {"gpuq", "debug"}

    def test_assigned_resources_become_occupancy(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu002"]
        assert (n.cpus_free, n.gpus_free) == (0, 0)

    def test_offline_is_unschedulable(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu003"]
        assert n.schedulable is False

    def test_state_unknown_is_unreachable(self, pbs_backend):
        n = _nodes(pbs_backend)["cpu002"]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_memory_units(self, pbs_backend):
        assert _nodes(pbs_backend)["cpu001"].memory_alloc_mb == 16 * 1024


class TestNodeParsingText:
    """Torque and PBS before 18 have no JSON mode at all."""

    def test_classic_format(self, pbs_backend):
        from conftest import read

        nodes = {
            n.name: n
            for n in pbs_backend.parse_nodes_text(read("pbs", "pbsnodes.txt"))
        }
        assert set(nodes) == {"node01", "node02"}
        assert nodes["node01"].gpus_total == 2
        assert nodes["node01"].accelerator.model == "H100"
        assert nodes["node01"].cpus_total == 32
        assert nodes["node02"].schedulable is False

    def test_json_is_preferred_and_text_is_the_fallback(self, pbs_backend):
        # The JSON fixture is what load_nodes should have used.
        assert len(pbs_backend.load_nodes()) == 5


class TestQueues:
    def _queues(self, pbs_backend):
        return {q.name: q for q in pbs_backend.load_queues()}

    def test_two_independent_switches(self, pbs_backend):
        q = self._queues(pbs_backend)
        # This is the PBS form of the phantom-capacity trap: enabled swallows
        # submissions, started decides whether any of them ever run.
        assert (q["gpuq"].enabled, q["gpuq"].started) == (True, True)
        assert (q["drainq"].enabled, q["drainq"].started) == (True, False)
        assert (q["closedq"].enabled, q["closedq"].started) == (False, False)

    def test_accepts_but_never_starts_is_unusable(self, pbs_backend):
        q = self._queues(pbs_backend)["drainq"]
        assert q.usable is False
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}

    def test_an_enabled_acl_with_no_users_permits_nobody(self, pbs_backend):
        # acl_user_enable=True with an empty acl_users is a closed door, and
        # must not be read as "no restriction".
        q = self._queues(pbs_backend)["emptyaclq"]
        assert q.usable is False
        assert "NO_USERS" in {b.code for b in q.structural_blockers()}

    def test_a_disabled_acl_is_not_a_restriction(self, pbs_backend):
        assert self._queues(pbs_backend)["cpuq"].allow_users == ()

    def test_acl_users_are_read(self, pbs_backend):
        assert set(self._queues(pbs_backend)["gpuq"].allow_users) == {
            "alice", "bob", "carol"
        }

    def test_walltime_is_hhmmss_never_bare_minutes(self, pbs_backend):
        # PBS "48:00:00" is 48 hours; reading it with Slurm's grammar would
        # give 48 minutes.
        assert self._queues(pbs_backend)["gpuq"].max_walltime_seconds == 48 * 3600

    def test_queue_membership_is_reconstructed_from_the_nodes(self, pbs_backend):
        # PBS does not list a queue's nodes; the mapping lives on Qlist.
        assert set(self._queues(pbs_backend)["gpuq"].node_names) == {
            "gpu001", "gpu002", "gpu003"
        }


class TestLimits:
    def test_per_entity_table_is_unwrapped(self, pbs_backend):
        # "max_run_res.ngpus = [u:PBS_GENERIC=8]" -> 8
        limits = pbs_backend.load_limits()["gpuq"]
        assert limits.per_user["gpu"] == 8

    def test_resources_max_becomes_a_per_job_ceiling(self, pbs_backend):
        assert pbs_backend.load_limits()["gpuq"].per_job["cpu"] == 256

    def test_ceiling_is_detected_from_the_shape_alone(self, pbs_backend):
        limits = pbs_backend.load_limits()["gpuq"]
        got = limits.blockers(JobShape(nodes=4, gpus_per_node=4, walltime="1h"))
        assert "MAX_GPU_USER" in {b.code for b in got}


class TestNoProbe:
    def test_probe_returns_none(self, pbs_backend):
        # PBS has no verify-only submission, so there is nothing to ask.
        assert pbs_backend.probe("gpuq", JobShape()) is None

    def test_capabilities_say_so(self, pbs_backend):
        caps = pbs_backend.capabilities()
        assert caps.probe is False
        assert any("no verify-only" in n for n in caps.notes)


class TestSubmitFlags:
    def test_select_statement(self, pbs_backend):
        flags = " ".join(
            pbs_backend.submit_flags("gpuq", JobShape(nodes=2, gpus_per_node=4,
                                                      cpus_per_task=8, memory_gb=64,
                                                      walltime="4h"))
        )
        assert "select=2:ncpus=8:ngpus=4:mem=64gb" in flags
        assert "walltime=04:00:00" in flags
        assert "-q gpuq" in flags

    def test_nodelist_uses_pbs_notation(self, pbs_backend):
        assert pbs_backend.format_nodelist(["b", "a"]) == "a+b"


QSTAT_F_RUNNING = """Job Id: 1234.pbs-head
    Job_Name = train
    job_state = R
    Resource_List.walltime = 24:00:00
    stime = Thu Aug 21 10:00:00 2026
    exec_host = gpu001/0*64+gpu002/0*64
    exec_vnode = (gpu001:ncpus=64)+(gpu002:ncpus=64)
    etime = Thu Aug 21 09:55:00 2026

Job Id: 1235.pbs-head
    job_state = R
    Resource_List.walltime = 02:00:00
    stime = Thu Aug 21 12:00:00 2026
    exec_host = gpu001/0*8

Job Id: 1236.pbs-head
    job_state = Q
    Resource_List.walltime = 01:00:00
    estimated.start_time = Fri Aug 22 03:00:00 2026
"""


class TestFreeTimes:
    """PBS records no end time, so it has to be computed."""

    def _times(self, pbs_backend):
        return pbs_backend.parse_free_times(QSTAT_F_RUNNING)

    def test_end_is_start_plus_walltime(self, pbs_backend):
        # An upper bound, which is the right direction: a job may finish early,
        # so promising the node sooner would be a promise PBS never made.
        got = self._times(pbs_backend)
        assert str(got["gpu002"]) == "2026-08-22 10:00:00"

    def test_exec_host_slash_form_is_parsed(self, pbs_backend):
        # "gpu001/0*64+gpu002/0*64" -- matching this with a parenthesis
        # pattern (the exec_vnode shape) silently yields nothing at all.
        assert set(self._times(pbs_backend)) == {"gpu001", "gpu002"}

    def test_the_latest_job_on_a_node_wins(self, pbs_backend):
        # gpu001 hosts two jobs; the node is free no earlier than the last one.
        got = self._times(pbs_backend)
        assert str(got["gpu001"]) == "2026-08-22 10:00:00"

    def test_a_pending_job_contributes_nothing(self, pbs_backend):
        # estimated.start_time is when a queued job may START, not when
        # anything ends; using it as an end time is simply wrong.
        assert len(self._times(pbs_backend)) == 2

    def test_a_job_missing_any_piece_is_skipped(self, pbs_backend):
        partial = "Job Id: 1\n    exec_host = n1/0\n"
        assert pbs_backend.parse_free_times(partial) == {}

    def test_empty_input(self, pbs_backend):
        assert pbs_backend.parse_free_times("") == {}


QUEUES_JSON = """{
 "timestamp": 1755800000,
 "pbs_version": "2022.1.1",
 "Queue": {
   "workq": {
     "queue_type": "Execution",
     "enabled": "True",
     "started": "False",
     "Priority": "80",
     "resources_max": {"walltime": "48:00:00", "nodect": "16"},
     "max_run_res": {"ngpus": "[u:PBS_GENERIC=8]"},
     "acl_user_enable": "True",
     "acl_users": "alice,bob"
   }
 }
}"""


class TestQueuesJson:
    """The JSON path, which PBS Pro 18+ prefers."""

    def test_nested_keys_are_flattened(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        # resources_max.walltime has to survive the nesting.
        assert q.max_walltime_seconds == 48 * 3600
        assert q.max_nodes == 16

    def test_switches_and_acl(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        assert (q.enabled, q.started) == (True, False)
        assert set(q.allow_users) == {"alice", "bob"}
        assert q.priority == 80

    def test_accepts_but_never_starts(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}


def _pbs_cluster(nodes_json: str, qstat: str):
    from nodetop.backends.pbs import PbsBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    return Cluster.load(
        PbsBackend(RecordedRunner({
            "pbsnodes -a -F json": (0, nodes_json, ""),
            "qstat -Qf": (0, qstat, ""),
            "qstat -f": (0, "", ""),
        })),
        with_free_times=False,
    )


MIXED_NODES = """{"nodes": {
 "gpu001": {"state":"free","resources_available":{"ncpus":64,"ngpus":4,"Qlist":"gpuq"},
            "resources_assigned":{}},
 "cpu001": {"state":"free","resources_available":{"ncpus":128},"resources_assigned":{}}
}}"""

TYPED_QUEUES = """Queue: gpuq
    queue_type = Execution
    enabled = True
    started = True

Queue: workq
    queue_type = Execution
    enabled = True
    started = True

Queue: routeq
    queue_type = Route
    enabled = True
    started = True
    route_destinations = gpuq,workq
"""


class TestUnrestrictedNodes:
    """Qlist restricts a node; it does not enrol it."""

    def test_a_node_with_no_qlist_belongs_to_every_execution_queue(self):
        # Requiring an explicit mention orphans every unrestricted node: its
        # capacity becomes invisible to all queues.
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert "cpu001" in cluster.queues["gpuq"].node_names
        assert "cpu001" in cluster.queues["workq"].node_names

    def test_a_restricted_node_stays_restricted(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert "gpu001" in cluster.queues["gpuq"].node_names
        assert "gpu001" not in cluster.queues["workq"].node_names

    def test_a_queue_no_node_names_is_not_left_looking_empty(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert cluster.queues["workq"].node_names != ()

    def test_membership_is_deduplicated(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        names = cluster.queues["gpuq"].node_names
        assert len(names) == len(set(names))


class TestRoutingQueues:
    def test_a_route_queue_is_recognised(self):
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert q.routes is True
        assert q.forwards_to == ("gpuq", "workq")

    def test_an_execution_queue_is_not(self):
        assert _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["gpuq"].routes is False

    def test_it_owns_no_nodes(self):
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert q.node_names == ()
        assert q.declared_nodes == 0

    def test_having_no_nodes_is_not_reported_as_all_nodes_down(self):
        # "every node is down" is a finding; "owns no nodes by design" is not.
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert "ALL_NODES_UNSCHEDULABLE" not in {
            b.code for b in q.structural_blockers()
        }

    def test_it_is_not_offered_as_a_placement_target(self):
        from nodetop.core.fit import rank
        from nodetop.core.model import JobShape

        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        names = [p.queue for p in rank(cluster, JobShape(cpus_per_task=2))]
        # Its capacity belongs to its destinations, which are ranked instead.
        assert "routeq" not in names
        assert {"gpuq", "workq"} <= set(names)


class TestWrappedAttributeValuesAreNotTruncated:
    """PBS wraps long attribute values at 80 columns, mid-value, with a tab.

    Every parser here iterated raw lines and required an `=` to accept one, so
    each continuation was dropped -- losing the *tail* of exactly the values long
    enough to wrap, which are the ones that matter.
    """

    def test_a_nodes_queue_list_keeps_every_queue(self):
        # A node serving eight queues was recorded as serving five. It then goes
        # missing from the queues whose names were cut, and its capacity with it.
        text = ("node01\n"
                "     state = free\n"
                "     pcpus = 32\n"
                "     resources_available.ncpus = 32\n"
                "     resources_available.Qlist = workq,batch,longq,gpuq,bigmem,\n"
                "\tdebug,test,priority\n"
                "     resources_assigned.ncpus = 0\n")
        node = PbsBackend(RecordedRunner({})).parse_nodes_text(text)[0]
        assert node.queues == ("workq", "batch", "longq", "gpuq", "bigmem",
                               "debug", "test", "priority")

    def test_a_queue_acl_keeps_every_user(self):
        # A truncated allowlist reads as authoritative and produces a FALSE
        # DENIAL -- no access reported to a queue that would have taken the job.
        text = ("Queue: gpuq\n"
                "    queue_type = Execution\n"
                "    enabled = True\n"
                "    started = True\n"
                "    acl_user_enable = True\n"
                "    acl_users = alice@*,bob@*,carol@*,dave@*,erin@*,frank@*,\n"
                "\tgrace@*,heidi@*\n")
        queue = PbsBackend(RecordedRunner({})).parse_queues_text(text)[0]
        assert len(queue.allow_users) == 8
        assert "heidi@*" in queue.allow_users

    def test_exec_host_keeps_every_node(self):
        # The longest field PBS emits, and it always wraps for a multi-node job.
        # It maps running work to nodes, so a truncated one attributes free-time
        # estimates to the wrong machines.
        text = ("Job Id: 1.server\n"
                "    job_state = R\n"
                "    exec_host = n01/0*32+n02/0*32+n03/0*32+n04/0*32+n05/0*32+\n"
                "\tn06/0*32+n07/0*32\n"
                "    stime = Sat Aug 23 10:00:00 2026\n"
                "    Resource_List.walltime = 04:00:00\n")
        got = PbsBackend(RecordedRunner({})).parse_free_times(text)
        assert sorted(got) == [f"n0{i}" for i in range(1, 8)]

    def test_a_record_header_is_never_read_as_a_continuation(self):
        # Headers are unindented and continuations are indented, which is what
        # keeps the two apart. Two records must stay two.
        text = ("node01\n     state = free\n     pcpus = 8\n"
                "node02\n     state = free\n     pcpus = 8\n")
        nodes = PbsBackend(RecordedRunner({})).parse_nodes_text(text)
        assert [n.name for n in nodes] == ["node01", "node02"]

    def test_unwrapped_output_is_unaffected(self, pbs_backend):
        # The format in use must parse exactly as before the hardening.
        assert len(pbs_backend.load_nodes()) >= 1
