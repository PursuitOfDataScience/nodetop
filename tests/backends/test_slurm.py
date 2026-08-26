"""Slurm -- the reference backend, validated against a live cluster."""

from __future__ import annotations

import pytest
from fixtures.slurm import probe_outputs as po

from nodetop.backends.slurm import SlurmBackend, parse_probe, parse_slurm_duration
from nodetop.core.model import JobShape, Queue, VerdictCategory
from nodetop.runner import RecordedRunner

B1 = "gn-0001"
BIGMEM = "gn-bigmem1"
DEGRADED = "cn-0385"


def _parse(text, queue="p", account="a", rc=1):
    # sbatch writes its verdict to stderr and exits non-zero even when the
    # answer is informative, so that is the realistic call shape.
    return parse_probe(queue, account, rc, "", text)


class TestSlurmDuration:
    @pytest.mark.parametrize("text,seconds", [
        ("60", 3600),          # bare number is minutes
        ("2:00", 120),         # MM:SS, not HH:MM
        ("1:00:00", 3600),
        ("2-00:00:00", 172800),
        ("2-12", 216000),      # D-HH
        ("UNLIMITED", None),
    ])
    def test_grammar(self, text, seconds):
        assert parse_slurm_duration(text) == seconds


class TestNodes:
    def test_all_records(self, slurm_backend):
        assert len(slurm_backend.load_nodes()) == 9

    def test_fields(self, slurm_backend):
        n = {x.name: x for x in slurm_backend.load_nodes()}[B1]
        assert (n.cpus_total, n.cpus_alloc) == (32, 32)
        assert n.memory_mb == 256000
        assert n.gpus_total == 4
        assert n.accelerator.model == "A100"

    def test_compound_state_is_decomposed(self, slurm_backend):
        # "DOWN*+DRAIN" is a base state, a not-responding marker and a flag
        # fused into one token.
        n = {x.name: x for x in slurm_backend.load_nodes()}[DEGRADED]
        assert n.state_raw == "DOWN*+DRAIN"
        assert n.conditions == frozenset({"DOWN", "DRAIN"})
        assert n.unreachable is True
        assert n.schedulable is False

    def test_reason_with_spaces_is_not_truncated(self, slurm_backend):
        n = {x.name: x for x in slurm_backend.load_nodes()}[DEGRADED]
        assert n.reason.startswith("maintenance: hardware issue")

    @pytest.mark.parametrize("state", ["DOWN*", "DOWN+NOT_RESPONDING"])
    def test_either_spelling_of_lost_contact_is_read(self, state):
        # One fact, two spellings, and a Slurm version picks. `sinfo` writes the
        # `*` suffix on every release, but `scontrol show node` -- what this
        # backend reads -- writes `+NOT_RESPONDING` on 25.11 and no `*`
        # anywhere in the record: `grep -c '*'` over every node came back 0.
        # Reading only the marker left a node 23 days out of contact looking
        # merely DOWN, and "the control plane has lost contact with this node"
        # could not be said on that cluster at all.
        n = SlurmBackend(RecordedRunner({
            "scontrol show node": (
                0, f"NodeName=n1 CPUTot=8 State={state} Partitions=p\n", ""),
        })).load_nodes()[0]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_accelerator_ness_is_from_gres_not_hostname(self, slurm_backend):
        # This host sits among 44 accelerator nodes and has none.
        n = {x.name: x for x in slurm_backend.load_nodes()}[BIGMEM]
        assert n.name.startswith("gn")
        assert n.is_gpu_node is False


class TestTheFastFieldSplitAgreesWithTheRegex:
    """Parsing is on every run, and it was 35% one function.

    `_fields` ended each value with a lazy match plus a lookahead, which
    re-evaluates the lookahead at every character of every value: profiling
    `parse_nodes` over 607 live nodes put it at a third of the total, with
    460,680 `re.Match.group` calls under it. Splitting on spaces cannot
    backtrack. 42.4 ms -> 32.9 ms for those nodes.

    The whole value of the change rests on the two paths agreeing, so that is
    what is pinned here -- including the case the fast path cannot do, where it
    has to hand back to the regex.
    """

    #: Records chosen to break a naive splitter: a key-shaped token inside a
    #: value (the first-wins case this file already documents), `=` inside a
    #: value, doubled spaces, an empty value, leading and trailing space, and
    #: the `#` and `:` that `OS=` and `Gres=` carry.
    RECORDS = [
        "NodeName=n1 CPUTot=8 State=IDLE",
        "NodeName=n1 Reason=replacing NodeName=n2 per ticket State=DOWN",
        "NodeName=n1 Reason=two  spaces inside State=IDLE",
        "NodeName=n1 OS=Linux 5.14.0 #1 SMP Thu State=IDLE",
        "NodeName=n1 Comment= State=IDLE",
        "  NodeName=n1 State=IDLE",
        "NodeName=n1 State=IDLE  ",
        "NodeName=n1 Reason=a=b=c State=IDLE",
        "NodeName=n1 Gres=gpu:a100:4(S:0-1) AllocTRES=cpu=28,gres/gpu=1 State=MIXED",
        # Whitespace that is a separator but not a space: the fast path cannot
        # see these, so the record must go back to the regex.
        "NodeName=n1\tCPUTot=8\tState=IDLE",
        "NodeName=n1 Reason=tab\there State=IDLE",
    ]

    @staticmethod
    def _regex_only(pattern, record):
        out: dict[str, str] = {}
        for m in pattern.finditer(record):
            out.setdefault(m.group("key"), m.group("val").strip())
        return out

    @pytest.mark.parametrize("record", RECORDS, ids=range(len(RECORDS)))
    def test_both_paths_read_the_same_fields(self, record):
        from nodetop.backends.slurm import _NODE_FIELD, _PART_FIELD, _fields

        for pattern in (_NODE_FIELD, _PART_FIELD):
            assert _fields(pattern, record) == self._regex_only(pattern, record)

    def test_a_tab_separated_record_still_parses(self):
        # The guard exists because the failure would be the worst available:
        # the whole record read as one field, so a node with no state, which
        # `parse_nodes` then reports as UNKNOWN and unschedulable.
        n = SlurmBackend(RecordedRunner({
            "scontrol show node": (
                0, "NodeName=n1\tCPUTot=8\tState=IDLE\tPartitions=p\n", ""),
        })).load_nodes()[0]
        assert (n.name, n.cpus_total, n.state_raw) == ("n1", 8, "IDLE")
        assert n.schedulable is True

    def test_the_key_test_is_cached_and_bounded(self):
        from nodetop.backends.slurm import _is_key

        _is_key.cache_clear()
        for _ in range(200):
            _is_key("node", "NodeName")
        info = _is_key.cache_info()
        assert info.hits == 199 and info.misses == 1
        assert info.maxsize == 512
        # An operator `Reason` full of unique `word=` tokens must not grow it
        # without limit -- this process can outlive a parse by hours.
        for i in range(2000):
            _is_key("node", f"ticket{i}")
        assert _is_key.cache_info().currsize <= 512


class TestPlannedIsAPlanNotAnOutage:
    """`PLANNED` is the backfill scheduler's intent, not a node's condition.

    Slurm writes it on a node it means to hand to a queued job -- `sinfo`'s `-`
    suffix, beside `*` for unreachable and `$` for a maintenance reservation --
    and goes on running work there and placing more. Reading it as MAINT is how
    a healthy node becomes an outage: on a Slurm 25.11 cluster two
    `MIXED+PLANNED` H100 nodes, all 8 GPUs allocated to other people's running
    jobs, were reported as "all 2 nodes are down, drained or unreachable", both
    their partitions were struck out as ALL_NODES_UNSCHEDULABLE, and
    `accelerators` announced "no GPUs found in this cluster" one line under a
    header counting 8 of them.
    """

    def _node(self, state: str):
        return SlurmBackend(RecordedRunner({
            "scontrol show node": (
                0,
                f"NodeName=mgpu03 CPUTot=64 CPUAlloc=28 RealMemory=8000 AllocMem=2000 "
                f"Gres=gpu:h100:4 AllocTRES=cpu=28,mem=2000M,gres/gpu=1 "
                f"State={state} Partitions=gpu\n",
                "",
            ),
            "scontrol show config": (
                0, "SelectTypeParameters    = CR_CORE_MEMORY\n", ""),
        })).load_nodes()[0]

    def test_a_planned_node_still_takes_work(self):
        n = self._node("MIXED+PLANNED")
        assert n.state_raw == "MIXED+PLANNED"   # verbatim, so the reader sees it
        assert n.schedulable is True
        assert n.cpus_free == 36
        assert n.gpus_total == 4
        assert n.effective_free_gpus == 3

    def test_the_flag_is_carried_but_blocks_nothing(self):
        # Informational, not dropped: "why is this idle node not filling up?"
        # is a question the flag answers, and `--json` hands it over.
        assert self._node("MIXED+PLANNED").conditions == frozenset({"PLANNED"})
        assert self._node("IDLE+PLANNED").schedulable is True

    @pytest.mark.parametrize("state", ["MIXED+MAINT", "MIXED+DRAIN", "DOWN+PLANNED"])
    def test_a_real_condition_alongside_it_still_counts(self, state):
        # The flag is not a blanket amnesty: PLANNED riding along with a state
        # that does block must not rescue the node.
        assert self._node(state).schedulable is False

    @pytest.mark.parametrize("state", [
        # `man sinfo`, verbatim: "not capable of running any jobs".
        "IDLE+POWERED_DOWN",
        # "blocked by exclusive topo job"
        "IDLE+BLOCKED",
        # "rendering this node as not usable for any other jobs"
        "MIXED+PERFCTRS",
        # "A reboot request has been sent to the agent"
        "IDLE+REBOOT_ISSUED",
    ])
    def test_a_flag_that_means_not_usable_is_not_read_as_room(self, state):
        # An unmapped flag fails in the expensive direction: it contributes no
        # condition, so the node stays schedulable and its idle cores are
        # offered as room. A powered-down node advertises a full complement of
        # both while being, in `man sinfo`'s words, "not capable of running any
        # jobs".
        n = self._node(state)
        assert n.schedulable is False, n.state_raw
        queue = Queue(name="p", nodes=[n], node_names=(n.name,))
        assert queue.effective_free_cpus == 0
        assert queue.effective_free_gpus == 0

    def test_powering_up_is_still_somewhere_to_send_work(self):
        # The one powersave state Slurm does place work on. Calling it
        # unschedulable would erase real capacity on any cluster that cycles
        # nodes; the job waits for the boot instead.
        assert self._node("IDLE+POWERING_UP").schedulable is True


class TestMemoryIsAConsumableResourceOrItIsNot:
    """Whether a full `AllocMem` is a ceiling depends on the cluster's config.

    `CR_CORE_MEMORY` makes it one: a node whose memory is fully allocated can
    host nothing more, however many cores are idle. Without the `_MEMORY`
    suffix Slurm never decrements memory, so `AllocMem` records what jobs
    asked for and reading it as a ceiling would report a whole cluster as full.
    """

    NODE = ("NodeName=n1 CPUTot=48 CPUAlloc=4 RealMemory=1000 AllocMem=1000 "
            "State=MIXED Partitions=p\n")

    def _backend(self, select: str | None):
        recorded = {"scontrol show node": (0, self.NODE, "")}
        if select is not None:
            recorded["scontrol show config"] = (
                0, f"SelectType              = select/cons_tres\n"
                   f"SelectTypeParameters    = {select}\n", "")
        return SlurmBackend(RecordedRunner(recorded))

    def test_memory_tracking_config_makes_a_full_node_full(self):
        backend = self._backend("CR_CORE_MEMORY,CR_ONE_TASK_PER_CORE")
        assert backend.memory_is_consumable() is True
        node = backend.load_nodes()[0]
        assert node.cpus_free == 44          # what the scheduler claims
        assert node.effective_free_cpus == 0  # what it can hand out
        assert node.has_room is False

    def test_a_config_without_memory_leaves_the_cores_countable(self):
        backend = self._backend("CR_CORE,CR_ONE_TASK_PER_CORE")
        assert backend.memory_is_consumable() is False
        node = backend.load_nodes()[0]
        assert node.memory_consumable is False
        assert node.effective_free_cpus == 44
        assert node.has_room is True

    def test_unreadable_config_claims_less_capacity(self):
        # No recording for `show config`, so the query raises. Applying the
        # constraint is the safe direction: the alternative is recommending a
        # node the scheduler will refuse.
        backend = self._backend(None)
        assert backend.memory_is_consumable() is True
        assert backend.load_nodes()[0].has_room is False

    def test_config_without_the_parameter_line_claims_less_too(self):
        backend = SlurmBackend(RecordedRunner({
            "scontrol show node": (0, self.NODE, ""),
            "scontrol show config": (0, "ClusterName = x\n", ""),
        }))
        assert backend.memory_is_consumable() is True

    def test_the_config_is_asked_for_once(self):
        backend = self._backend("CR_CORE_MEMORY")
        backend.load_nodes()
        config_calls = [c for c in backend.runner.calls if "config" in c]
        assert len(config_calls) == 1


class TestTheSitesMemoryFloorReachesTheNodes:
    """`DefMemPerCPU` is what one core of a default job costs, so it is the floor.

    Without it the exhaustion test compares against zero and misses the node it
    exists to catch: 97 idle cores, 250 MB free, and a cluster that hands out
    3810 MB per core. Measured on a 607-node cluster after wiring it through:
    **3,461 phantom cores removed across 153 nodes** -- a quarter of the free
    cores the tool had been reporting.
    """

    NODE = ("NodeName=n1 CPUTot=128 CPUAlloc=31 RealMemory=250000 "
            "AllocMem=249750 State=MIXED Partitions=p\n")

    def _nodes(self, config):
        return SlurmBackend(RecordedRunner({
            "scontrol show node": (0, self.NODE, ""),
            "scontrol show config": (0, config, ""),
        })).load_nodes()

    CONSUMABLE = "SelectTypeParameters    = CR_CORE_MEMORY\n"

    def test_the_floor_is_read_and_applied(self):
        n = self._nodes(self.CONSUMABLE + "DefMemPerCPU             = 3810\n")[0]
        assert n.memory_floor_mb == 3810
        assert n.memory_exhausted is True
        assert n.effective_free_cpus == 0

    @pytest.mark.parametrize("line", [
        "",                                        # field absent
        "DefMemPerCPU             = UNLIMITED\n",  # no floor published
        "DefMemPerCPU             = (null)\n",
    ])
    def test_no_published_floor_claims_none(self, line):
        n = self._nodes(self.CONSUMABLE + line)[0]
        assert n.memory_floor_mb == 0
        assert n.memory_exhausted is False        # back to the zero test

    def test_a_non_consumable_cluster_is_left_alone(self):
        # `AllocMem` there is a record of requests, not a ceiling; there is
        # nothing for a floor to be compared against.
        n = self._nodes("SelectTypeParameters = CR_CORE\nDefMemPerCPU = 3810\n")[0]
        assert n.memory_consumable is False
        assert n.memory_exhausted is False

    def test_the_config_is_still_asked_for_once(self):
        # Three consumers now: memory-consumability, this floor, and whether
        # associations are enforced.
        backend = SlurmBackend(RecordedRunner({
            "scontrol show node": (0, self.NODE, ""),
            "scontrol show config": (
                0, self.CONSUMABLE + "DefMemPerCPU = 3810\n"
                   "AccountingStorageEnforce = associations\n", ""),
            "sacctmgr": (0, "", ""),
        }))
        backend.load_nodes()
        backend.load_identity()
        asked = [c for c in backend.runner.calls if "config" in " ".join(c)]
        assert len(asked) == 1, asked


class TestGresCountsAreNotDeviceIndices:
    """`Gres=` carries the devices, and the suffix looks like the count.

    Slurm appends which accelerators, not just how many -- `gpu:2(IDX:0,3)` on
    a job's allocation, `gpu:v100:4(S:0-1)` on a node's socket affinity -- and
    the contents hold both colons and commas, the two characters the field is
    split on. Splitting first read the device *index* as the count. Measured on
    one node's three jobs: 0, 2, 1 where the truth was 2, 1, 1.
    """

    @pytest.mark.parametrize("gres,expected", [
        ("gpu:4", 4),                                   # plain
        ("gpu:a30:4", 4),                               # typed
        ("gpu:2(IDX:0,3)", 2),                          # the comma case: read 0
        ("gpu:1(IDX:2)", 1),                            # read 2 -- another job's
        ("gpu:1(IDX:1)", 1),                            # right by coincidence
        ("gpu:v100:4(S:0-1)", 4),                       # socket affinity: read 0
        ("gpu:v100:2(IDX:0-1),gpu:a100:1(IDX:0)", 3),   # two models on one node
        ("gpu:2,mps:1", 2),                             # another gres beside it
        ("mps:1", 0),
        ("(null)", 0),
        ("", 0),
        (None, 0),
    ])
    def test_it_counts_devices_not_their_ids(self, gres, expected):
        from nodetop.backends.slurm import _gres_gpus

        assert _gres_gpus(gres) == expected

    def test_a_node_reporting_socket_affinity_still_has_its_accelerators(self):
        # This cluster does not print the suffix on nodes, so the node-level
        # half of the bug was latent -- and it would have made every GPU node
        # on a cluster that does print it look like a CPU node.
        backend = SlurmBackend(RecordedRunner({
            "scontrol show node": (
                0, "NodeName=g1 CPUTot=48 CPUAlloc=0 RealMemory=1000 AllocMem=0 "
                   "State=IDLE Gres=gpu:v100:4(S:0-1) Partitions=p\n", ""),
        }))
        node = backend.load_nodes()[0]
        assert node.gpus_total == 4
        assert node.is_gpu_node is True

    def test_a_jobs_share_of_the_accelerators_is_the_count(self):
        backend = SlurmBackend(RecordedRunner({}))
        got = {(a.job, a.node): a for a in backend.parse_allocations(
            "JobId=1 JobName=x\n"
            "     Nodes=n1 CPU_IDs=0-15 Mem=60960 GRES=gpu:2(IDX:0,3)\n"
            "JobId=2 JobName=y\n"
            "     Nodes=n1 CPU_IDs=32-39 Mem=16384 GRES=gpu:1(IDX:2)\n")}
        assert got[("1", "n1")].gpus == 2
        assert got[("2", "n1")].gpus == 1

    def test_the_shares_of_one_node_sum_to_what_it_holds(self):
        # The invariant that would have caught this at once: 0 + 2 + 1 = 3 on a
        # node reporting all four of its accelerators allocated.
        backend = SlurmBackend(RecordedRunner({}))
        shares = [a for a in backend.parse_allocations(
            "JobId=1 JobName=x\n"
            "     Nodes=n1 CPU_IDs=0-15 Mem=1 GRES=gpu:2(IDX:0,3)\n"
            "JobId=2 JobName=y\n"
            "     Nodes=n1 CPU_IDs=32-39 Mem=1 GRES=gpu:1(IDX:2)\n"
            "JobId=3 JobName=z\n"
            "     Nodes=n1 CPU_IDs=1 Mem=1 GRES=gpu:1(IDX:1)\n")
            if a.node == "n1"]
        assert sum(a.gpus for a in shares) == 4


class TestPerNodeShares:
    """`scontrol show job -d` is the only source for a job's share of a node.

    A job list reports totals over every node held, so a 42-node job read as
    512 cores on a 48-core machine.
    """

    DUMP = """JobId=53272514 JobName=_interactive
   NumNodes=42 NumCPUs=512
   JOB_GRES=(null)
     Nodes=cn-0002 CPU_IDs=33,35,37,39 Mem=4096 GRES=
     Nodes=cn-0114 CPU_IDs=41-47 Mem=7168 GRES=
JobId=54465084 ArrayJobId=54462542 ArrayTaskId=65 JobName=ffgs
   NumNodes=1 NumCPUs=1
     Nodes=cn-0500 CPU_IDs=3 Mem=28672 GRES=gpu:1(IDX:1)
JobId=54480000 ArrayJobId=54480000 ArrayTaskId=1-20%10 JobName=pending-array
     Nodes=cn-[0521-0522] CPU_IDs=78-94 Mem=29750 GRES=gpu:2
"""

    def _by_key(self):
        backend = SlurmBackend(RecordedRunner({}))
        return {(a.job, a.node): a for a in backend.parse_allocations(self.DUMP)}

    def test_a_range_of_core_ids_is_counted_not_read(self):
        # `CPU_IDs=41-47` is seven cores, not the number 41.
        got = self._by_key()[("53272514", "cn-0114")]
        assert (got.cpus, got.memory_mb) == (7, 7168)

    def test_a_comma_list_is_counted_too(self):
        assert self._by_key()[("53272514", "cn-0002")].cpus == 4

    def test_an_array_task_is_registered_under_squeues_spelling(self):
        # `squeue` says `54462542_65`; `scontrol` says JobId=54465084 with the
        # array recorded separately. 1864 of 2928 jobs on the reference cluster
        # are array tasks, so keying on JobId alone found a share for none.
        keys = self._by_key()
        assert ("54462542_65", "cn-0500") in keys
        assert ("54465084", "cn-0500") in keys

    def test_a_pending_array_range_is_not_mistaken_for_a_task(self):
        keys = self._by_key()
        assert not any(k[0].startswith("54480000_") for k in keys)

    def test_a_collapsed_nodelist_applies_to_every_node_in_it(self):
        # Slurm collapses consecutive nodes that got the same shape.
        keys = self._by_key()
        for node in ("cn-0521", "cn-0522"):
            assert keys[("54480000", node)].cpus == 17, node

    def test_gres_becomes_an_accelerator_count(self):
        assert self._by_key()[("54462542_65", "cn-0500")].gpus == 1
        assert self._by_key()[("54480000", "cn-0521")].gpus == 2

    def test_the_whole_cluster_is_one_call(self):
        # 0.6s for 2928 jobs against 0.13s for one: asking about five jobs
        # already pays for asking about all of them, and a node with 49 tasks
        # would otherwise stall an interactive repaint for six seconds.
        backend = SlurmBackend(RecordedRunner({
            "scontrol show job -d": (0, self.DUMP, ""),
        }))
        assert backend.load_allocations()
        assert len(backend.runner.calls) == 1


class TestPartitions:
    def _queues(self, slurm_backend):
        return {q.name: q for q in slurm_backend.load_queues()}

    def test_hidden_partitions_are_included(self, slurm_backend):
        # A hidden partition is exactly the kind that has quietly been taken
        # out of service, so omitting it hides the diagnosis.
        assert self._queues(slurm_backend)["test"].hidden is True

    def test_all_four_kill_switches(self, slurm_backend):
        q = self._queues(slurm_backend)["test"]
        codes = {b.code for b in q.structural_blockers()}
        assert "QUEUE_DISABLED" in codes
        assert "NO_ACCOUNTS" in codes
        assert "NO_QOS" in codes
        assert q.hidden is True
        assert q.usable is False

    def test_the_dead_partition_still_declares_its_nodes(self, slurm_backend):
        assert self._queues(slurm_backend)["test"].declared_nodes == 610

    def test_bracket_notation_in_an_unusual_position(self, slurm_backend):
        # Nodes=gn-00[01-44]
        assert len(self._queues(slurm_backend)["gn"].node_names) == 44

    def test_allow_accounts(self, slurm_backend):
        assert "rcc-staff" in self._queues(slurm_backend)["gn"].allow_accounts

    def test_drain_state_accepts_but_does_not_start(self):
        # Slurm has one switch where PBS has two; DRAIN is the
        # accepts-but-never-starts case.
        b = SlurmBackend(RecordedRunner({"scontrol show partition": (
            0, "PartitionName=d\n   State=DRAIN AllowGroups=ALL\n   TotalNodes=1\n", ""
        )}))
        q = b.load_queues()[0]
        assert (q.enabled, q.started) == (True, False)


class TestQos:
    def test_ceilings(self, slurm_backend):
        limits = slurm_backend.load_limits()["gn"]
        assert limits.max_walltime_seconds == 172800
        assert limits.per_job == {"gpu": 4}
        assert limits.per_user == {"cpu": 256, "gpu": 32, "node": 8}

    def test_empty_columns_do_not_raise(self, slurm_backend):
        # "test|7-00:00:00|||||100000000|" -- five empty fields in a row.
        limits = slurm_backend.load_limits()["test"]
        assert limits.max_walltime_seconds == 604800
        assert limits.per_job == {}

    def test_the_ceiling_the_dry_run_misses(self, slurm_backend):
        # sbatch --test-only returns PASSED with a start time for this shape,
        # and it then pends indefinitely.
        limits = slurm_backend.load_limits()["gn"]
        got = limits.blockers(JobShape(nodes=3, gpus_per_node=4, walltime="1h"))
        assert "MAX_GPU_JOB" in {b.code for b in got}


class TestProbeAccepted:
    def test_allowed(self):
        v = _parse(po.ACCEPTED, rc=0)
        assert v.allowed is True
        assert v.category == VerdictCategory.OK

    def test_predicted_start_and_nodes(self):
        v = _parse(po.ACCEPTED, rc=0)
        assert v.predicted_start.isoformat() == "2026-08-21T17:00:12"
        assert v.predicted_nodes == ("gn-0006",)

    def test_effective_qos_is_what_the_controller_chose(self):
        # Asked for "gn"; the site auto-promoted to "gn-prio".
        # Checking ceilings against the requested name checks the wrong ones.
        v = _parse(po.ACCEPTED, rc=0)
        assert v.effective_qos == "gn-prio"
        assert v.effective_account == "rcc-staff"

    def test_shared_partition_preamble(self):
        v = _parse(po.SHARED_PARTITION, rc=0)
        assert v.allowed is True
        assert v.predicted_nodes == ("cn-0278",)

    @pytest.mark.parametrize("line,nodes", [
        # Slurm 25.11, byte for byte off `cat -A`: a stray `a` sits between the
        # timestamp and `using`. One sentence-shaped regex matched the time,
        # failed at `using`, and dropped the nodelist with it -- so `check`
        # reported a confirmed start and `predicted_nodes: []` on every
        # accepted probe on that cluster.
        ("sbatch: Job 556812 to start at 2026-08-24T17:32:50 a using 1 processors "
         "on nodes mcn53 in partition standard", ("mcn53",)),
        ("sbatch: Job 556813 to start at 2026-08-24T17:32:50 a using 2 processors "
         "on nodes mcn[52-53] in partition standard", ("mcn52", "mcn53")),
        # The older wording has to keep working: this is a tolerance, not a
        # migration.
        ("sbatch: Job 42 to start at 2026-08-24T17:32:50 using 1 processors "
         "on nodes mcn53 in partition standard", ("mcn53",)),
        # And a line that says only when, which Slurm also emits.
        ("sbatch: Job 42 to start at 2026-08-24T17:32:50", ()),
    ])
    def test_the_prediction_survives_a_reworded_sentence(self, line, nodes):
        v = _parse(line, rc=0)
        assert v.allowed is True
        assert v.predicted_start.isoformat() == "2026-08-24T17:32:50"
        assert v.predicted_nodes == nodes

    def test_a_nodelist_from_another_line_is_not_borrowed(self):
        # Three facts are read off the line the verdict is on, so an unrelated
        # sentence mentioning nodes cannot be attributed to this prediction.
        v = _parse(
            "sbatch: Job 42 to start at 2026-08-24T17:32:50\n"
            "sbatch: some other note on nodes mcn99\n", rc=0)
        assert v.predicted_nodes == ()


class TestTwoLayerDisagreement:
    """The case a single-source check gets exactly backwards."""

    def test_filter_passed_but_core_refused_is_not_allowed(self):
        v = _parse(po.PLUGIN_PASSED_CORE_REFUSED)
        assert v.filter_verdict == "PASSED"
        assert v.allowed is False
        assert v.category == VerdictCategory.ACCOUNT_MISMATCH

    def test_the_disagreement_is_surfaced(self):
        v = _parse(po.PLUGIN_PASSED_CORE_REFUSED)
        assert "site submit filter reported PASSED" in v.reason

    def test_filter_rejection_is_classified_from_its_own_reason(self):
        v = _parse(po.PLUGIN_REJECTED)
        assert v.filter_verdict == "REJECTED"
        assert v.category == VerdictCategory.NOT_ENTITLED
        assert "pi-pedramh" in v.reason


class TestProbeTaxonomy:
    @pytest.mark.parametrize("text,category", [
        (po.NO_ACCOUNT, VerdictCategory.NO_ACCOUNT),
        (po.SHAPE_TOO_BIG, VerdictCategory.SHAPE_UNAVAILABLE),
        (po.CONTROLLER_IO_ERROR, VerdictCategory.CONTROL_PLANE_DOWN),
        (po.STOCK_QOS_VIOLATION, VerdictCategory.QUOTA_EXCEEDED),
        (po.STOCK_BAD_PARTITION, VerdictCategory.UNKNOWN_QUEUE),
        (po.STOCK_NODE_CONFIG, VerdictCategory.SHAPE_UNAVAILABLE),
        (po.STOCK_TIME_LIMIT, VerdictCategory.TIME_LIMIT),
    ])
    def test_categories(self, text, category):
        assert _parse(text).category == category

    def test_a_controller_outage_is_not_a_statement_about_access(self):
        # A controller that cannot write job scripts fails EVERY submission,
        # including a bare --wrap=hostname.
        v = _parse(po.CONTROLLER_IO_ERROR)
        assert v.allowed is False
        assert v.durable is False

    def test_membership_refusal_is_durable(self):
        assert _parse(po.PLUGIN_REJECTED).durable is True

    def test_unknown_output_is_labelled_unknown_not_guessed(self):
        assert _parse("something nobody has seen").category == VerdictCategory.UNKNOWN


class TestProbeInvocation:
    def test_test_only_is_hard_coded(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        SlurmBackend(runner).probe("gn", JobShape(gpus_per_node=1), "acct")
        cmd = runner.calls[0]
        assert cmd[0] == "sbatch"
        assert "--test-only" in cmd

    def test_shape_becomes_sbatch_flags(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        shape = JobShape(nodes=2, gpus_per_node=4, cpus_per_task=8,
                         memory_gb=64, walltime="2-00:00:00")
        SlurmBackend(runner).probe("gn", shape, "acct")
        joined = " ".join(runner.calls[0])
        for flag in ["--nodes=2", "--gres=gpu:4", "--cpus-per-task=8",
                     "--time=2-00:00:00", "--account=acct", "--partition=gn"]:
            assert flag in joined

    def test_the_probe_account_wins_over_the_shape_account(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        SlurmBackend(runner).probe("gn", JobShape(account="from-shape"), "from-probe")
        joined = " ".join(runner.calls[0])
        assert "--account=from-probe" in joined
        assert "--account=from-shape" not in joined

    def test_a_runner_failure_becomes_a_finding_not_a_crash(self):
        v = SlurmBackend(RecordedRunner({})).probe("gn", JobShape())
        assert v.category == VerdictCategory.CONTROL_PLANE_DOWN
        assert v.durable is False


class TestNodelist:
    def test_collapsed_bracket_notation(self, slurm_backend):
        names = ["cn-0298", "cn-0377", "cn-0378", "cn-0423"]
        assert slurm_backend.format_nodelist(names) == "cn-[0298,0377-0378,0423]"


class TestTresMemoryScale:
    """Slurm TRES memory defaults to MB, with an optional suffix."""

    @pytest.mark.parametrize("text,mb", [
        ("mem=500K", 0),          # ~0.49 MB
        ("mem=2048K", 2),
        ("mem=500M", 500),
        ("mem=4G", 4096),
        ("mem=1T", 1024 ** 2),
        ("mem=100", 100),         # no suffix means MB
    ])
    def test_scales(self, text, mb):
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map(text)["mem_mb"] == mb

    def test_a_kilobyte_ceiling_is_not_read_as_megabytes(self):
        # An integer 1 // 1024 is 0, and dodging that with `or 1` makes K equal
        # M -- so a mem=500K ceiling reads 1024x too large, and an over-limit
        # job is never flagged.
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map("mem=500K")["mem_mb"] < _parse_tres_map("mem=500M")["mem_mb"]

    def test_an_over_limit_job_is_flagged_against_a_kilobyte_ceiling(self):
        from nodetop.backends.slurm import _parse_tres_map
        from nodetop.core.model import JobShape, Limits

        limits = Limits(name="q", per_job=_parse_tres_map("mem=500K"))
        codes = {b.code for b in limits.blockers(JobShape(memory_gb=1))}
        assert "MAX_MEM_MB_JOB" in codes

    def test_other_resources_are_unaffected(self):
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map("cpu=8,mem=2G,gres/gpu=4") == {
            "cpu": 8, "mem_mb": 2048, "gpu": 4
        }


class TestIdentityQuery:
    """The argv shape matters, and getting it wrong fails silently."""

    def test_where_and_the_condition_are_separate_arguments(self):
        # Passing "where user=X" as one element makes sacctmgr answer
        # `Unknown condition`, so the identity comes back empty -- and an
        # empty identity silently disables every account and QOS access check,
        # because those are tri-state and read "nothing to compare against" as
        # "no verdict". Nothing errors; a whole analysis layer just stops.
        runner = RecordedRunner({"sacctmgr": (0, "acct||gn\n", "")})
        SlurmBackend(runner).load_identity()
        argv = next(c for c in runner.calls if c[0] == "sacctmgr")
        assert "where" in argv
        assert any(a.startswith("user=") for a in argv)
        assert not any(a.startswith("where user=") for a in argv)

    def test_the_query_asks_for_the_fields_it_parses(self):
        runner = RecordedRunner({"sacctmgr": (0, "", "")})
        SlurmBackend(runner).load_identity()
        argv = next(c for c in runner.calls if c[0] == "sacctmgr")
        assert "format=Account,Partition,QOS,DefaultQOS" in argv

    def test_a_rejected_query_is_raised_not_swallowed(self):
        # It used to return an empty identity, and that is indistinguishable
        # from a user who genuinely holds no associations. The account and QOS
        # checks downstream are tri-state and read "nothing to compare against"
        # as "no verdict", so a failed query silently disabled all of them.
        # Measured consequence: with `sacctmgr` down, all 34 accounts vanished,
        # every probe ran with no --account, and the overview reported
        # "0 open to you, 83 refused" -- total loss of access, asserted
        # confidently, during a database hiccup.
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({"sacctmgr": (1, "", "Unknown condition")})
        with pytest.raises(CommandError):
            SlurmBackend(runner).load_identity()

    def test_the_cluster_records_it_rather_than_crashing(self):
        # Raising is only correct because Cluster.load catches it: the report
        # still renders, `identity` is None, and the entitlement filter already
        # treats that as "cannot filter" rather than "entitled to nothing".
        from nodetop.core.cluster import Cluster

        runner = RecordedRunner({"sacctmgr": (1, "", "Unknown condition")})
        cluster = Cluster.load(SlurmBackend(runner))
        assert cluster.identity is None
        assert "identity" in cluster.errors

    def test_a_real_association_dump_is_parsed(self):
        # Two accounts, an identical queue list: the templated-entitlements
        # signal this backend exists to surface.
        dump = (
            "acct-a||aaz,gn,wide\n"
            "acct-b||aaz,gn,wide\n"
        )
        ident = SlurmBackend(RecordedRunner({"sacctmgr": (0, dump, "")})).load_identity()
        assert set(ident.accounts) == {"acct-a", "acct-b"}
        assert "gn" in ident.qos
        assert ident.entitlements_look_templated is True

    def _ident(self, dump):
        return SlurmBackend(RecordedRunner({"sacctmgr": (0, dump, "")})).load_identity()

    def test_the_default_qos_is_read_because_it_sets_the_ceilings(self):
        # The set a job runs under when it names none, so it is the set whose
        # MaxTRES decides whether the job is accepted -- and on a cluster whose
        # partitions declare no QOS it is the only one that does.
        ident = self._ident("basic||clay|clay\nstaff|||\n")
        assert ident.default_qos == "clay"

    def test_two_accounts_disagreeing_yields_no_default(self):
        # Different ceilings per account and no way to know which the job will
        # use: naming one would invent a limit for the other.
        assert self._ident("a||x,y|x\nb||x,y|y\n").default_qos is None

    def test_the_older_three_column_answer_still_parses(self):
        # The field is read positionally, so an install that does not know the
        # column must not disturb the three before it.
        ident = self._ident("acct-a||gn,wide\n")
        assert ident.default_qos is None
        assert set(ident.qos) == {"gn", "wide"}
        assert ident.accounts == ("acct-a",)


class TestAnEmptyAccountListIsTwoDifferentFacts:
    """"This site does not use accounts" and "you have none" look identical.

    `AccountingStorageEnforce=associations` means Slurm refuses a submission
    from a caller with no association row, so an empty list on such a cluster is
    not a shrug -- it is the reason nothing will run. Observed on a Slurm 24.11
    cluster where `sacctmgr show assoc` returned nothing for the caller: four
    partitions came back ACCOUNT_MISMATCH, two were refused by a site plugin,
    and the overview said "2 open to you (2 unconfirmed)" without naming the
    cause the control plane had already given.
    """

    ENFORCED = "AccountingStorageEnforce = associations,limits,qos\n"

    def _identity(self, config, assoc=""):
        return SlurmBackend(RecordedRunner({
            "sacctmgr": (0, assoc, ""),
            "scontrol show config": (0, config, ""),
        })).load_identity()

    def test_enforced_associations_are_recorded(self):
        ident = self._identity(self.ENFORCED)
        assert ident.accounts == ()
        assert ident.accounts_required is True

    @pytest.mark.parametrize("config", [
        "AccountingStorageEnforce = (null)\n",
        "AccountingStorageEnforce = limits,qos\n",     # enforced, but not this
        "ClusterName = x\n",                           # field absent
    ])
    def test_anything_else_claims_nothing(self, config):
        assert self._identity(config).accounts_required is False

    def test_an_unreadable_config_claims_nothing(self):
        # The conservative direction here is silence: asserting "nothing you
        # submit will run" from a failed query would be a fabrication.
        backend = SlurmBackend(RecordedRunner({"sacctmgr": (0, "", "")}))
        assert backend.load_identity().accounts_required is False

    def test_the_config_is_read_once_for_both_consumers(self):
        # Memory-consumability asks the same query. Two readings of one file is
        # two instants in one report, which is what the caching is for.
        backend = SlurmBackend(RecordedRunner({
            "sacctmgr": (0, "", ""),
            "scontrol show config": (0, self.ENFORCED + "SelectTypeParameters = CR_CORE\n", ""),
            "scontrol show node": (0, "NodeName=n1 CPUTot=8 State=IDLE Partitions=p\n", ""),
        }))
        backend.load_identity()
        backend.load_nodes()
        asked = [c for c in backend.runner.calls if "config" in " ".join(c)]
        assert len(asked) == 1, asked


class TestLocalGroupLookupIsAllOrNothing:
    """A half-collected group list is worse than none at all.

    The membership check downstream is tri-state: an EMPTY set reads as "cannot
    tell" and yields no verdict, while a NON-EMPTY one is taken as
    authoritative. So a lookup that gathered the supplementary groups and then
    failed on the primary had the tool assert "none of your groups are permitted
    here" from a list it knew to be incomplete -- a false denial, which hides a
    queue the caller can use.
    """

    def test_a_failure_on_the_primary_group_yields_nothing(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.slurm import _unix_groups

        class _G:
            gr_name = "supplementary"
            gr_mem = ["testuser"]

        monkeypatch.setattr(grp_mod, "getgrall", lambda: [_G()])
        import pwd as pwd_mod
        monkeypatch.setattr(pwd_mod, "getpwnam",
                            lambda _u: (_ for _ in ()).throw(KeyError("no such user")))
        # Not ("supplementary",): that would be a verdict from a partial read.
        assert _unix_groups("testuser") == ()

    def test_a_complete_lookup_returns_every_group(self, monkeypatch):
        import grp as grp_mod
        import pwd as pwd_mod

        from nodetop.backends.slurm import _unix_groups

        class _G:
            def __init__(self, name, mem):
                self.gr_name, self.gr_mem = name, mem

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: [_G("sup", ["testuser"]), _G("other", ["nobody"])])
        monkeypatch.setattr(pwd_mod, "getpwnam", lambda _u: type("P", (), {"pw_gid": 7})())
        monkeypatch.setattr(grp_mod, "getgrgid",
                            lambda _g: type("G", (), {"gr_name": "primary"})())
        assert _unix_groups("testuser") == ("primary", "sup")


class TestTheExitStatusOutranksTheText:
    """`sbatch --test-only` exits 0 if and only if the job would be accepted.

    That makes the return code the one authoritative, wording-independent
    signal, and `filter_verdict == "PASSED"` used to be sufficient for
    `allowed=True` on its own -- so a run that exited 1 while its site plugin
    printed PASSED was reported as allowed. It inverts the reliability ordering
    this module is built on: the filter's PASSED is exactly the claim documented
    as generous fiction.

    It was reachable because the core-refusal branch recognises only two message
    prefixes (`allocation failure:`, `Batch job submission failed:`). Any other
    refusal wording -- Slurm has many, and site plugins add their own -- fell
    through and was granted.
    """

    def test_a_nonzero_exit_is_not_an_acceptance(self):
        v = parse_probe("q", "a", 1, "sbatch: error: Verification: ***PASSED***\n", "")
        assert not v.allowed

    def test_it_is_unsettled_rather_than_a_durable_refusal(self):
        # We know it failed; we do not know why. Claiming a durable refusal
        # would assert more than the output established, and that is what gets a
        # usable partition hidden.
        v = parse_probe("q", "a", 1, "sbatch: error: Verification: ***PASSED***\n", "")
        assert not v.durable

    def test_a_predicted_start_does_not_outvote_a_failure(self):
        # Contradictory output -- a start time printed, then a non-zero exit.
        # The conservative reading is the one this module commits to everywhere.
        out = ("sbatch: Verification: ***PASSED***\n"
               "sbatch: Job 1 to start at 2026-08-23T10:00:00 using 1 processors\n")
        assert not parse_probe("q", "a", 1, out, "").allowed

    def test_a_genuine_acceptance_is_unaffected(self):
        out = ("sbatch: Verification: ***PASSED***\n"
               "sbatch: Job 1 to start at 2026-08-23T10:00:00 using 1 processors\n")
        v = parse_probe("q", "a", 0, out, "")
        assert v.allowed and v.confirmed
        assert v.predicted_start is not None

    def test_a_recognised_refusal_keeps_its_specific_category(self):
        # The rc veto sits after the core-refusal branch so a known message
        # still supplies the better category and reason.
        v = parse_probe(
            "q", "a", 1, "sbatch: error: Verification: ***PASSED***\n",
            "sbatch: error: Batch job submission failed: Invalid account or "
            "account/partition combination specified\n")
        assert not v.allowed and v.durable
        assert v.category == VerdictCategory.ACCOUNT_MISMATCH
        assert "site submit filter reported PASSED" in v.reason


class TestAnExhaustedAllocationIsNotADeniedOne:
    """Different problem, opposite remedy.

    Sites reject an empty allocation through the same channel as a permission
    failure -- the observed pair is `Reason: No sufficient SU allocations for a
    shared partition` with `allocation failure: Access/permission denied` on
    stderr -- so it fell through to ACCESS_DENIED and read as "you are not
    allowed here". The answer to one is to ask for service units; the answer to
    the other is to ask for access.
    """

    def test_su_exhaustion_is_a_quota(self):
        v = parse_probe(
            "wide", "acct", 1,
            "sbatch: error: Verification: ***REJECTED***\n"
            "sbatch: error: Reason: No sufficient SU allocations for a shared partition\n",
            "allocation failure: Access/permission denied\n")
        assert v.category == VerdictCategory.QUOTA_EXCEEDED
        assert v.durable and not v.allowed
        # And the operator's own wording survives, since that is what says why.
        assert "SU allocations" in v.reason

    def test_a_plain_permission_denial_is_still_access_denied(self):
        # The generic pattern must keep working; the quota entry sits before it
        # in a first-match-wins table, which is the whole mechanism.
        v = parse_probe("q", "a", 1, "",
                        "sbatch: error: Batch job submission failed: "
                        "Access/permission denied\n")
        assert v.category == VerdictCategory.ACCESS_DENIED


class TestEitherOutputShapeParsesTheSame:
    """`scontrol` emits one record per line or one field per line, one flag apart.

    This backend asks for `--oneliner` when listing nodes and not when listing
    partitions, and each parser understood only the shape its own command
    happened to produce. Both failed silently on the other, in opposite
    directions:

    * partitions were split on blank lines, so oneliner input became a *single*
      record whose field dictionary kept only the last value for each key --
      2000 partitions collapsing to 1, with nothing reported;
    * nodes were read one per line, so multi-line input gave a record per line
      and every node came back with 0 CPUs and no state: a cluster that appears
      to own no resources at all.

    Neither is reachable while the argv here is fixed, which is precisely why it
    would go unnoticed. The way in is a replayed snapshot recorded where
    `scontrol` behaved differently, a site wrapper, or a version whose output
    changed shape -- and a parser that cannot tell "no records" from "one merged
    record" has no way to complain.
    """

    NODES_ONELINE = (
        "NodeName=n1 CPUAlloc=4 CPUTot=8 State=MIXED RealMemory=1000 Gres=gpu:2\n"
        "NodeName=n2 CPUAlloc=0 CPUTot=8 State=IDLE RealMemory=1000 Gres=gpu:2\n"
    )
    NODES_MULTILINE = (
        "NodeName=n1 CPUAlloc=4 CPUTot=8\n"
        "   State=MIXED RealMemory=1000\n"
        "   Gres=gpu:2\n"
        "\n"
        "NodeName=n2 CPUAlloc=0 CPUTot=8\n"
        "   State=IDLE RealMemory=1000\n"
        "   Gres=gpu:2\n"
    )

    def test_nodes_parse_the_same_either_way(self):
        b = SlurmBackend(RecordedRunner({}))
        a = [(n.name, n.cpus_total, n.cpus_free, n.gpus_total, n.state_raw)
             for n in b.parse_nodes(self.NODES_ONELINE)]
        c = [(n.name, n.cpus_total, n.cpus_free, n.gpus_total, n.state_raw)
             for n in b.parse_nodes(self.NODES_MULTILINE)]
        assert a == c
        assert a == [("n1", 8, 4, 2, "MIXED"), ("n2", 8, 8, 2, "IDLE")]

    def test_a_multiline_node_record_is_not_truncated(self):
        # The specific old failure: fields after the first line were dropped, so
        # every node read as 0 CPUs with no state.
        nodes = SlurmBackend(RecordedRunner({})).parse_nodes(self.NODES_MULTILINE)
        assert all(n.cpus_total == 8 and n.state_raw for n in nodes)

    def test_oneliner_partitions_do_not_collapse_into_one(self):
        text = "\n".join(
            f"PartitionName=p{i} AllowGroups=ALL AllowAccounts=ALL "
            f"State=UP MaxTime=UNLIMITED Nodes=n-{i}" for i in range(5))
        queues = SlurmBackend(RecordedRunner({})).parse_queues(text)
        assert [q.name for q in queues] == [f"p{i}" for i in range(5)]

    def test_the_real_multiline_fixture_still_parses(self, slurm_backend):
        # The format actually in use must be unaffected by the hardening.
        assert len(slurm_backend.load_queues()) == 3

    def test_the_header_keyword_inside_free_text_does_not_split_a_record(self):
        # A node's Reason is operator-authored, so it can contain anything. The
        # header is required at a line start for this reason.
        text = ("NodeName=n1 CPUTot=8 State=DOWN RealMemory=1000 "
                "Reason=replacing NodeName=n2 per ticket\n")
        nodes = SlurmBackend(RecordedRunner({})).parse_nodes(text)
        assert [n.name for n in nodes] == ["n1"]

    def test_empty_and_junk_input_yield_nothing_rather_than_a_phantom(self):
        b = SlurmBackend(RecordedRunner({}))
        for junk in ("", "\n\n", "no records here", "   \n   "):
            assert b.parse_nodes(junk) == []
            assert b.parse_queues(junk) == []
