"""Malformed scheduler output.

A parser that raises is fine: the failure is recorded in `Cluster.errors` and
the report says it is partial. The dangerous case is a parser that *succeeds*
on garbage and hands back a plausible wrong answer -- and two of those were
real. A truncated Slurm record carried no state at all, which made it read as
schedulable AND empty, the most attractive thing in the cluster to a placement
search. And a garbage numeric produced `CPUAlloc=-5` against `CPUTot=0`, which
reported five free CPUs that did not exist.
"""

from __future__ import annotations

import pytest

from nodetop.backends.slurm import SlurmBackend
from nodetop.core.cluster import Cluster
from nodetop.runner import RecordedRunner

GOOD = (
    "NodeName=n1 CPUTot=32 CPUAlloc=8 RealMemory=256000 AllocMem=1024 "
    "Gres=gpu:4 AvailableFeatures=a100 State=MIXED Partitions=p\n"
)
PARTITION = (
    "PartitionName=p\n   State=UP AllowGroups=ALL\n   Nodes=n1\n   TotalNodes=1\n"
)


@pytest.fixture
def slurm():
    return SlurmBackend(RecordedRunner({}))


class TestTolerantParsing:
    """Cosmetic damage must not change the answer."""

    @pytest.mark.parametrize("text,note", [
        (GOOD.replace("\n", "\r\n"), "CRLF from a Windows-side pipe"),
        (GOOD.rstrip("\n"), "no trailing newline"),
        ("\n\n" + GOOD + "\n\n", "blank lines around it"),
        ("   " + GOOD, "leading whitespace"),
        ("slurm_load_node: partial results\n" + GOOD, "a warning line first"),
    ])
    def test_the_node_still_parses_correctly(self, slurm, text, note):
        nodes = slurm.parse_nodes(text)
        assert len(nodes) == 1, note
        node = nodes[0]
        assert node.name == "n1", note
        assert (node.cpus_alloc, node.cpus_total) == (8, 32), note
        assert node.memory_mb == 256000, note
        assert node.schedulable is True, note


class TestTruncatedRecords:
    """A partial read is not a healthy node."""

    @pytest.mark.parametrize("text", [
        "NodeName=n1 CPUTot=32 CPUAlloc=8 RealMem\n",
        "NodeName=n1 CPUTot=3\n",
        "NodeName=n1\n",
    ])
    def test_a_record_with_no_state_is_not_schedulable(self, slurm, text):
        # With no conditions it would read as schedulable and empty, which is
        # the most attractive thing in the cluster to a placement search.
        node = slurm.parse_nodes(text)[0]
        assert node.schedulable is False
        assert "UNKNOWN" in node.conditions

    def test_it_says_why(self, slurm):
        node = slurm.parse_nodes("NodeName=n1 CPUTot=32\n")[0]
        assert "truncated" in node.reason

    def test_it_claims_no_resources(self, slurm):
        # Half a record is not evidence of capacity.
        node = slurm.parse_nodes("NodeName=n1 CPUTot=32 RealMem\n")[0]
        assert node.cpus_free == 0
        assert node.gpus_free == 0

    def test_a_record_with_no_name_is_dropped(self, slurm):
        assert slurm.parse_nodes("CPUTot=32 State=IDLE\n") == []


class TestNoNegativeResources:
    """A count below zero is meaningless and actively harmful."""

    def test_a_negative_allocation_invents_no_free_capacity(self, slurm):
        # cpus_free is total - alloc, so alloc=-5 against total=0 would report
        # five free CPUs that do not exist.
        node = slurm.parse_nodes("NodeName=n1 CPUTot=abc CPUAlloc=-5 State=IDLE\n")[0]
        assert node.cpus_alloc == 0
        assert node.cpus_free == 0

    @pytest.mark.parametrize("field", ["CPUTot", "CPUAlloc", "RealMemory", "AllocMem"])
    def test_no_field_can_go_negative(self, slurm, field):
        node = slurm.parse_nodes(f"NodeName=n1 {field}=-99 State=IDLE\n")[0]
        for value in (node.cpus_total, node.cpus_alloc,
                      node.memory_mb, node.memory_alloc_mb):
            assert value >= 0

    def test_kubernetes_cpu_is_clamped_too(self):
        from nodetop.backends.kubernetes import _quantity_to_cpu

        assert _quantity_to_cpu("-8") == 0
        assert _quantity_to_cpu("8") == 8

    @pytest.mark.parametrize("parser", [
        "nodetop.backends.kubernetes._quantity_to_mb",
        "nodetop.backends.lsf._mem_to_mb",
        "nodetop.backends.pbs._mem_to_mb",
        "nodetop.backends.sge._mem_to_mb",
        "nodetop.backends.slurm._int",
        "nodetop.backends.base.count",
    ])
    def test_every_numeric_parser_clamps(self, parser):
        import importlib

        module, name = parser.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        for probe in ["-8", "-8G", "-99999"]:
            got = fn(probe)
            assert got is None or got >= 0, f"{parser}({probe!r}) = {got}"


class TestACountIsNotACrash:
    """A field that is not a number must not empty the node list.

    PBS reports ``resources_available.ngpus = unlimited`` for an uncapped
    resource, and site scripts emit ``4x`` and ``8gb``. `int()` on any of those
    raised through the node parser, so one odd field on one node took down the
    whole listing -- and an empty listing is reported as "wrong backend, or the
    control plane is down", a misdiagnosis rather than a gap.
    """

    @pytest.mark.parametrize("value,expected", [
        (4, 4), ("4", 4), (4.0, 4), ("4x", 4), ("48 cores", 48),
        ("unlimited", 0), ("n/a", 0), ("", 0), (None, 0),
        (-2, 0), ("-2", 0),                      # a negative count is not one
        (True, 0), (False, 0),                   # a flag is not a count
    ])
    def test_count_reads_what_it_can_and_refuses_the_rest(self, value, expected):
        from nodetop.backends.base import count

        assert count(value) == expected

    @pytest.mark.parametrize("value", ["unlimited", "4x", -2, None, "n/a"])
    def test_a_pbs_node_survives_an_odd_resource_value(self, value):
        import json

        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        doc = {"nodes": {"n1": {"state": "free",
               "resources_available": {"ncpus": 48, "ngpus": value},
               "resources_assigned": {}}}}
        nodes = PbsBackend(RecordedRunner({})).parse_nodes_json(json.dumps(doc))
        assert [n.name for n in nodes] == ["n1"]
        assert nodes[0].cpus_total == 48          # the good field still lands
        assert nodes[0].gpus_total >= 0

    def test_a_pbs_text_node_survives_it_too(self):
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        nodes = PbsBackend(RecordedRunner({})).parse_nodes_text(
            "n1\n     state = free\n"
            "     resources_available.ncpus = 48\n"
            "     resources_available.ngpus = unlimited\n")
        assert [(n.name, n.cpus_total, n.gpus_total) for n in nodes] == [("n1", 48, 0)]


class TestDuplicateRecords:
    def test_a_node_listed_twice_is_counted_once(self):
        # Otherwise every count and every gauge in the report is inflated.
        backend = SlurmBackend(RecordedRunner({
            "scontrol show node": (0, GOOD + GOOD.replace("CPUTot=32", "CPUTot=64"), ""),
            "scontrol show partition": (0, PARTITION, ""),
            "show qos": (0, "", ""), "show assoc": (0, "", ""), "squeue": (0, "", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        assert len(cluster.nodes) == 1
        assert cluster.summary()["nodes"] == 1

    def test_the_first_record_wins(self):
        backend = SlurmBackend(RecordedRunner({
            "scontrol show node": (0, GOOD + GOOD.replace("CPUTot=32", "CPUTot=64"), ""),
            "scontrol show partition": (0, PARTITION, ""),
            "show qos": (0, "", ""), "show assoc": (0, "", ""), "squeue": (0, "", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes[0].cpus_total == 32


class TestUnparseableOutputIsReported:
    """A parser that gives up must not look like an empty cluster."""

    @pytest.mark.parametrize("payload", [
        '{"items":[{"metadata":{"name"',        # truncated mid-object
        "error: connection refused",            # not JSON at all
        "",                                     # nothing
        '{"items": {"n1": {}}}',                # items is not a list
    ])
    def test_kubernetes_records_the_failure(self, payload):
        from nodetop.backends.kubernetes import KubernetesBackend

        backend = KubernetesBackend(RecordedRunner({
            "get nodes": (0, payload, ""),
            "get pods": (0, "", ""),
            "get namespaces": (0, '{"items":[]}', ""),
            "get resourcequota": (0, "", ""),
            "auth whoami": (0, "{}", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes == []
        # The report says it is partial rather than claiming an empty cluster.
        assert "nodes" in cluster.errors

    def test_valid_json_with_no_items_is_genuinely_empty_not_an_error(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        backend = KubernetesBackend(RecordedRunner({
            "get nodes": (0, '{"kind":"List"}', ""),
            "get pods": (0, "", ""),
            "get namespaces": (0, '{"items":[]}', ""),
            "get resourcequota": (0, "", ""),
            "auth whoami": (0, "{}", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes == []
        assert "nodes" not in cluster.errors
