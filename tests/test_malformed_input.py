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
PARTITION = "PartitionName=p\n   State=UP AllowGroups=ALL\n   Nodes=n1\n   TotalNodes=1\n"


@pytest.fixture
def slurm():
    return SlurmBackend(RecordedRunner({}))


class TestTolerantParsing:
    """Cosmetic damage must not change the answer."""

    @pytest.mark.parametrize(
        "text,note",
        [
            (GOOD.replace("\n", "\r\n"), "CRLF from a Windows-side pipe"),
            (GOOD.rstrip("\n"), "no trailing newline"),
            ("\n\n" + GOOD + "\n\n", "blank lines around it"),
            ("   " + GOOD, "leading whitespace"),
            ("slurm_load_node: partial results\n" + GOOD, "a warning line first"),
        ],
    )
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

    @pytest.mark.parametrize(
        "text",
        [
            "NodeName=n1 CPUTot=32 CPUAlloc=8 RealMem\n",
            "NodeName=n1 CPUTot=3\n",
            "NodeName=n1\n",
        ],
    )
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
        for value in (node.cpus_total, node.cpus_alloc, node.memory_mb, node.memory_alloc_mb):
            assert value >= 0

    def test_kubernetes_cpu_is_clamped_too(self):
        from nodetop.backends.kubernetes import _quantity_to_cpu

        assert _quantity_to_cpu("-8") == 0
        assert _quantity_to_cpu("8") == 8

    @pytest.mark.parametrize(
        "parser",
        [
            "nodetop.backends.kubernetes._quantity_to_mb",
            "nodetop.backends.lsf._mem_to_mb",
            "nodetop.backends.pbs._mem_to_mb",
            "nodetop.backends.sge._mem_to_mb",
            "nodetop.backends.slurm._int",
            "nodetop.backends.base.count",
        ],
    )
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

    @pytest.mark.parametrize(
        "value,expected",
        [
            (4, 4),
            ("4", 4),
            (4.0, 4),
            ("4x", 4),
            ("48 cores", 48),
            ("unlimited", 0),
            ("n/a", 0),
            ("", 0),
            (None, 0),
            (-2, 0),
            ("-2", 0),  # a negative count is not one
            (True, 0),
            (False, 0),  # a flag is not a count
        ],
    )
    def test_count_reads_what_it_can_and_refuses_the_rest(self, value, expected):
        from nodetop.backends.base import count

        assert count(value) == expected

    @pytest.mark.parametrize("value", ["unlimited", "4x", -2, None, "n/a"])
    def test_a_pbs_node_survives_an_odd_resource_value(self, value):
        import json

        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        doc = {
            "nodes": {
                "n1": {
                    "state": "free",
                    "resources_available": {"ncpus": 48, "ngpus": value},
                    "resources_assigned": {},
                }
            }
        }
        nodes = PbsBackend(RecordedRunner({})).parse_nodes_json(json.dumps(doc))
        assert [n.name for n in nodes] == ["n1"]
        assert nodes[0].cpus_total == 48  # the good field still lands
        assert nodes[0].gpus_total >= 0

    def test_a_pbs_text_node_survives_it_too(self):
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        nodes = PbsBackend(RecordedRunner({})).parse_nodes_text(
            "n1\n     state = free\n"
            "     resources_available.ncpus = 48\n"
            "     resources_available.ngpus = unlimited\n"
        )
        assert [(n.name, n.cpus_total, n.gpus_total) for n in nodes] == [("n1", 48, 0)]


class TestDuplicateRecords:
    def test_a_node_listed_twice_is_counted_once(self):
        # Otherwise every count and every gauge in the report is inflated.
        backend = SlurmBackend(
            RecordedRunner(
                {
                    "scontrol show node": (0, GOOD + GOOD.replace("CPUTot=32", "CPUTot=64"), ""),
                    "scontrol show partition": (0, PARTITION, ""),
                    "show qos": (0, "", ""),
                    "show assoc": (0, "", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        assert len(cluster.nodes) == 1
        assert cluster.summary()["nodes"] == 1

    def test_the_first_record_wins(self):
        backend = SlurmBackend(
            RecordedRunner(
                {
                    "scontrol show node": (0, GOOD + GOOD.replace("CPUTot=32", "CPUTot=64"), ""),
                    "scontrol show partition": (0, PARTITION, ""),
                    "show qos": (0, "", ""),
                    "show assoc": (0, "", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes[0].cpus_total == 32

    def test_a_queue_member_listed_twice_is_counted_once(self):
        # The same inflation as above, one level down and missed there: the
        # cluster node list was deduplicated but a *queue's* membership list was
        # not, so `Nodes=n1,n1` resolved the one record twice and every
        # per-queue gauge doubled. A queue cannot hold more than the cluster
        # holds.
        backend = SlurmBackend(
            RecordedRunner(
                {
                    "scontrol show node": (0, GOOD, ""),
                    "scontrol show partition": (
                        0,
                        "PartitionName=p\n   State=UP AllowGroups=ALL\n"
                        "   Nodes=n1,n1\n   TotalNodes=1\n",
                        "",
                    ),
                    "show qos": (0, "", ""),
                    "show assoc": (0, "", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        q = cluster.queues["p"]
        assert [n.name for n in q.nodes] == ["n1"]
        assert q.cpus_total == 32
        assert q.gpus_total == 4
        assert len(q.schedulable_nodes) == 1

    def test_distinct_members_survive_and_a_phantom_is_still_reported(self):
        # Control for the test above: deduplication must not swallow genuinely
        # different members, reorder them, or hide a member that does not
        # resolve. This cluster's `test` partition really does name three
        # decommissioned nodes, and "+N claimed but unresolved" is how the
        # report says so.
        backend = SlurmBackend(
            RecordedRunner(
                {
                    "scontrol show node": (
                        0,
                        GOOD + GOOD.replace("NodeName=n1", "NodeName=n2"),
                        "",
                    ),
                    "scontrol show partition": (
                        0,
                        "PartitionName=p\n   State=UP AllowGroups=ALL\n"
                        "   Nodes=n1,n2,ghost\n   TotalNodes=3\n",
                        "",
                    ),
                    "show qos": (0, "", ""),
                    "show assoc": (0, "", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        q = cluster.queues["p"]
        assert [n.name for n in q.nodes] == ["n1", "n2"]
        assert q.cpus_total == 64
        assert q.unresolved_nodes == 1


class TestUnparseableOutputIsReported:
    """A parser that gives up must not look like an empty cluster."""

    @pytest.mark.parametrize(
        "payload",
        [
            '{"items":[{"metadata":{"name"',  # truncated mid-object
            "error: connection refused",  # not JSON at all
            "",  # nothing
            '{"items": {"n1": {}}}',  # items is not a list
        ],
    )
    def test_kubernetes_records_the_failure(self, payload):
        from nodetop.backends.kubernetes import KubernetesBackend

        backend = KubernetesBackend(
            RecordedRunner(
                {
                    "get nodes": (0, payload, ""),
                    "get pods": (0, "", ""),
                    "get namespaces": (0, '{"items":[]}', ""),
                    "get resourcequota": (0, "", ""),
                    "auth whoami": (0, "{}", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes == []
        # The report says it is partial rather than claiming an empty cluster.
        assert "nodes" in cluster.errors

    def test_valid_json_with_no_items_is_genuinely_empty_not_an_error(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        backend = KubernetesBackend(
            RecordedRunner(
                {
                    "get nodes": (0, '{"kind":"List"}', ""),
                    "get pods": (0, "", ""),
                    "get namespaces": (0, '{"items":[]}', ""),
                    "get resourcequota": (0, "", ""),
                    "auth whoami": (0, "{}", ""),
                }
            )
        )
        cluster = Cluster.load(backend, with_free_times=False)
        assert cluster.nodes == []
        assert "nodes" not in cluster.errors


class TestASchedulerThatAnswersWithGarbageIsNotSuccess:
    """A client that exits 0 and returns unreadable output records NO error.

    `_reject_broken_snapshot` exists so that "we could not ask" is never rendered
    as "there is nothing there" — but it opened with `if not cluster.errors:
    return 0`, and a scheduler that *succeeds* while answering with something this
    parser cannot read reports no error at all. Measured against clients emitting
    one field per line, extra columns, non-numeric CPUs, negative counts, absurd
    counts, and a header with no rows:

        nodes=0 queues=0 errors=[]      every time

    So the shortcut fired, `status` printed "no nodes -- wrong backend, or the
    control plane is down", and it exited **0**. A script reading `$?` saw success
    from the one sentence that says the tool cannot see the cluster.

    This is the misdetected-backend case the surrounding comment already describes
    — forcing a backend whose client is somebody else's shim — reached by the route
    where the shim answers instead of failing. Version skew reaches it too.
    """

    MODES = {
        "one field per line": "onefield\nalso-one\n",
        "far too many columns": "|".join("abcdefghijklmnopqrstuvwxyz") + "\n",
        "non-numeric cpus": "node1|IDLE|banana|melon|kiwi|plum|cherry|fig\n",
        "negative counts": "node1|IDLE|-8|-16384|-1|-1|none|j\n",
        "absurd counts": "node1|IDLE|" + "9" * 20 + "|" + "9" * 20 + "|0|0|none|j\n",
        "a header and no rows": "NODELIST|STATE|CPUS\n",
        "nothing but blank lines": "\n\n\n",
    }

    @staticmethod
    def _run(tmp_path, payload, argv=("status",)):
        import os
        import pathlib
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        for name in ("sinfo", "squeue", "scontrol", "sacct", "sstat", "sacctmgr", "sbatch"):
            stub = bindir / name
            stub.write_text(f"#!/usr/bin/env python3\nimport sys\nsys.stdout.write({payload!r})\n")
            stub.chmod(0o755)
        return subprocess.run(
            [sys.executable, "-m", "nodetop", *argv],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                "PATH": f"{bindir}:/usr/bin:/bin",
                "HOME": os.environ.get("HOME", "/tmp"),
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "110",
            },
        )

    @pytest.mark.parametrize("label", sorted(MODES))
    def test_it_exits_three_and_says_which_cause(self, label, tmp_path):
        done = self._run(tmp_path, self.MODES[label])
        assert "Traceback" not in done.stderr, done.stderr[-300:]
        assert done.returncode == 3, (
            f"{label!r} produced rc={done.returncode}: a report the control plane "
            f"never supplied\n{done.stdout[-300:]}"
        )
        # And the RIGHT cause: nothing failed, so "every query failed" would lie.
        assert "the queries answered" in done.stderr, done.stderr[-300:]
        assert "every query failed" not in done.stderr, done.stderr[-300:]
        assert "not an empty cluster" in done.stderr

    @pytest.mark.parametrize("command", ["queues", "nodes", "health"])
    def test_the_other_zeroed_views_agree(self, command, tmp_path):
        """The views the docstring above names as having printed confident zeros."""
        done = self._run(tmp_path, self.MODES["non-numeric cpus"], (command,))
        assert done.returncode == 3, done.stdout[-200:]
        assert "Traceback" not in done.stderr

    def test_a_genuinely_failing_client_still_says_so(self, tmp_path):
        """The control: the original message must survive for the original cause.

        "every query failed" is right when they did, and telling that reader to
        check `--backend` would send them somewhere useless.
        """
        import os
        import pathlib
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        for name in ("sinfo", "squeue", "scontrol", "sacct", "sstat", "sacctmgr"):
            stub = bindir / name
            stub.write_text("#!/bin/bash\necho 'slurm_load: Unable to contact' >&2\nexit 1\n")
            stub.chmod(0o755)
        done = subprocess.run(
            [sys.executable, "-m", "nodetop", "status"],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                "PATH": f"{bindir}:/usr/bin:/bin",
                "HOME": os.environ.get("HOME", "/tmp"),
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "110",
            },
        )
        assert done.returncode == 3, done.stdout[-200:]
        assert "every query failed" in done.stderr, done.stderr[-300:]
