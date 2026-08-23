"""The same fact, reported by different commands, must agree.

Each view computes its own totals -- `status` from `Cluster.summary`, `gpus`
from its own per-model grouping, `nodes` row by row -- so they can and do drift.
This is where `Cluster.summary` was found counting free accelerators on nodes
reachable only through a dead queue while `Queue.effective_free_gpus` correctly
reported zero for the same hardware: two views, two answers, no test that
compared them.

Run against the recorded fixture, so the cluster cannot move between calls the
way it does live.
"""

from __future__ import annotations

import json

import pytest

import nodetop.cli as cli
from nodetop.cli import build_parser
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


@pytest.fixture
def stranded(cluster):
    """The recorded cluster plus an idle GPU node behind a DOWN partition.

    Without one, every view computes the same *wrong* number and they agree
    with each other -- a consistency test catches divergence, not shared error.
    Reverting the reachability fix left this file entirely green until the
    fixture contained hardware the bug could strand.
    """
    import dataclasses

    from nodetop.core.model import Node, Queue

    node = Node(name="stranded1", state_raw="IDLE", cpus_total=16,
                memory_mb=64000, gpus_total=4, queues=("morgue",))
    morgue = Queue(name="morgue", state_raw="DOWN", enabled=False,
                   node_names=("stranded1",), declared_nodes=1, nodes=[node])
    return dataclasses.replace(
        cluster,
        nodes=[*cluster.nodes, node],
        queues={**cluster.queues, "morgue": morgue},
    )


@pytest.fixture
def views(stranded, capsys):
    """Every JSON view of one cluster, captured once."""
    cluster = stranded
    out = {}
    for key, fn, argv in (
        ("status", cli.cmd_status, ["--json", "status"]),
        ("gpus", cli.cmd_accelerators, ["--json", "accelerators", "--all"]),
        ("nodes", cli.cmd_nodes, ["--json", "nodes", "--all"]),
        ("queues", cli.cmd_queues, ["--json", "queues", "--all"]),
        ("health", cli.cmd_health, ["--json", "health"]),
    ):
        fn(cluster, _args(argv), PLAIN)
        out[key] = json.loads(capsys.readouterr().out)
    return out


class TestTheViewsAgree:
    def test_installed_accelerators(self, views):
        assert (views["status"]["accelerators_total"]
                == views["gpus"]["accelerators_installed"]
                == sum(n["accelerators"][1] for n in views["nodes"]))

    def test_free_accelerators(self, views):
        # The one that was wrong: summary counted hardware behind dead queues.
        assert (views["status"]["accelerators_free"]
                == sum(m["free"] for m in views["gpus"]["models"].values()))

    def test_node_count(self, views):
        assert views["status"]["nodes"] == len(views["nodes"])

    def test_accelerator_node_count(self, views):
        assert (views["status"]["accelerator_nodes"]
                == sum(m["nodes"] for m in views["gpus"]["models"].values()))

    def test_queue_count(self, views):
        assert views["status"]["queues"] == len(views["queues"])

    def test_unusable_queues(self, views):
        assert (len(views["status"]["unusable_queues"])
                == sum(1 for q in views["queues"] if not q["usable"]))

    def test_unschedulable_nodes(self, views):
        assert (views["status"]["unschedulable_nodes"]
                == len(views["health"]["unschedulable"])
                == sum(1 for n in views["nodes"] if not n["schedulable"]))

    def test_degraded_nodes(self, views):
        assert (len(views["status"]["degraded_nodes"])
                == len(views["health"]["degraded"]))

    def test_phantom_capacity_matches_the_unusable_queues(self, views):
        phantom = views["status"]["phantom_capacity"]
        unusable = set(views["status"]["unusable_queues"])
        assert set(phantom) <= unusable, "phantom capacity on a usable queue"

    def test_no_free_count_exceeds_its_total(self, views):
        # A free figure above the installed one means two different
        # populations were counted -- exactly the dead-queue bug's signature.
        assert (views["status"]["accelerators_free"]
                <= views["status"]["accelerators_total"])
        for model, m in views["gpus"]["models"].items():
            assert m["free"] <= m["installed"], model
        for q in views["queues"]:
            assert q["effective_free_accelerators"] <= q["accelerators_total"], q["name"]


class TestTheHeaderDescribesTheTableBeneathIt:
    """One panel, one population.

    The header reported cluster-wide totals while the table below it was
    filtered to the caller's slice, so `status` read "358 GPUs, 117 free" above
    five partitions holding 230 of them. Two populations in one box, which is
    the same defect `accelerators` had when it announced a total nobody could
    use.
    """

    @staticmethod
    def _build():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Identity, Node, Queue

        nodes, queues = [], {}
        # 2 nodes / 8 GPUs yours; 3 nodes / 12 GPUs not.
        for name, count, gpus, allow in (("mine", 2, 4, ("mine",)),
                                         ("theirs", 3, 4, ("other",))):
            mine = [
                Node(name=f"{name}{i}", state_raw="IDLE", cpus_total=8,
                     memory_mb=16000, gpus_total=gpus, queues=(name,))
                for i in range(count)
            ]
            nodes += mine
            queues[name] = Queue(name=name, declared_nodes=count, nodes=mine,
                                 node_names=tuple(n.name for n in mine),
                                 allow_accounts=allow)
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues,
                       identity=Identity(user="me", accounts=("mine",),
                                         qos=("x",)))

    def _header(self, capsys):
        cli.cmd_status(self._build(), _args(["status"]), PLAIN)
        out = capsys.readouterr().out
        return next(ln for ln in out.splitlines() if "node" in ln and "up" in ln)

    def test_it_counts_only_the_nodes_you_can_use(self, capsys):
        assert "2 of 5 nodes" in self._header(capsys)

    def test_it_counts_only_the_gpus_you_can_use(self, capsys):
        assert "8 of 20 GPUs" in self._header(capsys)

    def test_the_free_figure_is_scoped_too(self, capsys):
        # Every node in the fixture is idle, so cluster-wide free is 20 and
        # yours is 8. Without this, counting free accelerators over the whole
        # cluster went unnoticed -- the installed figure was scoped and the
        # free one beside it was not.
        assert "8 free" in self._header(capsys)

    def test_it_keeps_the_cluster_total_as_context(self, capsys):
        # Your slice is the subject; the cluster size is the qualifier.
        head = self._header(capsys)
        assert " of 5 " in head and " of 20 " in head

    def test_all_reports_the_whole_cluster_without_the_qualifier(self, capsys):
        cli.cmd_status(self._build(), _args(["status", "--all"]), PLAIN)
        out = capsys.readouterr().out
        head = next(ln for ln in out.splitlines() if "node" in ln and "up" in ln)
        assert "5 nodes" in head
        assert " of 5 " not in head   # nothing is hidden, so nothing to qualify

    def test_the_header_matches_the_rows(self, capsys):
        # The invariant, not a fixed string: whatever is shown, the header
        # counts exactly those partitions' nodes.
        cluster = self._build()
        cli.cmd_status(cluster, _args(["status"]), PLAIN)
        out = capsys.readouterr().out
        shown = [q for q in cluster.queues.values() if q.name in out]
        assert shown, "no partition rendered"
        expected = len({n.name for q in shown for n in q.nodes})
        head = next(ln for ln in out.splitlines() if "node" in ln and "up" in ln)
        assert f"{expected} of" in head or f"{expected} node" in head
