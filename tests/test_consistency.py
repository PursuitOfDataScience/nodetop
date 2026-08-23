"""Do the commands agree with each other, and with a replay of themselves?

Three properties that no single-command test can express, each of which has
already been violated once:

* **Cross-command agreement.** `status` reported "220 of 358 GPUs" while `gpus`
  reported "230 of 358" from the same snapshot -- two numbers for one fact,
  because the two views scoped their denominator differently. A reader cannot
  reconcile that, and will assume one of them is broken.
* **Replay fidelity.** The whole post-mortem feature rests on a snapshot
  reproducing the report. Nothing checked that it did.
* **Ordering.** `where` is the command you act on, so an inversion in the sort
  key sends someone at their second-best option.
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from nodetop.cli import (
    _verdict_label,
    build_parser,
    cmd_accelerators,
    cmd_exclude,
    cmd_health,
    cmd_nodes,
    cmd_queues,
    cmd_zoom,
)
from nodetop.core.cluster import Cluster
from nodetop.core.fit import rank
from nodetop.core.model import JobShape
from nodetop.hostlist import expand
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _json(cluster, fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(cluster, build_parser().parse_args(argv), PLAIN)
    return json.loads(buf.getvalue())


class TestEveryCommandAgreesWithEveryOther:
    """One snapshot, one set of facts, however it is sliced."""

    def test_the_unschedulable_count_is_the_same_everywhere(self, cluster):
        nodes = _json(cluster, cmd_nodes, ["nodes", "--all", "--json"])
        health = _json(cluster, cmd_health, ["health", "--json"])
        excl = _json(cluster, cmd_exclude, ["exclude", "--unschedulable", "--json"])
        out = sum(1 for n in nodes if not n["schedulable"])
        assert len(health["unschedulable"]) == out
        assert excl["count"] == out

    def test_the_degraded_count_is_the_same_everywhere(self, cluster):
        nodes = _json(cluster, cmd_nodes, ["nodes", "--all", "--json"])
        health = _json(cluster, cmd_health, ["health", "--json"])
        assert len(health["degraded"]) == sum(1 for n in nodes if n["degraded"])

    def test_the_accelerator_total_is_the_same_everywhere(self, cluster):
        nodes = _json(cluster, cmd_nodes, ["nodes", "--all", "--json"])
        gpus = _json(cluster, cmd_accelerators, ["gpus", "--all", "--json"])
        assert gpus["accelerators_installed"] == sum(
            n["accelerators"][1] for n in nodes)

    def test_the_per_model_counts_sum_to_the_total(self, cluster):
        gpus = _json(cluster, cmd_accelerators, ["gpus", "--all", "--json"])
        named = sum(m["installed"] for m in gpus["models"].values())
        assert named + gpus["accelerators_unidentifiable"] == \
            gpus["accelerators_installed"]

    def test_per_queue_figures_match_the_nodes_they_are_built_from(self, cluster):
        by_name = {q["name"]: q for q in
                   _json(cluster, cmd_queues, ["queues", "--all", "--json"])}
        for name, queue in cluster.queues.items():
            js = by_name.get(name)
            if js is None:
                continue
            assert js["nodes"] == len(queue.nodes), name
            assert js["accelerators_total"] == sum(
                n.gpus_total for n in queue.nodes), name
            assert js["effective_free_nodes"] == queue.effective_free_nodes, name

    def test_the_exclusion_list_expands_back_to_exactly_that_set(self, cluster):
        # The output is fed to `sbatch --exclude`, so a nodelist that expands to
        # a different set than it was built from silently excludes the wrong
        # machines -- or, worse, none of them.
        nodes = _json(cluster, cmd_nodes, ["nodes", "--all", "--json"])
        excl = _json(cluster, cmd_exclude, ["exclude", "--unschedulable", "--json"])
        assert set(expand(excl["nodelist"])) == {
            n["name"] for n in nodes if not n["schedulable"]}


class TestZoomAgreesWithTheListingItZoomsOutTo:
    """A zoom view whose figures disagree with `nodes -q` is worse than none.

    The header comes from `_queues_detail` and the rows from `_node_rows`, both
    shared with the commands they came from -- but sharing a builder does not by
    itself guarantee the same *selection*, which is what this pins.
    """

    def test_the_same_nodes_are_listed(self, cluster):
        for name in cluster.queues:
            zoomed = _json(cluster, cmd_zoom, ["zoom", name, "--json"])
            listed = _json(cluster, cmd_nodes,
                           ["nodes", "-q", name, "--all", "--json"])
            assert {m["name"] for m in zoomed["members"]} == \
                {n["name"] for n in listed}, name

    def test_the_same_figures_are_reported(self, cluster):
        for name in cluster.queues:
            zoomed = _json(cluster, cmd_zoom, ["zoom", name, "--json"])
            listed = {n["name"]: n for n in _json(
                cluster, cmd_nodes, ["nodes", "-q", name, "--all", "--json"])}
            for member in zoomed["members"]:
                other = listed[member["name"]]
                assert member["cpus"] == other["cpus"], member["name"]
                assert member["accelerators"] == other["accelerators"], member["name"]

    def test_the_counts_add_up(self, cluster):
        for name, queue in cluster.queues.items():
            zoomed = _json(cluster, cmd_zoom, ["zoom", name, "--json"])
            assert zoomed["nodes"] == len(queue.nodes), name
            # with_room and wholly_idle are both subsets of the schedulable set,
            # and every wholly idle node trivially has room.
            assert zoomed["wholly_idle"] <= zoomed["with_room"] <= zoomed["nodes"]
            assert zoomed["unschedulable"] + zoomed["with_room"] <= zoomed["nodes"]

    def test_free_capacity_excludes_unschedulable_nodes(self, cluster):
        # The whole point of the phantom-capacity rule, at the summary level.
        for name, queue in cluster.queues.items():
            zoomed = _json(cluster, cmd_zoom, ["zoom", name, "--json"])
            assert zoomed["cpus"][0] == sum(
                n.cpus_free for n in queue.nodes if n.schedulable), name
            assert zoomed["accelerators"][0] == sum(
                n.gpus_free for n in queue.nodes if n.schedulable), name


class TestAReplayReproducesTheReport:
    """A snapshot that does not reproduce the report is not a post-mortem."""

    @staticmethod
    def _pair(slurm_backend):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.runner import CapturingRunner, RecordedRunner

        capture = CapturingRunner(slurm_backend.runner)
        live = Cluster.load(SlurmBackend(capture))
        replayed = Cluster.load(
            SlurmBackend(RecordedRunner(dict(capture.captured))), replayed=True)
        return live, replayed

    @pytest.mark.parametrize("fn,argv", [
        (cmd_nodes, ["nodes", "--all", "--json"]),
        (cmd_queues, ["queues", "--all", "--json"]),
        (cmd_health, ["health", "--json"]),
        (cmd_accelerators, ["gpus", "--all", "--json"]),
    ])
    def test_it_is_identical(self, slurm_backend, fn, argv):
        live, replayed = self._pair(slurm_backend)
        assert _json(live, fn, argv) == _json(replayed, fn, argv)


class TestTheRankingHasNoInversions:
    """`where` is the command you act on, so the sort key has to hold."""

    SHAPES = [
        JobShape(nodes=1, cpus_per_task=1),
        JobShape(nodes=1, cpus_per_task=48),
        JobShape(nodes=1, gpus_per_node=1, cpus_per_task=1),
        JobShape(nodes=1, gpus_per_node=4, gpu_memory_gb=40, cpus_per_task=1),
        JobShape(nodes=4, gpus_per_node=4, cpus_per_task=1),
        JobShape(nodes=1, gpus_per_node=1, requires=("bf16",), cpus_per_task=1),
    ]

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.describe())
    def test_reachability_tiers_never_interleave(self, cluster, shape):
        places = rank(cluster, shape, include_unusable=True)
        tiers = [0 if p.runnable_now else 1 if p.reachable else 2 for p in places]
        assert tiers == sorted(tiers)

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.describe())
    def test_room_is_non_increasing_among_the_runnable(self, cluster, shape):
        room = [p.nodes_available for p in
                rank(cluster, shape, include_unusable=True) if p.runnable_now]
        assert room == sorted(room, reverse=True)

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.describe())
    def test_a_run_now_row_never_sits_below_another_label(self, cluster, shape):
        # No skip when nothing runs now: the property holds trivially there,
        # and a skipped parametrisation reads as coverage it is not.
        labels = [_verdict_label(p) for p in
                  rank(cluster, shape, include_unusable=True)]
        runnable = [i for i, x in enumerate(labels) if x == "RUN NOW"]
        others = [i for i, x in enumerate(labels) if x != "RUN NOW"]
        assert all(i > max(runnable) for i in others) if runnable else True

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.describe())
    def test_the_verdict_never_contradicts_itself(self, cluster, shape):
        for p in rank(cluster, shape, include_unusable=True):
            assert not (p.runnable_now and p.fatal_blockers), p.queue
            assert not (p.confirmed and not p.reachable), p.queue
