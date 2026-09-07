"""The GPU inventory's `--json` carries the denominator its headline prints.

`cmd_accelerators` filters its node list to the partitions the caller's account
is on, and the entitlement helper states what that costs in its own words --
"358 accelerators, 230" (`_entitled_queues`), and again at the filter itself:
"'358 GPUs' is true of the cluster and false of what the reader can use; 230 of
them are in partitions this account is on."

The rendered surface honours that: it prints `230 GPUs of 358 on the cluster`.
The payload published only the 230, under the name `accelerators_installed`, so
a consumer could not tell a whole-cluster inventory from the slice of it this
account can reach -- the one decision the phrase exists to make. Same shape as
the `health --json` gap, where the payload listed only the unhealthy nodes and
the command's own headline figures were not derivable from it at all.
"""

from __future__ import annotations

import json
import re

import pytest

from nodetop.cli import build_parser, cmd_accelerators
from nodetop.core.cluster import Cluster
from nodetop.core.model import Identity, Node, Queue
from nodetop.render import Style

PLAIN = Style(enabled=False)


def _args(argv: list[str]):
    return build_parser().parse_args(argv)

GPUS_PER_NODE = 4


def _cluster(spec: list[tuple[str, int, bool]]) -> Cluster:
    """GPU nodes in `(partition, node count, my account is allowed)` groups."""
    nodes: list[Node] = []
    queues: dict[str, Queue] = {}
    for name, count, mine_allowed in spec:
        group = [
            Node(name=f"n-{name}-{i}", state_raw="IDLE", cpus_total=16,
                 memory_mb=64000, gpus_total=GPUS_PER_NODE, queues=(name,))
            for i in range(count)
        ]
        nodes += group
        queues[name] = Queue(
            name=name, node_names=tuple(n.name for n in group),
            declared_nodes=count, nodes=group,
            # An empty allowlist is open to everyone, so the excluded partition
            # has to name an account that is not one of ours.
            allow_accounts=() if mine_allowed else ("someone-else",),
        )
    return Cluster(
        backend_name="synthetic", queue_term="partition",
        nodes=nodes, queues=queues,
        identity=Identity(user="me", accounts=("mine",), qos=("q",)),
    )


# 2 nodes I can reach, 3 I cannot: 8 GPUs of 20.
SPLIT = [("open", 2, True), ("theirs", 3, False)]


def _payload(cluster: Cluster, argv: list[str], capsys) -> dict:
    cmd_accelerators(cluster, _args(argv), PLAIN)
    return json.loads(capsys.readouterr().out)


def _headline(cluster: Cluster, argv: list[str], capsys) -> str:
    cmd_accelerators(cluster, _args(argv), PLAIN)
    out = capsys.readouterr().out
    return next(ln for ln in out.splitlines() if "GPUs" in ln)


class TestTheDenominatorReachesBothSurfaces:
    def test_the_payload_carries_the_cluster_wide_total(self, capsys) -> None:
        data = _payload(_cluster(SPLIT), ["--json", "accelerators"], capsys)
        assert data["accelerators_installed"] == 2 * GPUS_PER_NODE
        assert data["accelerators_on_cluster"] == 5 * GPUS_PER_NODE

    def test_both_figures_the_headline_prints_are_in_the_payload(self, capsys) -> None:
        """The two surfaces of one measurement, compared against each other."""
        cluster = _cluster(SPLIT)
        line = _headline(cluster, ["accelerators"], capsys)
        shown, total = (int(x) for x in re.search(
            r"(\d+) GPUs of (\d+) on the cluster", line).groups())
        data = _payload(cluster, ["--json", "accelerators"], capsys)
        assert data["accelerators_installed"] == shown
        assert data["accelerators_on_cluster"] == total

    def test_a_reader_can_tell_its_view_is_partial(self, capsys) -> None:
        """The decision the figure exists for: is anything outside my view?"""
        data = _payload(_cluster(SPLIT), ["--json", "accelerators"], capsys)
        assert data["accelerators_on_cluster"] > data["accelerators_installed"]



class TestControls:
    """Each passes with `accelerators_on_cluster` removed as well as with it.

    They cover the figures the fix did not touch, so a neuter that drops the new
    key must leave all of these green -- verified by running it.
    """

    def test_the_in_view_figure_still_counts_only_the_reachable_nodes(
        self, capsys
    ) -> None:
        """`accelerators_installed` keeps its meaning; the fix added, not moved."""
        data = _payload(_cluster(SPLIT), ["--json", "accelerators"], capsys)
        assert data["accelerators_installed"] == 2 * GPUS_PER_NODE
        assert data["accelerators_identified"] <= data["accelerators_installed"]

    def test_the_models_still_sum_to_the_in_view_figure(self, capsys) -> None:
        """The per-model rows are a partition of the in-view total, not of the cluster."""
        data = _payload(_cluster(SPLIT), ["--json", "accelerators"], capsys)
        assert sum(m["installed"] for m in data["models"].values()) == \
            data["accelerators_installed"]

    def test_all_drops_the_filter_and_the_headline_drops_the_phrase(
        self, capsys
    ) -> None:
        """With `--all` the two figures coincide, so the phrase is omitted."""
        line = _headline(_cluster(SPLIT), ["accelerators", "--all"], capsys)
        assert "on the cluster" not in line, line
        assert f"{5 * GPUS_PER_NODE} GPUs" in line, line

    def test_filtering_to_nothing_declines_to_filter(self, capsys) -> None:
        """An allowlist that admits nothing is read as unusable data, not as a
        cluster the caller may use nothing of -- `_entitled_queues` returns
        "do not filter" there, because "a blank screen is a worse failure than
        an over-full one, and it hides the very data that would explain it".

        Asserted on the in-view figure alone, so it holds with the new key
        removed. My first version of the finding test above assumed this case
        filtered to zero; it does not, and the code is right.
        """
        data = _payload(
            _cluster([("theirs", 3, False)]), ["--json", "accelerators"], capsys)
        assert data["accelerators_installed"] == 3 * GPUS_PER_NODE

    @pytest.mark.parametrize("flag", [[], ["--all"]])
    def test_the_payload_is_valid_json_either_way(self, flag, capsys) -> None:
        data = _payload(_cluster(SPLIT), ["--json", "accelerators", *flag], capsys)
        assert isinstance(data["models"], dict)
