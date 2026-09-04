"""`health --json` did not carry the denominator its own headline is built from.

The text view of `nodetop health` leads with four figures::

    552 schedulable   ·   0 degraded   ·   56 out   ·   608 total

Two of them were reachable from the JSON payload -- `degraded` and the
unschedulable count are the lengths of the two lists it publishes. The other two
were not there at all, because the payload names the **unhealthy** nodes only:
nothing in it said how many nodes the cluster has, so a consumer of
`health --json` had to call `status --json` as well to state the command's own
headline. The JSON branch returned before `total = len(cluster.nodes)` was even
computed.

That is the gap the note beside the payload says the payload exists to close --
`reason_text`/`reason_set_by`/`reason_set_at` are published as parsed halves of
the raw `reason` specifically "so the text view and a script agree on what one
cause is". A figure on screen that a script cannot get is the same defect one
field further out.

The two figures are asserted against the RENDERED headline rather than against
constants, so this compares the two surfaces to each other. `nodes` is the name
`status --json` already uses for the same fact, checked below.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

import nodetop.cli as cli
from nodetop.cli import build_parser
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _health(cluster, capsys, *, as_json: bool):
    argv = (["--json", "health"] if as_json else ["health"])
    cli.cmd_health(cluster, _args(argv), PLAIN)
    out = capsys.readouterr().out
    return json.loads(out) if as_json else out


#: The headline, as four named figures. Anchored on the words rather than on
#: position: the panel is padded to the terminal width and the separator is a
#: glyph that `--ascii` changes, neither of which is what this reads.
_HEADLINE = {
    "schedulable": re.compile(r"(\d+)\s+schedulable"),
    "degraded": re.compile(r"(\d+)\s+degraded"),
    "out": re.compile(r"(\d+)\s+out\b"),
    "total": re.compile(r"(\d+)\s+total"),
}


def _rendered_headline(text):
    found = {}
    for name, pattern in _HEADLINE.items():
        match = pattern.search(text)
        assert match, f"the headline no longer prints {name!r}: {text}"
        found[name] = int(match.group(1))
    return found


class TestTheHeadlineIsReachableFromTheJson:
    def test_the_rendered_headline_still_prints_four_figures(self, cluster, capsys):
        """Vacuity guard. Every assertion below reads its expectation out of this
        text, so a headline that stopped printing would make them all trivially
        true rather than red."""
        headline = _rendered_headline(_health(cluster, capsys, as_json=False))
        assert set(headline) == {"schedulable", "degraded", "out", "total"}
        # And the four are one arithmetic, on the text side too.
        assert headline["schedulable"] + headline["out"] == headline["total"]

    def test_every_headline_figure_is_reachable_from_the_payload(self, cluster, capsys):
        """The finding: all four, from the JSON alone."""
        text = _health(cluster, capsys, as_json=False)
        doc = _health(cluster, capsys, as_json=True)
        headline = _rendered_headline(text)

        assert doc["nodes"] == headline["total"]
        assert doc["schedulable"] == headline["schedulable"]
        # The two that were always reachable, as list lengths -- asserted so the
        # payload cannot answer the new pair while the old pair drifts.
        assert len(doc["degraded"]) == headline["degraded"]
        assert len(doc["unschedulable"]) == headline["out"]

    def test_the_payload_is_internally_consistent(self, cluster, capsys):
        doc = _health(cluster, capsys, as_json=True)
        assert doc["schedulable"] == doc["nodes"] - len(doc["unschedulable"])
        assert doc["nodes"] >= len(doc["unschedulable"])

    def test_the_two_views_use_one_name_for_the_cluster_size(self, cluster, capsys):
        """`status --json` already published this fact as `nodes`. Reusing the
        name is the point -- a second spelling would be a mapping a consumer has
        to maintain, which is what having two views of one number costs."""
        health = _health(cluster, capsys, as_json=True)
        cli.cmd_status(cluster, _args(["--json", "status"]), PLAIN)
        status = json.loads(capsys.readouterr().out)
        assert health["nodes"] == status["nodes"]
        assert health["nodes"] - health["schedulable"] == status["unschedulable_nodes"]


class TestTheQuietCluster:
    """The degenerate direction: nothing wrong anywhere.

    This is the shape the payload is least informative in -- both lists empty --
    and so the one where a missing total is most costly: with no `nodes`, an
    all-healthy cluster and a cluster of no nodes produce the identical
    document.
    """

    @pytest.fixture
    def healthy(self, cluster):
        return dataclasses.replace(
            cluster,
            nodes=[n for n in cluster.nodes
                   if n not in cluster.unschedulable_nodes
                   and n not in cluster.degraded_nodes],
        )

    def test_a_cluster_with_nothing_wrong_still_says_how_big_it_is(self, healthy, capsys):
        doc = _health(healthy, capsys, as_json=True)
        assert doc["degraded"] == [] and doc["unschedulable"] == []
        assert doc["nodes"] == len(healthy.nodes) > 0
        assert doc["schedulable"] == doc["nodes"]

    def test_an_empty_cluster_is_distinguishable_from_a_healthy_one(self, healthy, capsys):
        """The pair the old payload could not tell apart."""
        empty = dataclasses.replace(healthy, nodes=[])
        healthy_doc = _health(healthy, capsys, as_json=True)
        empty_doc = _health(empty, capsys, as_json=True)
        assert healthy_doc["unschedulable"] == empty_doc["unschedulable"] == []
        assert healthy_doc != empty_doc
        assert (healthy_doc["nodes"], empty_doc["nodes"]) == (len(healthy.nodes), 0)


class TestControls:
    """What the fix must not have moved."""

    def test_control_the_rendered_view_is_unchanged(self, cluster, capsys):
        """The text surface takes its figures from `cluster`, not from the
        payload, so adding keys to the payload must leave it byte-identical.
        Pinned as the whole headline line."""
        text = _health(cluster, capsys, as_json=False)
        line = next(ln for ln in text.splitlines() if "schedulable" in ln)
        headline = _rendered_headline(text)
        assert line.strip().strip("│").strip() == (
            f"{headline['schedulable']} schedulable   ·   "
            f"{headline['degraded']} degraded   ·   "
            f"{headline['out']} out   ·   "
            f"{headline['total']} total"
        ), line

    def test_control_the_keys_that_were_already_published_are_untouched(
        self, cluster, capsys
    ):
        """The lists and the compressed nodelist, unchanged in name and content.

        Written as a subset check on the keys plus equality on the values, so it
        passes both before and after the fix: the old payload's three keys must
        still be there and still say the same thing.
        """
        doc = _health(cluster, capsys, as_json=True)
        assert {"degraded", "unschedulable", "unschedulable_nodelist"} <= set(doc)
        assert doc["unschedulable_nodelist"] == cluster.format_nodelist(
            [n.name for n in cluster.unschedulable_nodes]
        )
        assert [n["name"] for n in doc["unschedulable"]] == [
            n.name for n in cluster.unschedulable_nodes
        ]
        assert [n["name"] for n in doc["degraded"]] == [
            n.name for n in cluster.degraded_nodes
        ]
        # The parsed reason halves the note beside the payload is about.
        for entry in doc["unschedulable"]:
            assert {"reason", "reason_text", "reason_set_by", "reason_set_at"} <= set(entry)

    def test_control_the_json_view_prints_one_document_and_nothing_else(
        self, cluster, capsys
    ):
        """`--json` on this command has always been the whole of stdout, and a
        headline accidentally printed alongside it would break every consumer."""
        cli.cmd_health(cluster, _args(["--json", "health"]), PLAIN)
        out = capsys.readouterr().out
        json.loads(out)
        assert "schedulable   ·" not in out
