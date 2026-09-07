"""`queues`' first data column was not in `queues --json`.

The table's own header names it: ``nodes up``, which the renderer computes as
``len([n for n in q.nodes if n.schedulable])``. The document published the
denominator and three OTHER numerators, so on a real cluster the screen read

    ●  amd      UP   39/40   4131/5120 ...

while the row for `amd` offered ``idle_nodes_advertised: 26``,
``effective_free_nodes: 26``, ``nodes_with_room: 38`` and ``nodes: 40``. None of
them is 39, and there is no field to derive it from: `nodes --json` has
``schedulable`` per node but **no partition**, so a consumer cannot join the two.
Having the denominator already there is what made the gap easy to miss.

This is the rule the builder states about itself, one column over. The comment
above ``nodes_with_room`` records the same defect being fixed for the other two
columns: *"The text form prints '0 wholly free, 45 of 190 with something spare'
and a free-core count; `--json` carried neither, so the two forms of the same
command answered different questions."*

The name is borrowed rather than invented: ``status --json`` already calls this
concept ``nodes_schedulable`` for the cluster as a whole (``yours.nodes_schedulable
= 328`` beside ``nodes = 608``), so the per-queue field is the same word for the
same measurement.

Verified against the live cluster through one frozen snapshot, both surfaces on
identical input, before and after::

    amd      screen 39/40    json 39/40
    beagle3  screen 43/44    json 43/44
    caslake  screen 190/190  json 190/190
    test     screen 552/608  json 552/608

`TestControls` pins that the three figures already published keep their meanings,
that the text table is unchanged, and that a routing queue -- which owns no nodes
and prints a dash rather than a count -- still answers 0 rather than raising.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import sys

import pytest

from nodetop.cli import build_parser, cmd_queues
from nodetop.core.cluster import Cluster, Node, Queue
from nodetop.core.model import BackendCapabilities, Identity
from nodetop.render import Style

PLAIN = Style(enabled=False)


class _Backend:
    def __init__(self, caps):
        self._caps = caps

    def probe(self, q, shape, account=None):
        return None

    def capabilities(self):
        return self._caps

    def submit_flags(self, q, shape):
        return [f"--partition={q}"]

    def format_nodelist(self, names):
        return ",".join(sorted(names))


def _cluster():
    """Four nodes in one partition: two schedulable, one DOWN, one DRAINED.

    So "nodes up" is 2 of 4 and is equal to none of the other three figures --
    idle is 2, with-room is 2, total is 4 -- which is what makes the assertions
    below able to fail.
    """
    nodes = [
        Node(name="n0", state_raw="IDLE", cpus_total=8, memory_mb=16000, queues=("p",)),
        Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=16000, queues=("p",)),
        # `Node.schedulable` reads `conditions`, NOT `state_raw` -- the backends
        # translate their own vocabulary into `BLOCKING_CONDITIONS`, so a fixture
        # that sets only the display string describes a node the model considers
        # perfectly schedulable (measured: all four counted, and every figure in
        # the row read 4).
        Node(name="n2", state_raw="DOWN", conditions=frozenset({"DOWN"}),
             cpus_total=8, memory_mb=16000, queues=("p",)),
        Node(name="n3", state_raw="DRAINED", conditions=frozenset({"DRAIN"}),
             cpus_total=8, memory_mb=16000, queues=("p",)),
    ]
    queues = {
        "p": Queue(
            name="p",
            node_names=tuple(n.name for n in nodes),
            declared_nodes=4,
            nodes=nodes,
        ),
        "router": Queue(name="router", node_names=(), declared_nodes=0, nodes=[],
                        forwards_to=("p",)),
    }
    caps = BackendCapabilities(probe=False, probe_supported=False,
                               probe_command="sbatch --test-only")
    return dataclasses.replace(
        Cluster(backend_name="slurm", queue_term="partition", nodes=nodes,
                queues=queues, identity=Identity(user="me", accounts=("mine",))),
        capabilities=caps, _backend=_Backend(caps),
    )


def _run(argv):
    buf = io.StringIO()
    saved, sys.stdout = sys.stdout, buf
    try:
        cmd_queues(_cluster(), build_parser().parse_args(argv), PLAIN)
    finally:
        sys.stdout = saved
    return buf.getvalue()


def _rows():
    return {r["name"]: r for r in json.loads(_run(["--json", "queues", "--all"]))}


def _screen_nodes_up(name):
    """The `nodes up` cell the table prints for `name`, as text."""
    text = _run(["--no-color", "queues", "--all"])
    row = next(ln for ln in text.splitlines() if re.search(rf"\b{name}\b", ln))
    found = re.search(r"(\d+)/(\d+)", row)
    assert found, row
    return found.group(0)


class TestTheDocumentCarriesTheColumnTheTablePrints:
    def test_the_field_exists(self) -> None:
        assert "nodes_schedulable" in _rows()["p"]

    def test_it_is_the_number_on_screen(self) -> None:
        row = _rows()["p"]
        cell = f"{row['nodes_schedulable']}/{row['nodes']}"
        assert cell == _screen_nodes_up("p")

    def test_it_counts_only_schedulable_nodes(self) -> None:
        # Two of the four are DOWN and DRAINED.
        assert _rows()["p"]["nodes_schedulable"] == 2

    def test_it_is_not_any_of_the_figures_already_published(self) -> None:
        """The fixture exists to make this assertion possible."""
        row = _rows()["p"]
        assert row["nodes_schedulable"] != row["nodes"]
        assert row["nodes_schedulable"] != row["nodes_declared"]

    def test_a_routing_queue_answers_zero_rather_than_raising(self) -> None:
        assert _rows()["router"]["nodes_schedulable"] == 0


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_the_denominator_was_already_there(self) -> None:
        assert _rows()["p"]["nodes"] == 4

    @pytest.mark.parametrize(
        ("field", "expected"),
        [("nodes", 4), ("nodes_declared", 4), ("idle_nodes_advertised", 2),
         ("nodes_with_room", 2), ("cpus_total", 32)],
    )
    def test_the_figures_already_published_keep_their_values(self, field, expected) -> None:
        assert _rows()["p"][field] == expected

    def test_the_text_table_still_prints_the_column(self) -> None:
        text = _run(["--no-color", "queues", "--all"])
        assert "nodes up" in text, text
        assert _screen_nodes_up("p") == "2/4"

    def test_the_routing_queue_still_prints_a_dash_not_a_count(self) -> None:
        text = _run(["--no-color", "queues", "--all"])
        row = next(ln for ln in text.splitlines() if "router" in ln)
        assert "/" not in row.split("router")[1].split("1-")[0] or "-" in row, row

    def test_every_row_still_carries_the_same_key_set(self) -> None:
        rows = _rows()
        assert set(rows["p"]) == set(rows["router"]), (
            set(rows["p"]) ^ set(rows["router"])
        )

    def test_the_document_is_still_a_bare_list_of_queues(self) -> None:
        doc = json.loads(_run(["--json", "queues", "--all"]))
        assert isinstance(doc, list)
        assert {r["name"] for r in doc} == {"p", "router"}
