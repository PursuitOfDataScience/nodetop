"""`blocked_reasons` was computed, ranked, and read by nothing.

`assess_capacity` builds two histograms of "why not here" and sorts both by count,
descending:

* `hardware_reasons` — no node CAN take this shape. Thoroughly consumed: the table
  prints its top four, `--json` publishes all of them, and `EXCLUDED_REASON` is
  exported so a consumer can test for that case "without matching on a sentence".
* `blocked_reasons` — the capable nodes are busy. Consumed by **nothing**. Before
  this change its only appearances in `src/` were its declaration
  (`core/capacity.py:216`) and its assignment (`core/capacity.py:362`).

The two are disjoint, which is what makes the omission cost something. Measured on a
two-node A100 partition with 1 and 3 of 4 accelerators already allocated, asked for a
4-GPU shape:

    hardware_reasons  {}                                  <- the published field
    blocked_reasons   {'N/N accelerators free': 2}        <- the only explanation

So for a shape that no node can currently fit, the field the surfaces publish is empty
and the field holding the answer was unreachable. `--json` now carries it, all of it,
for the reason `cli.py` already gives about its sibling: a document has no width to run
out of.

**The table is deliberately NOT changed, and this is the interesting half.** The
`hardware` item that prints reasons is gated on
`hw_note = p.hardware_incompatible or p.capacity.capable_but_all_unavailable`, and for
the capable-but-busy case both are **False** — the nodes are up, merely allocated. An
earlier draft of this change put the fallback inside that branch, where it was dead
code for exactly the case it was written for; measuring the rendered table (no
"accelerators free" line at all) is what caught it. Widening that gate changes when an
existing item renders, which is more than this round is for. Recorded as the open half.
"""

import io
import json
import sys

from nodetop.cli import build_parser, cmd_where
from nodetop.core.capacity import assess_capacity
from nodetop.core.cluster import Cluster, Queue
from nodetop.core.model import JobShape
from nodetop.render import Glyphs, Style
from tests.test_capacity import _node

PLAIN = Style(depth=0, glyphs=Glyphs())


def _busy_cluster():
    """Capable hardware, every node partly allocated — so nothing fits a 4-GPU shape."""
    nodes = [
        _node(name="a", model="A100", gpus_alloc=1),
        _node(name="b", model="A100", gpus_alloc=3),
    ]
    queue = Queue(name="gpuq", node_names=("a", "b"), nodes=nodes)
    return Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes, queues={"gpuq": queue}
    )


def _where_json(cluster, argv):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        cmd_where(cluster, build_parser().parse_args(argv), PLAIN)
    finally:
        sys.stdout = saved
    return json.loads(buf.getvalue())


class TestTheJsonPublishesBothHalvesOfWhyNot:
    def test_blocked_reasons_reaches_the_document(self):
        rows = _where_json(_busy_cluster(), ["--json", "where", "-g", "4", "--all"])
        assert rows, rows
        assert rows[0]["blocked_reasons"] == {"N/N accelerators free": 2}, rows[0]

    def test_it_is_the_only_half_with_anything_to_say_here(self):
        """The asymmetry that made the omission cost something."""
        rows = _where_json(_busy_cluster(), ["--json", "where", "-g", "4", "--all"])
        assert rows[0]["hardware_reasons"] == {}, rows[0]
        assert rows[0]["blocked_reasons"], rows[0]

    def test_every_row_carries_the_key(self):
        # A schema promise, not a happens-to-be: a consumer may read it blind.
        rows = _where_json(_busy_cluster(), ["--json", "where", "-g", "1", "--all"])
        assert rows, rows
        assert all("blocked_reasons" in r for r in rows), rows


class TestControls:
    """None of these reads the new key, so each holds with the fix in or out."""

    def test_the_producer_still_ranks_by_count_descending(self):
        # A property of `assess_capacity`, untouched by publishing it.
        nodes = [
            _node(name="a", model="A100", gpus_alloc=1),
            _node(name="b", model="A100", gpus_alloc=3),
        ]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=4))
        counts = list(cap.blocked_reasons.values())
        assert counts == sorted(counts, reverse=True), cap.blocked_reasons

    def test_the_sibling_field_is_still_published(self):
        rows = _where_json(_busy_cluster(), ["--json", "where", "-g", "4", "--all"])
        assert "hardware_reasons" in rows[0], rows[0]

    def test_the_hardware_half_still_answers_when_it_is_the_reason(self):
        # The pre-existing path: wrong hardware, so `hardware_reasons` speaks and
        # this row is not about being busy at all.
        nodes = [_node(name="a", model="V100"), _node(name="b", model="V100")]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=1, requires=("bf16",)))
        assert cap.hardware_reasons, cap.hardware_reasons

    def test_the_fixture_really_leaves_nothing_fitting(self):
        # If anything fitted, both histograms would be empty and every assertion
        # above would pass vacuously.
        nodes = [
            _node(name="a", model="A100", gpus_alloc=1),
            _node(name="b", model="A100", gpus_alloc=3),
        ]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=4))
        assert cap.fitting_nodes == () and cap.capable_nodes == ("a", "b")
