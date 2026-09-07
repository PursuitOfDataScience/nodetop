"""The caveat says "check the node labels" and never said which nodes.

`assess_capacity` sets aside a node whose accelerator model it cannot identify —
deliberately, with the reason recorded at the site: counting it as satisfied "would
let the tool recommend a node that fails at run time". The names go into
`Capacity.unverified_nodes`, and `fit.py` then tells the reader:

    N node(s) were set aside because their accelerator model could not be
    identified, so bf16 cannot be confirmed -- they may well be capable;
    check the node labels

**Which node labels?** Every read of `unverified_nodes` was `len()` or a truthiness
test — `cli.py:3609`, `:3612`, `:3883`, `fit.py:328`, `:445-451` — so the names never
reached a surface. The instruction was actionable and its target was withheld, while
the tool held the list.

Its sibling `hardware_nodes` shows the pattern is both possible and wanted:
`cli.py:3629` iterates those names and looks each one up. `fitting_nodes` and
`capable_nodes` are read only by `len()`/truthiness too, but neither carries an
instruction to go and look at something, so only this one is published here.

`--json` gets the names, all of them, for the reason the `hardware_reasons` note in
the same payload gives: a document has no width to run out of. The prose caveat is
deliberately NOT changed — it renders into a width-constrained cell area, and an
earlier round in this repo shipped a table addition that turned out to be dead code
behind a gate, so prose additions there are measured before they are made, not
bundled into a JSON fix.
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
#: `model=None` leaves `accelerator` unset while the GPUs are still present, which
#: is exactly "the model could not be identified".
UNIDENTIFIABLE = {"model": None, "gpus": 4}


def _cluster():
    nodes = [_node(name="n1", **UNIDENTIFIABLE), _node(name="n2", **UNIDENTIFIABLE)]
    queue = Queue(name="gpuq", node_names=("n1", "n2"), nodes=nodes)
    return Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes, queues={"gpuq": queue}
    )


def _where_json(argv):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        cmd_where(_cluster(), build_parser().parse_args(argv), PLAIN)
    finally:
        sys.stdout = saved
    return json.loads(buf.getvalue())


class TestTheNamesReachTheDocument:
    def test_the_set_aside_nodes_are_named(self):
        rows = _where_json(["--json", "where", "-g", "1", "--needs", "bf16", "--all"])
        assert rows, rows
        assert rows[0]["unverified_node_names"] == ["n1", "n2"], rows[0]

    def test_the_names_agree_with_the_count_beside_them(self):
        rows = _where_json(["--json", "where", "-g", "1", "--needs", "bf16", "--all"])
        assert len(rows[0]["unverified_node_names"]) == rows[0]["nodes_unverified"], rows[0]

    def test_every_row_carries_the_key(self):
        # A schema promise, so a consumer may read it blind.
        rows = _where_json(["--json", "where", "-g", "1", "--all"])
        assert rows, rows
        assert all("unverified_node_names" in r for r in rows), rows


class TestControls:
    """None of these reads the new key, so each holds with the fix in or out."""

    def test_the_count_key_is_still_published(self):
        # Deliberately NOT the finding's argv: a first draft shared it, and when
        # that argv turned out to be invalid this "control" failed with the
        # findings -- which is precisely what a control must not do.
        rows = _where_json(["--json", "where", "-g", "1", "--all"])
        assert "nodes_unverified" in rows[0], rows[0]

    def test_the_producer_still_sets_the_names_aside(self):
        # A property of `assess_capacity`, untouched by publishing it.
        nodes = [_node(name="n1", **UNIDENTIFIABLE), _node(name="n2", **UNIDENTIFIABLE)]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=1, requires=("bf16",)))
        assert cap.unverified_nodes == ("n1", "n2"), cap.unverified_nodes

    def test_the_caveat_still_tells_the_reader_to_look(self):
        # The sentence this fix makes followable; unchanged by it.
        import pathlib

        import nodetop.core.fit as fit_mod

        assert "check the node labels" in pathlib.Path(fit_mod.__file__).read_text()

    def test_a_verifiable_fleet_sets_nothing_aside(self):
        # If every fleet produced unverified nodes, the assertions above would
        # pass without the guard meaning anything.
        nodes = [_node(name="a", model="A100"), _node(name="b", model="A100")]
        cap = assess_capacity(nodes, JobShape(gpus_per_node=1, requires=("bf16",)))
        assert cap.unverified_nodes == (), cap.unverified_nodes
