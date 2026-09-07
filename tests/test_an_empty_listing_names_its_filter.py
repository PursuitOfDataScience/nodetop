"""An empty node listing says which filter emptied it.

`cmd_zoom` has said so, and said why, for several rounds:

    # "(nothing to show)" alone reads as "this queue is empty", which is a
    # different claim from "your filter excluded all of it".

`cmd_nodes` printed exactly that string. Measured on one snapshot:
`nodes --gpu --cpu` rendered a headline, a rule, and `(nothing to show)` --
the phrase its sibling's comment names as the wrong claim -- while
`zoom amd --gpu --cpu` forty lines away rendered `nothing matches --gpu --cpu`.

`--gpu --cpu` is what makes this more than tidiness. The two are complements by
definition (`--gpu` keeps `is_gpu_node`, `--cpu` keeps `not is_gpu_node`), so
the pair is unsatisfiable on **every** cluster and `(nothing to show)` is never
the right answer for it. nodetop refuses the cluster-DEPENDENT unsatisfiable
filter already -- `nodes -q bogus` exits 2 with "no such partition: bogus / run
'nodetop partitions' to list them" -- so the always-unsatisfiable one being
silent was the inconsistency.

The sentence now has one home (`_nothing_matches`), because a fourth node filter
added to one listing would otherwise be missing from the other's explanation.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from nodetop.cli import build_parser, cmd_nodes, cmd_zoom
from nodetop.core.cluster import Cluster
from nodetop.core.model import Identity, Node, Queue
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _cluster() -> Cluster:
    """Two GPU nodes and three CPU nodes in one partition the account may use."""
    nodes = [
        Node(name=f"gpu{i}", state_raw="IDLE", cpus_total=16, memory_mb=64000,
             gpus_total=4, queues=("amd",))
        for i in range(2)
    ] + [
        Node(name=f"cpu{i}", state_raw="IDLE", cpus_total=16, memory_mb=64000,
             queues=("amd",))
        for i in range(3)
    ]
    queues = {
        "amd": Queue(name="amd", node_names=tuple(n.name for n in nodes),
                     declared_nodes=len(nodes), nodes=nodes, allow_accounts=()),
    }
    return Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes, queues=queues,
        identity=Identity(user="me", accounts=("a",), qos=("q",)),
    )


def _out(fn, argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(_cluster(), build_parser().parse_args(argv), PLAIN)
    return buf.getvalue()


class TestTheFilterIsNamed:
    def test_the_unsatisfiable_pair_is_explained(self) -> None:
        out = _out(cmd_nodes, ["nodes", "--gpu", "--cpu"])
        assert "nothing matches --gpu --cpu" in out, out
        assert "(nothing to show)" not in out, out

    def test_the_queue_is_named_with_its_value(self) -> None:
        """`-q` is a flag on `nodes`, so the value belongs in the sentence --
        the way `no such partition: X` names its value."""
        out = _out(cmd_nodes, ["nodes", "-q", "amd", "--gpu", "--cpu"])
        assert "nothing matches -q amd --gpu --cpu" in out, out

    @pytest.mark.parametrize("argv", [["--gpu", "--cpu"], ["--cpu", "--gpu"]])
    def test_the_order_typed_does_not_change_the_sentence(self, argv) -> None:
        """The list is built from the parser, not from argv order."""
        out = _out(cmd_nodes, ["nodes", *argv])
        assert "nothing matches --gpu --cpu" in out, out

    def test_both_listings_use_the_same_sentence(self) -> None:
        """The two surfaces of one explanation, compared against each other."""
        nodes_out = _out(cmd_nodes, ["nodes", "--gpu", "--cpu"])
        zoom_out = _out(cmd_zoom, ["zoom", "amd", "--gpu", "--cpu"])
        assert "nothing matches --gpu --cpu" in nodes_out
        assert "nothing matches --gpu --cpu" in zoom_out

    def test_the_sentence_has_one_home(self) -> None:
        """A fourth filter must not reach one listing's explanation only."""
        import pathlib

        import nodetop.cli as cli_mod

        code = "\n".join(
            ln for ln in pathlib.Path(str(cli_mod.__file__)).read_text().splitlines()
            if not ln.strip().startswith("#")
        )
        # An f-string, not `%s`: nodetop's gate selects UP031, so percent format
        # is a lint error here. The sibling packages allow it; this one does not.
        assert code.count('f"nothing matches {named}"') == 1, code
        # Three, not two: the `def` line matches the same substring as the two
        # call sites. A single-source pin has to allow the definition -- asserting
        # 2 here failed on the correct code, which is the tell.
        assert code.count("_nothing_matches(args") == 3
        assert code.count("def _nothing_matches(args") == 1


class TestControls:
    """Each passes with the `cmd_nodes` block removed as well as with it.

    They cover the listings that are NOT empty and `zoom`'s own wording, which
    this round shares rather than changes -- verified by running the neuter.
    """

    def test_a_matching_listing_still_renders_its_table(self) -> None:
        out = _out(cmd_nodes, ["nodes", "--gpu"])
        assert "nothing matches" not in out, out
        assert "gpu0" in out, out

    def test_a_bare_listing_still_renders_every_node(self) -> None:
        out = _out(cmd_nodes, ["nodes"])
        assert "nothing matches" not in out, out
        for name in ("gpu0", "cpu0"):
            assert name in out, out

    def test_zoom_does_not_gain_a_dash_q_it_never_had(self) -> None:
        """The regression guard on this round's own scoping.

        `zoom` takes its partition POSITIONALLY, so naming it `-q amd` would
        misdescribe how the caller typed it. The first version of the shared
        helper read `args.queue` unconditionally and zoom's sentence silently
        gained a `-q amd`; the queue term is now opt-in per caller.
        """
        out = _out(cmd_zoom, ["zoom", "amd", "--gpu", "--cpu"])
        assert "nothing matches --gpu --cpu" in out, out
        assert "-q amd" not in out, out

    def test_the_funnel_still_reports_how_many_were_filtered(self) -> None:
        """The count is a separate disclosure and this round does not touch it:
        the headline says HOW MANY, the sentence says WHICH."""
        out = _out(cmd_nodes, ["nodes", "--gpu", "--cpu"])
        assert "filtered out" in out, out
