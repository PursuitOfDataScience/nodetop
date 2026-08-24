"""Boundary shapes: clusters no property test would construct.

Surviving is not the same as saying something sensible. An empty cluster that
reports "0 nodes" with a straight face invites the reader to believe it, when
in practice that reading almost always means the wrong backend or an
unreachable control plane.
"""

from __future__ import annotations

import io
import json
import time

import pytest

from nodetop.cli import _COMMANDS, build_parser
from nodetop.core.cluster import Cluster
from nodetop.core.model import BackendCapabilities, Identity, Node, Queue
from nodetop.render import MIN_WIDTH, Glyphs, Style, width

PLAIN = Style(depth=0, glyphs=Glyphs())
ASCII = Style(depth=0, glyphs=Glyphs.ascii())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _node(name: str, cpus: int = 8, gpus: int = 0, conditions=()) -> Node:
    return Node(
        name=name, cpus_total=cpus, memory_mb=1024, gpus_total=gpus,
        conditions=frozenset(conditions), state_raw="up",
    )


def _cluster(nodes: list[Node], queues: list[Queue]) -> Cluster:
    for q in queues:
        q.nodes = [n for n in nodes if n.name in q.node_names]
    return Cluster(
        backend_name="test", nodes=nodes, queues={q.name: q for q in queues},
        capabilities=BackendCapabilities(probe=False), identity=Identity(user="u"),
    )


SHAPES = {
    "empty": _cluster([], []),
    "one node one queue": _cluster(
        [_node("n1")], [Queue(name="q", node_names=("n1",))]),
    "every node down": _cluster(
        [_node("n1", conditions=("DOWN",))], [Queue(name="q", node_names=("n1",))]),
    "queue with no nodes": _cluster([_node("n1")], [Queue(name="empty")]),
    "node in no queue": _cluster([_node("orphan")], [Queue(name="q")]),
    "zero-cpu node": _cluster(
        [_node("z", cpus=0)], [Queue(name="q", node_names=("z",))]),
    "huge accelerator count": _cluster(
        [_node("h", gpus=100_000)], [Queue(name="q", node_names=("h",))]),
    "very long names": _cluster(
        [_node("n" * 200)], [Queue(name="q" * 200, node_names=("n" * 200,))]),
}

COMMANDS = [
    ["status"], ["status", "--all"], ["queues"], ["queues", "--detail"],
    ["nodes"], ["nodes", "--gpu"], ["nodes", "--free"], ["health"],
    ["accelerators"], ["where", "-g", "1"], ["where", "-c", "1"],
    ["where", "-c", "1", "--all"], ["exclude", "--gpu-nodes"],
]


def _run(cluster, argv, style=PLAIN):
    import contextlib

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = _COMMANDS[argv[0]](cluster, _args(argv), style)
    return rc, out.getvalue()


@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
@pytest.mark.parametrize("argv", COMMANDS, ids=[" ".join(c) for c in COMMANDS])
class TestEveryShapeSurvives:
    def test_it_does_not_raise(self, shape, argv):
        rc, _ = _run(SHAPES[shape], argv)
        assert rc in (0, 1, 2)

    @pytest.mark.parametrize("size", [MIN_WIDTH, 60, 100, 200])
    def test_it_fits_the_window(self, shape, argv, monkeypatch, size):
        # Every width, not just 100. "very long names" carries 200-character
        # node and queue names on purpose, and a single generous width is the
        # one place they are least likely to cause trouble.
        monkeypatch.setenv("COLUMNS", str(size))
        _, out = _run(SHAPES[shape], argv)
        for line in out.splitlines():
            if line.startswith("  --"):
                continue  # the copy-me submit line
            assert width(line) <= size, f"@{size} {width(line)}: {line[:60]!r}"

    def test_ascii_stays_ascii(self, shape, argv):
        _, out = _run(SHAPES[shape], argv, ASCII)
        out.encode("ascii")

    def test_json_stays_valid(self, shape, argv):
        # The sub-command must stay first; --json is accepted after it.
        _, out = _run(SHAPES[shape], [argv[0], "--json", *argv[1:]])
        if out.strip():
            json.loads(out)


class TestEmptyClusterIsTreatedAsAFinding:
    def test_status_warns_rather_than_reporting_zero_blandly(self):
        _, out = _run(SHAPES["empty"], ["status"])
        text = " ".join(out.split())
        assert "no nodes" in text
        assert "wrong backend" in text
        assert "nodetop backends" in text

    def test_a_populated_cluster_gets_no_such_warning(self):
        _, out = _run(SHAPES["one node one queue"], ["status"])
        assert "wrong backend" not in out

    def test_accelerators_does_not_print_an_empty_heading(self):
        _, out = _run(SHAPES["empty"], ["accelerators"])
        text = " ".join(out.split())
        assert "no GPUs found" in text
        # A heading with nothing under it says less than nothing.
        assert "what these GPUs can do" not in text

    def test_where_says_plainly_that_nothing_fits(self):
        rc, out = _run(SHAPES["empty"], ["where", "-g", "1"])
        assert rc == 1
        assert "no queue can run this shape" in out


class TestGpusYouCannotReachAreNotAbsentGpus:
    """"None" and "none for you" are different answers, and one is a lie.

    Observed on a cluster whose only accelerators sat in two partitions closed
    to the caller: the inventory header counted "0 GPUs of 8 on the cluster"
    and the note one line below it said "no GPUs found in this cluster".
    """

    CLOSED = _cluster(
        [_node("c1"), _node("g1", gpus=8)],
        [Queue(name="open", node_names=("c1",)),
         Queue(name="theirs", node_names=("g1",), enabled=False)],
    )

    def test_it_says_where_they_are_instead_of_denying_them(self):
        _, out = _run(self.CLOSED, ["accelerators"])
        text = " ".join(out.split())
        assert "no GPUs found in this cluster" not in text
        assert "8 GPUs on this cluster" in text
        assert "theirs" in text
        assert "--all" in text        # and what to type to see them anyway

    def test_all_shows_them(self):
        _, out = _run(self.CLOSED, ["accelerators", "--all"])
        assert "8 GPUs" in out
        assert "no GPUs" not in out

    def test_a_queue_with_none_of_them_says_so_without_generalising(self):
        _, out = _run(self.CLOSED, ["accelerators", "-q", "open"])
        text = " ".join(out.split())
        assert "no GPUs in that queue" in text
        assert "the cluster has 8 elsewhere" in text

    def test_a_cluster_with_no_gpus_at_all_still_says_that(self):
        # The old message was not wrong everywhere -- only where it was.
        _, out = _run(SHAPES["one node one queue"], ["accelerators"])
        assert "no GPUs found in this cluster" in " ".join(out.split())


class TestGrammar:
    def test_one_queue_is_not_reported_as_one_queues(self):
        _, out = _run(SHAPES["one node one queue"], ["where", "-c", "1"])
        assert "1 queue considered" in out
        assert "1 queues" not in out


class TestScale:
    """Ten thousand nodes is a real cluster, not a stress test."""

    # A classmethod, because a class-scoped fixture written as an instance
    # method runs once against a throwaway instance -- pytest 9 deprecates it
    # and pytest 10 removes it.
    @pytest.fixture(scope="class")
    @classmethod
    def big(cls):
        nodes = [_node(f"n{i:05d}", gpus=4 if i % 3 == 0 else 0) for i in range(10_000)]
        queue = Queue(name="big", node_names=tuple(n.name for n in nodes))
        return _cluster(nodes, [queue])

    @pytest.mark.parametrize("argv", [
        ["status"], ["queues"], ["health"], ["accelerators"], ["where", "-g", "4"],
    ], ids=lambda a: " ".join(a))
    def test_it_stays_responsive(self, big, argv):
        start = time.time()
        rc, _ = _run(big, argv)
        elapsed = time.time() - start
        assert rc in (0, 1, 2)
        # Rendering is not where a cluster tool should spend its time.
        assert elapsed < 5.0, f"{' '.join(argv)} took {elapsed:.1f}s"

    def test_the_nodelist_uses_bracket_notation(self, big):
        _, out = _run(big, ["exclude", "--gpu-nodes"])
        assert "[" in out
        # Every third node has an accelerator, so the set is scattered and
        # cannot compress into runs. It still beats spelling out each name.
        spelled_out = sum(len(n.name) + 1 for n in big.nodes if n.is_gpu_node)
        assert len(out) < spelled_out

    def test_an_incompressible_set_warns_that_it_is_too_long(self, big, capsys):
        import contextlib

        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            _COMMANDS["exclude"](big, _args(["exclude", "--gpu-nodes"]), PLAIN)
        text = " ".join(err.getvalue().split())
        assert "too long for a command line" in text

    def test_counts_are_right_at_scale(self, big):
        rc, out = _run(big, ["status", "--json"])
        data = json.loads(out)
        assert data["nodes"] == 10_000
        assert data["accelerator_nodes"] == len(
            [n for n in big.nodes if n.is_gpu_node])
