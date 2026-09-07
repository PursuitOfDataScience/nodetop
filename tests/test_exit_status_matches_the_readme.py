"""The README's exit-status contract, checked against what the commands return.

`README.md` documents the codes because a pipeline branches on them. The
sentence used to read:

    `where` and `check` return 0 only when somewhere could actually take the
    job, 1 when nothing can, 2 when the system has no dry-run to ask, and 3
    when the queries themselves failed.

`cmd_where` returns **only 0 or 1** -- it has no other `return` in it -- so two
of those four claims were never true of it. Measured on one snapshot:
`nodetop --replay snap.json where -c 1` exits **0** while the same replay through
`check` exits 2.

**The code's own message is the authority for which side was wrong.**
`cmd_check`'s no-probe guard says: "nothing to ask: {reason}. Entitlement can
only be read from declared ACLs -- use 'nodetop where' for that." So `where`
answering from a replay is the design, and `where` returning 0 there is correct;
the README was lumping the two commands together. Corrected there, not here.

Two halves, because a doc fix invites a wrong code fix. The paragraph is now
parsed and checked, AND the behaviour is pinned against the "correction" the old
sentence invites -- making `where` return 2 when it cannot probe.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re

import pytest

from nodetop.cli import build_parser, cmd_check, cmd_where
from nodetop.core.cluster import Cluster
from nodetop.core.model import BackendCapabilities, Identity, Node, Queue
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())
README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _paragraph() -> str:
    text = README.read_text()
    start = text.index("Exit status is usable in a pipeline")
    return text[start : text.index("\n\n", start)]


def _cluster(*, probe: bool) -> Cluster:
    """A cluster that can or cannot be dry-run. Four roomy 32-core nodes, so a
    1-core shape fits and a 99999-core one cannot -- which is what separates the
    0 and 1 answers without needing a probe."""
    nodes = [
        Node(name=f"n{i}", state_raw="IDLE", cpus_total=32, memory_mb=128000,
             queues=("open",))
        for i in range(4)
    ]
    queues = {
        "open": Queue(name="open", node_names=tuple(n.name for n in nodes),
                      declared_nodes=len(nodes), nodes=nodes, allow_accounts=()),
    }
    return Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes, queues=queues,
        identity=Identity(user="me", accounts=("a",), qos=("q",)),
        replayed=not probe,
        capabilities=BackendCapabilities(
            probe=probe, probe_supported=True, probe_command="sbatch --test-only"),
    )


def _run(fn, cluster: Cluster, argv: list[str]) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(cluster, build_parser().parse_args(argv), PLAIN)


class TestTheParagraphSaysWhichCommand:
    def test_it_gives_the_no_dry_run_code_to_check_and_says_what_where_does(
        self,
    ) -> None:
        """The claim that was wrong: `where` was credited with code 2 too.

        Asserted as the POSITIVE pair rather than as `"where" not in sentence`.
        The first version did that and failed on the corrected text, because the
        paragraph now names both commands in one sentence in order to contrast
        them -- which is the whole point of the correction. A negative
        containment check cannot tell "credited with the code" apart from
        "mentioned while being excluded from it".
        """
        para = _paragraph()
        # `check` owns code 2 for a missing probe, and the reason is named.
        clause = para[para.index("no dry-run to ask") - 120 : para.index("no dry-run to ask")]
        assert "`check`" in clause, clause
        assert "2" in clause, clause
        # And the paragraph says what `where` does in that same situation.
        assert "`where` answers from the declared ACLs" in para, para
        assert "0 or 1" in para, para

    def test_it_still_documents_zero_and_one_for_both(self) -> None:
        para = _paragraph()
        first = para.split(".")[0]
        assert "`where`" in first and "`check`" in first, first
        assert re.search(r"\b0\b", first) and re.search(r"\b1\b", first), first

    def test_every_code_it_names_is_one_the_tool_can_return(self) -> None:
        """0, 1, 2, 3 and 130 -- and nothing invented."""
        codes = {int(m) for m in re.findall(r"\b(\d{1,3})\b", _paragraph())}
        assert codes <= {0, 1, 2, 3, 130}, codes


class TestTheBehaviourMatchesIt:
    def test_check_returns_two_when_it_cannot_probe(self) -> None:
        assert _run(cmd_check, _cluster(probe=False), ["check", "-c", "1"]) == 2

    @pytest.mark.parametrize("cpus", ["1", "99999"])
    def test_where_never_returns_two_when_it_cannot_probe(self, cpus: str) -> None:
        """The mechanism pin, against the fix the old sentence invited.

        Making `where` return 2 here would satisfy the old README and break the
        one thing `check` tells the reader to fall back to.
        """
        rc = _run(cmd_where, _cluster(probe=False), ["where", "-c", cpus])
        assert rc in (0, 1), rc

    def test_where_answers_from_a_replay_rather_than_declining(self) -> None:
        """0 when something fits, 1 when nothing does -- with no probe at all."""
        assert _run(cmd_where, _cluster(probe=False), ["where", "-c", "1"]) == 0
        assert _run(cmd_where, _cluster(probe=False), ["where", "-c", "99999"]) == 1


class TestControls:
    """Each passes with the README paragraph reverted as well as corrected.

    The doc fix changes no behaviour, so these cover the codes both commands
    already returned -- verified by running the neuter.
    """

    def test_a_probe_capable_cluster_still_answers_zero_and_one(self) -> None:
        cluster = _cluster(probe=True)
        assert _run(cmd_where, cluster, ["where", "-c", "1"]) == 0
        assert _run(cmd_where, cluster, ["where", "-c", "99999"]) == 1

    def test_check_still_answers_rather_than_declining_when_it_can_probe(self) -> None:
        """It must not return 2 merely because a probe is available."""
        rc = _run(cmd_check, _cluster(probe=True), ["check", "-c", "1"])
        assert rc in (0, 1), rc

    def test_cmd_where_has_no_return_other_than_zero_or_one(self) -> None:
        """The structural fact the README got wrong, read off the source.

        This round has two halves and this test belongs to both, which running
        the neuters is what established. It is a CONTROL for the doc fix -- it
        holds with the old sentence restored, because the sentence being wrong
        was never a fact about the source -- and a FINDING test for the
        mechanism half: injecting `if not cluster.can_probe: return 2` into
        `cmd_where` reddens it along with three tests above.
        """
        import ast

        import nodetop.cli as cli_mod

        tree = ast.parse(pathlib.Path(str(cli_mod.__file__)).read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "cmd_where"
        )
        literals = {
            n.value.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, int)
        }
        assert literals <= {0, 1}, literals

    def test_the_paragraph_is_still_findable(self) -> None:
        """Vacuity guard: every doc assertion above rests on locating it."""
        assert len(_paragraph()) > 80
