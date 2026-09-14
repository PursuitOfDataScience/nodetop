"""DESIGN.md said the outage guard covers *every* command. It covers nine of twelve.

Section 1c is the tool's own account of the failure mode it exists to catch: with
every Slurm command failing, four commands printed clean, confident, wrong
answers and exited 0, and `health` — the command whose entire purpose is to say
whether something is wrong — reported a perfectly healthy cluster. The remedy is
`_reject_broken_snapshot`, and the document's claim was "one guard at dispatch now
covers every command".

Measured by spying on the guard while running every command the parser offers:
**nine reach it, `backends`, `snapshot` and `mcp` do not** — all three return
before the loop the guard sits in. None is a bug in itself. `backends` answers
*which batch systems are usable here*, so it is the one command whose job is to
reply when nothing was detected. `snapshot` records the queries for analysis after
the fact, and a recording of a broken cluster is a legitimate artifact — `errors`
is a field of it, and replaying it *is* rejected by the guard. `mcp` reads nothing
at all: it serves, and every tool call it serves goes through `main()` and meets
the guard there — once per call, rather than once per process.

What was wrong is that the exemption lived only in the shape of two early returns,
with a document asserting the opposite and nothing reading either. A command added
after those returns would inherit the exact bug the guard exists to prevent. The
exemptions are now named (`_GUARD_EXEMPT`) and every command the parser offers is
classified here, so a new one has to be declared one way or the other.

`snapshot` also gained the one thing its exemption was missing: it was the only
command that recorded an outage in silence, and it now names each failed query on
stderr like the guarded nine do.
"""

from __future__ import annotations

import io
import pathlib
import sys

from nodetop import cli
from nodetop.cli import build_parser, cmd_snapshot
from nodetop.core.cluster import Cluster
from nodetop.core.model import Node
from nodetop.render import Glyphs, Style
from nodetop.runner import CapturingRunner, RecordedRunner

PLAIN = Style(depth=0, glyphs=Glyphs())

#: The commands that must reach `_reject_broken_snapshot`. Canonical names only;
#: the aliases below resolve into these.
GUARDED = frozenset(
    {
        "status",
        "queues",
        "zoom",
        "nodes",
        "health",
        "where",
        "check",
        "exclude",
        "accelerators",
    }
)

#: Alias → the command it dispatches as, so adding an alias does not read as
#: adding an unclassified command.
ALIASES = {
    "partitions": "queues",
    "in": "zoom",
    "fit": "where",
    "probe": "check",
    "accel": "accelerators",
    "gpus": "accelerators",
}


def _parser_commands() -> set[str]:
    parser = build_parser()
    subs = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    return set(subs.choices)


def _cluster_with_a_failed_query(why: str = "CommandError: scontrol exited 1") -> Cluster:
    return Cluster(
        backend_name="slurm",
        queue_term="partition",
        nodes=[Node(name="n1", state_raw="free", cpus_total=8)],
        queues={},
        errors={"queues": why},
    )


class TestEveryCommandIsClassified:
    def test_the_parser_offers_nothing_unclassified(self) -> None:
        canonical = {ALIASES.get(name, name) for name in _parser_commands()}
        assert canonical == GUARDED | set(cli._GUARD_EXEMPT), sorted(
            canonical ^ (GUARDED | set(cli._GUARD_EXEMPT))
        )

    def test_the_two_sets_do_not_overlap(self) -> None:
        assert not (GUARDED & set(cli._GUARD_EXEMPT))

    def test_the_exemption_set_is_not_empty(self) -> None:
        # Otherwise the classification above would be satisfied by declaring
        # everything guarded, which is the false claim this file replaces.
        assert cli._GUARD_EXEMPT


class TestSnapshotNamesTheFailedQueries:
    def _stderr_of_a_snapshot(self, tmp_path: pathlib.Path) -> str:
        cluster = _cluster_with_a_failed_query()
        cluster.capture = CapturingRunner(RecordedRunner({}))
        args = build_parser().parse_args(["snapshot", "-o", str(tmp_path / "s.json")])
        buffer = io.StringIO()
        stashed, sys.stderr = sys.stderr, buffer
        try:
            assert cmd_snapshot(cluster, args, PLAIN) == 0
        finally:
            sys.stderr = stashed
        return buffer.getvalue()

    def test_the_failed_query_is_named_on_stderr(self, tmp_path: pathlib.Path) -> None:
        text = self._stderr_of_a_snapshot(tmp_path)
        assert "query failed: queues" in text, text

    def test_it_still_writes_the_recording_and_exits_zero(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The exemption is the point: a recording of a broken cluster is a
        # legitimate artifact, so this must not become an exit 3.
        import json

        self._stderr_of_a_snapshot(tmp_path)
        payload = json.loads((tmp_path / "s.json").read_text())
        assert payload["errors"] == {"queues": "CommandError: scontrol exited 1"}


class TestControls:
    """These hold whether or not the exemptions were named or snapshot made noisy."""

    def test_a_guarded_command_still_exits_three_on_an_empty_broken_snapshot(self) -> None:
        # The guard itself, untouched: `queues` empty with a failed query is the
        # unobtainable case, not an empty cluster.
        assert cli._reject_broken_snapshot(_cluster_with_a_failed_query(), "queues") == 3

    def test_a_clean_snapshot_still_passes_the_guard(self) -> None:
        from nodetop.core.cluster import Queue

        node = Node(name="n1", state_raw="free", cpus_total=8)
        cluster = Cluster(
            backend_name="slurm",
            queue_term="partition",
            nodes=[node],
            queues={"q": Queue(name="q", node_names=("n1",), nodes=[node])},
            errors={},
        )
        assert cli._reject_broken_snapshot(cluster, "queues") == 0

    def test_the_guard_still_names_the_query_for_a_guarded_command(self) -> None:
        # `_name_failed_queries` was extracted from the guard; the guard must
        # still call it.
        buffer = io.StringIO()
        stashed, sys.stderr = sys.stderr, buffer
        try:
            cli._reject_broken_snapshot(_cluster_with_a_failed_query(), "queues")
        finally:
            sys.stderr = stashed
        assert "query failed: queues" in buffer.getvalue()
