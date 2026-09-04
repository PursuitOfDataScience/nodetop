"""The line that reports a failed query crashed reporting it, on an ASCII stderr.

`truncate`'s `ellipsis` defaults to U+2026, and twelve of its fourteen call sites
pass a glyph from the detected set instead. The two that did not were both error
paths, and one writes to stderr with no `Style` in scope:

    print(f"query failed: {name}: {truncate(why, 120)}", file=sys.stderr)

Under `LC_ALL=C` that raises `UnicodeEncodeError` -- but only once a message is
long enough to be cut, so a short failure reported fine and a verbose one killed
the reporter. `Glyphs.detect` had already written down why: a terminal that cannot
encode the glyph "would raise or print replacement characters; ASCII is strictly
better than either."

Both sites now take the mark from a glyph set -- `Glyphs.detect(sys.stderr)` for
the stderr line, `st.g.ellipsis` for the one inside `cmd_status`, which already
holds the `Style` that owns them. `truncate`'s own default is deliberately
unchanged; see `TestControls`.
"""

import io
import sys

from nodetop.cli import _reject_broken_snapshot, build_parser, cmd_status
from nodetop.core.cluster import Cluster, Node
from nodetop.render import Glyphs, Style, truncate

#: Long enough to be cut at both call sites' limits (120 and 60).
LONG_WHY = "CommandError: qstat -Qf exited 2: " + "Unknown option --very-long-flag " * 6
SHORT_WHY = "CommandError: sacctmgr exited 1"


def _cluster(why, *, with_queues=False):
    node = Node(name="n1", state_raw="free", cpus_total=8)
    queues = {}
    if with_queues:
        from nodetop.core.cluster import Queue

        queues = {"q": Queue(name="q", node_names=("n1",), nodes=[node])}
    return Cluster(
        backend_name="pbs",
        queue_term="queue",
        nodes=[node],
        queues=queues,
        errors={"queues": why},
    )


def _stderr_bytes(why, encoding):
    """Run the reporter with a REAL encoded stream, and return (rc, decoded text).

    A `StringIO` cannot show this defect: it accepts any str. The encoding has to
    be enforced at write time, which is what a terminal does.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors="strict", write_through=True)
    saved = sys.stderr
    sys.stderr = stream
    try:
        rc = _reject_broken_snapshot(_cluster(why), "status")
    finally:
        sys.stderr = saved
        stream.flush()
    return rc, raw.getvalue().decode(encoding)


class TestTheReporterSurvivesAnAsciiTerminal:
    def test_a_long_failure_no_longer_raises_on_an_ascii_stderr(self):
        rc, text = _stderr_bytes(LONG_WHY, "ascii")
        assert rc == 3
        assert "query failed: queues" in text, text
        assert "..." in text, text

    def test_a_utf8_terminal_still_gets_the_nicer_mark(self):
        """The fix is per-stream detection, not a downgrade for everyone."""
        _rc, text = _stderr_bytes(LONG_WHY, "utf-8")
        assert "…" in text, text

    def test_the_status_panel_row_folds_with_its_style(self):
        args = build_parser().parse_args(["status"])
        style = Style(enabled=False, glyphs=Glyphs.ascii())
        buf = io.StringIO()
        saved = sys.stdout
        sys.stdout = buf
        try:
            cmd_status(_cluster(LONG_WHY, with_queues=True), args, style)
        finally:
            sys.stdout = saved
        out = buf.getvalue()
        assert "FAILED" in out, out
        assert "…" not in out, [c for c in out if not c.isascii()]


class TestControls:
    """Independent of the two call sites, so each holds with the fix in or out."""

    def test_truncates_default_is_still_the_unicode_ellipsis(self):
        """The fix changed two CALLERS, not this shared default.

        Flipping the default would have been the smaller diff and the wrong one:
        twelve other sites already pass a detected glyph, and a helper that
        silently prefers ASCII would make those twelve look redundant.
        """
        assert truncate("x" * 200, 20).endswith("…")
        assert truncate("x" * 200, 20, "...").endswith("...")

    def test_a_short_message_is_not_truncated_so_carries_no_mark(self):
        rc, text = _stderr_bytes(SHORT_WHY, "ascii")
        assert rc == 3
        assert SHORT_WHY in text, text
        assert "..." not in text and "…" not in text, text

    def test_the_exit_codes_are_untouched(self):
        # 3 when the queue query is what failed, 0 when queues survived.
        assert _reject_broken_snapshot(_cluster(SHORT_WHY, with_queues=True), "status") == 0

    def test_the_glyph_sets_disagree_about_this_mark(self):
        # If these ever coincided, every assertion above would pass vacuously.
        assert Glyphs().ellipsis != Glyphs.ascii().ellipsis
