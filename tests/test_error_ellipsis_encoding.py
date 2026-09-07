"""An unfolded U+2026 crashes on an ASCII **stdout**, and litters an ASCII stderr.

`truncate`'s `ellipsis` defaults to U+2026, and twelve of its fourteen call sites
pass a glyph from the detected set instead. The two that did not were both error
paths: one inside `cmd_status` (stdout) and one writing to stderr with no `Style`
in scope. Both now take the mark from a glyph set -- `st.g.ellipsis` and
`Glyphs.detect(sys.stderr)` respectively.

**The two streams behave differently, and an earlier version of this file had it
backwards.** Measured with the handlers CPython actually installs under an ASCII
locale (`LC_ALL=C`, UTF-8 mode off):

    stdout   ascii / surrogateescape    -> UnicodeEncodeError
    stderr   ascii / backslashreplace   -> succeeds, emits the text `\u2026`

So the **stdout** site is the real crash, and the stderr one is cosmetic: the
reader gets `query failed: queues: ...\u2026` instead of an ellipsis. Python has
guaranteed `backslashreplace` on stderr since 3.5, so a `UnicodeEncodeError` there
is not reachable at all -- the earlier claim that it was came from a harness that
forced `errors="strict"`, which no real stream uses.

Both halves are still worth fixing, and `Glyphs.detect` says why in one line: a
terminal that cannot encode the glyph "would raise or print replacement
characters; ASCII is strictly better than either." `truncate`'s own default is
deliberately unchanged; see `TestControls`.
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

    `errors="backslashreplace"` because that is what CPython installs on stderr,
    not `"strict"`. Forcing strict here is what made an earlier version of this
    file report a crash that cannot happen.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(
        raw, encoding=encoding, errors="backslashreplace", write_through=True
    )
    saved = sys.stderr
    sys.stderr = stream
    try:
        rc = _reject_broken_snapshot(_cluster(why), "status")
    finally:
        sys.stderr = saved
        stream.flush()
    return rc, raw.getvalue().decode(encoding)


class TestTheReporterSurvivesAnAsciiTerminal:
    def test_an_ascii_stderr_gets_a_readable_mark_not_an_escape(self):
        """stderr cannot raise, so what is at stake here is legibility.

        Without the fix this line reads `...\u2026` -- the escape CPython's
        `backslashreplace` produces -- which is noise in the one message that
        exists to explain a failed query.
        """
        rc, text = _stderr_bytes(LONG_WHY, "ascii")
        assert rc == 3
        assert "query failed: queues" in text, text
        assert "..." in text, text
        assert "\\u2026" not in text, text

    def test_a_utf8_terminal_still_gets_the_nicer_mark(self):
        """The fix is per-stream detection, not a downgrade for everyone."""
        _rc, text = _stderr_bytes(LONG_WHY, "utf-8")
        assert "…" in text, text

    def test_the_status_panel_does_not_crash_on_an_ascii_stdout(self):
        """The real crash, on a REAL encoded stream.

        stdout gets `surrogateescape`, which raises on a character it cannot
        encode -- unlike stderr. A `StringIO` accepts any `str`, so the earlier
        version of this test could only see the glyph, never the exception it
        matters for.
        """
        args = build_parser().parse_args(["status"])
        style = Style(enabled=False, glyphs=Glyphs.ascii())
        raw = io.BytesIO()
        stream = io.TextIOWrapper(
            raw, encoding="ascii", errors="surrogateescape", write_through=True
        )
        saved = sys.stdout
        sys.stdout = stream
        try:
            cmd_status(_cluster(LONG_WHY, with_queues=True), args, style)
        finally:
            sys.stdout = saved
            stream.flush()
        out = raw.getvalue().decode("ascii")
        assert "FAILED" in out, out
        assert "\u2026" not in out, out


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
