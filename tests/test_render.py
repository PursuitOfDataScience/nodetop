"""Terminal rendering: width, glyph fallback, colour depth, meters."""

from __future__ import annotations

import re

import pytest

from nodetop.render import (
    MIN_WIDTH,
    RAMP_STEPS,
    Glyphs,
    Style,
    badge,
    bar,
    colorize_help,
    columns,
    flow,
    gauge,
    heat_step,
    heat_steps,
    kv,
    panel,
    rule,
    sanitize,
    section,
    sparkline,
    table,
    tree,
    truncate,
    width,
    wrap_indent,
)

PLAIN = Style(depth=0, glyphs=Glyphs())
ASCII = Style(depth=0, glyphs=Glyphs.ascii())
COLOR = Style(depth=24, glyphs=Glyphs())


class TestWidth:
    def test_plain_ascii(self):
        assert width("abc") == 3

    def test_ansi_escapes_are_invisible(self):
        # Padding a coloured cell with len() is what makes a table drift.
        assert width("\033[31mabc\033[0m") == 3
        assert width("\033[38;2;1;2;3mabc\033[0m") == 3
        assert width(COLOR.ok("abc")) == 3

    def test_east_asian_wide_counts_two(self):
        assert width("日本語") == 6

    def test_combining_marks_count_zero(self):
        # "e" plus a combining acute is one column, not two.
        assert width("é") == 1

    def test_box_drawing_is_single_width(self):
        assert width("─│╭╮╰╯") == 6

    def test_empty(self):
        assert width("") == 0


class TestTruncate:
    def test_no_cut_when_it_fits(self):
        assert truncate("abc", 10) == "abc"

    def test_cut_respects_display_width(self):
        got = truncate("abcdefghij", 5)
        assert width(got) == 5

    def test_wide_characters_do_not_overflow(self):
        got = truncate("日本語日本語", 5)
        assert width(got) <= 5

    def test_ascii_ellipsis(self):
        assert truncate("abcdefghij", 6, "...").endswith("...")

    def test_zero_limit(self):
        assert truncate("abc", 0) == ""


class TestGlyphFallback:
    def test_unicode_set_is_unicode(self):
        assert Glyphs().unicode is True
        assert Glyphs().ok == "●"

    def test_ascii_set_is_pure_ascii(self):
        g = Glyphs.ascii()
        assert g.unicode is False
        for name in ("h", "v", "tl", "branch", "ok", "bad", "warn", "sep",
                     "ellipsis", "blocks", "empty", "arrow", "bullet"):
            value = getattr(g, name)
            value.encode("ascii")  # raises if not representable

    def test_detect_falls_back_on_a_non_utf8_stream(self):
        class Latin1:
            encoding = "latin-1"

        # A terminal that cannot encode the glyph would print replacement
        # characters; ASCII is strictly better than mojibake.
        assert Glyphs.detect(Latin1()).unicode is False

    def test_detect_uses_unicode_on_utf8(self):
        class Utf8:
            encoding = "UTF-8"

        assert Glyphs.detect(Utf8()).unicode is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NODETOP_ASCII", "1")
        assert Glyphs.detect().unicode is False


class TestColourDepth:
    def test_disabled_emits_no_escapes(self):
        assert "\033" not in PLAIN.ok("x")
        assert PLAIN.enabled is False

    def test_truecolor_uses_rgb(self):
        assert "38;2;" in Style(depth=24).ok("x")

    def test_256_uses_indexed(self):
        assert "38;5;" in Style(depth=8).ok("x")

    def test_16_colour_uses_plain_sgr(self):
        got = Style(depth=4).ok("x")
        assert "38;" not in got
        assert "\033[" in got

    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        assert Style().enabled is False

    def test_dumb_terminal_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("TERM", "dumb")
        assert Style().enabled is False

    @pytest.mark.parametrize("role", ["accent", "ok", "warn", "bad", "info", "dim"])
    def test_every_role_survives_every_depth(self, role):
        for depth in (4, 8, 24):
            painted = getattr(Style(depth=depth), role)("x")
            assert width(painted) == 1


class TestBar:
    def test_empty_and_full(self):
        assert width(bar(0.0, 10, PLAIN)) == 10
        assert width(bar(1.0, 10, PLAIN)) == 10

    @pytest.mark.parametrize("fraction", [0.0, 0.01, 0.25, 0.5, 0.99, 1.0])
    def test_width_is_always_exact(self, fraction):
        # A meter that changes width breaks the column it sits in.
        assert width(bar(fraction, 14, PLAIN)) == 14
        assert width(bar(fraction, 14, ASCII)) == 14

    def test_a_tiny_nonzero_fraction_still_shows_something(self):
        # 2 of 176 free must not round away to an empty bar: the difference
        # between 0 and 2 free accelerators is the whole question.
        drawn = bar(2 / 176, 14, PLAIN)
        assert drawn[0] != PLAIN.g.empty

    def test_out_of_range_is_clamped(self):
        assert width(bar(-5, 8, PLAIN)) == 8
        assert width(bar(9, 8, PLAIN)) == 8


class TestGauge:
    def test_shows_numbers(self):
        assert "12/176" in gauge(12, 176, 10, PLAIN)

    def test_zero_total_is_a_dash_not_a_division_error(self):
        assert gauge(0, 0, 10, PLAIN)

    def test_unit_label(self):
        assert "gpu" in gauge(1, 2, 6, PLAIN, "gpu")


class TestSparkline:
    def test_length_matches_input(self):
        assert width(sparkline([1, 2, 3, 4], PLAIN)) == 4

    def test_empty_input(self):
        assert sparkline([], PLAIN) == ""

    def test_all_zeros_does_not_divide_by_zero(self):
        assert width(sparkline([0, 0, 0], PLAIN)) == 3

    def test_ascii_ramp(self):
        out = sparkline([1, 5, 9], ASCII)
        out.encode("ascii")


class TestTable:
    def test_columns_align_in_display_width(self):
        # First column is max("A","x","zzz") = 3 wide, joined by a 2-space gap,
        # so the second column must begin at offset 5 on every line. Looking
        # for the first double space instead finds padding, not the boundary.
        out = table(["A", "B"], [["x", "yy"], ["zzz", "w"]], style=PLAIN)
        for line in out.splitlines():
            assert line[5] not in " ", line

    def test_colour_does_not_shift_columns(self):
        plain = table(["A", "B"], [["xx", "y"]], style=PLAIN)
        painted = table(["A", "B"], [[COLOR.ok("xx"), "y"]], style=PLAIN)
        assert [width(x) for x in plain.splitlines()] == [
            width(x) for x in painted.splitlines()
        ]

    def test_wide_characters_do_not_shift_columns(self):
        out = table(["A", "B"], [["日本", "y"], ["ab", "z"]], style=PLAIN)
        assert len({width(line) for line in out.splitlines()}) == 1

    def test_none_cells_render_empty(self):
        assert "None" not in table(["A"], [[None]], style=PLAIN)

    def test_non_string_cells_are_coerced(self):
        assert "42" in table(["A"], [[42]], style=PLAIN)

    def test_limits_truncate_a_runaway_column(self):
        out = table(["A"], [["x" * 80]], style=PLAIN, limits=[10])
        assert max(width(line) for line in out.splitlines()) <= 10

    def test_missing_cells_are_padded(self):
        out = table(["A", "B", "C"], [["x"]], style=PLAIN)
        assert len(out.splitlines()) == 3

    def test_empty(self):
        assert "nothing to show" in table(["A"], [], style=PLAIN)


class TestStructure:
    def test_panel_lines_are_uniform_width(self):
        # Uniform, not equal to the window: the frame now sizes to its content.
        out = panel(["short", "a much longer line here"], "title", PLAIN, size=40)
        assert len({width(line) for line in out.splitlines()}) == 1

    def test_panel_without_a_title(self):
        out = panel(["x"], "", PLAIN, size=20)
        assert len({width(line) for line in out.splitlines()}) == 1

    def test_panel_shrinks_to_its_content(self):
        # A box ruled out to the full window around a short line reads as an
        # empty room rather than as one object.
        out = panel(["x"], "", PLAIN, size=100)
        assert width(out.splitlines()[0]) < 100

    def test_panel_never_exceeds_the_window(self):
        out = panel(["y" * 500], "", PLAIN, size=60)
        for line in out.splitlines():
            assert width(line) <= 60

    def test_panel_fits_a_long_title(self):
        out = panel(["x"], "t" * 50, PLAIN, size=100)
        assert len({width(line) for line in out.splitlines()}) == 1
        assert width(out.splitlines()[0]) <= 100

    def test_panel_shrink_can_be_disabled(self):
        out = panel(["x"], "", PLAIN, size=50, shrink=False)
        assert width(out.splitlines()[0]) == 50

    def test_panel_in_ascii(self):
        out = panel(["x"], "t", ASCII, size=20)
        out.encode("ascii")

    def test_a_divider_inside_a_panel_spans_the_panel(self):
        # `_grid` rules to its own widest column. Inside a frame whose width
        # comes from a longer line -- the cluster facts header -- that left a
        # separator stopping short of the border, which reads as a rendering
        # fault: "i hope the ui is completely sealed".
        rows = panel(["a much longer line than the rule", "  " + PLAIN.g.h * 4],
                     "", PLAIN, size=60).splitlines()
        dashes = [line for line in rows[1:-1] if PLAIN.g.h * 4 in line]
        assert len(dashes) == 1
        # Same right edge as every other content line.
        assert len({width(line) for line in rows}) == 1
        # Filled out to the inner width, less the indent it came with.
        assert dashes[0].count(PLAIN.g.h) == width(rows[0]) - 4 - 2

    def test_a_divider_stretches_in_ascii_too(self):
        rows = panel(["a much longer line than the rule", "  " + ASCII.g.h * 4],
                     "", ASCII, size=60).splitlines()
        assert len({width(line) for line in rows}) == 1
        assert rows[2].count(ASCII.g.h) == width(rows[0]) - 4 - 2

    def test_content_that_merely_starts_with_a_dash_is_left_alone(self):
        # The stretch must key on "this line is nothing but rule characters",
        # not on "this line contains one", or a node named `a-b` or a bar of
        # dashes would be blown out to the full width.
        body = PLAIN.g.h + " left alone"
        rows = panel(["a much longer line than the rule", body],
                     "", PLAIN, size=60).splitlines()
        assert "left alone" in rows[2]
        assert rows[2].count(PLAIN.g.h) == 1

    def test_a_frame_is_one_row_shorter_than_the_window(self, monkeypatch):
        """The spare line is load-bearing, not taste.

        A frame exactly as tall as the terminal scrolls it by one on its final
        newline, so the repaint's cursor-up lands a line low and every keypress
        orphans the top border -- a growing stack of `╭────╮`.
        """
        import os
        import shutil

        from nodetop.render import term_height

        monkeypatch.setattr(shutil, "get_terminal_size",
                            lambda *_a, **_k: os.terminal_size((100, 24)))
        assert term_height() == 23

    def test_a_frame_is_capped_like_the_width_is(self, monkeypatch):
        # A box ruled out to sixty rows around eight rows of content reads as
        # an empty room.
        import os
        import shutil

        from nodetop.render import MAX_HEIGHT, term_height

        monkeypatch.setattr(shutil, "get_terminal_size",
                            lambda *_a, **_k: os.terminal_size((100, 200)))
        assert term_height() == MAX_HEIGHT

    def test_a_frame_has_a_floor(self, monkeypatch):
        # Below its own chrome the frame cannot be drawn at all; the caller
        # falls back to the static print rather than asking for two rows.
        import os
        import shutil

        from nodetop.render import MIN_HEIGHT, term_height

        monkeypatch.setattr(shutil, "get_terminal_size",
                            lambda *_a, **_k: os.terminal_size((100, 3)))
        assert term_height() == MIN_HEIGHT

    def test_rule_fills_the_width(self):
        assert width(rule("", PLAIN, size=30)) == 30

    def test_rule_with_a_title_fills_the_width(self):
        assert width(rule("hello", PLAIN, size=30)) == 30

    def test_tree_closes_the_last_branch(self):
        out = tree([("a", ""), ("b", "")], PLAIN)
        assert PLAIN.g.branch in out
        assert PLAIN.g.last in out

    def test_tree_single_item_uses_the_closing_branch(self):
        assert PLAIN.g.branch not in tree([("only", "")], PLAIN)

    def test_tree_wraps_long_detail(self):
        out = tree([("tag", "word " * 60)], PLAIN)
        assert len(out.splitlines()) > 2

    def test_section_marker(self):
        assert "title" in section("title", PLAIN)

    def test_badge_without_colour_is_bracketed(self):
        assert badge("OK", "ok", PLAIN) == "[OK]"

    def test_badge_with_colour_contains_the_text(self):
        assert "OK" in badge("OK", "ok", COLOR)

    def test_kv_aligns_keys(self):
        out = kv([("a", "1"), ("longer", "2")], PLAIN)
        assert len({line.index("  ") for line in out.splitlines()}) == 1

    def test_kv_empty(self):
        assert kv([], PLAIN) == ""

    def test_columns_lays_out_multiple_per_row(self):
        out = columns(["a", "b", "c", "d"], PLAIN, size=20)
        assert len(out.splitlines()) < 4

    def test_columns_empty(self):
        assert columns([], PLAIN) == ""


class TestWrapIndent:
    def test_prose_is_wrapped_and_indented(self):
        out = wrap_indent("word " * 40, indent="    ", size=40)
        assert all(line.startswith("    ") for line in out.splitlines())
        assert all(width(line) <= 40 for line in out.splitlines())

    def test_raw_prefix_keeps_a_coloured_indent_out_of_the_measurement(self):
        indent = COLOR.dim("| ")
        out = wrap_indent("word " * 30, indent=indent, size=40, raw_prefix=True)
        # Every line fits once the escape sequence is discounted.
        assert all(width(line) <= 40 for line in out.splitlines())


class TestAnsiSafeTruncate:
    """Cutting a styled string must not cut the escape sequence."""

    def test_visible_length_is_respected_with_colour(self):
        painted = COLOR.ok("abcdefghij")
        got = truncate(painted, 5)
        assert width(got) == 5

    def test_a_reset_is_appended_when_cutting_inside_a_styled_run(self):
        got = truncate(COLOR.ok("abcdefghij"), 5)
        assert got.endswith("\033[0m")

    def test_escape_bytes_are_not_counted_against_the_budget(self):
        # If escapes were counted, a coloured cell would lose visible text and
        # a plain one would not, so the two would no longer line up.
        plain = truncate("abcdefghij", 6)
        painted = truncate(COLOR.ok("abcdefghij"), 6)
        assert width(plain) == width(painted)

    def test_no_reset_added_when_nothing_was_styled(self):
        assert "\033" not in truncate("abcdefghij", 5)


class TestFitToWindow:
    """Nothing may be wider than the window it is drawn in."""

    @pytest.mark.parametrize("size", [40, 60, 80, 100])
    def test_panel_clips_overlong_content(self, size):
        out = panel(["x" * 300], "t", PLAIN, size=size)
        assert {width(line) for line in out.splitlines()} == {size}

    @pytest.mark.parametrize("size", [40, 60, 80, 100, 200])
    def test_table_shrinks_to_fit(self, size):
        rows = [[f"value-{i}-" + "y" * 40 for i in range(6)] for _ in range(3)]
        out = table(["one", "two", "three", "four", "five", "six"], rows,
                    style=PLAIN, indent="  ", size=size)
        assert max(width(line) for line in out.splitlines()) <= size

    def test_table_headers_shrink_with_their_columns(self):
        # Truncating only the data leaves the header row wider than every row
        # beneath it -- the one line guaranteed to overflow.
        rows = [["a", "b", "c"]]
        out = table(["a-very-long-header-one", "and-another-long-one",
                     "third-long-header"], rows, style=PLAIN, size=40)
        widths = {width(line) for line in out.splitlines()}
        assert max(widths) <= 40

    def test_table_fit_can_be_disabled(self):
        rows = [["y" * 60]]
        out = table(["h"], rows, style=PLAIN, fit=False, size=20)
        assert max(width(line) for line in out.splitlines()) > 20

    @pytest.mark.parametrize("size", [40, 60, 80])
    def test_kv_wraps_a_long_value(self, size):
        out = kv([("key", "word " * 60)], PLAIN, size=size)
        assert max(width(line) for line in out.splitlines()) <= size

    def test_kv_leaves_a_prestyled_value_alone(self):
        # A gauge lays itself out; re-wrapping would split an escape sequence.
        value = COLOR.ok("#" * 30)
        assert value in kv([("k", value)], PLAIN, size=20)

    @pytest.mark.parametrize("size", [40, 60, 80])
    def test_tree_clips_its_tag(self, size):
        out = tree([("T" * 200, "detail")], PLAIN, size=size)
        assert max(width(line) for line in out.splitlines()) <= size

    @pytest.mark.parametrize("size", [40, 60, 80])
    def test_section_clips_its_note(self, size):
        out = section("title", PLAIN, "n" * 200, size=size)
        assert width(out) <= size

    def test_section_drops_the_note_when_there_is_no_room(self):
        out = section("a-fairly-long-section-title", PLAIN, "note", size=32)
        assert "note" not in out

    def test_term_width_has_a_floor(self, monkeypatch):
        from nodetop.render import MIN_WIDTH, term_width

        monkeypatch.setenv("COLUMNS", "10")
        # Below the floor the layout cannot place a table at all; computing
        # ever-smaller frames just produces rubble.
        assert term_width() == MIN_WIDTH


class TestHyphenSafeWrapping:
    """Wrapping must not invent a different flag."""

    def test_a_flag_is_never_split_at_its_hyphen(self):
        text = "re-run with --check to confirm (sbatch --test-only, read-only)"
        for size in range(24, 80):
            out = wrap_indent(text, indent="  ", size=size)
            joined = " ".join(out.split())
            assert "--test-only" in joined, size
            assert "read-only" in joined, size
            # A trailing hyphen at a line end is the tell-tale of a bad break.
            for line in out.splitlines():
                assert not line.rstrip().endswith("-"), (size, line)

    def test_a_hyphenated_node_name_survives(self):
        text = "the node gn-0001 is drained until maintenance completes here"
        for size in range(24, 80):
            joined = " ".join(wrap_indent(text, indent="  ", size=size).split())
            assert "gn-0001" in joined, size

    def test_kv_values_wrap_without_splitting_flags(self):
        out = kv([("k", "pass --gres=gpu:4 and --cpus-per-task=8 to sbatch here")],
                 PLAIN, size=36)
        joined = " ".join(out.split())
        assert "--cpus-per-task=8" in joined


class TestPlural:
    """The noun comes from the backend, so it is never a literal to eyeball."""

    @pytest.mark.parametrize("count,word,expected", [
        (0, "queue", "0 queues"),
        (1, "queue", "1 queue"),
        (2, "queue", "2 queues"),
        (1, "partition", "1 partition"),
        (3, "namespace", "3 namespaces"),
        (1, "pool", "1 pool"),
    ])
    def test_agreement(self, count, word, expected):
        from nodetop.render import plural

        assert plural(count, word) == expected

    def test_a_custom_suffix(self):
        from nodetop.render import plural

        assert plural(2, "box", "es") == "2 boxes"


class TestNoStrayNonAsciiLiterals:
    """Every drawn character must come from the glyph set.

    A hardcoded `"\u2014"` in `gauge` leaked an em dash straight through the
    ASCII fallback -- the tests in place did not catch it because they only
    exercised gauges with a nonzero total.
    """

    def _source(self, name: str) -> list[tuple[int, str]]:
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "nodetop" / name
        )
        return list(enumerate(path.read_text().splitlines(), 1))

    def _offenders(self, name: str, skip_from: str | None = None):
        import re

        out = []
        in_glyphs = False
        for lineno, line in self._source(name):
            if skip_from and line.startswith(skip_from):
                in_glyphs = True
            if in_glyphs and line and not line[0].isspace() and "class" not in line:
                in_glyphs = False
            stripped = line.strip()
            if stripped.startswith("#") or in_glyphs:
                continue
            # Docstrings legitimately show rendered output.
            if '"""' in line or stripped.startswith(("*", "|", ":", "``")):
                continue
            for match in re.finditer(r'"([^"]*)"', line):
                if any(ord(ch) > 127 for ch in match.group(1)):
                    out.append((lineno, stripped[:70]))
        return out

    def test_render_has_no_stray_literals(self):
        offenders = [
            o for o in self._offenders("render.py", skip_from="class Glyphs")
            # The encoding probe deliberately tests a real glyph.
            if "encode(" not in o[1]
        ]
        assert not offenders, f"non-ASCII literals outside Glyphs: {offenders}"

    def test_the_dash_comes_from_the_glyph_set(self):
        from nodetop.render import Glyphs, Style, gauge

        assert gauge(0, 0, 8, Style(depth=0, glyphs=Glyphs.ascii())) == "--"
        gauge(0, 0, 8, Style(depth=0, glyphs=Glyphs.ascii())).encode("ascii")


class TestFlow:
    """The legend/hint joiner. It exists because a hand-joined legend overflowed."""

    def test_short_items_stay_on_one_line(self):
        assert flow(["a", "b", "c"], PLAIN, size=40).count("\n") == 0

    def test_it_wraps_rather_than_overflowing(self):
        out = flow(["aaaaaaaaaa"] * 6, PLAIN, size=40)
        assert out.count("\n") >= 1
        for line in out.splitlines():
            assert width(line) <= 40

    def test_it_does_not_pad_short_items(self):
        # The difference from columns(): a two-item legend must not become a
        # ragged grid the width of its longest entry.
        out = flow(["x", "a much longer entry"], PLAIN, size=80)
        assert "x   " not in out

    def test_an_item_too_wide_for_a_line_of_its_own_is_clipped(self):
        # Starting a fresh line for it would still overflow, which is how the
        # `where` legend broke at 40 columns.
        out = flow(["y" * 200], PLAIN, size=40)
        assert out.count("\n") == 0
        assert width(out) <= 40

    def test_it_clips_an_oversized_item_without_dropping_the_rest(self):
        out = flow(["z" * 200, "kept"], PLAIN, size=40)
        assert "kept" in out
        for line in out.splitlines():
            assert width(line) <= 40

    def test_empty_items_render_nothing(self):
        assert flow([], PLAIN) == ""

    @pytest.mark.parametrize("size", [40, 41, 60, 80])
    def test_it_never_overflows_at_any_width(self, size):
        items = ["short", "a medium length one", "x" * 90, "tail"]
        for line in flow(items, PLAIN, size=size).splitlines():
            assert width(line) <= size


class TestRuleClipsItsTitle:
    def test_a_long_title_does_not_overflow(self):
        # The fill going to zero does not stop the line growing, so the title
        # itself has to be clipped.
        out = rule("t" * 200, PLAIN, size=40)
        assert width(out) <= 40

    def test_a_title_with_no_room_degrades_to_a_plain_rule(self):
        out = rule("title", PLAIN, size=5)
        assert width(out) <= 5

    def test_a_short_title_is_untouched(self):
        assert "keepme" in rule("keepme", PLAIN, size=60)


class TestTableDropsColumnsWhenItMustReduce:
    # Headers long enough that every floor is the full 6 columns: 7 x 6 plus
    # six 2-space gaps is 54, so 40 is genuinely unreachable by shrinking.
    # (Short headers have small floors and *do* fit -- the first version of
    # this fixture used ONE/TWO/... and was asserting a drop that correctly
    # never happened.)
    HEADERS = ["PARTITION", "NODES", "IDLE", "ACCELERATORS", "MODELS",
               "MAXTIME", "BLOCKERS"]
    ROW = ["aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee", "ffffff", "gggggg"]

    def test_it_fits_a_narrow_window_by_dropping_trailing_columns(self):
        out = table(self.HEADERS, [self.ROW], style=PLAIN, size=40)
        for line in out.splitlines():
            assert width(line) <= 40

    def test_the_drop_is_disclosed(self):
        # A table that silently loses columns reads as a table that had none.
        out = table(self.HEADERS, [self.ROW], style=PLAIN, size=40)
        assert "more column" in out

    def test_it_drops_from_the_right(self):
        out = table(self.HEADERS, [self.ROW], style=PLAIN, size=40)
        assert "PARTIT" in out       # first column survives (clipped)
        assert "BLOCKERS" not in out  # last column is gone

    def test_the_leading_columns_are_never_dropped(self):
        # They identify the row; a table of anonymous numbers is useless.
        out = table(self.HEADERS, [self.ROW], style=PLAIN, size=MIN_WIDTH, keep=2)
        head = out.splitlines()[0]
        assert "PARTIT" in head and "NODES" in head

    def test_a_wide_window_drops_nothing(self):
        out = table(self.HEADERS, [self.ROW], style=PLAIN, size=200)
        assert "BLOCKERS" in out
        assert "more column" not in out

    def test_a_narrow_table_that_already_fits_is_untouched(self):
        out = table(["A", "B"], [["1", "2"]], style=PLAIN, size=40)
        assert "more column" not in out

    def test_short_headers_shrink_instead_of_dropping(self):
        # Shrinking is always preferred: dropping is the last resort, so a
        # table whose floors fit must keep every column.
        out = table(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN"],
                    [self.ROW], style=PLAIN, size=40)
        assert "SEVEN" in out
        assert "more column" not in out
        for line in out.splitlines():
            assert width(line) <= 40


class TestWrapIndentHangingBullet:
    """``first`` gives the first line its own prefix, e.g. a bullet."""

    TEXT = ("a sentence long enough that it has to wrap at least once at the "
            "narrow width this test uses for it")

    def test_the_first_line_takes_the_bullet(self):
        out = wrap_indent(self.TEXT, indent="    ", first="  - ", size=50)
        assert out.splitlines()[0].startswith("  - ")

    def test_the_continuation_lines_do_not(self):
        out = wrap_indent(self.TEXT, indent="    ", first="  - ", size=50)
        rest = out.splitlines()[1:]
        assert rest
        assert not any(ln.lstrip().startswith("- ") for ln in rest)

    @pytest.mark.parametrize("bullet", ["  \u2014 ", "  -- ", "  * ", "  \u2022 "])
    def test_every_line_starts_at_the_same_column(self, bullet):
        # The ASCII glyph twin is a different width from the Unicode one
        # ("--" against an em dash), so a caller-supplied hanging indent lines
        # up in exactly one of the two spellings. The padding is done here so
        # neither can be wrong.
        out = wrap_indent(self.TEXT, indent="    ", first=bullet, size=50)
        starts = {len(ln) - len(ln.lstrip()) for ln in out.splitlines()[1:]}
        assert len(starts) == 1
        assert starts.pop() == width(bullet)

    def test_it_still_respects_the_window(self):
        for line in wrap_indent(self.TEXT, indent="    ", first="  -- ",
                                size=48).splitlines():
            assert width(line) <= 48

    def test_without_first_it_behaves_as_before(self):
        plain = wrap_indent(self.TEXT, indent="  ", size=50)
        assert all(ln.startswith("  ") for ln in plain.splitlines())

    def test_a_single_line_needs_no_continuation(self):
        assert wrap_indent("short", indent="    ", first="  - ", size=50) == "  - short"


class TestTableDropsBlankColumns:
    """A column blank on every row costs width and says nothing.

    Distinct from the width-driven drop and deliberately silent: that one hides
    data and announces itself, this one removes a column with no data in it.
    Real cases on a live cluster were ``nodes --free`` (REASON blank on all 411
    free nodes) and ``check`` (CATEGORY blank when the dry-run passed).
    """

    HEADERS = ["", "NODE", "STATE", "REASON"]
    ROWS = [["*", "n1", "IDLE", ""], ["*", "n2", "MIXED", ""]]

    def test_a_column_blank_everywhere_is_removed(self):
        out = table(self.HEADERS, self.ROWS, style=PLAIN, size=100)
        assert "REASON" not in out

    def test_one_populated_cell_keeps_the_column(self):
        rows = [r[:] for r in self.ROWS]
        rows[1][3] = "maintenance"
        out = table(self.HEADERS, rows, style=PLAIN, size=100)
        assert "REASON" in out
        assert "maintenance" in out

    def test_whitespace_only_counts_as_blank(self):
        rows = [["*", "n1", "IDLE", "   "], ["*", "n2", "MIXED", ""]]
        assert "REASON" not in table(self.HEADERS, rows, style=PLAIN, size=100)

    def test_a_coloured_blank_still_counts_as_blank(self):
        # The cell may carry escape sequences with no visible text; measuring
        # the raw string would keep the column forever.
        coloured = COLOR.dim("")
        rows = [["*", "n1", "IDLE", coloured], ["*", "n2", "MIXED", coloured]]
        assert "REASON" not in table(self.HEADERS, rows, style=COLOR, size=100)

    def test_the_leading_identity_columns_are_never_dropped(self):
        # A blank NODE is a data problem the reader should see, not a layout
        # one to tidy away.
        out = table(["", "NODE", "STATE"], [["*", "", "IDLE"]], style=PLAIN,
                    size=100, keep=2)
        assert "NODE" in out

    def test_dropping_is_silent(self):
        # Nothing was lost, so there is nothing to disclose -- unlike the
        # width-driven drop, which says how many columns went.
        out = table(self.HEADERS, self.ROWS, style=PLAIN, size=100)
        assert "more column" not in out

    def test_it_can_be_turned_off(self):
        out = table(self.HEADERS, self.ROWS, style=PLAIN, size=100,
                    drop_empty=False)
        assert "REASON" in out

    def test_the_freed_width_goes_to_the_remaining_columns(self):
        wide = ["a" * 40, "b" * 40, "c" * 40, ""]
        with_blank = table(["", "ONE", "TWO", "GONE"], [wide], style=PLAIN, size=60)
        assert "GONE" not in with_blank
        for line in with_blank.splitlines():
            assert width(line) <= 60

    def test_an_empty_body_is_unaffected(self):
        assert "nothing to show" in table(self.HEADERS, [], style=PLAIN, size=100)


class TestSectionClipsItsTitle:
    """A section heading must not be the thing that overflows the window."""

    def test_a_long_title_is_clipped(self):
        out = section("the submit filter and the scheduler disagree", PLAIN,
                      size=40)
        assert width(out) <= 40

    def test_a_short_title_is_untouched(self):
        assert "placements" in section("placements", PLAIN, size=80)

    @pytest.mark.parametrize("size", [MIN_WIDTH, 45, 60, 100])
    def test_it_fits_at_every_width(self, size):
        out = section("t" * 200, PLAIN, note="n" * 200, size=size)
        for line in out.splitlines():
            assert width(line) <= size

    def test_a_window_with_no_room_still_returns_something(self):
        out = section("anything", PLAIN, size=4)
        assert width(out) <= 4

    def test_the_note_is_still_clipped_too(self):
        out = section("short", PLAIN, note="n" * 200, size=60)
        assert width(out) <= 60

    def test_the_bullet_survives_clipping(self):
        # It is the visual anchor of the heading; losing it turns a section
        # into a stray line of prose.
        out = section("x" * 200, PLAIN, size=40)
        assert out.startswith(PLAIN.g.bullet)


class TestBarColourIsAScaleNotAVerdict:
    """The fill runs along an ordered ramp, and the ramp is never an alarm.

    Two colours either side of a threshold is a verdict: an earlier version
    painted bars green above half and amber below, and the same amber then meant
    "40% of GPUs are free", which is a warning about nothing. Going flat -- one
    fill colour whatever the value -- fixed the false alarm by giving up on
    colour carrying any quantity at all, which is the other extreme.

    Twelve ordered steps are neither. They read as a scale the way a heatmap
    legend does, and because no step is red at either end, neither a full bar
    nor an empty one can be mistaken for an alarm. Red stays reserved for the
    things that are actually wrong, said with a glyph and a word.
    """

    def test_the_fill_colour_tracks_the_value(self):
        colours = [bar(f, 16, COLOR)[: bar(f, 16, COLOR).index("m") + 1]
                   for f in (0.05, 0.25, 0.49, 0.51, 0.75, 1.0)]
        assert len(set(colours)) > 1, "the ramp is not being used"

    def test_the_ramp_is_monotone_in_the_value(self):
        # Adjacent steps may repeat -- twelve steps over a continuum -- but the
        # sequence must never go backwards, or the colour stops meaning "more".
        steps = [heat_step(f) for f in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
        assert steps == sorted(steps)
        assert steps[0] == 0 and steps[-1] == RAMP_STEPS - 1

    def test_no_step_of_the_ramp_is_the_alarm_colour(self):
        alarm = COLOR.bad("x")[: COLOR.bad("x").index("m") + 1]
        for step in range(RAMP_STEPS):
            for fill in (False, True):
                out = COLOR.tint("x", step, fill=fill)
                assert out[: out.index("m") + 1] != alarm

    def test_the_fill_is_a_darker_twin_of_the_text_tone(self):
        # A bar is a slab and text is a line: the tone that reads as bright in a
        # number reads as shouting across sixteen filled cells.
        for step in range(RAMP_STEPS):
            assert COLOR.tint("x", step, fill=True) != COLOR.tint("x", step)

    def test_a_number_and_its_bar_can_be_pinned_to_one_tone(self):
        # The point of the `step` override: the row's number and the meter
        # beside it are one reading, so they must not be two colours.
        number = COLOR.tint("128", 7, fill=True)
        meter = bar(0.2, 8, COLOR, step=7)
        assert number[: number.index("m") + 1] == meter[: meter.index("m") + 1]

    def test_the_fill_and_the_trough_are_coloured_differently(self):
        out = bar(0.5, 16, COLOR)
        assert out.count("\x1b[") >= 2

    def test_an_explicit_role_still_wins(self):
        # Callers that genuinely mean "this is bad" can still say so.
        assert bar(0.9, 16, COLOR, role="bad") != bar(0.9, 16, COLOR)

    def test_the_geometry_is_unchanged_by_the_colouring(self):
        for fraction in (0.0, 0.1, 0.5, 1.0):
            assert width(bar(fraction, 16, COLOR)) == 16
            assert width(bar(fraction, 16, PLAIN)) == 16

    def test_ascii_mode_is_coloured_the_same_way(self):
        out = bar(0.5, 16, Style(depth=24, glyphs=Glyphs.ascii()))
        assert out.count("\x1b[") >= 2
        assert width(out) == 16

class TestHeatIsColouredAsASet:
    """A column is ranked as a whole, never row by row.

    Colouring each row against a fixed set of bands is what produces "these
    three look the same": nothing in a per-row rule can know that 1336, 128 and
    128 happen to land inside one band, so a partition with ten times another's
    room comes out the same colour as it.
    """

    def test_two_different_values_never_share_a_step(self):
        steps = heat_steps([5120, 1408, 128, 128, 48])
        assert steps[0] > steps[1] > steps[2]
        assert steps[3] > steps[4]

    def test_equal_values_do_share_a_step(self):
        # The property that makes the ramp mean anything: same size, same colour.
        steps = heat_steps([100, 100, 100])
        assert len(set(steps)) == 1

    def test_the_tail_bottoms_out_instead_of_wrapping(self):
        # A long listing must end in flat blue, not start over in amber.
        steps = heat_steps([1000] + [1] * 40)
        assert min(steps) == 0
        assert steps[0] == max(steps)

    def test_the_largest_value_is_the_warmest(self):
        values = [3, 91, 17, 44]
        steps = heat_steps(values)
        assert steps[values.index(max(values))] == max(steps)

    def test_an_empty_column_and_an_all_zero_column_are_safe(self):
        assert heat_steps([]) == []
        assert heat_steps([0, 0]) == [0, 0]

    def test_a_share_is_not_gamma_corrected(self):
        # Gamma belongs to ranking a skewed set of raw counts. Applied to a
        # share it would be a lie about the number: half of the cores free has
        # to land in the middle of the ramp, not two thirds up it.
        assert heat_step(0.5) == RAMP_STEPS // 2


class TestFrameGradient:
    """The border sweeps diagonally, and it closes at every depth."""

    def test_the_sweep_uses_more_than_one_tone(self):
        out = panel(["a"] * 6, "t", COLOR, size=60)
        tones = set(re.findall(r"\x1b\[38;2;[0-9;]+m", out))
        assert len(tones) > 2

    def test_every_row_is_the_same_visible_width(self):
        for st in (COLOR, PLAIN, ASCII, Style(depth=8), Style(depth=4)):
            rows = panel(["a", "bb", "ccc"], "title", st, size=50).splitlines()
            assert len({width(r) for r in rows}) == 1, st.depth

    def test_the_frame_closes_on_every_row(self):
        rows = panel(["x", "y"], "t", PLAIN, size=40).splitlines()
        for row in rows[1:-1]:
            assert row.startswith(PLAIN.g.v) and row.endswith(PLAIN.g.v)

    def test_a_role_forces_a_flat_border(self):
        # For a frame that has to mean something. Not the default: chrome that
        # shouts a semantic colour competes with the numbers inside it.
        out = panel(["a"] * 6, "", COLOR, size=60, role="bad")
        assert len(set(re.findall(r"\x1b\[38;2;[0-9;]+m", out))) == 1

    def test_no_colour_means_no_escapes(self):
        assert "\x1b" not in panel(["a", "b"], "t", PLAIN, size=40)


class TestHelpIsPaintedAfterFormatting:
    """argparse measures its columns with len(), so colour goes on last.

    Painting the strings argparse is *handed* throws every column off by the
    width of the escape sequences in it. This is the property that makes the
    whole approach work, so it is asserted rather than assumed.
    """

    HELP = (
        "usage: nodetop [-h] [--backend NAME]\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --backend NAME        force a batch system\n"
        "  -n N, --top N         how many rows to list (default: 20)\n"
        "\n"
        "examples:\n"
        "  nodetop where -g 4     rank the queues that fit this job\n"
        "\n"
        "  A queue that looks fine is not one that will take your job.\n"
    )

    def test_the_layout_is_byte_identical_once_stripped(self):
        out = colorize_help(self.HELP, COLOR)
        assert re.sub(r"\x1b\[[0-9;]*m", "", out) == self.HELP

    def test_every_line_keeps_its_visible_width(self):
        painted = colorize_help(self.HELP, COLOR).split("\n")
        for before, after in zip(self.HELP.split("\n"), painted, strict=True):
            assert width(after) == len(before)

    def test_colour_off_is_a_passthrough(self):
        assert colorize_help(self.HELP, PLAIN) == self.HELP

    def test_a_flag_and_its_placeholder_are_different_colours(self):
        out = colorize_help(self.HELP, COLOR)
        assert COLOR.paint("info", "--backend") in out
        assert COLOR.paint("warn", "NAME") in out

    def test_the_separator_between_two_spellings_is_not_a_value(self):
        # "-h, --help" read as a flag taking an argument when the comma was
        # painted amber.
        out = colorize_help(self.HELP, COLOR)
        assert COLOR.dim(", ") in out
        assert COLOR.paint("warn", ", ") not in out

    def test_a_default_is_context_not_content(self):
        assert COLOR.dim("(default: 20)") in colorize_help(self.HELP, COLOR)

    def test_the_note_closing_the_examples_is_not_painted_as_a_command(self):
        # Both the note and the examples are indented, so the indent cannot
        # decide it; the note came out with every word painted as a value.
        out = colorize_help(self.HELP, COLOR)
        assert COLOR.paint("warn", "queue") not in out
        assert "A queue that looks fine" in re.sub(r"\x1b\[[0-9;]*m", "", out)

    def test_a_prose_noun_in_capitals_is_not_a_placeholder(self):
        # Amber has to mean "you substitute this". This tool's prose is full of
        # GPU, QOS and GRES, and painting those drains the colour of meaning.
        out = colorize_help("options:\n  --json      GPU and QOS totals\n", COLOR)
        assert COLOR.paint("warn", "GPU") not in out


class TestSchedulerTextCannotRepaintTheTerminal:
    """A node's Reason is operator-authored free text that reaches a table cell.

    Left alone it does damage `width` cannot see, because `width` measures what
    text *occupies* and control characters act instead: ESC [ 2 J clears the
    caller's terminal mid-report, CR returns to column zero so the rest of the
    row overwrites what was drawn -- silently, hiding content rather than
    mangling it -- LF splits one row into two, and TAB measures as one column
    then expands to a tab stop.

    This is also the `--replay` boundary: a snapshot is a JSON file that may
    have come from someone else, and reading one must not repaint your terminal.
    """

    @pytest.mark.parametrize("ch", [
        "\x1b", "\r", "\n", "\t", "\x00", "\a", "\b", "\x7f", "\x9b",
    ])
    def test_every_control_character_becomes_a_space(self, ch):
        assert sanitize(f"before{ch}after") == "before after"

    def test_ordinary_text_is_untouched(self):
        for text in ("drained for maintenance", "\u65e5\u672c\u8a9e", "e\u0301 combining",
                     "gpu:4(S:0-1)", ""):
            assert sanitize(text) == text

    def test_the_tools_own_styling_survives(self):
        # Sanitizing happens on the way IN; styling is applied afterwards, so
        # the escapes nodetop emits deliberately are never touched.
        assert "\x1b[" in COLOR.bad(sanitize("clean text"))

    def test_a_hostile_cell_does_not_break_the_table(self):
        nasty = sanitize("drained \x1b[2J\rhidden\nnewline")
        out = table(["node", "reason"], [["n1", nasty]], style=PLAIN, size=60)
        assert len(out.splitlines()) == 3          # header, rule, one row
        for ch in ("\x1b", "\r", "\n"):
            assert ch not in nasty
        assert "\x1b" not in out

    def test_it_is_applied_at_the_model_boundary(self):
        # One rule at the boundary rather than six in the backends: every
        # adapter is covered, and so is a replayed snapshot.
        from nodetop.core.model import Node, Queue, Verdict

        node = Node(name="n\x00", state_raw="IDLE\r", memory_mb=1,
                    reason="bad \x1b[2J node", labels=("a\tb",))
        assert node.reason == "bad  [2J node"
        assert node.state_raw == "IDLE " and node.labels == ("a b",)

        queue = Queue(name="q\n", state_raw="UP\r", allow_accounts=("a\x1bb",))
        assert queue.name == "q " and queue.allow_accounts == ("a b",)

        verdict = Verdict(queue="q", reason="no \rway", category="X\x00")
        assert verdict.reason == "no  way" and verdict.category == "X "


class TestTheRampReadsAsAvailability:
    """Three properties a meter needs, none of which the ramp had by accident.

    The ramp used to end in amber, so the *emptiest* node drew the most alarming
    colour on screen -- "why use the orange colour to denote an unoccupied cpu?"
    And the coldest fill was darker than the track it sat in, so a barely-full
    bar read inverted: the emptiness looked more present than the fill.
    """

    @staticmethod
    def _lum(rgb):
        red, green, blue = rgb
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    def test_every_fill_is_brighter_than_the_empty_track(self):
        from nodetop.render import _PALETTE, _WASH

        track = self._lum(_PALETTE["track"][0])
        for step, wash in enumerate(_WASH):
            assert self._lum(wash[0]) > track, f"step {step} fill is darker"

    def test_every_fill_is_darker_than_its_own_text_tone(self):
        # A bar is a slab and text is a line: the tone that reads as bright in a
        # number reads as shouting across sixteen filled cells.
        from nodetop.render import _RAMP, _WASH

        for step, (text, wash) in enumerate(zip(_RAMP, _WASH, strict=True)):
            assert self._lum(wash[0]) < self._lum(text[0]), f"step {step}"

    def test_every_text_tone_is_legible_against_the_background(self):
        from nodetop.render import _PALETTE, _RAMP

        track = self._lum(_PALETTE["track"][0])
        for step, tone in enumerate(_RAMP):
            assert self._lum(tone[0]) > track, f"step {step} is dimmer than chrome"

    def test_the_warm_end_is_gone(self):
        # Amber and gold are warning colours. The top of this ramp is the most
        # AVAILABLE resource, which is the opposite of a warning.
        from nodetop.render import _RAMP, _WASH

        for tone in list(_RAMP) + list(_WASH):
            red, green, blue = tone[0]
            if red > 150:                     # anything that could read as warm
                assert green > red or blue > red, f"rgb{tone[0]} reads as amber"

    def test_more_free_is_greener_and_less_is_bluer(self):
        # The ordering has to be visible as hue, since that is what carries it.
        from nodetop.render import _RAMP

        coldest, warmest = _RAMP[0][0], _RAMP[-1][0]
        assert coldest[2] > coldest[1], "the low end should read blue"
        assert warmest[1] > warmest[2], "the high end should read green"
