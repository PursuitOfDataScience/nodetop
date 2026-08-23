"""Every text renderer, in every output mode.

The JSON paths are asserted elsewhere; this file exists because the *text*
paths are where an unguarded ``None``, a missing attribute or a width bug
crashes, and none of that shows up in a JSON test.  Each command is rendered
in plain, coloured and ASCII modes, since the three take different branches.
"""

from __future__ import annotations

import pytest

from nodetop.cli import (
    build_parser,
    cmd_accelerators,
    cmd_backends,
    cmd_exclude,
    cmd_health,
    cmd_nodes,
    cmd_queues,
    cmd_status,
    cmd_where,
)
from nodetop.render import MIN_WIDTH, Glyphs, Style, width

MODES = {
    "plain": Style(depth=0, glyphs=Glyphs()),
    "truecolor": Style(depth=24, glyphs=Glyphs()),
    "ansi16": Style(depth=4, glyphs=Glyphs()),
    "ascii": Style(depth=0, glyphs=Glyphs.ascii()),
}


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


COMMANDS = [
    ("status", cmd_status, ["status"]),
    ("status --all", cmd_status, ["status", "--all"]),
    ("queues", cmd_queues, ["queues"]),
    ("queues --unusable-only", cmd_queues, ["queues", "--unusable-only"]),
    ("queues --detail", cmd_queues, ["queues", "--detail"]),
    ("nodes", cmd_nodes, ["nodes"]),
    ("nodes --gpu", cmd_nodes, ["nodes", "--gpu"]),
    ("nodes --cpu", cmd_nodes, ["nodes", "--cpu"]),
    ("nodes --free", cmd_nodes, ["nodes", "--free"]),
    ("health", cmd_health, ["health"]),
    ("where", cmd_where, ["where", "-g", "1"]),
    ("where --all", cmd_where, ["where", "-g", "1", "--all"]),
    ("where cpu-only", cmd_where, ["where", "-c", "2"]),
    ("where impossible", cmd_where, ["where", "-g", "8", "--needs", "fp8",
                                     "--gpu-mem", "999"]),
    ("exclude --gpu-nodes", cmd_exclude, ["exclude", "--gpu-nodes"]),
    ("exclude --unschedulable", cmd_exclude, ["exclude", "--unschedulable"]),
    ("exclude --degraded", cmd_exclude, ["exclude", "--degraded"]),
    ("accelerators", cmd_accelerators, ["accelerators"]),
    ("accel -q", cmd_accelerators, ["accel", "-q", "beagle3"]),
]


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
def test_every_renderer_runs(cluster, capsys, mode, _label, fn, argv):
    rc = fn(cluster, _args(argv), MODES[mode])
    assert rc in (0, 1, 2)
    capsys.readouterr()


@pytest.mark.parametrize("mode", list(MODES))
def test_backends_renders(capsys, mode):
    assert cmd_backends(None, _args(["backends"]), MODES[mode]) == 0
    capsys.readouterr()


class TestAsciiPurity:
    """The ASCII path must emit no non-ASCII bytes at all.

    A ``LANG=C`` terminal cannot encode a box-drawing character, and printing
    mojibake is worse than printing plain punctuation.
    """

    @pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
    def test_output_is_pure_ascii(self, cluster, capsys, _label, fn, argv):
        fn(cluster, _args(argv), MODES["ascii"])
        out = capsys.readouterr().out
        out.encode("ascii")  # raises if anything slipped through

    def test_backends_output_is_pure_ascii(self, capsys):
        cmd_backends(None, _args(["backends"]), MODES["ascii"])
        capsys.readouterr().out.encode("ascii")


class TestNoAnsiWhenDisabled:
    @pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
    def test_no_escape_sequences(self, cluster, capsys, _label, fn, argv):
        fn(cluster, _args(argv), MODES["plain"])
        assert "\033" not in capsys.readouterr().out


class TestLayoutStability:
    """Colour must not change what the terminal lays out."""

    @pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
    def test_colour_does_not_change_line_widths(self, cluster, capsys, _label, fn, argv):
        fn(cluster, _args(argv), MODES["plain"])
        plain = capsys.readouterr().out.splitlines()
        fn(cluster, _args(argv), MODES["truecolor"])
        painted = capsys.readouterr().out.splitlines()
        assert len(plain) == len(painted)
        assert [width(x) for x in plain] == [width(x) for x in painted]


def _prose(out: str) -> str:
    """Collapse whitespace before asserting on wrapped prose.

    Explanatory text is wrapped to the window, so a phrase can be split across
    lines. Asserting on the raw substring makes the test depend on the
    terminal width of whoever runs it.
    """
    return " ".join(out.split())


class TestContent:
    """A few assertions on meaning, so the renderers cannot go silently blank."""

    def test_status_names_the_backend_and_accounts_for_every_queue(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), MODES["plain"])
        out = _prose(capsys.readouterr().out)
        assert "slurm" in out
        # The funnel, not a DEAD block: what is broken is a count on this view,
        # and `queues` / `health` carry the detail. See TestStatus.
        assert "partitions" in out and "dead" in out
        assert "advertised" not in out

    def test_queues_table_summarises_the_blockers(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues", "--unusable-only"]), MODES["plain"])
        out = _prose(capsys.readouterr().out)
        assert "QUEUE_DISABLED" in out
        assert "blocked by" in out

    def test_queues_detail_shows_advertised_against_usable(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues", "--detail", "--unusable-only"]),
                   MODES["plain"])
        out = _prose(capsys.readouterr().out)
        assert "advertised" in out
        assert "QUEUE_DISABLED" in out

    def test_where_legend_describes_exactly_the_states_shown(self, cluster, capsys):
        # The legend used to be a fixed list of four, so `where` advertised
        # "wrong hardware" over a table containing no such row. It is now built
        # from the labels present -- both directions matter: an unexplained
        # glyph is a puzzle, an explained absent one is noise.
        from nodetop.cli import _VERDICT_LEGEND

        cmd_where(cluster, _args(["where", "-g", "1"]), MODES["plain"])
        out = capsys.readouterr().out
        in_table = {label for label in _VERDICT_LEGEND if label in out}
        assert in_table, "no verdict label rendered at all"
        for label, (_g, _r, gloss) in _VERDICT_LEGEND.items():
            if label in in_table:
                assert gloss in out, f"{label} shown with no legend entry"
            else:
                assert gloss not in out, f"{label} explained but never shown"

    def test_where_marks_a_self_computed_start_estimate(self, cluster, capsys):
        cmd_where(cluster, _args(["where", "-g", "1", "--all"]), MODES["plain"])
        out = capsys.readouterr().out
        # The marker and its key travel together: a bare asterisk in a table
        # with no key is a puzzle, and a key with no asterisk is noise. Which
        # of the two states this fixture produces is not the point.
        assert ("*" in out) == ("our estimate" in out)

    def test_where_offers_the_flags_for_the_best_option(self, cluster, capsys):
        # Not only for a run-now one: "it would queue on X" is still the answer
        # you act on, and rebuilding the flags by hand is where a mismatch with
        # what was actually checked creeps in.
        cmd_where(cluster, _args(["where", "-g", "1"]), MODES["plain"])
        out = capsys.readouterr().out
        assert "submit" in _prose(out)
        assert "--partition=" in out

    def test_health_admits_the_limits_of_a_keyword_scan(self, cluster, capsys):
        cmd_health(cluster, _args(["health"]), MODES["plain"])
        assert "reason-field" in _prose(capsys.readouterr().out)

    def test_nodes_flags_inferred_accelerator_memory(self, cluster, capsys):
        cmd_nodes(cluster, _args(["nodes", "--gpu"]), MODES["plain"])
        # A100 could be 40 or 80 GB; the "?" is the disclosure.
        assert "?" in capsys.readouterr().out

    def test_accelerators_reports_capability_reach(self, cluster, capsys):
        cmd_accelerators(cluster, _args(["accelerators"]), MODES["plain"])
        out = _prose(capsys.readouterr().out)
        assert "features" in out
        assert "bf16" in out and "fp8" in out

    def test_exclude_output_is_a_bare_nodelist_for_piping(self, cluster, capsys):
        cmd_exclude(cluster, _args(["exclude", "--gpu-nodes"]), MODES["plain"])
        out = capsys.readouterr().out.strip()
        assert "\n" not in out
        assert " " not in out


class TestFitsTheTerminal:
    """No rendered line may exceed the window width.

    A line wider than the terminal soft-wraps, and a wrapped row destroys the
    column alignment that makes the output readable.

    The exemptions are the lines that exist to be **copied** -- the submit-flag
    line and the bare host list from ``exclude``. An ellipsis or a hanging
    indent in either hands the reader a broken command, so those are data on
    stdout rather than display, and width does not apply to them.

    MIN_WIDTH is in the parameter list on purpose. It used to start at 60, and
    everything between the renderer's own floor and 60 went unmeasured: at 40
    a seven-column table, a hand-joined legend and a hand-built gauge row all
    overflowed, and the sweep never looked.
    """

    #: Prefix of the submit-flag line as *Slurm* spells it. Kept for the
    #: slurm-only sweeps below, but see tests/test_backend_render.py: PBS opens
    #: its flag line with "-q" and Kubernetes with "-n", so this prefix is not a
    #: general test for "a line meant to be copied". Point a sweep at another
    #: backend and use `_submit_lines` from that module instead.
    EXEMPT = ("  --",)
    #: Commands whose output is a value to paste, not a layout.
    DATA_COMMANDS = ("exclude",)

    @pytest.mark.parametrize("size", [MIN_WIDTH, 45, 50, 60, 80, 100, 120])
    @pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
    def test_no_line_overflows(self, cluster, capsys, monkeypatch, size, _label, fn, argv):
        monkeypatch.setenv("COLUMNS", str(size))
        fn(cluster, _args(argv), MODES["plain"])
        if _label.startswith(self.DATA_COMMANDS):
            return
        for line in capsys.readouterr().out.splitlines():
            if line.startswith(self.EXEMPT):
                continue
            assert width(line) <= size, f"{width(line)} > {size}: {line!r}"

    # The same sweep against a recording. `replayed=True` reaches rendering
    # paths a live cluster never does, and one of them -- the "access is
    # DECLARED, not confirmed" explanation -- was 148 columns wide precisely
    # because it was only ever rendered on a replay.
    @pytest.mark.parametrize("size", [MIN_WIDTH, 60, 100])
    @pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
    def test_no_line_overflows_on_a_replay(
        self, replayed_cluster, capsys, monkeypatch, size, _label, fn, argv
    ):
        monkeypatch.setenv("COLUMNS", str(size))
        fn(replayed_cluster, _args(argv), MODES["plain"])
        if _label.startswith(self.DATA_COMMANDS):
            return
        for line in capsys.readouterr().out.splitlines():
            if line.startswith(self.EXEMPT):
                continue
            assert width(line) <= size, f"{width(line)} > {size}: {line!r}"

    def test_the_replay_path_really_renders_its_own_explanation(
        self, replayed_cluster, capsys
    ):
        # Guards the guard: if the fixture stopped reaching that branch, the
        # sweep above would pass by rendering nothing new.
        cmd_where(replayed_cluster, _args(["where", "-g", "1"]), MODES["plain"])
        out = " ".join(capsys.readouterr().out.split())
        assert "DECLARED, not confirmed" in out
        assert "replayed snapshot" in out

    @pytest.mark.parametrize("size", [MIN_WIDTH, 60, 80, 100])
    def test_backends_fits(self, capsys, monkeypatch, size):
        monkeypatch.setenv("COLUMNS", str(size))
        cmd_backends(None, _args(["backends"]), MODES["plain"])
        for line in capsys.readouterr().out.splitlines():
            assert width(line) <= size


class TestTheHouseStyleIsUniform:
    """Every table looks the same, because they all go through one helper.

    Six commands predate the style and each kept the old look: bold capitals,
    and a rule both above and below the header row. The style is three keyword
    arguments deep, so relying on each call site to remember it did not work.
    `_grid` applies it, and this asserts no caller bypasses it.
    """

    TABLE_COMMANDS = [c for c in COMMANDS
                      if not c[0].startswith(("exclude", "health"))]

    @pytest.mark.parametrize("_label,fn,argv", TABLE_COMMANDS,
                             ids=[c[0] for c in TABLE_COMMANDS])
    def test_no_table_header_shouts(self, cluster, capsys, _label, fn, argv):
        fn(cluster, _args(argv), MODES["plain"])
        out = capsys.readouterr().out
        # A header row is the line directly under a rule of dashes. Any
        # all-caps word in it means a call site skipped _grid.
        lines = out.splitlines()
        for i, line in enumerate(lines[:-1]):
            bare = line.strip()
            if bare and set(bare) <= {"─", "-", " "} and len(bare) > 8:
                header = lines[i + 1]
                shouty = [w for w in header.split()
                          if w.isalpha() and w.isupper() and len(w) > 2]
                assert not shouty, f"{_label}: shouting header {shouty}"

    def test_grid_lowercases_whatever_it_is_given(self):
        from nodetop.cli import _grid

        out = _grid(["LOUD", "ALSO LOUD"], [["a", "b"]], style=MODES["plain"])
        assert "loud" in out
        assert "LOUD" not in out

    def test_grid_puts_one_rule_above_and_none_below(self):
        from nodetop.cli import _grid

        lines = _grid(["a", "b"], [["1", "2"]], style=MODES["plain"]).splitlines()
        assert set(lines[0].strip()) == {"─"}
        assert set(lines[2].strip()) != {"─"}   # the data row, not a second rule

    def test_grid_output_still_fits_the_window(self, monkeypatch):
        from nodetop.cli import _grid

        monkeypatch.setenv("COLUMNS", "50")
        out = _grid(["a" * 30, "b" * 30], [["x" * 30, "y" * 30]],
                    style=MODES["plain"])
        for line in out.splitlines():
            assert width(line) <= 50
