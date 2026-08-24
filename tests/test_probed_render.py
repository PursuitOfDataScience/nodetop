"""Rendering once a dry-run has actually answered.

A probe adds an ACCESS column and unlocks headings that exist nowhere else --
"the submit filter and the scheduler disagree" among them. The main width sweep
runs against a cluster with no probe, so none of that had ever been measured,
and that heading is 46 columns wide: `section` was clipping its note and not
its title.

The four fixtures cover the four answers a control plane gives: accepted,
refused, and the two-layer case where a site submit filter contradicts the
scheduler underneath it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from nodetop.cli import cmd_check, cmd_where
from nodetop.render import MIN_WIDTH, Glyphs, Style, width
from tests.test_cli_render import _args

PLAIN = Style(depth=0, glyphs=Glyphs())

#: Every line these commands emit purely to be copied.
COPY_ME = ("  --", "  -q ", "  -n ")

ARGVS = [
    ["where", "-g", "1"],
    ["where", "-g", "1", "--all"],
    ["where", "-c", "2"],
    ["check", "-q", "beagle3", "-g", "1"],
    ["check", "-q", "beagle3", "-g", "8", "--needs", "fp8", "--gpu-mem", "999"],
]

FIXTURES = ["accepting_cluster", "refusing_cluster", "disagreeing_cluster",
            "cpu_only_cluster"]


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("size", [MIN_WIDTH, 50, 60, 80, 100])
@pytest.mark.parametrize("argv", ARGVS, ids=[" ".join(a) for a in ARGVS])
class TestProbedPathsFitTheTerminal:
    def test_no_line_overflows(self, request, capsys, monkeypatch, fixture, size, argv):
        monkeypatch.setenv("COLUMNS", str(size))
        cluster = request.getfixturevalue(fixture)
        fn = cmd_check if argv[0] == "check" else cmd_where
        fn(cluster, _args(argv), PLAIN)
        for line in capsys.readouterr().out.splitlines():
            if line.startswith(COPY_ME):
                continue
            assert width(line) <= size, (
                f"{fixture} {' '.join(argv)} @{size}: {width(line)}: {line!r}")

    def test_it_renders_something(self, request, capsys, monkeypatch, fixture,
                                  size, argv):
        # A command that silently produced nothing would pass the width check.
        monkeypatch.setenv("COLUMNS", str(size))
        cluster = request.getfixturevalue(fixture)
        fn = cmd_check if argv[0] == "check" else cmd_where
        fn(cluster, _args(argv), PLAIN)
        assert capsys.readouterr().out.strip()


class TestTheProbedPathsAreActuallyReached:
    """Guards the guard.

    If these fixtures stopped producing verdicts, the sweep above would still
    pass -- by rendering the same un-probed output the main sweep already
    covers, and measuring nothing new.
    """

    def test_a_probe_adds_the_access_column(self, accepting_cluster, capsys):
        cmd_where(accepting_cluster, _args(["where", "-g", "1"]), PLAIN)
        assert "access" in capsys.readouterr().out

    def test_the_disagreement_heading_is_reached(self, disagreeing_cluster, capsys):
        # The 46-column heading that section() was not clipping.
        cmd_check(disagreeing_cluster,
                  _args(["check", "-q", "beagle3", "-g", "1"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "disagree" in out

    def test_a_refusal_is_reported_as_such(self, refusing_cluster, capsys):
        cmd_check(refusing_cluster,
                  _args(["check", "-q", "beagle3", "-g", "1"]), PLAIN)
        assert capsys.readouterr().out.strip()


class TestAQueueAheadOfYouIsNotARunNow:
    """The scheduler's own start time outranks "there are free nodes".

    The `ACCEPTED` fixture predicts a start at 2026-08-22T18:00. Read against a
    snapshot taken six hours earlier, that is a wait -- and it used to render
    as RUN NOW / `now` because the row was decided by free hardware alone.
    """

    @staticmethod
    def _out(cluster, capsys, taken):
        cluster.taken_at = taken
        cmd_where(cluster, _args(["where", "-g", "1"]), PLAIN)
        return " ".join(capsys.readouterr().out.split())

    def test_a_predicted_wait_renders_as_a_queue(self, accepting_cluster, capsys):
        out = self._out(accepting_cluster, capsys, datetime(2026, 8, 22, 12))
        assert "QUEUE" in out
        assert "RUN NOW" not in out
        # And the wait itself, not a bare "now".
        assert "6h" in out

    def test_the_headline_says_nothing_starts_immediately(self, accepting_cluster,
                                                          capsys):
        out = self._out(accepting_cluster, capsys, datetime(2026, 8, 22, 12))
        assert "NOWHERE NOW" in out

    def test_the_submit_line_does_not_claim_it_starts_now(self, accepting_cluster,
                                                          capsys):
        out = self._out(accepting_cluster, capsys, datetime(2026, 8, 22, 12))
        assert "submit (queues)" in out

    def test_a_prediction_already_past_still_reads_as_now(self, accepting_cluster,
                                                          capsys):
        out = self._out(accepting_cluster, capsys, datetime(2026, 8, 22, 23))
        assert "RUN NOW" in out
        assert "QUEUE" not in out
