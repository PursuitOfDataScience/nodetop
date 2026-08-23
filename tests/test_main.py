"""Entry-point behaviour: backend selection, dispatch and exit status."""

from __future__ import annotations

import json

import pytest

from nodetop import backends as registry
from nodetop.cli import main
from nodetop.exceptions import NoBackendError


@pytest.fixture(autouse=True)
def _detect_finds_the_recorded_backend(request, monkeypatch, slurm_backend):
    """Make `main()` detect a recorded cluster instead of the host's own.

    **These tests were not hermetic and only passed where they were written.**
    `main()` does live backend detection and then queries it, so every
    `main(["status"])` below ran real `scontrol` calls -- fine on a login node,
    guaranteed to fail on CI, which has no batch system at all. That is the sort
    of green suite that goes red the first time somebody else runs it.

    Only `detect` is replaced, not `get`: the `--backend NAME` tests are about
    the real registry -- an unknown name, or forcing an adapter whose client is
    absent -- and substituting that would test nothing.
    """
    if request.node.get_closest_marker("live_detect"):
        # Opted out: a test that simulates a broken control plane has to own
        # detection itself, or it would be handed a working recorded one.
        return
    monkeypatch.setattr(registry, "detect", lambda: slurm_backend)


class TestOneReportIsOneSnapshot:
    """Every number in a report must describe the same instant.

    A documented guarantee with no test behind it. Re-reading a measurement
    mid-report is how "607 nodes, 549 up" ends up beside a table that adds to
    606: the two readings straddle a job finishing. Measured against the real
    backend the count is exactly one per query, so this pins it.

    Dry-runs are exempt and counted separately: a probe is an action taken
    against the control plane, not a measurement of it.
    """

    def test_each_measurement_is_fetched_once(self, monkeypatch, slurm_backend):
        import collections
        import contextlib
        import io as _io

        from nodetop.cli import build_parser, cmd_nodes, cmd_queues, cmd_status
        from nodetop.core.cluster import Cluster
        from nodetop.render import Glyphs, Style

        seen: collections.Counter = collections.Counter()
        inner = slurm_backend.runner
        original = inner.run

        def counting(cmd, timeout=None):
            seen[" ".join(cmd)] += 1
            return original(cmd, timeout=timeout)

        monkeypatch.setattr(inner, "run", counting)

        for fn, argv in ((cmd_status, ["status"]), (cmd_nodes, ["nodes"]),
                         (cmd_queues, ["queues"])):
            seen.clear()
            cluster = Cluster.load(slurm_backend)
            with contextlib.redirect_stdout(_io.StringIO()):
                fn(cluster, build_parser().parse_args(argv),
                   Style(depth=0, glyphs=Glyphs()))
            repeated = {k: v for k, v in seen.items()
                        if v > 1 and not k.startswith("sbatch")}
            assert not repeated, f"{argv[0]} re-read {repeated}"


@pytest.mark.live_detect
class TestABrokenControlPlaneIsNotAnEmptyCluster:
    """The failure this tool exists to prevent, reachable in the tool itself.

    With every scheduler command failing, four commands printed clean,
    confident, wrong answers and exited 0: `queues` said "0 shown, 0 usable",
    `nodes` said "all 0", `exclude --unschedulable` emitted an empty nodelist,
    and `health` -- the command whose entire purpose is to find out whether
    anything is wrong -- reported "0 degraded, 0 out", i.e. a perfectly healthy
    cluster. Only `status` mentioned the failures at all.
    """

    @staticmethod
    def _dead(monkeypatch):
        """A Slurm backend whose every query fails."""
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.exceptions import CommandError

        # Underscored because the double has to match the real signature
        # positionally and uses none of it.
        def boom(_self, _argv, _timeout=None):
            raise CommandError("cannot contact slurm controller")

        monkeypatch.setattr("nodetop.runner.SubprocessRunner.run", boom)
        monkeypatch.setattr("nodetop.runner.SubprocessRunner.run_full",
                            lambda _self, _argv, _timeout=None: (1, "", "dead"))
        monkeypatch.setattr(SlurmBackend, "detect", staticmethod(lambda: True))

    @pytest.mark.parametrize("argv", [
        ["status"], ["queues"], ["nodes"], ["health"], ["gpus"],
        ["where", "-g", "1"], ["exclude", "--unschedulable"],
    ])
    def test_every_command_exits_three(self, monkeypatch, capsys, argv):
        self._dead(monkeypatch)
        assert main(argv) == 3

    def test_it_prints_nothing_to_stdout(self, monkeypatch, capsys):
        # An empty nodelist on stdout is worse than no output: a script doing
        # `sbatch --exclude=$(nodetop exclude --unschedulable)` would submit
        # with no exclusions and believe it had them.
        self._dead(monkeypatch)
        main(["exclude", "--unschedulable"])
        assert capsys.readouterr().out == ""

    def test_it_names_the_queries_that_failed(self, monkeypatch, capsys):
        self._dead(monkeypatch)
        main(["health"])
        err = capsys.readouterr().err
        assert "query failed" in err
        assert "no data" in err

    def test_three_is_not_one(self, monkeypatch, capsys):
        # 1 means "nothing fits", which is a real answer. Reusing it for "we
        # could not ask" makes the two indistinguishable to a caller.
        self._dead(monkeypatch)
        assert main(["where", "-g", "1"]) == 3


class TestAMistypedQueueNameIsAnError:
    """A dropped keystroke used to produce a confident empty answer.

    `nodetop queues -q caslakke` printed nothing whatsoever and exited 0.
    `nodes -q caslakke` reported "0 of 607 nodes", which reads as a partition
    that exists and is idle. Both are the failure this tool is written against:
    an answer to a question that was never asked.
    """

    def test_it_exits_two_and_says_which_name(self, capsys):
        rc = main(["queues", "-q", "definitely-not-a-partition"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "no such partition" in err
        assert "definitely-not-a-partition" in err

    def test_a_near_miss_is_suggested(self, capsys):
        # With 87 partitions the intended one is usually one edit away, and
        # printing all 87 instead is how a helpful error becomes unreadable.
        rc = main(["queues", "-q", "tst"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "did you mean" in err and "test" in err

    def test_a_real_name_still_works(self, capsys):
        assert main(["queues", "-q", "test"]) == 0
        assert "no such" not in capsys.readouterr().err

    @pytest.mark.parametrize("verb", ["queues", "nodes", "where", "check", "exclude"])
    def test_every_command_taking_the_flag_is_guarded(self, capsys, verb):
        # Checked once at dispatch rather than in five commands, because five
        # copies of a validation is four chances to omit it.
        assert main([verb, "-q", "nope-not-here"]) == 2
        assert "no such" in capsys.readouterr().err

    def test_one_bad_name_among_good_ones_is_still_an_error(self, capsys):
        assert main(["queues", "-q", "test,nope-not-here"]) == 2
        assert "nope-not-here" in capsys.readouterr().err


class TestBackendSelection:
    def test_an_unknown_backend_name_exits_2_and_lists_the_options(self, capsys):
        rc = main(["--backend", "torque-classic", "status"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "unknown backend" in err
        assert "slurm" in err  # the options are named, not just rejected

    def test_forcing_an_undetected_backend_warns_but_proceeds(self, capsys, monkeypatch):
        # The detector is deliberately conservative and a caller may know
        # better; a missing client should not surface as five cryptic query
        # failures with no explanation.
        # The client has to be genuinely absent for this to mean anything: the
        # conftest fixture reports every scheduler client as installed, which is
        # what the rest of the suite needs and the opposite of what this asserts.
        from nodetop.backends.lsf import LsfBackend

        monkeypatch.setattr(LsfBackend, "detect", staticmethod(lambda: False))
        rc = main(["--backend", "lsf", "--json", "status"])
        err = capsys.readouterr().err
        assert "does not detect its system here" in err
        # And exits 3, because on this host it reported *nothing*: every LSF
        # query failed, so there are no nodes. That used to be rc 0 -- an empty
        # answer presented as a successful one, which is the same conflation
        # that had `health` reporting "0 out" during a total outage.
        assert rc == 3
        assert "no data" in err

    def test_no_backend_at_all_exits_3(self, capsys, monkeypatch):
        def refuse() -> None:
            raise NoBackendError("nothing usable")

        monkeypatch.setattr(registry, "detect", refuse)
        rc = main(["status"])
        err = capsys.readouterr().err
        assert rc == 3
        assert "nodetop backends" in err  # tells you how to find out why

    def test_backends_needs_no_backend_at_all(self, capsys, monkeypatch):
        # It must work on a machine where nothing is installed -- that is
        # precisely when you would run it.
        def refuse() -> None:
            raise NoBackendError("nothing usable")

        monkeypatch.setattr(registry, "detect", refuse)
        assert main(["backends", "--json"]) == 0
        capsys.readouterr()


class TestDispatch:
    def test_no_subcommand_defaults_to_status(self, capsys):
        assert main(["--json"]) == 0
        assert "backend" in capsys.readouterr().out

    # A bare `nodetop` -- the first thing anyone types -- crashed with
    # AttributeError: 'Namespace' object has no attribute 'all', because the
    # root parser does not carry the sub-command's own flags and the namespace
    # was dispatched into cmd_status regardless.  The test above missed it: the
    # --json branch returns before the renderer reads args.all.  So the
    # regression test has to render TEXT.
    @pytest.mark.parametrize("argv", [
        [],
        ["--no-color"],
        ["--ascii"],
        ["--no-color", "--ascii"],
    ])
    def test_the_bare_invocation_renders_text_without_crashing(self, capsys, argv):
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "nodetop" in out
        assert out.count("\n") > 5  # a dashboard, not one stray line

    # And a flag belonging to the default verb has to work on the bare
    # invocation too. The root parser does not know `--all`, so a strict first
    # parse rejected it before the default-verb fallback could run: `nodetop
    # --all` exited 2 with "unrecognized arguments" while `nodetop status --all`
    # worked -- and `--all` is the command the overview's own footer prints.
    @pytest.mark.parametrize("argv", [
        ["--all"],
        ["--declared"],
        ["--all", "--no-color"],
        ["--no-color", "--all"],
    ])
    def test_a_default_verb_flag_works_without_the_verb(self, capsys, argv):
        assert main(argv) == 0
        assert "nodetop" in capsys.readouterr().out

    def test_a_misspelled_flag_is_still_an_error(self, capsys):
        # The hazard of parse_known_args is that it makes a typo look like a
        # successful run. It must not swallow one, with or without a verb.
        for argv in (["--nonsens"], ["nodes", "--nonsens"]):
            with pytest.raises(SystemExit) as exit:
                main(argv)
            assert exit.value.code == 2
            assert "unrecognized arguments" in capsys.readouterr().err

    def test_the_bare_invocation_matches_the_spelled_out_verb(
        self, capsys, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        # The fix re-parses as ["status", *argv].  Asserting the two render
        # identically pins the property that matters -- the default verb is
        # the verb, not a second code path with its own defaults.
        #
        # Against a replay, not the live cluster. This used to invoke the live
        # cluster twice and assert the two renders were byte-identical, so any
        # job starting or finishing in between failed it -- observed flipping
        # "358 GPUs, 123 free" to "122 free" between the two calls. The
        # property under test is dispatch, not cluster state, so freezing the
        # state does not weaken it: a snapshot is the one input that cannot
        # churn under the assertion.
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.cli import build_parser, cmd_snapshot
        from nodetop.core.cluster import Cluster
        from nodetop.render import Glyphs, Style
        from nodetop.runner import CapturingRunner, RecordedRunner

        capture = CapturingRunner(RecordedRunner({
            "scontrol show node": (0, slurm_nodes, ""),
            "scontrol show partition": (0, slurm_partitions, ""),
            "show qos": (0, slurm_qos, ""),
            "show assoc": (0, "acct||beagle3\n", ""),
            "squeue": (0, "", ""),
        }))
        cluster = Cluster.load(SlurmBackend(capture), with_free_times=True)
        cluster.capture = capture
        snap = tmp_path / "snap.json"
        assert cmd_snapshot(
            cluster,
            build_parser().parse_args(["snapshot", "-o", str(snap)]),
            Style(depth=0, glyphs=Glyphs()),
        ) == 0
        capsys.readouterr()  # discard the snapshot's own report

        assert main(["--replay", str(snap), "--no-color"]) == 0
        bare = capsys.readouterr().out
        assert main(["--replay", str(snap), "status", "--no-color"]) == 0
        assert capsys.readouterr().out == bare

    # The re-parse moves a root-level flag to the right of the verb, so every
    # global has to be accepted in both positions.  _add_global_args re-adds
    # them to each sub-parser (with SUPPRESS defaults, so the sub-parser does
    # not clobber a value given before the verb) -- that is the mechanism this
    # guards.  A global added to the root alone would break a bare invocation
    # that used it.
    #: Each global flag, with a predicate that is true only if it took effect.
    #: Comparing the three spellings' output byte-for-byte was the obvious test
    #: and the wrong one: every invocation re-queries a live cluster, so node
    #: counts and the snapshot time move underneath it. Assert the flag was
    #: honoured, which is the actual claim, and is stable.
    GLOBAL_FLAGS = {
        "--json": lambda out: json.loads(out) is not None,
        "--no-color": lambda out: "\x1b[" not in out,
        "--ascii": lambda out: out.isascii(),
    }

    @pytest.mark.parametrize("flag", sorted(GLOBAL_FLAGS))
    @pytest.mark.parametrize("position", ["before", "after", "bare"])
    def test_every_global_flag_works_in_every_position(self, capsys, flag, position):
        argv = {
            "before": [flag, "status"],
            "after": ["status", flag],
            "bare": [flag],            # via the default-verb re-parse
        }[position]
        assert main(argv) == 0
        assert self.GLOBAL_FLAGS[flag](capsys.readouterr().out), (
            f"{flag} had no effect when given {position} the verb")

    @pytest.mark.parametrize("argv", [
        ["--json", "queues"],
        ["--json", "partitions"],
        ["--json", "nodes"],
        ["--json", "health"],
        ["--json", "accelerators"],
        ["--json", "accel"],
        ["--json", "backends"],
    ])
    def test_every_verb_and_alias_dispatches(self, capsys, argv):
        assert main(argv) == 0
        capsys.readouterr()

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "nodetop" in capsys.readouterr().out


class TestExitStatus:
    def test_where_exits_1_when_no_hardware_could_ever_host_it(self, capsys):
        rc = main(["--json", "where", "-g", "64", "--gpu-mem", "9999",
                   "--needs", "fp8"])
        capsys.readouterr()
        assert rc == 1

    def test_exclude_without_a_selector_exits_2(self, capsys):
        assert main(["exclude"]) == 2
        assert "at least one of" in capsys.readouterr().err
