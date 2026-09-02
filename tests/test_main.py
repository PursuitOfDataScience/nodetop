"""Entry-point behaviour: backend selection, dispatch and exit status."""

from __future__ import annotations

import json
import time

import pytest

from nodetop import backends as registry
from nodetop.cli import main
from nodetop.exceptions import NoBackendError


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
        # ...and it reads as a sentence. The registry raises KeyError, whose
        # __str__ is repr(args[0]), so printing str(exc) wrapped the whole line
        # in quotes: "unknown backend 'torque-classic'; known: slurm, ...".
        first = err.strip().splitlines()[0]
        assert not first.startswith('"'), first
        assert not first.endswith('"'), first

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


class TestTheJobViewsWaitIsHiddenNotPaid:
    """Opening a node's job list was a visible stall, and it had two halves.

    That frame waits on two lazy queries, measured cold on a 2,116-job cluster:
    175 ms for `jobs` (from `squeue`) and 1042 ms for `allocations` (a 9 MB
    `scontrol show job -d` plus ~92 ms of parsing). Stepping into a partition's
    node list starts both, so the wait happens while the reader is choosing a
    row. With only the shares warmed, `jobs` still cost 126 ms on the keypress
    -- half a fix. With both: 0.3 ms after half a second of reading.
    """

    def test_it_warms_both_halves(self):
        # One of these used to be missed, and the frame needs both before it
        # can draw a single row.
        asked = []
        cluster = self._cluster(lambda: asked.append("allocations") or [])
        cluster._backend.load_jobs = lambda: asked.append("jobs") or []
        cluster.prefetch_job_view()
        for _ in range(200):
            if len(asked) == 2:
                break
            time.sleep(0.01)
        # `jobs` first: the frame reads it first and it is six times cheaper.
        assert asked == ["jobs", "allocations"], asked

    @staticmethod
    def _cluster(loader):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import BackendCapabilities

        class _Backend:
            name, queue_term = "test", "queue"

            def load_allocations(self):
                return loader()

            def load_jobs(self):
                return []

            def capabilities(self):
                return BackendCapabilities(probe=False)

        return Cluster(backend_name="test", capabilities=BackendCapabilities(),
                       _backend=_Backend())

    def test_it_returns_before_the_work_is_done(self):
        import threading
        import time

        started, release = threading.Event(), threading.Event()

        def slow():
            started.set()
            release.wait(5)
            return []

        cluster = self._cluster(slow)
        begun = time.time()
        cluster.prefetch_job_view()
        assert time.time() - begun < 0.5      # it must not block the frame
        assert started.wait(5), "the fetch never started"
        release.set()

    def test_the_view_and_the_prefetch_do_not_both_fetch(self):
        # One whole-cluster query is the most expensive duplicate available.
        calls = []

        def counted():
            calls.append(1)
            return []

        cluster = self._cluster(counted)
        cluster.prefetch_job_view()
        cluster.allocations()
        cluster.allocations()
        assert len(calls) == 1, calls

    def test_a_failing_prefetch_is_recorded_not_raised(self):
        def boom():
            raise RuntimeError("scontrol died")

        cluster = self._cluster(boom)
        cluster.prefetch_job_view()
        # The view still works, and says what went wrong.
        assert cluster.allocations() == {}
        assert "allocations" in cluster.errors

    def test_it_does_nothing_once_the_answer_is_in_hand(self):
        calls = []
        cluster = self._cluster(lambda: calls.append(1) or [])
        cluster.allocations()
        cluster.prefetch_job_view()
        cluster.prefetch_job_view()
        assert len(calls) == 1


class TestOneSnapshotIsTakenConcurrently:
    """The queries behind a snapshot are independent reads, so they overlap.

    Measured on a 607-node cluster, five interleaved rounds: 0.57s issuing them
    one after another against 0.23s together, out of a 10.26s run whose 3.53s
    of queries this was. Nothing races on the control plane's side -- these
    commands only look -- and the readings land closer together in time, which
    is the property this class exists to provide.
    """

    def test_a_failure_in_one_query_is_still_recorded_against_its_name(self):
        # The errors dict is written from several threads now.
        from nodetop.core.cluster import Cluster

        class _Half:
            name, queue_term = "half", "queue"

            def load_nodes(self):
                raise RuntimeError("nodes are gone")

            def load_queues(self):
                raise RuntimeError("queues are gone")

            def load_limits(self):
                return {}

            def load_identity(self):
                return None

            def load_node_free_times(self):
                raise RuntimeError("no free times")

            def capabilities(self):
                from nodetop.core.model import BackendCapabilities

                return BackendCapabilities(probe=False)

        cluster = Cluster.load(_Half(), with_free_times=True)
        assert sorted(cluster.errors) == ["free_times", "nodes", "queues"]
        assert "nodes are gone" in cluster.errors["nodes"]
        assert cluster.nodes == [] and cluster.queues == {}

    def test_a_shared_answer_is_still_fetched_once(self):
        # Slurm's `load_nodes` and `load_identity` both read `scontrol show
        # config`, and they now run together: without a lock they would each
        # fetch it, which is two readings of one file in one report.
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        runner = RecordedRunner({
            "scontrol show node": (
                0, "NodeName=n1 CPUTot=8 State=IDLE Partitions=p\n", ""),
            "scontrol show partition": (
                0, "PartitionName=p\n   State=UP\n   TotalNodes=1\n", ""),
            "scontrol show config": (
                0, "SelectTypeParameters = CR_CORE_MEMORY\n"
                   "DefMemPerCPU = 1000\nAccountingStorageEnforce = associations\n", ""),
            "sacctmgr": (0, "acct||qos\n", ""),
            "squeue": (0, "", ""),
        })
        Cluster.load(SlurmBackend(runner), with_free_times=True)
        config = [c for c in runner.calls if "config" in " ".join(c)]
        assert len(config) == 1, config


class TestABrowseCanAskForFresherNumbers:
    """A browse renders ONE read for as long as it is open, and clusters move.

    Every number in a report has to describe one instant, so the frame is a
    single snapshot and deliberately not re-rendered in place. That leaves the
    reader studying figures that stopped being true minutes ago, with nothing
    saying so. `r` -- and an idle interval where a re-read is cheap -- asks for
    the whole reading again; `main` takes it and the browse reopens on the row
    it was on.
    """

    def _run(self, monkeypatch, replies):
        """Drive `main` with scripted `select` results; count the loads."""
        import nodetop.interactive as inter
        from nodetop.core.cluster import Cluster

        loads = {"n": 0}
        real = Cluster.load

        @classmethod
        def counting(_cls, *a, **kw):
            loads["n"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(Cluster, "load", counting)
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        answers = iter(replies)
        seen: list[dict] = []

        def scripted(render, count, **kw):
            seen.append(kw)
            got = next(answers, inter.Key.QUIT)
            render(0 if not isinstance(got, int) else min(got, max(0, count - 1)))
            return got

        monkeypatch.setattr(inter, "select", scripted)
        rc = main(["status"])
        return rc, loads["n"], seen

    def test_a_reload_takes_the_reading_again(self, monkeypatch, capsys):
        import nodetop.interactive as inter

        rc, loads, _ = self._run(
            monkeypatch, [inter.Key.RELOAD, inter.Key.RELOAD, inter.Key.QUIT])
        capsys.readouterr()
        assert rc == 0            # and never the sentinel
        assert loads == 3         # the first read, then one per reload

    def test_it_reopens_where_the_reader_was(self, monkeypatch, capsys):
        # Two levels down, then a reload: the rebuilt browse must not dump the
        # reader back at the top of the partition list.
        import nodetop.interactive as inter
        from nodetop.cli import _RESUME_CURSORS, _RESUME_STACK

        _RESUME_STACK.clear()
        _RESUME_CURSORS.clear()
        rc, loads, _ = self._run(monkeypatch, [0, 0, inter.Key.RELOAD])
        capsys.readouterr()
        assert rc == 0
        assert loads == 2
        # Consumed by the reopened browse, so it cannot leak into the next run.
        assert not _RESUME_STACK and not _RESUME_CURSORS

    def test_the_sentinel_never_escapes_as_an_exit_status(self, monkeypatch, capsys):
        import nodetop.interactive as inter
        from nodetop.cli import RELOAD

        rc, _, _ = self._run(monkeypatch, [inter.Key.RELOAD, inter.Key.QUIT])
        capsys.readouterr()
        assert rc != RELOAD
        assert rc == 0

    def test_a_slow_rebuild_switches_the_timer_off(self, monkeypatch, capsys):
        """The dry-runs count towards the cost, even though the load does not.

        `status` re-runs a dry-run per partition, so a rebuild can cost far more
        than the queries it starts with -- ~20s against 3.3s on one 607-node
        cluster. Pacing off `load_seconds` alone would stall the view under the
        reader's hands, so the browse stamps what the rebuild *actually* took at
        the moment it has something to show. Simulated here by making the
        dry-run pass slow, which is the thing that is missing from
        `load_seconds`.
        """
        import nodetop.cli as cli
        import nodetop.interactive as inter
        from nodetop.cli import _TURNAROUND

        monkeypatch.setenv("NODETOP_ACCESS_TTL", "0")   # no remembered answer
        real = cli.rank

        def slow(cluster, shape, **kw):
            if kw.get("use_probe"):
                time.sleep(1.2)
            return real(cluster, shape, **kw)

        monkeypatch.setattr(cli, "rank", slow)
        _TURNAROUND.clear()
        try:
            _rc, _loads, seen = self._run(monkeypatch, [inter.Key.QUIT])
            capsys.readouterr()
            assert seen[0].get("idle") is None
        finally:
            _TURNAROUND.clear()

    def test_the_reader_sitting_there_is_not_part_of_the_cost(self, monkeypatch,
                                                              capsys):
        """What the previous version of this measured, and why it was wrong.

        `main` timed the whole dispatch, and for an interactive command that
        includes however long the reader looked at the screen. So the first idle
        refresh after five seconds concluded a rebuild costs five seconds and
        switched the refresh off for good: **one re-read in a hundred idle
        seconds**, where the point was a paced series of them. Found by counting
        query bursts against a live session, not by a test.
        """
        import nodetop.cli as cli
        import nodetop.interactive as inter

        monkeypatch.setenv("NODETOP_ACCESS_TTL", "0")
        cli._TURNAROUND.clear()
        cli._IDLE_BACKOFF[0] = 0

        def dawdling(render, _count, **_kw):
            render(0)
            time.sleep(0.6)          # the reader, thinking
            return inter.Key.QUIT

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", dawdling)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        cli.main(["status"])
        capsys.readouterr()
        assert cli._TURNAROUND, "the browse never stamped what the rebuild cost"
        assert cli._TURNAROUND[-1] < 0.6, cli._TURNAROUND

    def test_an_idle_interval_is_offered_only_where_a_re_read_is_cheap(
            self, monkeypatch, capsys):
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "0")   # see the test above
        # 0.07s on one cluster, 70s on another. A fixed interval would either
        # waste the first or wreck the second, so the browse asks what the last
        # read cost.
        import dataclasses

        import nodetop.interactive as inter
        from nodetop.core.cluster import Cluster

        real = Cluster.load
        for cost, expected in ((0.05, True), (9.0, False)):

            @classmethod
            def priced(_cls, *a, c=cost, **kw):
                return dataclasses.replace(real(*a, **kw), load_seconds=c)

            monkeypatch.setattr(Cluster, "load", priced)
            _rc, _loads, seen = self._run(monkeypatch, [inter.Key.QUIT])
            capsys.readouterr()
            assert bool(seen[0].get("idle")) is expected, (cost, seen[0])


class TestClosingThePipeIsNotAnError:
    """`nodetop health | head` is how a long report gets read.

    Python's default SIGPIPE disposition makes that print "Exception ignored
    in: <_io.TextIOWrapper name='<stdout>'> BrokenPipeError: [Errno 32]" after
    the output, because the interpreter flushes a closed stdout at shutdown.
    Seen on a 10,624-node cluster, where `health` lists 147 unschedulable nodes
    and piping is the only way to read it.
    """

    def test_the_default_disposition_is_restored(self, capsys):
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_IGN)   # the Python default
        main(["--json"])
        assert signal.getsignal(signal.SIGPIPE) == signal.SIG_DFL
        capsys.readouterr()

    def test_a_real_pipe_closing_early_stays_quiet(self):
        # End to end through a shell, because the noise this prevents is
        # emitted by the interpreter at shutdown rather than by any code path a
        # unit test can reach.
        import subprocess
        import sys

        out = subprocess.run(
            f"{sys.executable} -m nodetop backends --no-color | head -2",
            shell=True, capture_output=True, text=True,
        )
        assert "BrokenPipeError" not in out.stderr
        assert "Exception ignored" not in out.stderr


class TestABrowseNobodyIsWatchingBacksOff:
    """The refresh interval doubles while nobody is there, and any key resets it.

    A browse re-reads by itself when a re-read is cheap, and remembering the
    access answer made it cheap on a cluster where it never used to be. So a
    terminal that used to sit still started re-reading every 6.6s forever, six
    queries at a time -- fine for one reader, and 45 queries a second if fifty
    people leave one open. Counted with `ps` against a live session, which is
    how it was noticed at all.
    """

    def test_the_interval_doubles_each_untouched_refresh(self):
        from nodetop.cli import _IDLE_BACKOFF, _IDLE_CAP, _IDLE_FIRED, _touched

        class Keys:
            RELOAD = "\x00reload"

        _IDLE_BACKOFF[0] = 0
        _IDLE_FIRED[0] = False
        seen = []
        for _ in range(12):
            seen.append(min(6.6 * (2 ** _IDLE_BACKOFF[0]), _IDLE_CAP))
            # what `still_waiting` does when the quiet timer expires
            _IDLE_BACKOFF[0] += 1
            _IDLE_FIRED[0] = True
            _touched(Keys.RELOAD, Keys)
        assert seen[:4] == [6.6, 13.2, 26.4, 52.8]
        assert seen[-1] == _IDLE_CAP          # and it stops there
        _IDLE_BACKOFF[0] = 0

    def test_a_keypress_puts_it_back(self):
        from nodetop.cli import _IDLE_BACKOFF, _IDLE_FIRED, _touched

        class Keys:
            RELOAD = "\x00reload"

        _IDLE_BACKOFF[0] = 5
        _IDLE_FIRED[0] = False
        _touched(3, Keys)                     # a row was opened
        assert _IDLE_BACKOFF[0] == 0

    def test_an_explicit_reload_puts_it_back_too(self):
        # `r` and the timer both come back as RELOAD, so the timer flags itself
        # on the way out; anything else is a person.
        from nodetop.cli import _IDLE_BACKOFF, _IDLE_FIRED, _touched

        class Keys:
            RELOAD = "\x00reload"

        _IDLE_BACKOFF[0] = 5
        _IDLE_FIRED[0] = False                # not the timer: someone pressed r
        _touched(Keys.RELOAD, Keys)
        assert _IDLE_BACKOFF[0] == 0

    def test_a_browse_hands_the_backed_off_interval_to_select(self, monkeypatch,
                                                              capsys):
        import nodetop.cli as cli
        import nodetop.interactive as inter

        monkeypatch.setenv("NODETOP_ACCESS_TTL", "0")   # no recheck poll here
        cli._IDLE_BACKOFF[0] = 3
        seen = []

        def scripted(render, _count, **kw):
            seen.append(kw.get("idle"))
            render(0)
            return inter.Key.QUIT

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", scripted)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        monkeypatch.setattr(cli, "_TURNAROUND", [0.05])
        try:
            cli.main(["status"])
        finally:
            cli._IDLE_BACKOFF[0] = 0
        capsys.readouterr()
        # 8x whatever the base was, or the cap.
        assert seen and seen[0] is not None
        assert seen[0] >= 5.0 * 8 or seen[0] == cli._IDLE_CAP


class TestCtrlCIsAnAnswerNotACrash:
    """An interrupt before the browse opens had nowhere to land.

    `select` has always caught it -- Ctrl-C while browsing quits -- but the
    window a reader is most likely to use it in is the wait *before* the first
    frame: the dry-runs are 1.6s of a 1.93s `status`. There the interrupt
    arrived inside the probe pool's shutdown join and printed twenty lines of
    `threading.py ... waiter.acquire()`. Verified against the real controller
    afterwards: exit **130** in 0.5-1.0s (one in-flight `sbatch`), no traceback.
    """

    def test_the_load_being_interrupted_exits_130(self, monkeypatch, capsys):
        import nodetop.cli as cli

        def interrupted(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli.Cluster, "load", staticmethod(interrupted))
        assert cli.main(["queues"]) == 130
        # 130 is 128 + SIGINT, which is what a shell reports for Ctrl-C.
        assert "Traceback" not in capsys.readouterr().err

    def test_a_command_being_interrupted_exits_130(self, monkeypatch, capsys):
        # The other side of the same window: the load finished, the dry-runs are
        # running, and the reader gives up.
        import nodetop.cli as cli

        monkeypatch.setitem(cli._COMMANDS, "queues",
                            lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt))
        assert cli.main(["queues"]) == 130
        capsys.readouterr()

    def test_the_progress_line_is_closed_before_the_prompt_returns(
            self, monkeypatch, capsys):
        # The ticker writes with `\r` and no newline, so an interrupt mid-count
        # would leave the shell prompt inside "checking ... 5/19".
        import nodetop.cli as cli

        def interrupted(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli.Cluster, "load", staticmethod(interrupted))
        cli.main(["queues"])
        assert capsys.readouterr().err.endswith("\n")


class TestAnOldInterpreterSaysSoInsteadOfBlamingTheCluster:
    """The floor is stated where the failure happens.

    `pip` refuses the install on an old Python, but the way this tool reaches a
    login node is a clone and `PYTHONPATH=src python3 -m nodetop` -- and
    `python3` there is whatever the distribution ships, 3.9 on RHEL 9. Nothing
    fails to import, so on that site the tool ran, every scheduler query died
    on `TypeError: zip() takes no keyword arguments`, and the report read "every
    query failed, so there is nothing to report" -- an accusation against a
    perfectly healthy cluster.
    """

    def test_it_names_the_version_and_stops(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.version_info", (3, 9, 25, "final", 0))
        assert main(["status"]) == 2
        err = capsys.readouterr().err
        assert "3.10" in err          # what is needed
        assert "3.9.25" in err        # what is here
        assert "python3.11" in err    # and how to get out of it

    def test_a_current_interpreter_is_not_lectured(self, capsys):
        assert main(["--json"]) == 0
        assert "needs Python" not in capsys.readouterr().err


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
            "show assoc": (0, "acct||gn\n", ""),
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
