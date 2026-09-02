"""Opening on the last answer, and correcting it on screen.

Starting `status` on a 607-node cluster costs 1.93s and **1.60s of it is
dry-runs** -- and that 1.60s is the controller running the site's submit plugin,
serialised: `sbatch --help` is 14.8 ms, `scontrol ping` 15.8 ms, `sbatch
--test-only` 98 ms. It cannot be made smaller from here, so a session spends it
differently: it opens on what the last run was told, re-asks in the background,
and reloads itself if the answer moved. Measured through a pty against the live
cluster: **first frame 328 ms instead of 1.9s**, and the recheck lands a second
or two later.

The tests here are about the promises that makes:

* a printout never uses a remembered answer -- it gets one shot at being right;
* the recheck always runs, so the screen converges on the truth within seconds
  rather than within the cache's lifetime;
* an answer to a *different* question is never reused;
* and none of this can fail loudly: no HOME, a read-only directory, half a file.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest

from nodetop.core import access


class TestTheFileIsAConvenienceNeverADependency:
    def test_nowhere_to_put_it_is_not_an_error(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("HOME", raising=False)
        assert access.directory() is None
        assert access.path("k") is None
        assert access.load("k") is None
        assert access.save("k", {}) is False

    def test_half_a_file_is_a_miss_and_the_next_save_repairs_it(self, tmp_path,
                                                               monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert access.save("k", {"a": access.YES})
        target = pathlib.Path(access.path("k"))
        target.write_text('{"version": 1, "at": ')
        assert access.load("k") is None
        assert access.save("k", {"a": access.YES})
        assert access.load("k")[0] == {"a": access.YES}

    def test_a_document_from_another_version_is_left_alone(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        target = pathlib.Path(access.path("k"))
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"version": 99, "verdicts": {"a": "yes"}}))
        assert access.load("k") is None

    def test_an_unwritable_directory_is_not_an_error(self, tmp_path, monkeypatch):
        wall = tmp_path / "wall"
        wall.write_text("not a directory")
        monkeypatch.setenv("XDG_CACHE_HOME", str(wall))
        assert access.save("k", {}) is False
        assert access.load("k") is None

    def test_a_stale_answer_is_a_miss(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES},
                    now=time.time() - access.DEFAULT_TTL - 1)
        assert access.load("k") is None

    def test_a_partial_answer_is_still_an_answer(self, tmp_path, monkeypatch):
        """What it knows, even when that is not everything.

        The verdicts are independent questions, so a partition with no
        remembered verdict is a gap to probe rather than a reason to re-ask
        about all of them. Measured before this: one partition gaining room
        between two runs cost a full pass, **2.2s** where the runs either side
        took 0.33s.
        """
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES, "b": access.NO})
        known, age = access.load("k")
        assert known == {"a": access.YES, "b": access.NO}
        assert age < 5
        # The caller works out the gap; this function does not hide it.
        assert [n for n in ("a", "b", "c") if n not in known] == ["c"]

    def test_what_it_learns_is_merged_not_replaced(self, tmp_path, monkeypatch):
        # The gap-probe writes only the gap, so the rest has to survive.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES, "b": access.NO})
        access.save("k", {"c": access.MAYBE})
        assert access.load("k")[0] == {
            "a": access.YES, "b": access.NO, "c": access.MAYBE}

    def test_a_carried_forward_verdict_ages_on_its_own_clock(self, tmp_path,
                                                             monkeypatch):
        """A refusal nobody re-asks about still has to expire.

        `save` merges, because a run only asks about the partitions with room
        right now. One timestamp for the whole document meant the merge
        restamped the answers this run never re-asked: a partition that stops
        having room stops being a candidate, is never probed again, and its
        remembered `no` is renewed by every unrelated write. Running the tool
        once a day is enough -- the `no` below is twelve days old and came back
        as "checked 1s ago", the refusal-outliving-the-fix that `ttl()` rejects
        `inf` to prevent, arriving through the merge instead of the knob.
        """
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        window = access.DEFAULT_TTL
        start = 1_700_000_000.0
        access.save("k", {"gpu": access.NO, "amd": access.YES}, now=start)
        # Every later run finds room only in `amd`, so `gpu` is never re-asked.
        when = start
        for day in range(1, 15):
            when = start + day * 0.9 * window
            access.save("k", {"amd": access.YES}, now=when)
        known, age = access.load("k", now=when + 1)
        assert known == {"amd": access.YES}
        assert age < 5

    def test_a_verdict_the_run_did_re_ask_about_keeps_its_place(self, tmp_path,
                                                               monkeypatch):
        # The control for the test above. Ageing each verdict separately must
        # not throw away one the run *did* re-ask about: that would make every
        # gap-probe a full nineteen-partition pass again. Same twelve days,
        # `gpu` asked about at every step.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        window = access.DEFAULT_TTL
        start = 1_700_000_000.0
        when = start
        for day in range(15):
            when = start + day * 0.9 * window
            access.save("k", {"gpu": access.NO, "amd": access.YES}, now=when)
        assert access.load("k", now=when + 1)[0] == {
            "gpu": access.NO, "amd": access.YES}

    def test_a_file_from_before_the_per_verdict_clock_still_reads(
            self, tmp_path, monkeypatch):
        # The other control: the per-verdict times are additive, so there is no
        # version bump, so a document an earlier release wrote has none of them
        # and has to go on ageing as a whole -- fresh here, stale past the TTL.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        start = 1_700_000_000.0
        where = pathlib.Path(access.path("k"))
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps({
            "version": access.VERSION,
            "at": start,
            "verdicts": {"gpu": access.NO},
        }), encoding="utf-8")
        assert access.load("k", now=start + 100)[0] == {"gpu": access.NO}
        assert access.load("k", now=start + access.DEFAULT_TTL + 1) is None

    def test_a_fresh_verdict_wins_over_a_remembered_one(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES})
        access.save("k", {"a": access.NO})
        assert access.load("k")[0] == {"a": access.NO}

    def test_the_ttl_can_be_switched_off_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES})
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "0")
        assert access.load("k") is None
        assert access.save("k", {"a": access.YES}) is False

    def test_a_nonsense_ttl_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "soon")
        assert access.ttl() == access.DEFAULT_TTL

    def test_the_file_does_not_grow_without_bound(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        last = access.KEEP + 11
        for i in range(last + 1):
            access.save(f"k{i}", {"a": access.YES})
        kept = {p.name for p in pathlib.Path(access.directory()).iterdir()}
        assert len(kept) == access.KEEP
        # Pruned by when each was last written, so the ones a reader has been
        # using survive and the ones nobody has touched go.
        assert f"k{last}.json" in kept and "k0.json" not in kept

    def test_it_is_not_world_readable(self, tmp_path, monkeypatch):
        # It names the partitions an account may submit to, which is nobody
        # else's business on a shared login node.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"a": access.YES})
        mode = pathlib.Path(access.path("k")).stat().st_mode
        assert mode & 0o077 == 0, oct(mode)


class TestTwoOfThemAtOnce:
    """A login node runs more than one of these, and one of them is a daemon thread.

    The write is a temporary file renamed over the target, so the file a reader
    opens is always a whole document. What two writers can lose is each other's
    *newest* verdicts -- both merge onto whatever they read, and the later rename
    wins. That costs one dry-run next time, which is why it is acceptable and
    why it is written down here rather than locked against.
    """

    def test_a_dozen_writers_leave_a_readable_file(self, tmp_path, monkeypatch):
        import threading

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        done = []

        def writer(n):
            for i in range(12):
                access.save(f"k{n}", {f"p{n}-{i}": access.YES})
            done.append(n)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert len(done) == 12
        # Whole document, and every writer's key present -- they touch different
        # keys, so nothing here should be lost at all.
        # Separate files, so nothing can be lost: twelve writers, twelve
        # answers. The shared-document version lost five of them.
        kept = sorted(p.name for p in pathlib.Path(access.directory()).iterdir())
        assert len(kept) == 12, kept
        for n in range(12):
            assert access.load(f"k{n}") is not None

    def test_no_half_written_file_is_ever_visible(self, tmp_path, monkeypatch):
        # A reader that opens the file while a writer is inside `save` must
        # never see a truncated document. `os.replace` is what guarantees that;
        # this checks the guarantee rather than assuming it.
        import threading

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        access.save("k", {"p0": access.YES})
        stop = threading.Event()
        bad = []

        def hammer():
            while not stop.is_set():
                got = access.load("k")
                if got is None:
                    bad.append("unreadable")

        reader = threading.Thread(target=hammer)
        reader.start()
        try:
            for i in range(150):
                access.save("k", {f"p{i}": access.YES})
        finally:
            stop.set()
            reader.join(10)
        assert bad == []

    def test_no_stray_temporary_files_are_left(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        for i in range(5):
            access.save("k", {f"p{i}": access.YES})
        left = [p.name for p in pathlib.Path(access.directory()).iterdir()]
        assert left == ["k.json"], left


class TestOneKeyPerQuestion:
    BASE = {
        "backend": "slurm", "cluster": "p1|p2", "user": "youzhi",
        "accounts": ["a", "b"], "shape": "nodes=1,cpus=1",
    }

    def test_the_same_question_is_the_same_key(self):
        assert access.key(**self.BASE) == access.key(**{
            **self.BASE, "accounts": ["b", "a"]})

    @pytest.mark.parametrize("field,value", [
        ("backend", "pbs"),
        ("cluster", "p1|p2|p3"),
        ("user", "someone-else"),
        ("accounts", ["a"]),
        ("shape", "nodes=2,cpus=1"),
    ])
    def test_a_different_question_is_a_different_key(self, field, value):
        # A new partition, another user's home, a bigger job: all of them are
        # questions the remembered answer does not answer.
        assert access.key(**{**self.BASE, field: value}) != access.key(**self.BASE)


class TestWhatASessionMayOpenOnAndAPrintoutMayNot:
    """The wiring, driven through `main` with a scripted `select`."""

    @staticmethod
    def _run(monkeypatch, argv, *, interactive=True, replies=()):
        """Run `main`, counting dry-run passes and capturing select's kwargs."""
        import nodetop.cli as cli
        import nodetop.core.fit as fit
        import nodetop.interactive as inter

        probes = {"n": 0}
        real_rank = fit.rank

        def counting(cluster, shape, **kw):
            if kw.get("use_probe"):
                probes["n"] += 1
            return real_rank(cluster, shape, **kw)

        monkeypatch.setattr(cli, "rank", counting)
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: interactive)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        answers, seen = iter(replies), []

        def scripted(render, _count, **kw):
            seen.append(kw)
            got = next(answers, inter.Key.QUIT)
            render(0)
            return got

        monkeypatch.setattr(inter, "select", scripted)
        rc = cli.main(argv)
        return rc, probes["n"], seen

    def test_the_first_run_probes_and_writes_it_down(self, monkeypatch, capsys,
                                                    tmp_path):
        rc, probes, _ = self._run(monkeypatch, ["status"])
        capsys.readouterr()
        assert rc == 0 and probes >= 1
        # Written for the next run, whose first frame is what this buys.
        kept = list(pathlib.Path(access.directory()).glob("*.json"))
        assert len(kept) == 1, kept

    def test_the_second_run_opens_on_it_and_re_asks_behind_the_frame(
            self, monkeypatch, capsys):
        first, probes_first, _ = self._run(monkeypatch, ["status"])
        capsys.readouterr()
        second, probes_second, seen = self._run(monkeypatch, ["status"])
        capsys.readouterr()
        assert first == 0 and second == 0
        # The frame is drawn from the remembered answer, and the recheck runs
        # anyway -- so the probe count is not zero, it is just no longer in
        # front of the reader.
        assert probes_second >= 1
        # The browse polls while the recheck is in flight.
        assert seen and seen[0].get("on_idle") is not None
        assert seen[0].get("idle") == pytest.approx(0.3)

    def test_a_printout_never_opens_on_a_remembered_answer(self, monkeypatch,
                                                           capsys):
        self._run(monkeypatch, ["status"])
        capsys.readouterr()
        # Not a terminal: `status` prints once and must be right when it does.
        _rc, _probes, seen = self._run(monkeypatch, ["status"], interactive=False)
        out = capsys.readouterr().out
        assert seen == []                      # no browse at all
        assert "access checked" not in out     # and nothing to disclose

    def test_json_never_opens_on_a_remembered_answer(self, monkeypatch, capsys):
        self._run(monkeypatch, ["status"])
        capsys.readouterr()
        _rc, _probes, seen = self._run(monkeypatch, ["status", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert seen == []
        assert "access_age" not in payload

    def test_declared_asks_nothing_and_remembers_nothing(self, monkeypatch,
                                                         capsys):
        _rc, probes, _ = self._run(monkeypatch, ["status", "--declared"])
        capsys.readouterr()
        assert probes == 0
        assert not pathlib.Path(access.directory()).exists()


class TestTheRecheckCorrectsTheScreen:
    def _recheck(self, answer, shown):
        import nodetop.cli as cli

        cli._Recheck.live = None
        return cli._Recheck(lambda _ticker=None: answer, "unused-key", shown)

    def test_an_answer_that_moved_asks_for_a_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        r = self._recheck({"a": "yes", "b": "no"}, {"a": "yes", "b": "yes"})
        r.done.wait(5)
        assert r.moved() is True

    def test_an_answer_that_agrees_says_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        r = self._recheck({"a": "yes", "b": "no"}, {"a": "yes", "b": "no"})
        r.done.wait(5)
        assert r.moved() is False

    def test_a_partition_nobody_could_see_is_not_a_difference(self, tmp_path,
                                                              monkeypatch):
        # The recheck may learn about a partition that was not on screen. That
        # is not a reason to redraw what the reader is looking at.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        r = self._recheck({"a": "yes", "z": "no"}, {"a": "yes"})
        r.done.wait(5)
        assert r.moved() is False

    def test_it_writes_what_it_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        import nodetop.cli as cli

        key = access.key(backend="t", cluster="c", user="u", accounts=[],
                         shape="s")
        cli._Recheck.live = None
        r = cli._Recheck(lambda _ticker=None: {"a": "yes", "b": "no"}, key,
                         {"a": "yes", "b": "yes"})
        r.done.wait(5)
        assert access.load(key)[0] == {"a": "yes", "b": "no"}

    def test_only_one_runs_at_a_time(self, tmp_path, monkeypatch):
        # `r` held down would otherwise start a dry-run pass per keypress: three
        # rechecks at once is nine concurrent probes against a controller this
        # tool deliberately asks for three.
        import threading

        import nodetop.cli as cli

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        hold = threading.Event()
        monkeypatch.setattr(cli._Recheck, "live", None)
        monkeypatch.setattr(cli._Recheck, "last_finished", 0.0)
        monkeypatch.setattr(cli._Recheck, "last_cost", 0.0)

        def slow(_ticker=None):
            hold.wait(5)
            return {"a": "yes"}

        first = cli._Recheck.start(slow, "k", {"a": "yes"})
        second = cli._Recheck.start(slow, "k", {"a": "yes"})
        assert first is not None and second is None
        hold.set()
        first.done.wait(5)
        # And once it is finished the next browse may start one -- after the
        # pacing gap, which the next test is about.
        monkeypatch.setattr(cli._Recheck, "last_finished", 0.0)
        assert cli._Recheck.start(lambda _t=None: {"a": "yes"}, "k",
                                  {"a": "yes"}) is not None

    def test_it_will_not_ask_again_straight_away(self, tmp_path, monkeypatch):
        """Making the frame cheap made the browse refresh itself.

        The idle interval is paced off what the last pass cost, and with the
        dry-runs moved into the background they stopped counting -- so an idle
        terminal re-read every 7s and dragged nineteen dry-runs along each time.
        Measured that way, which is the only reason it was noticed. A recheck now
        waits twenty times its own cost before the next one, the same rule the
        refresh itself uses.
        """
        import nodetop.cli as cli

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.setattr(cli._Recheck, "live", None)
        monkeypatch.setattr(cli._Recheck, "last_finished", 0.0)
        monkeypatch.setattr(cli._Recheck, "last_cost", 0.0)
        first = cli._Recheck.start(lambda _t=None: {"a": "yes"}, "k", {"a": "yes"})
        assert first is not None
        first.done.wait(5)
        assert cli._Recheck.last_cost >= 0.0
        # Pretend it cost a second: the next one must wait twenty.
        monkeypatch.setattr(cli._Recheck, "last_cost", 1.0)
        assert cli._Recheck.start(lambda _t=None: {"a": "yes"}, "k",
                                  {"a": "yes"}) is None
        # A cheap cluster is paced proportionally, not by a fixed constant.
        monkeypatch.setattr(cli._Recheck, "last_cost", 0.0)
        assert cli._Recheck.start(lambda _t=None: {"a": "yes"}, "k",
                                  {"a": "yes"}) is not None

    def test_a_recheck_that_raises_is_not_a_crash(self, tmp_path, monkeypatch):
        # It runs in a daemon thread behind a live browse; an exception there
        # must leave the remembered answer alone, not take down the session.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        def angry(_ticker=None):
            raise RuntimeError("controller went away")

        import nodetop.cli as cli

        cli._Recheck.live = None
        r = cli._Recheck(angry, "k", {"a": "yes"})
        assert r.done.wait(5)
        assert r.moved() is False


class TestAnUnusableTtlIsIgnoredAndSaidSo:
    """Polish pass, 2026-08-28: one typo class, three different silent outcomes.

        NODETOP_ACCESS_TTL=abc    -> the default (86400s)
        NODETOP_ACCESS_TTL=-5     -> 0, i.e. caching DISABLED
        NODETOP_ACCESS_TTL=1e999  -> inf, i.e. answers never expire
        NODETOP_ACCESS_TTL=nan    -> 0, i.e. disabled (`max`'s NaN ordering)

    Same kind of mistake, three different behaviours, nothing said. `inf` is the
    one that could mislead: a remembered *refusal* would be reused for the life of
    the cache file, long after the account was fixed.

    Still not an error — this is a cache knob, and the sibling package errors only
    for one that can hang a run (`SLURMPAST_TIMEOUT`). The rule is now single: an
    unusable value means the variable is ignored, and the run says so.
    """

    BAD = ["abc", "-5", "1e999", "-inf", "nan", "", "   ", "1,5", "300s"]

    @pytest.mark.parametrize("raw", BAD)
    def test_it_falls_back_to_the_default(self, raw, monkeypatch, capsys):
        from nodetop.core import access

        access._TTL_COMPLAINED.clear()
        monkeypatch.setenv("NODETOP_ACCESS_TTL", raw)
        assert access.ttl() == access.DEFAULT_TTL, raw
        err = capsys.readouterr().err
        assert "ignoring NODETOP_ACCESS_TTL" in err, err
        # It says what to set instead, not merely that this was wrong.
        assert "number of seconds" in err and "0 to disable" in err

    @pytest.mark.parametrize("raw,expected", [
        ("0", 0.0), ("300", 300.0), ("  600  ", 600.0), ("1.5", 1.5),
    ])
    def test_a_usable_value_is_honoured_silently(self, raw, expected, monkeypatch, capsys):
        """The control, including `0`.

        `0` is the documented way to disable the remembered answers, so it must
        stay legal and stay quiet — folding it in with the negatives would remove
        the only way to turn the cache off.
        """
        from nodetop.core import access

        access._TTL_COMPLAINED.clear()
        monkeypatch.setenv("NODETOP_ACCESS_TTL", raw)
        assert access.ttl() == expected
        assert capsys.readouterr().err == ""

    def test_an_unset_variable_is_silent(self, monkeypatch, capsys):
        from nodetop.core import access

        access._TTL_COMPLAINED.clear()
        monkeypatch.delenv("NODETOP_ACCESS_TTL", raising=False)
        assert access.ttl() == access.DEFAULT_TTL
        assert capsys.readouterr().err == ""

    def test_the_note_is_said_once_not_per_call(self, monkeypatch, capsys):
        """`ttl()` is consulted up to three times a run.

        Three copies of the same advisory would be worse than one, and is the kind
        of thing that gets a warning suppressed wholesale later.
        """
        from nodetop.core import access

        access._TTL_COMPLAINED.clear()
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "abc")
        for _ in range(4):
            access.ttl()
        assert capsys.readouterr().err.count("ignoring NODETOP_ACCESS_TTL") == 1

    def test_inf_no_longer_means_forever(self, monkeypatch):
        """The specific outcome worth naming: a refusal cached indefinitely."""
        import math

        from nodetop.core import access

        access._TTL_COMPLAINED.clear()
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "1e999")
        assert math.isfinite(access.ttl())
