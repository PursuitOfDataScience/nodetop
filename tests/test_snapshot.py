"""Snapshot capture and replay.

The point is post-mortem. When a partition goes down mid-run the evidence is
gone by the time anyone looks, so nodetop can record what its queries returned
and replay every command against that recording later.

Replay needs no special path in the analysis layer: the same backend code runs
against a recorded runner instead of a live one, which is why it works for
every backend without per-backend support.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from nodetop.cli import build_parser, cmd_snapshot, main
from nodetop.core.cluster import Cluster
from nodetop.render import Glyphs, Style
from nodetop.runner import CapturingRunner, RecordedRunner

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


class TestCapturingRunner:
    def test_it_records_what_it_ran(self):
        inner = RecordedRunner({"hello": (0, "world", "")})
        runner = CapturingRunner(inner)
        assert runner.run(["hello"]) == "world"
        assert runner.captured["hello"] == (0, "world", "")

    def test_it_records_failures_too(self):
        # A snapshot of a broken cluster is the interesting kind, and replay
        # should reproduce the breakage rather than a clean cluster.
        runner = CapturingRunner(RecordedRunner({}))
        runner.run_full(["missing", "command"])
        rc, _, err = runner.captured["missing command"]
        assert rc != 0
        assert err

    def test_it_raises_on_a_failed_run_like_the_real_runner(self):
        from nodetop.exceptions import CommandError

        runner = CapturingRunner(RecordedRunner({"x": (1, "", "boom")}))
        with pytest.raises(CommandError):
            runner.run(["x"])

    def test_the_key_is_the_whole_command_line(self):
        runner = CapturingRunner(RecordedRunner({"a": (0, "", "")}))
        runner.run(["a", "-b", "--c=d"])
        assert "a -b --c=d" in runner.captured


class TestAFlagChangeDoesNotOrphanARecording:
    """A snapshot outlives the version that took it, so replay tolerates flags.

    Recordings are keyed by the exact argv of the version that captured them.
    Teaching the Slurm node query `--all` orphaned every recording of it, and an
    hour-old snapshot of a healthy cluster replayed as "every query failed, so
    there is nothing to report -- this is not an empty cluster": a claim about
    the cluster, made from a file that holds a complete picture of it.
    """

    def test_an_added_flag_still_finds_its_recording(self):
        r = RecordedRunner({"scontrol show node --oneliner": (0, "NodeName=n1", "")})
        assert r.run(["scontrol", "show", "node", "--all", "--oneliner"]) == "NodeName=n1"

    def test_a_removed_flag_too(self):
        r = RecordedRunner({"sacctmgr -nP show qos format=Name,MaxWall": (0, "q|", "")})
        assert r.run(["sacctmgr", "show", "qos"]) == "q|"

    def test_siblings_that_differ_only_in_flags_are_left_unmatched(self):
        # Every per-queue dry-run in a snapshot differs from its siblings ONLY
        # in `--partition=`/`--account=`. Answering one from another's recording
        # would file a verdict under the wrong queue's name -- a wrong answer,
        # where not finding it is merely a missing one.
        from nodetop.exceptions import CommandError

        r = RecordedRunner(
            {
                "sbatch --test-only --partition=a": (0, "queue a", ""),
                "sbatch --test-only --partition=b": (0, "queue b", ""),
            }
        )
        with pytest.raises(CommandError):
            r.run(["sbatch", "--test-only", "--partition=c"])
        # The exact keys still win, so the ordinary case is untouched.
        assert r.run(["sbatch", "--test-only", "--partition=b"]) == "queue b"

    def test_a_different_command_is_not_substituted(self):
        from nodetop.exceptions import CommandError

        r = RecordedRunner({"scontrol show node": (0, "nodes", "")})
        with pytest.raises(CommandError):
            r.run(["scontrol", "show", "partition", "--all"])


class TestRoundTrip:
    def _snapshot(self, cluster, tmp_path, capture):
        cluster.capture = capture
        path = tmp_path / "snap.json"
        assert cmd_snapshot(cluster, _args(["snapshot", "-o", str(path)]), PLAIN) == 0
        return json.loads(path.read_text()), path

    def _capturing_slurm(self, slurm_nodes, slurm_partitions, slurm_qos):
        from nodetop.backends.slurm import SlurmBackend

        inner = RecordedRunner(
            {
                "scontrol show node": (0, slurm_nodes, ""),
                "scontrol show partition": (0, slurm_partitions, ""),
                "show qos": (0, slurm_qos, ""),
                "show assoc": (0, "acct||gn\n", ""),
                "squeue": (0, "", ""),
            }
        )
        capture = CapturingRunner(inner)
        return SlurmBackend(capture), capture

    def test_the_snapshot_records_the_backend_and_the_queries(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        backend, capture = self._capturing_slurm(slurm_nodes, slurm_partitions, slurm_qos)
        cluster = Cluster.load(backend, with_free_times=True)
        data, _ = self._snapshot(cluster, tmp_path, capture)
        assert data["backend"] == "slurm"
        assert data["queue_term"] == "partition"
        assert data["commands"]
        assert any("scontrol show node" in k for k in data["commands"])

    def test_replay_reproduces_the_cluster(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        backend, capture = self._capturing_slurm(slurm_nodes, slurm_partitions, slurm_qos)
        live = Cluster.load(backend, with_free_times=True)
        _, path = self._snapshot(live, tmp_path, capture)

        from nodetop.cli import _load_replay

        replayed_backend, name, _captured = _load_replay(str(path))
        replayed = Cluster.load(replayed_backend, with_free_times=True, replayed=True)

        assert name == "slurm"
        assert len(replayed.nodes) == len(live.nodes)
        assert set(replayed.queues) == set(live.queues)
        assert replayed.summary()["accelerators_total"] == (live.summary()["accelerators_total"])
        assert replayed.summary()["unusable_queues"] == (live.summary()["unusable_queues"])

    def test_the_snapshot_is_written_where_asked(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos, capsys
    ):
        backend, capture = self._capturing_slurm(slurm_nodes, slurm_partitions, slurm_qos)
        cluster = Cluster.load(backend, with_free_times=True)
        _, path = self._snapshot(cluster, tmp_path, capture)
        assert path.exists()
        # Progress goes to stderr so `snapshot -o file` stays pipe-safe.
        assert capsys.readouterr().out == ""

    def test_stdout_mode_emits_the_json_itself(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos, capsys
    ):
        backend, capture = self._capturing_slurm(slurm_nodes, slurm_partitions, slurm_qos)
        cluster = Cluster.load(backend, with_free_times=True)
        cluster.capture = capture
        assert cmd_snapshot(cluster, _args(["snapshot", "-o", "-"]), PLAIN) == 0
        assert json.loads(capsys.readouterr().out)["backend"] == "slurm"


class TestReplayHonesty:
    """A recording cannot be dry-run against, and must not claim it can."""

    def _replayed(self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.cli import _load_replay

        capture = CapturingRunner(
            RecordedRunner(
                {
                    "scontrol show node": (0, slurm_nodes, ""),
                    "scontrol show partition": (0, slurm_partitions, ""),
                    "show qos": (0, slurm_qos, ""),
                    "show assoc": (0, "acct||gn\n", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(SlurmBackend(capture), with_free_times=True)
        cluster.capture = capture
        path = tmp_path / "s.json"
        cmd_snapshot(cluster, _args(["snapshot", "-o", str(path)]), PLAIN)
        backend, _, _captured = _load_replay(str(path))
        return Cluster.load(backend, with_free_times=True, replayed=True), path

    def test_can_probe_is_false_even_though_slurm_can(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        cluster, _ = self._replayed(tmp_path, slurm_nodes, slurm_partitions, slurm_qos)
        assert cluster.replayed is True
        assert cluster.can_probe is False
        # The backend itself still reports the capability; the cluster overrides.
        assert cluster.capabilities.probe is True

    def test_probe_returns_nothing_rather_than_a_confusing_failure(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        from nodetop.core.model import JobShape

        cluster, _ = self._replayed(tmp_path, slurm_nodes, slurm_partitions, slurm_qos)
        assert cluster.probe("gn", JobShape()) is None

    def test_check_says_it_is_a_recording_not_that_slurm_lacks_a_dry_run(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos, capsys
    ):
        from nodetop.cli import cmd_check

        cluster, _ = self._replayed(tmp_path, slurm_nodes, slurm_partitions, slurm_qos)
        assert cmd_check(cluster, _args(["check", "-g", "1"]), PLAIN) == 2
        out = " ".join(capsys.readouterr().out.split())
        # Telling a Slurm user that Slurm has no verify-only submission is false.
        assert "replayed snapshot" in out
        assert "slurm has no verify-only" not in out


class TestReplayCli:
    def test_a_missing_file_exits_2(self, capsys):
        assert main(["--replay", "/nonexistent/snap.json", "status"]) == 2
        assert "cannot replay" in capsys.readouterr().err

    def test_malformed_json_exits_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert main(["--replay", str(bad), "status"]) == 2
        assert "cannot replay" in capsys.readouterr().err

    def test_an_unknown_backend_name_exits_2(self, tmp_path, capsys):
        bad = tmp_path / "b.json"
        bad.write_text(json.dumps({"backend": "nosuch", "commands": {}}))
        assert main(["--replay", str(bad), "status"]) == 2
        assert "cannot replay" in capsys.readouterr().err

    def test_replay_flag_is_accepted_on_either_side_of_the_verb(self):
        assert _args(["--replay", "f", "status"]).replay == "f"
        assert _args(["status", "--replay", "f"]).replay == "f"


class TestReplayIsDatedWhenItWasCaptured:
    """A replay must carry the data's own clock, not the reader's.

    The snapshot wrote `captured_at` and nothing read it back, so a replay
    stamped itself with the moment it was opened. Two consequences, the second
    much worse than the first: a post-mortem of last week's outage was dated
    today, and every wait estimate was computed against today's clock over
    last week's node free times -- so a node recorded as free in three hours
    read as "overdue" once the snapshot aged past that.

    Fifteen snapshot tests passed while this was broken; none looked at the
    time.
    """

    @pytest.fixture
    def live_cluster(self, slurm_nodes, slurm_partitions, slurm_qos):
        """A cluster loaded through a CapturingRunner, so there is something
        to write: cmd_snapshot dumps what the queries actually returned."""
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.core.cluster import Cluster

        capture = CapturingRunner(
            RecordedRunner(
                {
                    "scontrol show node": (0, slurm_nodes, ""),
                    "scontrol show partition": (0, slurm_partitions, ""),
                    "show qos": (0, slurm_qos, ""),
                    "show assoc": (0, "acct||gn\n", ""),
                    "squeue": (0, "", ""),
                }
            )
        )
        cluster = Cluster.load(SlurmBackend(capture), with_free_times=True)
        cluster.capture = capture
        return cluster

    def _snapshot(self, tmp_path, cluster):
        from nodetop.cli import _load_replay

        path = tmp_path / "snap.json"
        assert cmd_snapshot(cluster, _args(["snapshot", "-o", str(path)]), PLAIN) == 0
        return path, _load_replay(str(path))

    def test_the_capture_time_survives_the_round_trip(self, tmp_path, capsys, live_cluster):
        cluster = live_cluster
        path, (_backend, _name, captured) = self._snapshot(tmp_path, cluster)
        capsys.readouterr()
        written = json.loads(path.read_text())["captured_at"]
        assert captured is not None
        assert captured.isoformat() == written

    def test_the_replayed_cluster_uses_it(self, tmp_path, capsys, live_cluster):
        cluster = live_cluster
        from nodetop.core.cluster import Cluster

        _path, (backend, _name, captured) = self._snapshot(tmp_path, cluster)
        capsys.readouterr()
        replayed = Cluster.load(backend, with_free_times=True, replayed=True, taken_at=captured)
        assert replayed.taken_at == captured

    def test_an_old_snapshot_is_not_dated_today(self, tmp_path, capsys, live_cluster):
        cluster = live_cluster
        from datetime import datetime, timedelta

        from nodetop.core.cluster import Cluster

        _path, (backend, _name, _c) = self._snapshot(tmp_path, cluster)
        capsys.readouterr()
        long_ago = datetime.now() - timedelta(days=7)
        replayed = Cluster.load(backend, with_free_times=True, replayed=True, taken_at=long_ago)
        assert replayed.taken_at == long_ago
        assert (datetime.now() - replayed.taken_at).days == 7

    def test_a_live_load_still_stamps_now(self, cluster):
        from datetime import datetime

        # The override must not change the live path.
        assert cluster.taken_at is not None
        assert abs((datetime.now() - cluster.taken_at).total_seconds()) < 300

    def test_a_snapshot_with_no_captured_at_replays_anyway(self, tmp_path):
        from nodetop.cli import _load_replay

        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"backend": "slurm", "commands": {}}))
        _backend, name, captured = _load_replay(str(path))
        assert name == "slurm"
        assert captured is None  # unknown, not fabricated

    def test_a_corrupt_captured_at_does_not_crash(self, tmp_path):
        from nodetop.cli import _load_replay

        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"backend": "slurm", "commands": {}, "captured_at": "not a time"})
        )
        assert _load_replay(str(path))[2] is None


class TestAFailedWriteKeepsThePreviousSnapshot:
    """Polish pass, 2026-08-28: `write_text` truncated first, so a failed write
    destroyed the capture that was already there.

    Measured under `ulimit -f 100` against a real 629,968-byte capture:

        before          629,968 bytes, parses
        after failure   102,400 bytes, JSONDecodeError: Unterminated string
        exit            1, with a raw `OSError: [Errno 27] File too large` traceback

    That matters more here than for most writers: a snapshot exists to be carried
    to another machine and replayed, so it is often the only copy of a cluster
    state that no longer exists. `core.access` next door already wrote its cache
    through `mkstemp` + `os.replace`; this used the one-liner.
    """

    @staticmethod
    def _write(target, limit=None):
        """Run the snapshot writer's path with an optional RLIMIT_FSIZE."""
        import os
        import subprocess
        import sys

        # A child process, because RLIMIT_FSIZE cannot be raised again once
        # lowered and would follow the test session for every later write.
        code = (
            "import sys, resource, pathlib\n"
            f"lim = {limit!r}\n"
            "if lim is not None:\n"
            "    resource.setrlimit(resource.RLIMIT_FSIZE, (lim, lim))\n"
            "sys.argv = ['nodetop', 'snapshot', '-o', sys.argv[1]]\n"
            "from nodetop.cli import main\n"
            "raise SystemExit(main())\n"
        )
        root = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, "-c", code, str(target)],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "NODETOP_HOSTS": "",
                "COLUMNS": "200",
            },
        )

    def test_a_capped_write_leaves_the_old_file_intact(self, tmp_path):
        target = tmp_path / "snap.json"
        good = self._write(target)
        if good.returncode != 0 or not target.exists():
            pytest.skip(f"no cluster to capture here: {good.stderr[-200:]}")
        before = target.read_bytes()
        json.loads(before)  # a real, parseable capture

        # The cap is DERIVED from the snapshot this environment actually
        # produces, not fixed at 100 KiB. A fixed cap encodes an assumption about
        # how much a cluster returns: on a CI runner with no scheduler the
        # capture is a single empty query of roughly a kilobyte, sails under
        # 100 KiB, exits 0, and this assertion fails for a reason that has
        # nothing to do with the behaviour under test. It passed here only
        # because this host is a real cluster -- the inverse of the runner
        # `ci.yml` deliberately provides, which is where it failed on all five
        # test jobs.
        #
        # Half of what was just written is guaranteed to be exceeded by writing
        # it again, whatever the size, so the cap bites everywhere.
        cap = len(before) // 2
        assert cap < len(before), "the cap must be smaller than the write it should refuse"
        failed = self._write(target, limit=cap)
        assert failed.returncode == 2, failed.stderr[-300:]
        assert "Traceback" not in failed.stderr, failed.stderr[-300:]
        assert "cannot write" in failed.stderr, failed.stderr[-300:]
        # The claim the message makes has to be true.
        assert "unchanged" in failed.stderr
        assert target.read_bytes() == before, "the previous snapshot was damaged"
        json.loads(target.read_bytes())

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        target = tmp_path / "snap.json"
        if self._write(target).returncode != 0:
            pytest.skip("no cluster to capture here")
        # Derived for the same reason as above: a cap the write cannot exceed
        # tests nothing about what a failed write leaves behind.
        self._write(target, limit=max(1, target.stat().st_size // 2))
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "snap.json"]
        assert not leftovers, leftovers

    @pytest.mark.parametrize("mask,expected", [(0o002, 0o664), (0o022, 0o644), (0o077, 0o600)])
    def test_the_mode_follows_the_umask(self, mask, expected, tmp_path):
        """Atomicity must not quietly tighten the file.

        `mkstemp` creates 0600 and `os.replace` keeps the temporary's mode, so the
        first version of the atomic write turned a 0664 snapshot into a 0600 one.
        That is the wrong default here, whatever it is for the access cache next
        door: the cache is private state and is chmod'ed 0600 on purpose, while a
        snapshot exists to be handed to somebody else and replayed — which is what
        the README shows. It exposes nothing new either; the capture is
        `squeue`/`sinfo` output anyone on the cluster can already run.
        """
        import os
        import subprocess
        import sys

        target = tmp_path / "snap.json"
        root = pathlib.Path(__file__).resolve().parent.parent
        code = (
            "import os, sys\n"
            f"os.umask({mask})\n"
            "sys.argv = ['nodetop', 'snapshot', '-o', sys.argv[1]]\n"
            "from nodetop.cli import main\n"
            "raise SystemExit(main())\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code, str(target)],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        if done.returncode != 0 or not target.exists():
            pytest.skip(f"no cluster to capture here: {done.stderr[-160:]}")
        assert oct(os.stat(target).st_mode & 0o777) == oct(expected)

    def test_the_cache_next_door_stays_private(self, tmp_path, monkeypatch):
        """The control, and the distinction the comment rests on.

        If both writers ever agree on a mode, one of them is wrong — so this fails
        if the cache loosens rather than if the snapshot tightens.
        """
        import os

        from nodetop.core import access

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.setenv("NODETOP_ACCESS_TTL", "3600")
        access.save("probe", {"q": access.YES})
        written = list(pathlib.Path(tmp_path).rglob("*.json"))
        if not written:
            pytest.skip("no cache directory available here")
        assert oct(os.stat(written[0]).st_mode & 0o777) == "0o600"

    def test_a_symlink_is_written_through_not_replaced(self, tmp_path):
        """`os.replace` would clobber the link; `write_text` wrote through it.

        `-o ~/snapshots/latest.json` pointing at a dated capture is an ordinary way
        to keep a "current" name. The atomicity fix turned that link into a regular
        file and left the dated capture untouched — a change to what the path means,
        which the fix was not licence to make.
        """
        import os

        dated = tmp_path / "dated.json"
        dated.write_text("original")
        link = tmp_path / "latest.json"
        os.symlink("dated.json", link)

        if self._write(link).returncode != 0:
            pytest.skip("no cluster to capture here")
        assert link.is_symlink(), "the symlink was replaced by a regular file"
        assert os.readlink(link) == "dated.json"
        json.loads(dated.read_text())  # the capture landed in the target

    def test_a_dangling_symlink_still_creates_its_target(self, tmp_path):
        """`realpath`, not `resolve()`: a broken link still names the intended file.

        This is what the old `write_text` did, so it is what must keep happening.
        """
        import os

        link = tmp_path / "link.json"
        os.symlink("target.json", link)
        if self._write(link).returncode != 0:
            pytest.skip("no cluster to capture here")
        assert (tmp_path / "target.json").exists()
        assert link.is_symlink()

    def test_a_successful_write_still_replays(self, tmp_path):
        # The control: atomicity must not change what lands in the file.
        import os
        import subprocess
        import sys

        target = tmp_path / "snap.json"
        if self._write(target).returncode != 0:
            pytest.skip("no cluster to capture here")
        payload = json.loads(target.read_text())
        assert payload["commands"], payload.keys()
        root = pathlib.Path(__file__).resolve().parent.parent
        replay = subprocess.run(
            [sys.executable, "-m", "nodetop", "--replay", str(target), "status"],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        assert replay.returncode == 0, replay.stderr[-300:]


class TestReplayRefusesAFileThatIsNotASnapshot:
    """A snapshot exists to be CARRIED, so the file reaching `--replay` is the one
    most likely to be damaged: truncated by an interrupted `scp`, hand-edited,
    committed next to a bug report, or simply some other tool's JSON.

    Eight kinds of damaged input already produced a clean `cannot replay ...` at
    rc=2 — truncated, empty, binary, a directory, a missing path, foreign JSON, a
    missing version, an unknown version. One did not: a top-level **list** escaped
    as `AttributeError: 'list' object has no attribute 'get'` with a traceback,
    because the caller catches `(OSError, ValueError, KeyError)` and `.get` on a
    list raises none of them.

    Fixed by shape-checking in the loader rather than by widening that tuple: the
    reader needs to be told the file is not a snapshot, and "no attribute 'get'"
    does not say that.
    """

    @staticmethod
    def _replay(path):
        import os
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, "-m", "nodetop", "--replay", str(path), "status"],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )

    @pytest.mark.parametrize(
        "payload,expect",
        [
            ([1, 2, 3], "the top level is list"),
            ("a string", "the top level is str"),
            (42, "the top level is int"),
            ({"backend": "slurm", "commands": [1, 2]}, "`commands` is list"),
            ({"backend": "slurm", "commands": {"a": [1, 2]}}, "are not objects"),
            ({"backend": "slurm", "commands": {"a": "text"}}, "are not objects"),
        ],
    )
    def test_a_wrong_shape_is_named_not_traced(self, payload, expect, tmp_path):
        target = tmp_path / "snap.json"
        target.write_text(json.dumps(payload))
        done = self._replay(target)
        assert done.returncode == 2, done.stdout[-200:] + done.stderr[-200:]
        assert "Traceback" not in done.stderr, done.stderr[-400:]
        assert "not a nodetop snapshot" in done.stderr, done.stderr[-300:]
        assert expect in done.stderr, done.stderr[-300:]

    @pytest.mark.parametrize("content", [b"", b"\x00\x01\x02binary\xff", b"{"])
    def test_unparseable_bytes_still_fail_cleanly(self, content, tmp_path):
        # The control for the paths that were already right: these raise
        # `ValueError` from `json`, which the caller already handled.
        target = tmp_path / "snap.json"
        target.write_bytes(content)
        done = self._replay(target)
        assert done.returncode == 2, done.stderr[-200:]
        assert "Traceback" not in done.stderr, done.stderr[-300:]
        assert "cannot replay" in done.stderr

    def test_a_real_snapshot_still_replays(self, tmp_path):
        """The control that matters: validation must not reject a good file."""
        import os
        import subprocess
        import sys

        target = tmp_path / "snap.json"
        root = pathlib.Path(__file__).resolve().parent.parent
        made = subprocess.run(
            [sys.executable, "-m", "nodetop", "snapshot", "-o", str(target)],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        if made.returncode != 0 or not target.exists():
            pytest.skip("no cluster to capture here")
        assert self._replay(target).returncode == 0

    def test_replay_never_reaches_the_live_cluster(self, tmp_path):
        """Even a snapshot that records nothing must not fall back to querying.

        A `commands`-less snapshot exits 3 with `query failed: ... exited 127`,
        which reads as if a real command had been tried. Counted with a logging
        wrapper on PATH: zero invocations — that is the RECORDED runner reporting
        it has no answer, which is the honest outcome.
        """
        import os
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "calls.log"
        for name in ("scontrol", "sinfo", "squeue", "sbatch", "sacctmgr"):
            stub = bindir / name
            stub.write_text(f"#!/bin/bash\necho call >> {log}\nexit 127\n")
            stub.chmod(0o755)
        target = tmp_path / "snap.json"
        target.write_text(json.dumps({"backend": "slurm", "nodetop": "0.0.0"}))
        subprocess.run(
            [sys.executable, "-m", "nodetop", "--replay", str(target), "status"],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        assert not log.exists(), f"replay ran real commands: {log.read_text()}"
