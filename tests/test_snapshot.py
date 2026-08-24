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


class TestRoundTrip:
    def _snapshot(self, cluster, tmp_path, capture):
        cluster.capture = capture
        path = tmp_path / "snap.json"
        assert cmd_snapshot(cluster, _args(["snapshot", "-o", str(path)]), PLAIN) == 0
        return json.loads(path.read_text()), path

    def _capturing_slurm(self, slurm_nodes, slurm_partitions, slurm_qos):
        from nodetop.backends.slurm import SlurmBackend

        inner = RecordedRunner({
            "scontrol show node": (0, slurm_nodes, ""),
            "scontrol show partition": (0, slurm_partitions, ""),
            "show qos": (0, slurm_qos, ""),
            "show assoc": (0, "acct||gn\n", ""),
            "squeue": (0, "", ""),
        })
        capture = CapturingRunner(inner)
        return SlurmBackend(capture), capture

    def test_the_snapshot_records_the_backend_and_the_queries(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        backend, capture = self._capturing_slurm(
            slurm_nodes, slurm_partitions, slurm_qos
        )
        cluster = Cluster.load(backend, with_free_times=True)
        data, _ = self._snapshot(cluster, tmp_path, capture)
        assert data["backend"] == "slurm"
        assert data["queue_term"] == "partition"
        assert data["commands"]
        assert any("scontrol show node" in k for k in data["commands"])

    def test_replay_reproduces_the_cluster(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos
    ):
        backend, capture = self._capturing_slurm(
            slurm_nodes, slurm_partitions, slurm_qos
        )
        live = Cluster.load(backend, with_free_times=True)
        _, path = self._snapshot(live, tmp_path, capture)

        from nodetop.cli import _load_replay

        replayed_backend, name, _captured = _load_replay(str(path))
        replayed = Cluster.load(replayed_backend, with_free_times=True, replayed=True)

        assert name == "slurm"
        assert len(replayed.nodes) == len(live.nodes)
        assert set(replayed.queues) == set(live.queues)
        assert replayed.summary()["accelerators_total"] == (
            live.summary()["accelerators_total"]
        )
        assert replayed.summary()["unusable_queues"] == (
            live.summary()["unusable_queues"]
        )

    def test_the_snapshot_is_written_where_asked(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos, capsys
    ):
        backend, capture = self._capturing_slurm(
            slurm_nodes, slurm_partitions, slurm_qos
        )
        cluster = Cluster.load(backend, with_free_times=True)
        _, path = self._snapshot(cluster, tmp_path, capture)
        assert path.exists()
        # Progress goes to stderr so `snapshot -o file` stays pipe-safe.
        assert capsys.readouterr().out == ""

    def test_stdout_mode_emits_the_json_itself(
        self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos, capsys
    ):
        backend, capture = self._capturing_slurm(
            slurm_nodes, slurm_partitions, slurm_qos
        )
        cluster = Cluster.load(backend, with_free_times=True)
        cluster.capture = capture
        assert cmd_snapshot(cluster, _args(["snapshot", "-o", "-"]), PLAIN) == 0
        assert json.loads(capsys.readouterr().out)["backend"] == "slurm"


class TestReplayHonesty:
    """A recording cannot be dry-run against, and must not claim it can."""

    def _replayed(self, tmp_path, slurm_nodes, slurm_partitions, slurm_qos):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.cli import _load_replay

        capture = CapturingRunner(RecordedRunner({
            "scontrol show node": (0, slurm_nodes, ""),
            "scontrol show partition": (0, slurm_partitions, ""),
            "show qos": (0, slurm_qos, ""),
            "show assoc": (0, "acct||gn\n", ""),
            "squeue": (0, "", ""),
        }))
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

        capture = CapturingRunner(RecordedRunner({
            "scontrol show node": (0, slurm_nodes, ""),
            "scontrol show partition": (0, slurm_partitions, ""),
            "show qos": (0, slurm_qos, ""),
            "show assoc": (0, "acct||gn\n", ""),
            "squeue": (0, "", ""),
        }))
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
        replayed = Cluster.load(
            backend, with_free_times=True, replayed=True, taken_at=captured)
        assert replayed.taken_at == captured

    def test_an_old_snapshot_is_not_dated_today(self, tmp_path, capsys, live_cluster):
        cluster = live_cluster
        from datetime import datetime, timedelta

        from nodetop.core.cluster import Cluster

        _path, (backend, _name, _c) = self._snapshot(tmp_path, cluster)
        capsys.readouterr()
        long_ago = datetime.now() - timedelta(days=7)
        replayed = Cluster.load(
            backend, with_free_times=True, replayed=True, taken_at=long_ago)
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
        path.write_text(json.dumps(
            {"backend": "slurm", "commands": {}, "captured_at": "not a time"}))
        assert _load_replay(str(path))[2] is None
