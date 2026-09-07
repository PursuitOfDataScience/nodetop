"""`runner.py`'s second stated rule had no test at all.

The module says it in its own docstring, as one of the two rules it exists to
enforce:

    **A timeout is evidence, not noise.**  A control plane that answers slowly
    is a finding about the cluster, so it gets its own exception type rather
    than being folded into a generic failure.

Across 4,607 tests, `CommandTimeoutError` was never constructed and
`TimeoutExpired` never raised -- branch coverage put `runner.py:155` (the raise)
and `exceptions.py:45-47` (the whole `__init__`) among the few unexecuted lines
in the package. So both ends of the rule were unpinned: nothing checked that a
slow command produces the distinct type, and -- the one a refactor could break
silently -- nothing checked that the caller's timeout reaches `subprocess.run`
at all. Dropping `timeout=timeout` from that call leaves every test green and
turns a hung `scontrol` into a hung `nodetop`.

The distinction is structural, not cosmetic: `CommandTimeoutError` is a sibling
of `CommandError` under `NodetopError`, so an `except CommandError` cannot
quietly absorb one. That is pinned here deliberately, because it is the part of
the rule that a well-meaning tidy-up ("make the timeout a kind of command
failure") would undo.

What the reader ends up seeing is checked too. `Cluster.load` records failures
as `f"{type(exc).__name__}: {exc}"` and `cli.py` prints those verbatim, so the
type name is the whole mechanism by which a slow control plane is
distinguishable from a broken one on screen.
"""

import subprocess
import sys

import pytest

from nodetop.backends.slurm import SlurmBackend
from nodetop.core.cluster import Cluster
from nodetop.exceptions import (
    CommandError,
    CommandTimeoutError,
    NodetopError,
    SchedulerUnavailableError,
)
from nodetop.runner import DEFAULT_TIMEOUT, RecordedRunner, Runner, SubprocessRunner

#: Portable and certain to outlast the timeout, without assuming `sleep` exists.
SLOW = [sys.executable, "-c", "import time; time.sleep(30)"]

#: Long enough that a loaded login node cannot make it flake, short enough to
#: pay for four real subprocesses without being noticed.
GRACE = 0.75


class TestASlowCommandRaisesTheDistinctType:
    """Against real subprocesses -- a mock here would only test the mock."""

    def test_run_full_raises_the_timeout_type(self):
        with pytest.raises(CommandTimeoutError):
            SubprocessRunner().run_full(SLOW, GRACE)

    def test_run_raises_the_timeout_type(self):
        with pytest.raises(CommandTimeoutError):
            SubprocessRunner().run(SLOW, GRACE)

    def test_it_is_not_a_command_error_so_that_handler_cannot_absorb_it(self):
        # The rule, as a type relation. If `CommandTimeoutError` were made a
        # subclass of `CommandError`, this is the assertion that would notice.
        with pytest.raises(CommandTimeoutError) as caught:
            SubprocessRunner().run(SLOW, GRACE)
        assert not isinstance(caught.value, CommandError)
        assert isinstance(caught.value, NodetopError)
        assert not issubclass(CommandTimeoutError, CommandError)

    def test_the_message_names_the_command_and_the_seconds(self):
        with pytest.raises(CommandTimeoutError) as caught:
            SubprocessRunner().run(SLOW, GRACE)
        text = str(caught.value)
        assert "timed out after" in text, text
        assert sys.executable in text, text
        assert str(GRACE) in text, text

    def test_the_machine_readable_fields_survive(self):
        with pytest.raises(CommandTimeoutError) as caught:
            SubprocessRunner().run_full(SLOW, GRACE)
        assert caught.value.cmd == SLOW
        assert caught.value.timeout == GRACE
        # Chained with `from exc`, so the traceback still shows the origin.
        assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)

    def test_a_whole_number_of_seconds_is_not_written_as_a_float(self):
        # `:g` -- "timed out after 30s", not "30.0s". DEFAULT_TIMEOUT is a float.
        assert str(CommandTimeoutError(["sinfo"], DEFAULT_TIMEOUT)).endswith(
            f"after {int(DEFAULT_TIMEOUT)}s"
        )


class TestTheCallersTimeoutActuallyReachesSubprocess:
    """The property nothing pinned, and the one a refactor drops silently."""

    @staticmethod
    def _capture(monkeypatch):
        seen = {}

        def fake_run(*args, **kwargs):
            seen.update(kwargs)
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

        monkeypatch.setattr(subprocess, "run", fake_run)
        return seen

    def test_an_explicit_timeout_is_handed_down_unchanged(self, monkeypatch):
        seen = self._capture(monkeypatch)
        with pytest.raises(CommandTimeoutError):
            SubprocessRunner().run_full(["sinfo"], 12.5)
        assert seen["timeout"] == 12.5

    def test_omitting_it_uses_the_module_default(self, monkeypatch):
        seen = self._capture(monkeypatch)
        with pytest.raises(CommandTimeoutError):
            SubprocessRunner().run_full(["sinfo"])
        assert seen["timeout"] == DEFAULT_TIMEOUT

    def test_the_reported_seconds_are_the_ones_that_were_enforced(self, monkeypatch):
        # Not two independent numbers: the message must not be able to claim a
        # limit different from the one `subprocess` was given.
        seen = self._capture(monkeypatch)
        with pytest.raises(CommandTimeoutError) as caught:
            SubprocessRunner().run(["scontrol", "show", "node"], 7.0)
        assert caught.value.timeout == seen["timeout"] == 7.0


class _TimesOutOn(RecordedRunner):
    """Answers from recordings, except one query that hangs."""

    def __init__(self, responses, slow_substring):
        super().__init__(responses)
        self._slow = slow_substring

    def run(self, cmd, timeout=DEFAULT_TIMEOUT):
        if self._slow in " ".join(cmd):
            raise CommandTimeoutError(list(cmd), timeout)
        return super().run(cmd, timeout)


class TestTheDistinctionReachesTheReader:
    def test_the_type_name_lands_in_cluster_errors(self):
        from conftest import read  # `tests/` is not a package; house style

        backend = SlurmBackend(_TimesOutOn({
            "scontrol show node": (0, read("slurm", "nodes.txt"), ""),
            "scontrol show partition": (0, read("slurm", "partitions.txt"), ""),
            "show qos": (0, read("slurm", "qos.txt"), ""),
            "show assoc": (0, "acct-a||gn\n", ""),
            "squeue": (0, "", ""),
        }, "show partition"))
        cluster = Cluster.load(backend, with_free_times=False)

        # The nodes query answered in full, which is why naming the failure
        # matters -- see `test_partial_failure_wording.py`.
        assert cluster.nodes
        assert "queues" in cluster.errors, cluster.errors
        why = cluster.errors["queues"]
        assert why.startswith("CommandTimeoutError: "), why
        assert "timed out after" in why, why
        # A slow control plane must not read as a broken one.
        assert "CommandError:" not in why, why


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_a_non_zero_exit_is_still_a_plain_command_error(self):
        cmd = [sys.executable, "-c", "import sys; sys.stderr.write('nope'); sys.exit(2)"]
        with pytest.raises(CommandError) as caught:
            SubprocessRunner().run(cmd, GRACE)
        assert not isinstance(caught.value, CommandTimeoutError)
        assert caught.value.returncode == 2
        assert "nope" in str(caught.value)

    def test_a_missing_binary_is_still_the_unavailable_error(self):
        with pytest.raises(SchedulerUnavailableError):
            SubprocessRunner().run(["nodetop-no-such-binary-anywhere"], GRACE)

    def test_a_command_that_answers_still_returns_its_stdout(self):
        out = SubprocessRunner().run(
            [sys.executable, "-c", "print('hello')"], GRACE
        )
        assert out.strip() == "hello"

    def test_ok_still_swallows_every_failure_mode(self):
        # Documented as such: a probe must answer False, not explode. This is
        # the one place a timeout is *deliberately* folded into a plain no.
        assert Runner.ok(SubprocessRunner(), SLOW, GRACE) is False
