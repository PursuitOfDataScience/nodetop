"""Command execution, in one place so nothing else touches subprocess.

Two rules hold throughout nodetop and are enforced here:

* **Read-only by default.**  The only mutating-looking commands any backend may
  invoke are dry-runs that create nothing -- ``sbatch --test-only``,
  ``qsub -w v``, ``kubectl --dry-run=server``.  Each backend hard-codes its
  dry-run flag so it cannot be omitted by a caller.
* **A timeout is evidence, not noise.**  A control plane that answers slowly is
  a finding about the cluster, so it gets its own exception type rather than
  being folded into a generic failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence

from .exceptions import CommandError, CommandTimeoutError, SchedulerUnavailableError

__all__ = [
    "DEFAULT_TIMEOUT",
    "CapturingRunner",
    "RecordedRunner",
    "Runner",
    "SubprocessRunner",
    "which",
]

DEFAULT_TIMEOUT = 30.0

#: Pinned so parsing is deterministic regardless of the caller's locale or an
#: inherited scheduler time-format variable.
_FORCED_ENV = {
    "LC_ALL": "C",
    "LANG": "C",
    "SLURM_TIME_FORMAT": "standard",
}


def which(binary: str) -> bool:
    """Whether a client binary is on PATH."""
    return shutil.which(binary) is not None


class Runner:
    """Executes a command and returns its output."""

    def run(self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT) -> str:
        raise NotImplementedError

    def run_full(
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, str, str]:
        """Return ``(returncode, stdout, stderr)`` without raising on failure.

        Dry-run commands routinely write their verdict to *stderr* and exit
        non-zero even when the answer is informative, so probe paths need the
        unfiltered triple rather than just stdout.
        """
        raise NotImplementedError

    def ok(self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT) -> bool:
        """Whether the command succeeded, swallowing every failure mode."""
        try:
            return self.run_full(cmd, timeout)[0] == 0
        except Exception:
            return False


class SubprocessRunner(Runner):
    """Executes real commands."""

    def run(self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT) -> str:
        rc, out, err = self.run_full(cmd, timeout)
        if rc != 0:
            raise CommandError(list(cmd), rc, err)
        return out

    def run_full(
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        env.update(_FORCED_ENV)
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, never a shell
                list(cmd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandTimeoutError(list(cmd), timeout) from exc
        except FileNotFoundError as exc:
            raise SchedulerUnavailableError(f"{cmd[0]} not found on PATH") from exc
        return proc.returncode, proc.stdout, proc.stderr


class CapturingRunner(Runner):
    """Runs commands for real and keeps every result.

    A snapshot is taken by running an ordinary query pass through this and
    dumping what was actually called.  Nothing has to declare its own command
    list, so a snapshot cannot drift out of sync with the queries a backend
    really makes -- which is exactly how the previous hand-maintained version
    went stale.
    """

    def __init__(self, inner: Runner | None = None) -> None:
        self.inner = inner or SubprocessRunner()
        self.captured: dict[str, tuple[int, str, str]] = {}

    def run(self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT) -> str:
        rc, out, err = self.run_full(cmd, timeout)
        if rc != 0:
            raise CommandError(list(cmd), rc, err)
        return out

    def run_full(
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, str, str]:
        key = " ".join(cmd)
        try:
            result = self.inner.run_full(cmd, timeout)
        except Exception as exc:
            # Record the failure too: a snapshot of a broken cluster is the
            # interesting kind, and replay should reproduce the breakage.
            result = (127, "", f"{type(exc).__name__}: {exc}")
        self.captured[key] = result
        return result


class RecordedRunner(Runner):
    """Replays canned output, keyed by a substring of the command line.

    Used by the tests and by snapshot replay, so a cluster state captured on a
    login node can be analysed anywhere -- and so the interesting states (a
    disabled queue, a throttled node, an admission webhook that disagrees with
    the scheduler) can be exercised on demand.
    """

    def __init__(self, responses: dict[str, tuple[int, str, str]]) -> None:
        self._responses = responses
        self.calls: list[list[str]] = []

    def _lookup(self, cmd: Sequence[str]) -> tuple[int, str, str]:
        self.calls.append(list(cmd))
        joined = " ".join(cmd)
        # Longest key first, so a specific recording beats a general one.
        for key in sorted(self._responses, key=len, reverse=True):
            if key in joined:
                return self._responses[key]
        raise CommandError(list(cmd), 127, f"no recorded response for {joined!r}")

    def run(  # `timeout` unused here but required by the interface
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        rc, out, err = self._lookup(cmd)
        if rc != 0:
            raise CommandError(list(cmd), rc, err)
        return out

    def run_full(  # `timeout` unused here but required by the interface
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, str, str]:
        return self._lookup(cmd)
