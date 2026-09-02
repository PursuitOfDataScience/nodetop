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
    "resolve",
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


def resolve(binary: str) -> str | None:
    """Where ``binary`` actually lives, following symlinks.

    Needed because a client binary's *presence* does not identify the system
    behind it: Slurm ships ``qstat``, ``qsub``, ``qdel`` and ``pbsnodes``
    wrappers in ``contribs/torque``, installed by default at a great many
    sites, and on such a host ``which("pbsnodes")`` is true on a cluster with
    no PBS at all.  The resolved path says which package it came from.
    """
    found = shutil.which(binary)
    if not found:
        return None
    try:
        return os.path.realpath(found)
    except OSError:  # pragma: no cover - realpath does not normally raise
        return found


def _failure(cmd: Sequence[str], rc: int, out: str, err: str) -> CommandError:
    """The error to raise for a non-zero exit, with a reason that is not blank.

    ``CommandError`` reports stderr, which is right for every client that puts
    its diagnosis there -- and ``scontrol``, the one this tool leans on hardest,
    does not.  Measured on live Slurm 20.11.8: ``scontrol show node nosuchnode``
    and ``scontrol show partition nosuchpart`` both exit 1 with ``Node
    nosuchnode not found`` on **stdout** and stderr EMPTY.  So the funnel raised
    ``scontrol show node --all --oneliner exited 1: `` -- a reported failure
    whose reason is the empty string -- and `Cluster.load` stored exactly that
    under ``errors["nodes"]``, which is what the reader is shown when a query
    dies.  The one thing a failure report has to carry is why, and stdout held
    the only copy of it.

    stderr still wins wherever it has anything at all: it is the documented
    place for a diagnosis, a dry-run's verdict lands there, and appending stdout
    to it would put a partial data dump into an error message.  stdout is read
    only when it is the sole account of the failure.
    """
    return CommandError(list(cmd), rc, err or out)


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
            raise _failure(cmd, rc, out, err)
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
                # Slurm/GPFS output is UTF-8 in practice, and the bytes in it are not
                # ours: a job name, a node's `Reason`, a fileset name and an account
                # description are all free text a user or an admin typed. `text=True`
                # alone decodes with the PARENT's locale, and under `LC_ALL=C` with
                # coercion off that is `ANSI_X3.4-1968` -- pure ASCII -- so one
                # accented character anywhere in the output killed the read.
                #
                # Measured: a node whose `Reason` read `disk caf\xc3\xa9 replaced`
                # made all four queries raise `UnicodeDecodeError`, and `status`
                # reported "every query failed, so there is nothing to report" with
                # exit 3 -- on a healthy cluster. `replace` reads the rest and shows
                # one lossy character instead, which is the honest trade: the field
                # is decoration, the cluster state is the answer. Sibling packages
                # already pass exactly this pair on their Slurm runners; this was the
                # one funnel that did not.
                encoding="utf-8",
                errors="replace",
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
            raise _failure(cmd, rc, out, err)
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

    @staticmethod
    def _signature(text: str) -> tuple[str, ...]:
        """The program and its sub-command, with every flag and value dropped.

        ``scontrol show node --all --oneliner`` and ``scontrol show node
        --oneliner`` reduce to the same thing, which is the point: see
        :meth:`_lookup`.
        """
        return tuple(
            w for w in text.split() if not w.startswith("-") and "=" not in w
        )

    def _lookup(self, cmd: Sequence[str]) -> tuple[int, str, str]:
        self.calls.append(list(cmd))
        joined = " ".join(cmd)
        # Longest key first, so a specific recording beats a general one.
        for key in sorted(self._responses, key=len, reverse=True):
            if key in joined:
                return self._responses[key]
        # Nothing matched literally, so try again ignoring flags -- but only
        # where that is unambiguous.
        #
        # A snapshot is recorded under the exact argv of the version that took
        # it, and adding one flag to a query orphans every recording of it.
        # Measured: teaching the node query `--all` made an hour-old snapshot
        # replay as "every query failed, so there is nothing to report -- this
        # is not an empty cluster", which is a claim about the cluster, from a
        # file that holds a complete and healthy picture of it. Snapshots
        # travel between machines and versions -- that is what they are for --
        # so a flag added on either side must not invalidate one.
        #
        # The ambiguity check is what makes this safe rather than merely
        # convenient: every per-queue dry-run in a snapshot differs from its
        # siblings ONLY in flags (`--partition=`, `--account=`), so they all
        # share a signature. Answering one of those from another's recording
        # would put a verdict under the wrong queue's name -- a wrong answer,
        # where failing to find it is only a missing one. Several candidates
        # therefore means no match, exactly as before.
        want = self._signature(joined)
        if want:
            hits = [k for k in self._responses if self._signature(k) == want]
            if len(hits) == 1:
                return self._responses[hits[0]]
        raise CommandError(list(cmd), 127, f"no recorded response for {joined!r}")

    def run(  # `timeout` unused here but required by the interface
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        rc, out, err = self._lookup(cmd)
        if rc != 0:
            raise _failure(cmd, rc, out, err)
        return out

    def run_full(  # `timeout` unused here but required by the interface
        self, cmd: Sequence[str], timeout: float = DEFAULT_TIMEOUT
    ) -> tuple[int, str, str]:
        return self._lookup(cmd)
