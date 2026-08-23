"""Exception hierarchy for nodetop."""

from __future__ import annotations

__all__ = [
    "CommandError",
    "CommandTimeoutError",
    "NodetopError",
    "NoBackendError",
    "SchedulerUnavailableError",
]


class NodetopError(Exception):
    """Base class for every error raised by nodetop."""


class NoBackendError(NodetopError):
    """No supported batch system could be detected."""


class SchedulerUnavailableError(NodetopError):
    """The selected system's client tools are not usable here."""


class CommandError(NodetopError):
    """A command exited non-zero in a way we cannot interpret."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr.strip()
        super().__init__(f"{' '.join(cmd)} exited {returncode}: {self.stderr[:400]}")


class CommandTimeoutError(NodetopError):
    """A command did not return in time.

    Worth distinguishing from a plain failure: a control plane that reads
    slowly (or not at all) is itself a cluster-health signal, not a bug in
    the query.
    """

    def __init__(self, cmd: list[str], timeout: float) -> None:
        self.cmd = cmd
        self.timeout = timeout
        super().__init__(f"{' '.join(cmd)} timed out after {timeout:g}s")
