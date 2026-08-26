"""Backend registry and autodetection.

Detection order matters.  A single machine can have several clients installed
-- a login node with both ``sbatch`` and ``kubectl`` on PATH is common -- so
the order runs from most-specific to most-general and ends at the ssh pool,
which needs nothing at all and is therefore always available.
"""

from __future__ import annotations

from .base import Backend, BackendCapabilities

__all__ = [
    "Backend",
    "BackendCapabilities",
    "available",
    "detect",
    "get",
    "names",
]


#: Every backend, in DETECTION ORDER, as (name, module, class).
#:
#: A table rather than a dict of imported classes, so that asking *which*
#: backends exist costs no imports at all and detecting one costs only the
#: modules it had to try. `build_parser` asks for the names, and asking used to
#: import all six -- with `subprocess`, `termios`, `getpass` and `grp` behind
#: them -- before argparse had looked at a single argument.
#:
#: Worth **8 ms of a 95 ms startup**, measured properly: two source trees
#: identical but for this file, fifteen runs each, medians. `--version` 95.0 ->
#: 87.2 ms, `--help` 96.1 -> 87.9 ms, and `nodetop backends` -- which must ask
#: every backend and so imports every backend -- unchanged at 95.8 -> 94.9 ms.
#:
#: An earlier attempt at this was reverted on a measurement of "3 ms, and one
#: command slower". That comparison was against a tree that differed in a dozen
#: other files; it was noise. Two trees, one file.
#:
#: The order is unchanged and still matters: most-specific first, ending at the
#: ssh pool, which needs nothing and is therefore always available.
_BACKENDS: tuple[tuple[str, str, str], ...] = (
    ("slurm", ".slurm", "SlurmBackend"),
    ("pbs", ".pbs", "PbsBackend"),
    ("lsf", ".lsf", "LsfBackend"),
    ("sge", ".sge", "SgeBackend"),
    ("kubernetes", ".kubernetes", "KubernetesBackend"),
    ("sshpool", ".sshpool", "SshPoolBackend"),
)


def _load(name: str) -> type[Backend]:
    """The class for one backend, importing just its module.

    Imported here rather than at module level so a missing optional dependency
    in one backend cannot stop the others from loading -- the reason the old
    registry imported inside a function -- and so that a host running Slurm
    never pays for the five adapters it will not use.
    """
    from importlib import import_module

    for known, module, attr in _BACKENDS:
        if known == name:
            return getattr(import_module(module, __package__), attr)
    raise KeyError(
        f"unknown backend {name!r}; known: {', '.join(n for n, _, _ in _BACKENDS)}")


def names() -> list[str]:
    """Every backend name, in detection order.  Imports nothing."""
    return [n for n, _, _ in _BACKENDS]


def get(name: str, runner: object | None = None) -> Backend:
    """Instantiate a backend by name, optionally with a specific runner.

    Passing a runner is how snapshot replay works: the same backend code runs
    against recorded output instead of a live control plane.
    """
    cls = _load(name)
    if runner is None:
        return cls()
    return cls(runner=runner)  # type: ignore[call-arg]


def available() -> list[str]:
    """Names of every backend that detects its system as present."""
    # This one has to ask every backend, so it imports every backend. It is
    # `nodetop backends`, whose entire output is that answer.
    return [n for n in names() if _load(n).detect()]


def detect() -> Backend:
    """Pick a backend for this machine.

    Raises :class:`~nodetop.exceptions.NoBackendError` only if even the ssh
    pool declines, which should not happen -- it is the universal fallback.
    """
    from ..exceptions import NoBackendError

    # One at a time, stopping at the first hit: on a Slurm login node that is
    # one import instead of six.
    for name in names():
        cls = _load(name)
        if cls.detect():
            return cls()
    raise NoBackendError(
        "no batch system detected: none of "
        f"{', '.join(names())} is usable here"
    )
