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


def _registry() -> dict[str, type[Backend]]:
    # Imported lazily so a missing optional dependency in one backend cannot
    # stop the others from loading.
    from .kubernetes import KubernetesBackend
    from .lsf import LsfBackend
    from .pbs import PbsBackend
    from .sge import SgeBackend
    from .slurm import SlurmBackend
    from .sshpool import SshPoolBackend

    return {
        b.name: b
        for b in (
            SlurmBackend,
            PbsBackend,
            LsfBackend,
            SgeBackend,
            KubernetesBackend,
            SshPoolBackend,
        )
    }


def names() -> list[str]:
    """Every backend name, in detection order."""
    return list(_registry())


def get(name: str, runner: object | None = None) -> Backend:
    """Instantiate a backend by name, optionally with a specific runner.

    Passing a runner is how snapshot replay works: the same backend code runs
    against recorded output instead of a live control plane.
    """
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(reg)}")
    if runner is None:
        return reg[name]()
    return reg[name](runner=runner)  # type: ignore[call-arg]


def available() -> list[str]:
    """Names of every backend that detects its system as present."""
    return [n for n, cls in _registry().items() if cls.detect()]


def detect() -> Backend:
    """Pick a backend for this machine.

    Raises :class:`~nodetop.exceptions.NoBackendError` only if even the ssh
    pool declines, which should not happen -- it is the universal fallback.
    """
    from ..exceptions import NoBackendError

    for cls in _registry().values():
        if cls.detect():
            return cls()
    raise NoBackendError(
        "no batch system detected: none of "
        f"{', '.join(_registry())} is usable here"
    )
