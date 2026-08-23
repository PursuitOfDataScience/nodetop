"""A pool of machines with no scheduler at all.

Plenty of real multi-node setups have no batch system: a rack of workstations,
a handful of cloud instances, a single large shared box people take turns on.
They have exactly the problem this tool addresses -- "is there room for this
job, on hardware that can actually run it?" -- and none of the machinery for
answering it.  Every other backend reads that answer out of a scheduler; this
one measures it directly.

The trade is honest and worth stating plainly:

* There is no entitlement model, so nothing is gated and nothing is confirmed.
* There is no queue, so ``pool`` is the single :class:`~nodetop.core.model.Queue`.
* Occupancy is what is *running right now*, not what is reserved.  On an
  unscheduled machine those are the same thing, which is precisely why two
  people can start a job on it simultaneously.

Hosts come from the constructor, then ``$NODETOP_HOSTS``, then
``~/.config/nodetop/hosts``, then the local machine.
"""

from __future__ import annotations

import getpass
import math
import os
import pathlib
import re
from collections.abc import Iterable
from datetime import datetime

from ..core.hardware import identify_accelerator
from ..core.model import Identity, Job, JobShape, Limits, Node, Queue, Verdict
from ..runner import Runner, SubprocessRunner
from .base import BackendCapabilities

__all__ = ["SshPoolBackend", "HOSTS_FILE"]

HOSTS_FILE = pathlib.Path.home() / ".config" / "nodetop" / "hosts"

#: One shell pipeline per host, so a machine costs a single round trip.
#: Every field is prefixed so a partial answer is still parseable -- a host
#: with no nvidia-smi must still report its CPUs.
_PROBE_SCRIPT = r"""
echo "CPUS=$(nproc 2>/dev/null || echo 0)"
echo "LOAD=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
awk '/MemTotal/{printf "MEMTOTAL=%d\n", $2/1024}
     /MemAvailable/{printf "MEMAVAIL=%d\n", $2/1024}' /proc/meminfo 2>/dev/null
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null | sed 's/^/GPU=/'
fi
if command -v rocm-smi >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --csv 2>/dev/null | sed 's/^/ROCM=/'
fi
"""


class SshPoolBackend:
    """Adapter for an unscheduled pool of machines."""

    name = "sshpool"
    queue_term = "pool"

    def __init__(
        self,
        hosts: Iterable[str] | None = None,
        runner: Runner | None = None,
        ssh_timeout: int = 10,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.hosts = list(hosts) if hosts else self._discover_hosts()
        self.ssh_timeout = ssh_timeout

    @staticmethod
    def _discover_hosts() -> list[str]:
        env = os.environ.get("NODETOP_HOSTS", "").strip()
        if env:
            return [h.strip() for h in re.split(r"[,\s]+", env) if h.strip()]
        if HOSTS_FILE.is_file():
            return [
                line.strip()
                for line in HOSTS_FILE.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        return ["localhost"]

    @classmethod
    def detect(cls) -> bool:
        # The universal fallback: a machine is always a pool of one.  Listed
        # last in the registry so a real scheduler always wins.
        return True

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=False,
            limits=False,
            identity=False,
            free_times=False,
            notes=(
                "no scheduler: nothing to check, nothing to queue in -- and occupancy is "
                "reported, not reserved",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_host(self, name: str, output: str) -> Node:
        fields: dict[str, str] = {}
        gpus: list[tuple[str, int, int, int]] = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("GPU="):
                parts = [p.strip() for p in line[4:].split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        (parts[0], _i(parts[1]), _i(parts[2]), _i(parts[3]))
                    )
            elif line.startswith("ROCM="):
                gpus.append((line[5:].strip(), 0, 0, 0))
            elif "=" in line:
                key, _, value = line.partition("=")
                fields[key] = value.strip()

        cpus = _i(fields.get("CPUS"))
        load = float(fields.get("LOAD") or 0)
        mem_total = _i(fields.get("MEMTOTAL"))
        mem_avail = _i(fields.get("MEMAVAIL"))

        # A GPU counts as busy when it holds memory, not when its utilisation
        # is high: a job between kernels reads 0% util while still owning the
        # card, and treating that as free is how two jobs collide.
        busy = sum(1 for _, _, used, _ in gpus if used > 512)
        model = gpus[0][0] if gpus else ""

        return Node(
            name=name,
            state_raw="up" if output.strip() else "unreachable",
            conditions=frozenset() if output.strip() else frozenset({"DOWN"}),
            cpus_total=cpus,
            # Load average is the only occupancy signal without a scheduler,
            # and it is rounded UP: understating occupancy overstates free
            # capacity, which is the direction that gets two jobs started on
            # the same cores. (Python's round() is also banker's rounding, so
            # round(2.5) would give 2.)
            cpus_alloc=min(cpus, math.ceil(load)),
            memory_mb=mem_total,
            memory_alloc_mb=max(0, mem_total - mem_avail),
            gpus_total=len(gpus),
            gpus_alloc=busy,
            accelerator=identify_accelerator(None, model) if model else None,
            labels=tuple(f"gpu{i}={g[0]}" for i, g in enumerate(gpus)),
            queues=("pool",),
            unreachable=not output.strip(),
            reason="" if output.strip() else "no response over ssh",
        )

    def _run_on(self, host: str) -> str:
        if host in {"localhost", "127.0.0.1", os.uname().nodename}:
            cmd = ["bash", "-c", _PROBE_SCRIPT]
        else:
            cmd = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={self.ssh_timeout}",
                "-o", "StrictHostKeyChecking=accept-new",
                host,
                _PROBE_SCRIPT,
            ]
        try:
            rc, out, _ = self.runner.run_full(cmd, timeout=self.ssh_timeout + 20)
            return out if rc == 0 else ""
        except Exception:
            return ""

    def load_nodes(self) -> list[Node]:
        return [self.parse_host(h, self._run_on(h)) for h in self.hosts]

    # -- queues -------------------------------------------------------------
    def load_queues(self) -> list[Queue]:
        names = tuple(self.hosts)
        return [
            Queue(
                name="pool",
                state_raw="no scheduler",
                node_names=names,
                declared_nodes=len(names),
                is_default=True,
            )
        ]

    def load_limits(self) -> dict[str, Limits]:
        return {}

    def load_jobs(self) -> list[Job]:
        """Not implemented for this system.

        The protocol requires the method; an empty list here means "this adapter
        cannot list jobs", which the caller tells apart from "this node has no
        jobs" by asking whether the cluster returned any jobs at all. Reporting
        an idle node because the query does not exist would be phantom capacity
        in a new place.
        """
        return []

    def load_identity(self) -> Identity:
        return Identity(user=os.environ.get("USER") or getpass.getuser())

    def load_node_free_times(self) -> dict[str, datetime]:
        # Nothing declares a finish time, so there is no estimate to give.
        return {}

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """No scheduler, so no entitlement question exists to ask."""
        return None

    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        # There is nothing to submit to; the useful output is where to ssh.
        return []

    def format_nodelist(self, names: Iterable[str]) -> str:
        return ",".join(sorted(names))


def _i(text: str | None) -> int:
    if not text:
        return 0
    m = re.match(r"^\s*(\d+)", str(text))
    return int(m.group(1)) if m else 0
