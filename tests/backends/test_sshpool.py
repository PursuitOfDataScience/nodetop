"""A pool of machines with no scheduler at all."""

from __future__ import annotations

import pathlib

from nodetop.backends.sshpool import SshPoolBackend
from nodetop.core.model import JobShape
from nodetop.runner import RecordedRunner

GPU_HOST = """CPUS=64
LOAD=2.5
MEMTOTAL=257000
MEMAVAIL=200000
GPU=NVIDIA A100-SXM4-40GB, 40960, 0, 0
GPU=NVIDIA A100-SXM4-40GB, 40960, 38000, 97
"""

CPU_HOST = """CPUS=8
LOAD=0.1
MEMTOTAL=32000
MEMAVAIL=30000
"""


def _backend(output, hosts=("node1",)):
    return SshPoolBackend(hosts=hosts, runner=RecordedRunner({"": (0, output, "")}))


class TestHostProbe:
    def test_cpu_and_memory(self):
        n = _backend(GPU_HOST).load_nodes()[0]
        assert n.cpus_total == 64
        assert n.memory_mb == 257000
        assert n.memory_alloc_mb == 57000

    def test_load_average_is_the_only_occupancy_signal(self):
        # Without a scheduler there is nothing else to go on.
        assert _backend(GPU_HOST).load_nodes()[0].cpus_alloc == 3

    def test_accelerator_identification(self):
        n = _backend(GPU_HOST).load_nodes()[0]
        assert n.gpus_total == 2
        assert n.accelerator.model == "A100"

    def test_an_accelerator_holding_memory_counts_as_busy(self):
        # A job between kernels reads 0% utilisation while still owning the
        # card; treating that as free is how two jobs collide.
        n = _backend(GPU_HOST).load_nodes()[0]
        assert n.gpus_alloc == 1
        assert n.gpus_free == 1

    def test_a_host_with_no_accelerator(self):
        n = _backend(CPU_HOST).load_nodes()[0]
        assert n.is_gpu_node is False
        assert n.cpus_total == 8

    def test_an_unresponsive_host_is_down_not_empty(self):
        n = _backend("").load_nodes()[0]
        assert n.schedulable is False
        assert n.unreachable is True
        assert "no response" in n.reason


class TestPool:
    def test_a_single_queue_named_pool(self):
        queues = _backend(CPU_HOST, hosts=("a", "b")).load_queues()
        assert len(queues) == 1
        assert queues[0].name == "pool"
        assert queues[0].node_names == ("a", "b")

    def test_the_pool_is_always_usable(self):
        # There is no entitlement model, so nothing can close it.
        assert _backend(CPU_HOST).load_queues()[0].usable is True

    def test_no_limits_exist(self):
        assert _backend(CPU_HOST).load_limits() == {}


class TestHonesty:
    def test_capabilities_state_every_gap(self):
        caps = _backend(CPU_HOST).capabilities()
        assert caps.probe is False
        assert caps.limits is False
        assert caps.identity is False
        assert any("no scheduler" in n.lower() for n in caps.notes)

    def test_the_notes_warn_that_nothing_is_reserved(self):
        caps = _backend(CPU_HOST).capabilities()
        assert any("not reserved" in n for n in caps.notes)

    def test_probe_returns_none(self):
        assert _backend(CPU_HOST).probe("pool", JobShape()) is None

    def test_no_free_time_estimate(self):
        assert _backend(CPU_HOST).load_node_free_times() == {}


class TestHostDiscovery:
    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("NODETOP_HOSTS", "a, b  c")
        assert SshPoolBackend._discover_hosts() == ["a", "b", "c"]

    def test_defaults_to_the_local_machine(self, monkeypatch):
        monkeypatch.delenv("NODETOP_HOSTS", raising=False)
        monkeypatch.setattr(
            "nodetop.backends.sshpool.HOSTS_FILE",
            pathlib.Path("/nonexistent/nodetop-hosts"),
        )
        assert SshPoolBackend._discover_hosts() == ["localhost"]

    def test_it_is_always_available_as_the_last_resort(self):
        assert SshPoolBackend.detect() is True
