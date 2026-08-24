"""Shared fixtures.

Every test runs against recorded output, so the suite passes on a laptop with
no batch system installed -- which is also the only way to exercise the
interesting states on demand: a disabled queue, a cordoned node, a submit
filter that disagrees with the scheduler, a quota that admits then refuses.

The Slurm fixtures are verbatim captures from a production cluster.  The PBS,
LSF, Grid Engine and Kubernetes fixtures are authored from those systems'
documented output formats -- a difference worth being explicit about, because
only the Slurm path has been validated end-to-end against a live control plane.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES.parent))


def read(*parts: str) -> str:
    return (FIXTURES.joinpath(*parts)).read_text()


# -- slurm ------------------------------------------------------------------
@pytest.fixture
def slurm_nodes() -> str:
    return read("slurm", "nodes.txt")


@pytest.fixture
def slurm_partitions() -> str:
    return read("slurm", "partitions.txt")


@pytest.fixture
def slurm_qos() -> str:
    return read("slurm", "qos.txt")


@pytest.fixture
def slurm_backend(slurm_nodes, slurm_partitions, slurm_qos):
    from nodetop.backends.slurm import SlurmBackend
    from nodetop.runner import RecordedRunner

    return SlurmBackend(
        RecordedRunner({
            "scontrol show node": (0, slurm_nodes, ""),
            "scontrol show partition": (0, slurm_partitions, ""),
            "show qos": (0, slurm_qos, ""),
            "show assoc": (0, "acct-a||gn\nacct-b||gpu\n", ""),
            "squeue": (0, "", ""),
        })
    )


@pytest.fixture
def cluster(slurm_backend):
    from nodetop.core.cluster import Cluster

    return Cluster.load(slurm_backend, with_free_times=False)


@pytest.fixture
def replayed_cluster(slurm_backend):
    """The same data, marked as a recording.

    Worth a fixture of its own because `replayed=True` unlocks rendering paths
    nothing else reaches -- the "access is DECLARED, not confirmed" explanation
    among them, which was 148 columns wide because every width sweep ran
    against a live cluster and never rendered it.
    """
    from datetime import datetime, timedelta

    from nodetop.core.cluster import Cluster

    return Cluster.load(
        slurm_backend, with_free_times=False, replayed=True,
        taken_at=datetime.now() - timedelta(days=6, hours=4),
    )


# -- pbs --------------------------------------------------------------------
@pytest.fixture
def pbs_backend():
    from nodetop.backends.pbs import PbsBackend
    from nodetop.runner import RecordedRunner

    return PbsBackend(
        RecordedRunner({
            "pbsnodes -a -F json": (0, read("pbs", "pbsnodes.json"), ""),
            "qstat -Qf": (0, read("pbs", "qstat_Qf.txt"), ""),
            "qstat -f": (0, "", ""),
        })
    )


# -- lsf --------------------------------------------------------------------
@pytest.fixture
def lsf_backend():
    from nodetop.backends.lsf import LsfBackend
    from nodetop.runner import RecordedRunner

    return LsfBackend(
        RecordedRunner({
            "bhosts -gpu": (0, read("lsf", "bhosts_gpu.txt"), ""),
            "bhosts -w": (0, read("lsf", "bhosts.txt"), ""),
            "lshosts": (0, read("lsf", "lshosts.txt"), ""),
            "bqueues": (0, read("lsf", "bqueues_l.txt"), ""),
        })
    )


# -- sge --------------------------------------------------------------------
@pytest.fixture
def sge_backend():
    from nodetop.backends.sge import SgeBackend
    from nodetop.runner import RecordedRunner

    return SgeBackend(
        RecordedRunner({
            "qhost": (0, read("sge", "qhost.txt"), ""),
            "qstat -f": (0, read("sge", "qstat_f.txt"), ""),
            "qconf -sql": (0, "all.q\ncpu.q\n", ""),
            "qconf -sq all.q": (0, read("sge", "qconf_sq_allq.txt"), ""),
            "qconf -sq cpu.q": (0, read("sge", "qconf_sq_cpuq.txt"), ""),
            "qconf -srqsl": (1, "", "no resource quota sets"),
            "qconf -sul": (0, "gpu_users\n", ""),
            "qconf -su gpu_users": (0, "name gpu_users\nentries alice,bob\n", ""),
            # qhost reports only what is AVAILABLE; the configured total lives
            # in the exec host definition.
            "qconf -se": (
                0, "hostname h\nload_scaling NONE\ncomplex_values gpu=4,slots=32\n", ""
            ),
        })
    )


# -- kubernetes -------------------------------------------------------------
@pytest.fixture
def k8s_backend():
    from nodetop.backends.kubernetes import KubernetesBackend
    from nodetop.runner import RecordedRunner

    return KubernetesBackend(
        RecordedRunner({
            "get nodes": (0, read("k8s", "nodes.json"), ""),
            "get pods": (0, read("k8s", "pods.json"), ""),
            "get namespaces": (0, read("k8s", "namespaces.json"), ""),
            "get resourcequota": (0, read("k8s", "resourcequota.json"), ""),
            "auth whoami": (
                0,
                '{"status":{"userInfo":{"username":"alice","groups":["dev","system:authenticated"]}}}',
                "",
            ),
        })
    )


# -- a cluster whose probe answers, for exercising the check path -----------
_NODES = (
    "NodeName=n1 State=IDLE CPUTot=32 CPUAlloc=0 RealMemory=256000 AllocMem=0 "
    "Gres=gpu:4 AvailableFeatures=gold,256g,a100 Partitions=alpha,beta\n"
)
_PARTS = (
    "PartitionName=alpha\n   State=UP AllowGroups=ALL\n   Nodes=n1\n"
    "   TotalNodes=1\n\n"
    "PartitionName=beta\n   State=UP AllowGroups=ALL\n   Nodes=n1\n"
    "   TotalNodes=1\n"
)

ACCEPTED = (
    "sbatch: Verify job submission ...\n"
    "sbatch: QOS-Flag: alpha-prio\n"
    "sbatch: Account: acct\n"
    "sbatch: Verification: ***PASSED***\n"
    "sbatch: Job 42 to start at 2026-08-22T18:00:00 using 4 processors on nodes n1\n"
)
REFUSED = (
    "sbatch: error: Verification: ***REJECTED***\n"
    "sbatch: error: Reason: Invalid membership to account [acct]\n"
)
DISAGREEING = (
    "sbatch: error: Verification: ***PASSED***\n"
    "allocation failure: Invalid account or account/partition combination specified\n"
)


def probing_cluster(sbatch_stderr: str, rc: int = 1):
    """A one-node Slurm cluster whose ``--test-only`` returns a fixed answer."""
    from nodetop.backends.slurm import SlurmBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    backend = SlurmBackend(RecordedRunner({
        "scontrol show node": (0, _NODES, ""),
        "scontrol show partition": (0, _PARTS, ""),
        "show qos": (0, "alpha|2-00:00:00||gres/gpu=4|||0|\n", ""),
        "show assoc": (0, "acct||alpha\n", ""),
        "squeue": (0, "", ""),
        "sbatch": (rc, "", sbatch_stderr),
    }))
    return Cluster.load(backend, with_free_times=False)


@pytest.fixture
def accepting_cluster():
    return probing_cluster(ACCEPTED, rc=0)


@pytest.fixture
def refusing_cluster():
    return probing_cluster(REFUSED)


@pytest.fixture
def disagreeing_cluster():
    return probing_cluster(DISAGREEING)


@pytest.fixture
def cpu_only_cluster():
    """A Slurm cluster with no accelerators at all."""
    from nodetop.backends.slurm import SlurmBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    backend = SlurmBackend(RecordedRunner({
        "scontrol show node": (
            0, "NodeName=c1 State=IDLE CPUTot=64 CPUAlloc=0 RealMemory=256000 "
               "AllocMem=0 Partitions=cpuq\n", "",
        ),
        "scontrol show partition": (
            0, "PartitionName=cpuq\n   State=UP AllowGroups=ALL\n   Nodes=c1\n"
               "   TotalNodes=1\n", "",
        ),
        "show qos": (0, "", ""),
        "show assoc": (0, "acct||cpuq\n", ""),
        "squeue": (0, "", ""),
        "sbatch": (0, "", ACCEPTED),
    }))
    return Cluster.load(backend, with_free_times=False)

#: Client binaries the backends look for on PATH.  Every backend's
#: `detect()` and `capabilities()` asks `runner.which`, which asks
#: `shutil.which`, so patching that one point covers all six.
_SCHEDULER_CLIENTS = frozenset({
    "sbatch", "scontrol", "sinfo", "squeue", "sacctmgr",     # slurm
    "pbsnodes", "qstat", "qsub",                             # pbs / torque
    "bhosts", "bqueues", "bsub", "bmgroup",                  # lsf
    "qhost", "qconf",                                        # grid engine
    "kubectl",                                               # kubernetes
})


@pytest.fixture(autouse=True)
def _scheduler_clients_exist(monkeypatch):
    """Pretend every scheduler client is installed, on every host.

    **The suite was not hermetic and CI would have been red.** Every backend
    derives `capabilities().probe` from whether its client is on PATH, which is
    right for the product -- claiming a dry-run you cannot run is the exact lie
    this tool exists to catch -- and wrong for a test: on a machine without
    Slurm, `probe()` short-circuited and 40 tests of the probe path failed.
    They passed only because they happened to be written on a login node.

    Every test drives a `RecordedRunner`, so nothing is executed either way;
    what this restores is the *capability* answer the tests were written
    against. A test that needs a client to be absent should monkeypatch
    `shutil.which` itself -- narrower and visible at the point it matters.
    """
    real = shutil.which

    def present(binary, *args, **kwargs):
        if binary in _SCHEDULER_CLIENTS:
            return f"/usr/bin/{binary}"
        return real(binary, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", present)
