"""Every command rendered against every backend, at every width.

The rest of the render suite runs against the Slurm fixture, which left the
other five adapters' output unmeasured -- and they differ in the ways that
matter to a layout: the queue term is a different length ("partition" /
"queue" / "namespace" / "pool"), the capability notes are different prose, and
`probe_supported=False` on PBS, LSF and the ssh pool reaches the "declared, not
confirmed" branch with different text in it.

Finding the gap here is what turned up an unwrapped `re-run with --all` line.
"""

from __future__ import annotations

import pytest

from nodetop.core.cluster import Cluster
from nodetop.core.model import JobShape
from nodetop.render import Glyphs, Style, width
from tests.test_cli_render import COMMANDS, _args

PLAIN = Style(depth=0, glyphs=Glyphs())

BACKENDS = ["pbs_backend", "lsf_backend", "sge_backend", "k8s_backend"]


def _submit_lines(cluster) -> set[str]:
    """Every line `where` may emit purely to be copied.

    Identified by asking the cluster, not by matching a prefix: the previous
    version of this exemption tested for a leading "--", which is what Slurm
    happens to emit. PBS opens with "-q", Kubernetes with "-n", so on every
    other backend the exemption silently did nothing and the copy-me line
    looked like an overflow.
    """
    out = set()
    for shape in (JobShape(nodes=1, cpus_per_task=1),
                  JobShape(nodes=1, cpus_per_task=2),
                  JobShape(nodes=1, gpus_per_node=1),
                  JobShape(nodes=1, gpus_per_node=8, gpu_memory_gb=999)):
        for name in cluster.queues:
            flags = cluster.submit_flags(name, shape)
            if flags:
                out.add("  " + " ".join(flags))
    return out


@pytest.fixture
def any_cluster(request, backend_name):
    return Cluster.load(request.getfixturevalue(backend_name), with_free_times=True)


@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("size", [40, 60, 100])
@pytest.mark.parametrize("_label,fn,argv", COMMANDS, ids=[c[0] for c in COMMANDS])
class TestEveryBackendFitsTheTerminal:
    def test_no_line_overflows(
        self, any_cluster, capsys, monkeypatch, backend_name, size, _label, fn, argv
    ):
        monkeypatch.setenv("COLUMNS", str(size))
        fn(any_cluster, _args(argv), PLAIN)
        out = capsys.readouterr().out
        if _label.startswith("exclude"):
            return  # a host list to paste, not a layout
        exempt = _submit_lines(any_cluster)
        for line in out.splitlines():
            if line in exempt:
                continue
            assert width(line) <= size, (
                f"{backend_name} {_label} @{size}: {width(line)} > {size}: "
                f"{line!r}")

    def test_it_renders_something(
        self, any_cluster, capsys, monkeypatch, backend_name, size, _label, fn, argv
    ):
        # Guards the guard: a command that silently produced nothing would sail
        # through the width check above.
        monkeypatch.setenv("COLUMNS", str(size))
        fn(any_cluster, _args(argv), PLAIN)
        assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("backend_name", BACKENDS)
class TestEveryBackendNamesItsOwnVocabulary:
    def test_the_queue_term_is_used(self, any_cluster, capsys, backend_name):
        from nodetop.cli import cmd_queues

        cmd_queues(any_cluster, _args(["queues"]), PLAIN)
        out = capsys.readouterr().out.lower()
        assert any_cluster.queue_term.lower() in out
