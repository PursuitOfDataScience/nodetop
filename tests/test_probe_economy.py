"""A dry-run is the most expensive thing this tool does, so it is not spent
where the answer cannot matter.

`rank` already declines to probe a queue with no accelerator when one is asked
for, and `evaluate` already declines for a queue carrying an *operational*
blocker, on the stated ground that "the answer could not change the verdict on
screen".  The same argument covers a queue whose hardware could never host the
shape, and it was not being made -- which is where the round trips actually
were.  Measured on an 87-partition cluster:

=========================  ==================  ==================
`where`                    before              after
=========================  ==================  ==================
``-c 2 --mem 2``           7.6 s, 86 probes    7.7 s, 86 probes
``-c 64 --mem 200``        11.5 s, 130 probes  **3.1 s, 32**
``-N 8 -c 32 --mem 100``   24.0 s, 119 probes  **1.7 s, 16**
=========================  ==================  ==================

The first row is the control: for a two-core job nothing is impossible, so
nothing is skipped.  The saving is not a trade -- the rows that stop being
probed read *better* afterwards, because `TOO FEW 1/8` is the fact the reader
needs and a permission verdict on the same queue was hiding it.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re

import pytest

from nodetop.core.cluster import Cluster
from nodetop.core.fit import evaluate, rank
from nodetop.core.model import (
    BackendCapabilities,
    Identity,
    JobShape,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)

CAPS = BackendCapabilities(probe=True, probe_supported=True, probe_command="sbatch --test-only")


class _Counting:
    """A backend that accepts everything and counts what it was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def probe(self, queue, shape, account=None):
        self.asked.append(queue)
        return Verdict(
            queue=queue,
            account=account,
            allowed=True,
            category=VerdictCategory.OK,
            reason="ok",
        )

    def capabilities(self):
        return CAPS

    def submit_flags(self, queue, shape):
        return []

    def format_nodelist(self, names):
        return ",".join(sorted(names))


def _cluster(*specs):
    """One queue per ``(name, node_count, cpus, memory_mb)``."""
    nodes, queues = [], {}
    for name, count, cpus, mem in specs:
        mine = [
            Node(
                name=f"{name}{i}", state_raw="IDLE", cpus_total=cpus, memory_mb=mem, queues=(name,)
            )
            for i in range(count)
        ]
        nodes += mine
        queues[name] = Queue(
            name=name,
            node_names=tuple(n.name for n in mine),
            declared_nodes=count,
            nodes=mine,
        )
    backend = _Counting()
    cluster = dataclasses.replace(
        Cluster(
            backend_name="slurm",
            queue_term="partition",
            nodes=nodes,
            queues=queues,
            identity=Identity(user="me", accounts=("a",)),
        ),
        capabilities=CAPS,
        _backend=backend,
    )
    return cluster, backend


class TestAnImpossibleQueueIsNotProbed:
    def test_a_queue_with_too_small_a_node_is_not_asked(self):
        cluster, backend = _cluster(("big", 4, 64, 256000), ("small", 4, 8, 16000))
        rank(cluster, JobShape(cpus_per_task=32, memory_gb=100), use_probe=True, accounts=["a"])
        assert backend.asked == ["big"], "a 32-core job has nothing to ask `small`"

    def test_a_queue_with_too_few_nodes_is_not_asked(self):
        # Both halves of `Capacity.ever_possible`: the right kind of node, and
        # enough of them. A one-node queue asked for eight cannot acquire seven
        # more, so the entitlement answer is not the finding.
        cluster, backend = _cluster(("wide", 8, 32, 128000), ("single", 1, 32, 128000))
        rank(
            cluster,
            JobShape(nodes=8, cpus_per_task=32, memory_gb=100),
            use_probe=True,
            accounts=["a"],
        )
        assert backend.asked == ["wide"]

    def test_a_shape_that_fits_everywhere_still_probes_everywhere(self):
        """The control, and the one that must not regress.

        This optimisation has to be invisible for the ordinary small job -- the
        measured case where nothing was saved because nothing was impossible.
        """
        cluster, backend = _cluster(("big", 4, 64, 256000), ("small", 4, 8, 16000))
        rank(cluster, JobShape(cpus_per_task=2, memory_gb=2), use_probe=True, accounts=["a"])
        assert sorted(backend.asked) == ["big", "small"]

    def test_the_row_is_still_reported_and_says_why(self):
        # Skipped, not hidden: `--all` and the default view both still carry the
        # queue, and the reason is the hardware rather than a permission.
        cluster, _backend = _cluster(("small", 1, 8, 16000))
        queue = cluster.queues["small"]
        place = evaluate(
            cluster,
            JobShape(nodes=8, cpus_per_task=32, memory_gb=100),
            queue,
            use_probe=True,
            accounts=["a"],
        )
        assert place.hardware_incompatible or place.capacity.too_few_nodes
        assert place.verdict is None, "no probe was spent"
        assert place.entitlement_unconfirmed, "and the report says it is unsettled"

    def test_an_unresolved_node_list_is_still_probed(self):
        """A queue we cannot see the nodes of is not a queue we may rule out.

        `assess_capacity` sets `required_nodes` to 0 when the node list is
        incomplete, precisely so a resolution failure cannot look like a verdict
        -- and this skip must inherit that caution rather than turn a missing
        node into an impossible queue.
        """
        cluster, backend = _cluster(("hidden", 1, 8, 16000))
        # The queue claims 40 nodes and resolved 1.
        cluster.queues["hidden"] = dataclasses.replace(cluster.queues["hidden"], declared_nodes=40)
        rank(
            cluster, JobShape(nodes=8, cpus_per_task=4, memory_gb=2), use_probe=True, accounts=["a"]
        )
        assert backend.asked == ["hidden"]


class TestTheDocumentedTestCountStaysHonest:
    """`README.md` and `DESIGN.md` both quote a test count, and both had drifted.

    Measured during a final check: README said `~3400`, DESIGN said `3140`, the
    suite ran **3871**. Nothing guarded either, so the numbers had been quietly
    wrong for a while -- and a figure a reader uses to sanity-check their own run
    is worth keeping true. A sibling package pins the same claim with a test, and
    it caught the same drift twice in one session.

    Tolerant on purpose: the README writes `~N`, so this fails on a real
    divergence rather than on every test added.
    """

    TOLERANCE = 0.05

    @staticmethod
    def _collected() -> int:
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "--co",
                "-q",
            ],
            capture_output=True,
            text=True,
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        found = re.search(r"^(\d+) tests collected", done.stdout, re.M)
        assert found, done.stdout[-2000:]
        return int(found.group(1))

    @pytest.mark.parametrize("doc", ["README.md", "DESIGN.md"])
    def test_the_quoted_count_is_near_the_real_one(self, doc):
        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / doc).read_text()
        quoted = re.search(r"pytest\s+#\s*~?(\d[\d,]*) tests", text)
        if quoted is None:
            pytest.skip(f"{doc} quotes no test count")
        claimed = int(quoted.group(1).replace(",", ""))
        real = self._collected()
        drift = abs(claimed - real) / real
        assert drift <= self.TOLERANCE, (
            f"{doc} says {claimed} tests, the suite collects {real} ({drift:.0%} off)"
        )
