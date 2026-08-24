"""The funnel line must actually account for every partition.

`status` prints `87 partitions -> 5 open to you · 65 no access · 14 refused ·
3 dead`, and the whole point of that line is that it answers "why five rows"
by arithmetic rather than by asking the reader to trust it. That only holds if
every partition lands in exactly one term -- which is a property of the filter
chain, not of the line, and it breaks silently the moment a filter is added
without a matching count. Several have been.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from nodetop.cli import build_parser, cmd_status
from nodetop.core.cluster import Cluster
from nodetop.core.model import (
    BackendCapabilities,
    Identity,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())
TERMS = ("open to you", "with nodes", "no access", "refused", "no nodes", "down")


def _args(argv):
    return build_parser().parse_args(argv)


def _build(specs, *, accepts=(), probe=False):
    """specs: (name, node_count, allow_accounts, dead)."""
    nodes, queues = [], {}
    for name, count, allow, dead in specs:
        mine = [
            Node(name=f"{name}{i}", state_raw="IDLE", cpus_total=8,
                 memory_mb=16000, queues=(name,))
            for i in range(count)
        ]
        nodes += mine
        queues[name] = Queue(
            name=name, node_names=tuple(n.name for n in mine),
            declared_nodes=count, nodes=mine, allow_accounts=allow,
            **({"state_raw": "DOWN", "enabled": False} if dead else {}))

    class _Backend:
        name = "synthetic"
        queue_term = "partition"

        def capabilities(self):
            return BackendCapabilities(probe=probe, probe_supported=probe,
                                       probe_command="stub")

        def probe(self, q, shape, account=None):
            ok = q in accepts
            return Verdict(queue=q, account=account, allowed=ok,
                           category=VerdictCategory.OK if ok
                           else VerdictCategory.NOT_ENTITLED, reason="x")

        def submit_flags(self, q, shape):
            return []

    cluster = Cluster(backend_name="synthetic", queue_term="partition",
                      nodes=nodes, queues=queues,
                      identity=Identity(user="me", accounts=("mine",),
                                        qos=("x",)))
    if probe:
        cluster = dataclasses.replace(
            cluster, capabilities=_Backend().capabilities(),
            _backend=_Backend())
    return cluster


def _funnel(cluster, capsys, argv=("status",)):
    cmd_status(cluster, _args(list(argv)), PLAIN)
    out = capsys.readouterr().out
    line = next((ln for ln in out.splitlines()
                 if re.search(r"\d+ partitions?\b", ln)), None)
    assert line is not None, f"no funnel line in:\n{out}"
    total = int(re.search(r"(\d+) partition", line).group(1))
    parts = {}
    for term in TERMS:
        m = re.search(rf"(\d+) {re.escape(term)}", line)
        if m:
            parts[term] = int(m.group(1))
    return total, parts


CASES = {
    "all reachable": [("a", 2, (), False), ("b", 1, (), False)],
    "one dead": [("a", 2, (), False), ("dead", 1, (), True)],
    "one not mine": [("a", 2, (), False), ("theirs", 1, ("other",), False)],
    "one empty": [("a", 2, (), False), ("empty", 0, (), False)],
    "one of each": [("a", 2, (), False), ("theirs", 1, ("other",), False),
                    ("empty", 0, (), False), ("dead", 1, (), True)],
    "nothing reachable": [("theirs", 1, ("other",), False),
                          ("dead", 1, (), True)],
}


@pytest.mark.parametrize("name", list(CASES))
class TestTheFunnelAccountsForEveryPartition:
    def test_the_terms_sum_to_the_total(self, name, capsys):
        total, parts = _funnel(_build(CASES[name]), capsys)
        assert sum(parts.values()) == total, (name, parts, total)

    def test_the_total_is_the_real_partition_count(self, name, capsys):
        cluster = _build(CASES[name])
        total, _ = _funnel(cluster, capsys)
        assert total == len(cluster.queues)


class TestTheFunnelWithAProbe:
    SPECS = [("yes", 1, (), False), ("no", 1, (), False),
             ("theirs", 1, ("other",), False), ("dead", 1, (), True)]

    def test_it_still_sums_with_refusals_in_play(self, capsys):
        cluster = _build(self.SPECS, accepts=("yes",), probe=True)
        total, parts = _funnel(cluster, capsys)
        assert sum(parts.values()) == total, parts

    def test_a_refusal_is_counted_as_refused(self, capsys):
        cluster = _build(self.SPECS, accepts=("yes",), probe=True)
        _total, parts = _funnel(cluster, capsys)
        assert parts.get("refused") == 1, parts

    def test_all_reports_the_whole_cluster(self, capsys):
        cluster = _build(self.SPECS, accepts=("yes",), probe=True)
        total, parts = _funnel(cluster, capsys, ("status", "--all"))
        assert sum(parts.values()) == total, parts
