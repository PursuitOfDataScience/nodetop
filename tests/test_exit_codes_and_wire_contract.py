"""Two honesty findings: a wire verdict that could not say "was this asked?",
and a branch that answered "success" for a cluster it could not see.

* `where --json` emitted `allowed/category/reason/filter_verdict/effective_qos`
  but not `durable`. The text surface distinguishes `BLOCKED` from `NO ANSWER`,
  so a `--json` consumer could only reproduce that by vendoring
  `TRANSIENT_CATEGORIES` from `model.py`. A wire vocabulary that cannot answer
  "is this refusal real?" forces every consumer to copy the table.
* `cmd_status`'s `if not cluster.nodes:` branch returned **0**. It is
  unreachable through `main()` — `_reject_broken_snapshot` computes
  `fatal = not cluster.nodes or ...` and returns 3 before dispatch — but it is
  reachable by a direct caller, which is what the README documents as the
  library entry point. "No nodes: wrong backend, or the control plane is down"
  is the exit-3 case by the guard's own docstring, and dead code carrying the
  wrong code is a trap for the next refactor that moves the guard.
"""

from __future__ import annotations

import argparse
import json

import nodetop.core.model as model_mod
from nodetop.cli import build_parser, cmd_status
from nodetop.core.cluster import Cluster, Queue
from nodetop.core.model import Node
from nodetop.render import Style

PLAIN = Style(enabled=False)


def _args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _node() -> Node:
    return Node(name="n1", state_raw="idle", cpus_total=8, cpus_alloc=0,
                memory_mb=1000, memory_alloc_mb=0)


def _empty_cluster() -> Cluster:
    return Cluster(backend_name="slurm", queue_term="partition",
                   nodes=[], queues={}, errors={})


def _populated_cluster() -> Cluster:
    node = _node()
    return Cluster(backend_name="slurm", queue_term="partition", nodes=[node],
                   queues={"q": Queue(name="q", node_names=("n1",), nodes=[node])},
                   errors={})


class TestTheWireSaysWhetherTheQuestionWasAnswered:
    def test_the_verdict_carries_durable(self, cluster, capsys) -> None:
        from nodetop.cli import cmd_where

        cmd_where(cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        places = json.loads(capsys.readouterr().out)
        assert places, "no placements to check"
        verdicts = [p["verdict"] for p in places if p["verdict"]]
        assert verdicts, "no verdict was produced, so nothing was checked"
        assert all("durable" in v for v in verdicts)

    def test_durable_agrees_with_the_category_table(self, cluster, capsys) -> None:
        # The point of publishing it: a consumer must not have to vendor
        # `TRANSIENT_CATEGORIES` to get the same answer.
        from nodetop.cli import cmd_where

        cmd_where(cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        checked = 0
        for place in json.loads(capsys.readouterr().out):
            verdict = place["verdict"]
            if verdict is None:
                continue  # nothing was asked of this queue
            assert verdict["durable"] == (verdict["category"] not in model_mod.TRANSIENT_CATEGORIES)
            checked += 1
        assert checked, "no verdict was produced, so nothing was compared"

    def test_the_keys_that_were_already_there_still_are(self, cluster, capsys) -> None:
        # Control on an additive change: nothing was displaced.
        from nodetop.cli import cmd_where

        cmd_where(cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        verdict = next(
            p["verdict"] for p in json.loads(capsys.readouterr().out) if p["verdict"]
        )
        for key in ("allowed", "category", "reason", "filter_verdict", "effective_qos"):
            assert key in verdict, key


class TestWhoOwnsTheNoNodesExitCode:
    """The audit read `cmd_status`'s no-nodes arm answering 0 as a bug.

    It is not, and the reason is worth pinning rather than arguing: **3 is a
    code `main()` owns.** Every `cmd_*` returns 0/1/2 and nothing else, which
    `test_degenerate.py::test_it_does_not_raise` asserts for every shape, and
    `main()` returns 3 for this exact input before ever dispatching. Widening
    this one function to 3 broke three existing tests, which is how the
    contract announced itself.

    What WAS wrong is that the difference was undocumented and untested, so the
    arm read as a trap for the next refactor that moves the guard. Both halves
    are pinned here instead.
    """

    def test_the_direct_call_answers_zero_and_says_why(self, capsys) -> None:
        assert cmd_status(_empty_cluster(), _args(["status"]), PLAIN) == 0
        assert "no nodes" in capsys.readouterr().out

    def test_the_direct_json_call_still_emits_the_whole_document(self, capsys) -> None:
        # A `--json` consumer given empty stdout cannot tell "no nodes" from
        # "the command crashed", which is why this arm exists at all.
        assert cmd_status(_empty_cluster(), _args(["--json", "status"]), PLAIN) == 0
        assert json.loads(capsys.readouterr().out)

    def test_the_same_cluster_through_main_exits_three(self, tmp_path, capsys) -> None:
        # The other half of the difference: the guard, not the command, is what
        # refuses to call an unreadable cluster a success.
        import json as json_mod

        from nodetop.cli import main

        snapshot = tmp_path / "empty.json"
        snapshot.write_text(json_mod.dumps({
            "nodetop": "0.5.2", "backend": "slurm", "queue_term": "partition",
            "captured_at": "2026-09-09T00:00:00", "errors": {"nodes": "boom"},
            "commands": {},
        }))
        assert main(["--replay", str(snapshot), "status"]) == 3

    def test_a_cluster_with_nodes_still_exits_zero(self, capsys) -> None:
        # The control that holds either way.
        assert cmd_status(_populated_cluster(), _args(["status"]), PLAIN) == 0
        assert capsys.readouterr().out.strip()
