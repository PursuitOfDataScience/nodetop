"""Re-recording a recording is refused cleanly, and that branch is reachable.

`cmd_snapshot`'s guard carried `# pragma: no cover - set by main() before
dispatch`, which claimed it could not be reached. It can:

* `main()` sets `cluster.capture` on the LIVE path only (it builds a
  `CapturingRunner`, loads through it, assigns, and calls `cmd_snapshot`
  directly);
* a replay reaches the same function through the `_COMMANDS` table with nothing
  captured, because a replay runs no commands to capture.

So `nodetop --replay snap.json snapshot` lands on the guard every time.
Measured on a real snapshot: stdout empty, the line on stderr, exit **2**.

Refusing is the right behaviour and is unchanged -- the capture holds the
command output a live run produced, and there is none to re-record. What
changed is the claim: the `no cover` came off, because the comment directly
above it says how the guard came to be duplicated in the first place ("an
unreachable branch is easy to duplicate precisely because no test exercises
it") and excluding a reachable branch is that same mechanism.

Swept the neighbours while here: of the eight `pragma: no cover` markers in
`cli.py`, this was the only false one. The rest are exception handlers for a
failed recheck, a closed stderr, a probe that must not break a listing, a job
that vanished between two reads, a defensive `node is None`, an `else` needing
an unknown `kind` that only this function pushes, and `__main__`.
"""

from __future__ import annotations

import contextlib
import io

from nodetop.cli import _COMMANDS, build_parser, cmd_snapshot
from nodetop.core.cluster import Cluster
from nodetop.core.model import Identity, Node, Queue
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())

MESSAGE = "snapshot needs a capturing runner; nothing was captured"


def _cluster() -> Cluster:
    """A loaded cluster with no capture -- exactly a replay's shape."""
    nodes = [
        Node(name=f"n{i}", state_raw="IDLE", cpus_total=8, memory_mb=64000,
             queues=("open",))
        for i in range(3)
    ]
    queues = {
        "open": Queue(name="open", node_names=tuple(n.name for n in nodes),
                      declared_nodes=len(nodes), nodes=nodes, allow_accounts=()),
    }
    return Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes,
        queues=queues, identity=Identity(user="me", accounts=("a",), qos=("q",)),
    )


class _Capture:
    """The shape `cmd_snapshot` reads off a live run's runner."""

    def __init__(self) -> None:
        self.captured = {"scontrol show config": (0, "SelectType=x", "")}


def _run(cluster: Cluster) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cmd_snapshot(cluster, build_parser().parse_args(["snapshot"]), PLAIN)
    return rc, out.getvalue(), err.getvalue()


class TestTheGuardIsReachableAndClean:
    def test_a_cluster_with_no_capture_is_refused(self) -> None:
        rc, out, err = _run(_cluster())
        assert rc == 2
        assert out == ""
        assert MESSAGE in err, err

    def test_it_is_reached_through_the_command_table(self) -> None:
        """The route a replay takes. `main()`'s live branch is the other one."""
        assert _COMMANDS["snapshot"] is cmd_snapshot

    def test_nothing_is_written_to_stdout(self) -> None:
        """A caller doing `> snap.json` gets an empty file AND a non-zero exit,
        not a half-written payload."""
        _, out, _ = _run(_cluster())
        assert out.strip() == ""

    def test_the_guard_is_not_excluded_from_coverage(self) -> None:
        """The source pin. Re-adding the exclusion is how the duplicate guard
        this comment describes survived unnoticed."""
        import pathlib

        import nodetop.cli as cli_mod

        source = pathlib.Path(str(cli_mod.__file__)).read_text()
        line = next(ln for ln in source.splitlines() if MESSAGE in ln)
        index = source.splitlines().index(line)
        guard = source.splitlines()[index - 1]
        assert "if runner is None" in guard, guard
        assert "no cover" not in guard, guard


class TestControls:
    """Each passes with the guard deleted as well as with it.

    Deleting the guard is the correction the old comment invited -- if the
    branch were truly unreachable it would be dead code -- so that is the
    neuter, and these must survive it.
    """

    def test_a_live_run_still_writes_a_payload(self) -> None:
        """The path `main()` takes, which sets `capture` before dispatching."""
        cluster = _cluster()
        cluster.capture = _Capture()
        rc, out, _ = _run(cluster)
        assert rc == 0
        assert '"nodetop"' in out
        assert '"commands"' in out

    def test_the_payload_names_the_backend_and_the_queue_term(self) -> None:
        cluster = _cluster()
        cluster.capture = _Capture()
        _, out, _ = _run(cluster)
        assert '"backend": "slurm"' in out
        assert '"queue_term": "partition"' in out

    def test_the_command_table_still_holds_every_verb(self) -> None:
        for verb in ("status", "queues", "nodes", "health", "where", "snapshot"):
            assert verb in _COMMANDS, verb
