"""The "no data" summary must not claim more failed than actually failed.

`_reject_broken_snapshot` prints one `query failed: <label>: <why>` line per
failed query and then a single summary sentence. The summary chose between two
sentences on `if cluster.errors` -- **any** failure -- while asserting that
**all** of them failed, so a cluster that answered the node query in full and
lost only `scontrol show partition` printed, two lines apart:

    query failed: queues: CommandTimeoutError: ... timed out after 30s
    no data: every query failed, so there is nothing to report

naming the one query that failed and then contradicting it. The function's own
comment already forbade exactly this ("Saying `every query failed` ... would be
false"), for the sibling case it *did* handle -- queries that answered and could
not be parsed.

`Cluster.load` labels six independent reads and `errors` records only the ones
that failed, so "every" is not derivable from it at all. The fix keeps the
universal sentence for the one shape where the control-plane diagnosis is the
right one -- both primary reads gone, which is what every view here hangs off --
and names the queries in every other shape.

Exit codes are untouched in all of them; see `TestControls`.
"""

import io
import sys

from nodetop.cli import _named_failures, _reject_broken_snapshot
from nodetop.core.cluster import Cluster, Node, Queue

WHY = "CommandError: x exited 1"


def _run(**kw):
    """Return ``(rc, per_query_labels, summary_line)`` off a real stderr write."""
    node = Node(name="n1", state_raw="free", cpus_total=8)
    kw.setdefault("nodes", [node])
    kw.setdefault("queues", {})
    cluster = Cluster(backend_name="slurm", queue_term="partition", **kw)
    saved, buf = sys.stderr, io.StringIO()
    sys.stderr = buf
    try:
        rc = _reject_broken_snapshot(cluster, "status")
    finally:
        sys.stderr = saved
    labels = [
        line.split(":")[1].strip()
        for line in buf.getvalue().splitlines()
        if line.startswith("query failed:")
    ]
    summary = next(
        (line for line in buf.getvalue().splitlines() if line.startswith("no data:")),
        "",
    )
    return rc, labels, summary


def _queue(name="q"):
    return {name: Queue(name=name)}


class TestTheSummaryAgreesWithTheLinesAboveIt:
    def test_one_failed_query_is_named_not_generalised(self):
        rc, labels, summary = _run(errors={"queues": WHY})
        assert rc == 3
        assert labels == ["queues"]
        assert "the queues query failed" in summary, summary
        # The whole defect in one assertion: the node query answered.
        assert "every query failed" not in summary, summary

    def test_the_node_query_failing_alone_is_also_named(self):
        # The mirror shape: queues answered, nodes did not.
        rc, labels, summary = _run(nodes=[], queues=_queue(), errors={"nodes": WHY})
        assert rc == 3
        assert "the nodes query failed" in summary, summary
        assert "every query failed" not in summary, summary

    def test_two_failed_queries_read_as_a_plural(self):
        rc, _, summary = _run(errors={"queues": WHY, "limits": WHY})
        assert rc == 3
        assert "the limits and queues queries failed" in summary, summary

    def test_three_failed_queries_take_commas_and_a_final_and(self):
        rc, _, summary = _run(
            errors={"queues": WHY, "limits": WHY, "identity": WHY}
        )
        assert "the identity, limits and queues queries failed" in summary, summary

    def test_a_universal_claim_requires_both_primary_reads_to_be_gone(self):
        """The invariant, stated directly rather than per-shape.

        `errors` cannot prove that *every* labelled read failed, so the only
        honest use of the universal sentence is the shape it was written to
        diagnose -- a control plane or PATH problem, which takes out both of
        the reads the whole report hangs off.
        """
        for errors in (
            {"queues": WHY},
            {"nodes": WHY},
            {"limits": WHY, "identity": WHY},
            {"nodes": WHY, "limits": WHY},
            {"queues": WHY, "free_times": WHY},
        ):
            _, labels, summary = _run(nodes=[], errors=errors)
            if "every query failed" in summary:
                assert {"nodes", "queues"} <= set(labels), (errors, summary)

    def test_both_primary_reads_gone_keeps_the_control_plane_diagnosis(self):
        rc, _, summary = _run(
            nodes=[], errors={"nodes": WHY, "queues": WHY, "limits": WHY}
        )
        assert rc == 3
        assert "every query failed" in summary, summary

    def test_the_summary_names_only_queries_that_were_reported_failed(self):
        # Nothing invented: every label in the sentence has a line of its own.
        _, labels, summary = _run(errors={"queues": WHY, "limits": WHY})
        for label in labels:
            assert label in summary, (label, summary)
        assert "nodes" not in summary, summary


class TestTheNamingIsStable:
    def test_the_order_does_not_depend_on_insertion_order(self):
        # `errors` is filled by six threads, so its order is a race. Two dicts
        # holding the same failures must produce the same sentence.
        a = _named_failures({"queues": WHY, "identity": WHY, "limits": WHY})
        b = _named_failures({"limits": WHY, "queues": WHY, "identity": WHY})
        assert a == b == "the identity, limits and queues queries"

    def test_a_single_failure_uses_the_singular(self):
        assert _named_failures({"queues": WHY}) == "the queues query"


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_the_exit_codes_are_untouched(self):
        # 3 whenever a view would be a confidently-shaped nothing...
        assert _run(errors={"queues": WHY})[0] == 3
        assert _run(nodes=[], errors={"nodes": WHY})[0] == 3
        assert _run(nodes=[], errors={})[0] == 3
        # ...and 0 for a partial failure that left both primary reads intact,
        # which is the documented "the report is usable" case.
        assert _run(queues=_queue(), errors={"limits": WHY})[0] == 0

    def test_the_no_error_branch_is_unchanged(self):
        _, labels, summary = _run(nodes=[], errors={})
        assert labels == []
        assert "the queries answered, but nothing could be read from them" in summary
        assert "wrong backend for this cluster" in summary
        assert "nodetop backends" in summary

    def test_every_fatal_message_still_refuses_to_read_as_an_empty_cluster(self):
        # The tool's whole thesis, and it hangs off this clause.
        for kw in (
            {"errors": {"queues": WHY}},
            {"nodes": [], "errors": {"nodes": WHY, "queues": WHY}},
            {"nodes": [], "errors": {}},
        ):
            rc, _, summary = _run(**kw)
            assert rc == 3
            assert summary.startswith("no data: ")
            assert summary.endswith(
                ", so there is nothing to report -- this is not an empty cluster"
            )

    def test_each_failed_query_still_gets_its_own_line(self):
        _, labels, _ = _run(errors={"queues": WHY, "limits": WHY, "identity": WHY})
        assert sorted(labels) == ["identity", "limits", "queues"]
