"""CLI argument handling, rendering and JSON output."""

from __future__ import annotations

import json
import os

import pytest

from nodetop.cli import (
    STATUS_ROWS,
    build_parser,
    cmd_backends,
    cmd_check,
    cmd_exclude,
    cmd_health,
    cmd_nodes,
    cmd_queues,
    cmd_status,
    cmd_where,
    cmd_zoom,
)
from nodetop.core.model import JobShape
from nodetop.render import Style, table, width

PLAIN = Style(enabled=False)


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


class TestParser:
    @pytest.mark.parametrize("argv", [
        ["--json", "status"], ["status", "--json"],
        ["--json", "nodes", "--gpu"], ["nodes", "--gpu", "--json"],
        ["--json", "where", "-g", "1"], ["where", "-g", "1", "--json"],
    ])
    def test_json_works_on_either_side_of_the_verb(self, argv):
        # Requiring one order and rejecting the other is a papercut with no
        # upside; getting it wrong made --json silently a no-op.
        assert _args(argv).json is True

    def test_json_defaults_off(self):
        assert _args(["status"]).json is False

    def test_backend_override(self):
        assert _args(["--backend", "pbs", "status"]).backend == "pbs"
        assert _args(["status", "--backend", "pbs"]).backend == "pbs"

    def test_shape_flags(self):
        a = _args(["where", "-N", "2", "-g", "4", "-c", "8", "--mem", "64",
                   "-t", "2-00:00:00", "--gpu-mem", "40", "--needs", "bf16,fp8"])
        assert (a.nodes, a.gpus, a.cpus, a.mem) == (2, 4, 8, 64.0)
        assert a.needs == "bf16,fp8"

    @pytest.mark.parametrize("alias,canonical", [
        ("partitions", "queues"), ("fit", "where"), ("probe", "check"),
    ])
    def test_vocabulary_aliases(self, alias, canonical):
        # Slurm users type "partitions"; PBS users type "queues".
        assert _args([alias]).command == alias

    def test_partition_flag_is_an_alias_for_queue(self):
        assert _args(["queues", "-p", "gn"]).queue == "gn"
        assert _args(["queues", "--queue", "gn"]).queue == "gn"


class TestRenderIntegration:
    """Only the table's use by the CLI lives here; see test_render.py."""

    def test_empty_table(self):
        assert "nothing to show" in table(["A"], [], style=PLAIN)

    def test_a_row_longer_than_the_headers_is_tolerated(self):
        assert "x" in table(["A"], [["x", "extra"]], style=PLAIN)


class TestBackends:
    def test_lists_every_backend(self, capsys):
        assert cmd_backends(None, _args(["backends"]), PLAIN) == 0
        out = capsys.readouterr().out
        for name in ("slurm", "pbs", "lsf", "sge", "kubernetes", "sshpool"):
            assert name in out

    def test_json_separates_the_capability_from_local_availability(self, capsys):
        # The previous version of this test asserted
        # kubernetes.can_confirm_entitlement is True unconditionally, which
        # CERTIFIED the over-claim: the backend hardcoded probe=True and so
        # advertised confirmability on a host with no kubectl. The two facts are
        # now separate keys and the local one is gated on the client.
        from nodetop.runner import which

        cmd_backends(None, _args(["--json", "backends"]), PLAIN)
        data = json.loads(capsys.readouterr().out)["backends"]

        # A system with no dry-run at all: both false, always.
        assert data["pbs"]["dry_run_supported"] is False
        assert data["pbs"]["can_confirm_entitlement_here"] is False

        # A system that has one: supported regardless of this host...
        assert data["kubernetes"]["dry_run_supported"] is True
        assert data["sge"]["dry_run_supported"] is True
        assert data["slurm"]["dry_run_supported"] is True
        # ...and confirmable here only when the client is installed.
        assert data["kubernetes"]["can_confirm_entitlement_here"] is which("kubectl")
        assert data["sge"]["can_confirm_entitlement_here"] is which("qsub")
        assert data["slurm"]["can_confirm_entitlement_here"] is which("sbatch")

    def test_the_text_table_shows_the_capability_not_the_local_state(self, capsys):
        # On a Slurm login node the old table printed "none" for SGE, which
        # reads as "SGE has no dry-run" when the truth was only that qsub is
        # not installed here.
        cmd_backends(None, _args(["backends"]), PLAIN)
        out = capsys.readouterr().out
        assert "qsub -w v" in out
        assert "kubectl auth can-i" in out

    def test_a_system_with_no_dry_run_still_says_none(self, capsys):
        cmd_backends(None, _args(["backends"]), PLAIN)
        out = capsys.readouterr().out
        # pbs / lsf / sshpool genuinely have none; the word must survive.
        assert "none" in out

    def test_it_explains_what_a_missing_dry_run_means(self, capsys):
        cmd_backends(None, _args(["backends"]), PLAIN)
        assert "unconfirmed" in capsys.readouterr().out


class TestZoomAnswersTheIdleZeroQuestion:
    """`idle 0` was being read as "nothing in there for me".

    The column counts *wholly* idle nodes, and on a busy cluster almost nothing
    is wholly idle -- `amd` shows `idle 0` while carrying 2105 free cores spread
    over 24 nodes that are each running something. A job that does not need a
    whole node can start there immediately. The overview cannot show that
    without becoming a node listing, so the number that fits in the column is
    the one most easily misread; the fix is a view that opens the partition up,
    plus a line giving both counts at once.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        # Nothing wholly idle, but plenty spare -- the exact misread case.
        nodes = [
            Node(name="busy-1", state_raw="MIXED", cpus_total=32, cpus_alloc=8,
                 memory_mb=1000, queues=("q",)),
            Node(name="busy-2", state_raw="MIXED", cpus_total=32, cpus_alloc=30,
                 memory_mb=1000, queues=("q",)),
            Node(name="full-1", state_raw="ALLOCATED", cpus_total=32,
                 cpus_alloc=32, memory_mb=1000, queues=("q",)),
            Node(name="dead-1", state_raw="DOWN", cpus_total=32, cpus_alloc=0,
                 memory_mb=1000, queues=("q",), conditions=frozenset({"DOWN"})),
        ]
        queues = {"q": Queue(name="q", node_names=tuple(n.name for n in nodes),
                             declared_nodes=4, nodes=nodes)}
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _out(self, capsys, argv=("zoom", "q")):
        cmd_zoom(self._cluster(), _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    def test_it_says_how_many_nodes_have_room(self, capsys):
        out = " ".join(self._out(capsys).split())
        assert "0 wholly free" in out
        assert "2 of 4 with something spare" in out

    def test_it_says_it_in_numbers_not_prose(self, capsys):
        # A sentence explaining that partly-used nodes have room was cut: the
        # `idle` line already carries both counts, and a line of prose beside
        # the numbers it restates is a line nobody reads.
        out = " ".join(self._out(capsys).split())
        assert "can start now" not in out
        assert "0 wholly free" in out and "with something spare" in out

    def test_the_nodes_are_listed_roomiest_first(self, capsys):
        out = self._out(capsys)
        assert out.index("busy-1") < out.index("busy-2") < out.index("full-1")

    def test_a_drained_node_never_leads_the_list(self, capsys):
        # It reports its full complement free and none of it is reachable.
        # Ranking on that put a DOWN node with "32/32 cores" at the top of the
        # answer to "where is there room".
        out = self._out(capsys)
        assert out.index("dead-1") > out.index("full-1")

    def test_the_header_is_the_same_block_queues_prints(self, capsys):
        # Same renderer, so the two views cannot drift apart.
        zoomed = self._out(capsys)
        cmd_queues(self._cluster(), _args(["queues", "-q", "q"]), PLAIN)
        assert capsys.readouterr().out.splitlines()[0] in zoomed

    def test_json_reports_both_counts(self, capsys):
        payload = json.loads(self._out(capsys, ("zoom", "q", "--json")))
        assert payload["nodes"] == 4
        assert payload["with_room"] == 2
        assert payload["wholly_idle"] == 0
        assert payload["unschedulable"] == 1
        assert payload["cpus"] == [26, 128]   # the drained node contributes none
        assert len(payload["members"]) == 4

    def test_the_cap_keeps_the_roomiest(self, capsys):
        out = self._out(capsys, ("zoom", "q", "-n", "1"))
        assert "busy-1" in out and "full-1" not in out
        assert "more" in out

    def test_free_only_drops_the_full_and_the_dead(self, capsys):
        out = self._out(capsys, ("zoom", "q", "--free"))
        assert "busy-1" in out and "busy-2" in out
        assert "full-1" not in out and "dead-1" not in out


class TestNavigationGoesInAndOut:
    """Three levels in one place: partitions, then nodes, then jobs.

    The mapping is what matters. Rows are the finished lines `cmd_status`
    already built -- highlighted in place, not re-rendered -- so row N of the
    display must be entry N of the selection list; off by one opens the wrong
    partition, which is worse than having no navigation at all.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        # Free cores strictly descending, so ranked order is name order and a
        # row can be named without reaching into the command's own sort.
        for i, name in enumerate(("most", "mid", "least")):
            node = Node(name=f"n-{name}", state_raw="MIXED", cpus_total=1000,
                        cpus_alloc=i, memory_mb=1000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node])
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    @staticmethod
    def _funnel_entries(render, count: int) -> int:
        """How many selectable entries live on the funnel line.

        Derived, not hardcoded. Every one of them shares that single display
        row -- the partition total, the "open to you" bucket, one per exclusion
        reason -- so an entry belongs to it exactly when the cursor renders
        inside it. The count has changed twice; a constant here rotted twice.
        """
        return sum(
            1 for i in range(count)
            if any(PLAIN.g.cursor in line and "partition" in line
                   for line in render(i))
        )

    def _walk(self, monkeypatch, capsys, replies):
        """Drive the stack with a scripted list of `select` results.

        Integer replies are *partition* positions at the top level; the funnel
        entries in front of them are skipped, so a test can say "the second
        row" without tracking how many buckets the funnel happens to have.
        """
        import nodetop.interactive as inter

        frames: list[list[str]] = []
        answers = iter(replies)
        depth = {"n": 0}

        def scripted(render, count, **_kw):
            got = next(answers, inter.Key.QUIT)
            if isinstance(got, int) and depth["n"] == 0:
                got += self._funnel_entries(render, count)
            depth["n"] += 1
            index = got if isinstance(got, int) else 0
            frames.append(list(render(min(index, max(0, count - 1)))))
            return got

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", scripted)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        assert cmd_status(self._cluster(), _args(["status"]), PLAIN) == 0
        capsys.readouterr()
        return frames

    def test_it_offers_one_row_per_partition(self, monkeypatch, capsys):
        import nodetop.interactive as inter

        seen = {}
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)

        def capture(render, n, **_kw):
            seen["count"] = n
            seen["funnel"] = self._funnel_entries(render, n)
            return inter.Key.QUIT

        monkeypatch.setattr(inter, "select", capture)
        cmd_status(self._cluster(), _args(["status"]), PLAIN)
        capsys.readouterr()
        # One row per partition, and every funnel term is a target too -- the
        # total and "open to you" at least, plus one per exclusion reason.
        assert seen["count"] - seen["funnel"] == 3
        assert seen["funnel"] >= 2

    @pytest.mark.parametrize("pick,expected", [(0, "most"), (1, "mid"), (2, "least")])
    def test_choosing_a_row_opens_that_partition(self, monkeypatch, capsys,
                                                 pick, expected):
        frames = self._walk(monkeypatch, capsys, [pick])
        # The second frame is the node level, headed by the chosen partition.
        assert len(frames) >= 2
        assert expected in frames[1][1]

    def test_the_cursor_glyph_marks_the_selected_row(self, monkeypatch, capsys):
        # Inverse video is a no-op with colour off, so without the glyph there
        # would be nothing on screen saying which row is current.
        frames = self._walk(monkeypatch, capsys, [1])
        marked = [ln for ln in frames[0] if PLAIN.g.cursor in ln]
        assert len(marked) == 1
        assert "mid" in marked[0]

    def test_stepping_out_of_nodes_returns_to_partitions(self, monkeypatch,
                                                         capsys):
        import nodetop.interactive as inter

        frames = self._walk(monkeypatch, capsys,
                            [0, inter.Key.BACK, inter.Key.QUIT])
        # partitions, nodes, partitions again
        assert len(frames) == 3
        assert "partition" in frames[0][5]
        assert "node" in frames[2][5] or "partition" in frames[2][5]

    def test_a_node_opens_its_job_list(self, monkeypatch, capsys):
        frames = self._walk(monkeypatch, capsys, [0, 0])
        assert len(frames) >= 3
        joined = "\n".join(frames[2])
        # No backend on the fixture cluster, so there is no job list to read --
        # and the view must say that rather than showing an empty table.
        assert "n-most" in joined
        assert "cannot list jobs" in joined or "nothing running" in joined

    def test_quitting_from_any_depth_unwinds(self, monkeypatch, capsys):
        import nodetop.interactive as inter

        for replies in ([inter.Key.QUIT], [0, inter.Key.QUIT],
                        [0, 0, inter.Key.QUIT]):
            frames = self._walk(monkeypatch, capsys, replies)
            assert len(frames) == len(replies)

    def test_it_falls_back_without_a_terminal(self, capsys):
        # `supported()` is False under pytest, so the static report must come
        # out and nothing may block waiting for a keystroke.
        assert cmd_status(self._cluster(), _args(["status"]), PLAIN) == 0
        assert "partition" in capsys.readouterr().out

    def test_static_forces_the_printout_even_on_a_terminal(self, monkeypatch,
                                                           capsys):
        # For a terminal that is not a person: `watch nodetop` allocates a pty
        # and would otherwise block on a keystroke forever.
        import nodetop.interactive as inter

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", lambda *_a, **_k: pytest.fail(
            "--static must not enter the selection loop"))
        assert cmd_status(self._cluster(), _args(["status", "--static"]),
                          PLAIN) == 0
        assert "partition" in capsys.readouterr().out


class TestAJobHasItsOwnView:
    """Enter on a job row used to pop straight back to the node listing.

    "when choosing any of the job here, it doesn't go into the job details but
    going back to the original node" -- so the row was the deepest the tool
    went, with its name truncated and its node list unseen.
    """

    @staticmethod
    def _cluster(job):
        import dataclasses

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        node = Node(name="n1", state_raw="MIXED", cpus_total=48, cpus_alloc=40,
                    memory_mb=1000, queues=("q",))
        queue = Queue(name="q", node_names=("n1",), declared_nodes=1,
                      nodes=[node])
        cluster = Cluster(backend_name="synthetic", queue_term="partition",
                          nodes=[node], queues={"q": queue})
        return dataclasses.replace(cluster, _jobs=[job])

    def _frames(self, monkeypatch, capsys, job, replies):
        import nodetop.interactive as inter

        frames: list[list[str]] = []
        answers = iter(replies)

        def scripted(render, _count, **_kw):
            got = next(answers, inter.Key.QUIT)
            frames.append(list(render(got if isinstance(got, int) else 0)))
            return got

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", scripted)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        assert cmd_status(self._cluster(job), _args(["status"]), PLAIN) == 0
        capsys.readouterr()
        return frames

    #: total, "open to you", then the partition row.
    ROW = 2

    def test_entering_a_job_opens_it(self, monkeypatch, capsys):
        from nodetop.core.model import Job

        job = Job(id="42", user="alice", account="pi-x", queue="q", cpus=8,
                  nodes=("n1",), name="a-name-far-too-long-for-any-table-cell",
                  state="RUNNING", elapsed="1:00:00", remaining="2:00:00")
        frames = self._frames(monkeypatch, capsys, job, [self.ROW, 0, 0])
        # partitions, nodes, jobs, then the job itself.
        assert len(frames) == 4
        deepest = "\n".join(frames[3])
        assert "42" in deepest
        # The name in full, which the listing cannot afford.
        assert job.name in deepest
        assert "RUNNING" in deepest

    def test_it_names_the_node_and_the_share(self, monkeypatch, capsys):
        from nodetop.core.model import Job

        job = Job(id="42", user="alice", cpus=8, nodes=("n1",))
        deepest = "\n".join(
            self._frames(monkeypatch, capsys, job, [self.ROW, 0, 0])[3])
        assert "on n1" in deepest
        # No `job total` and no `nodes` row on a one-node job: the share above
        # IS the total, and the node is named in its own label, so both would
        # restate what the reader just read.
        assert "job total" not in deepest
        assert "\nnodes" not in deepest

    def test_a_spanning_job_gets_its_totals_and_its_nodelist(self, monkeypatch,
                                                             capsys):
        from nodetop.core.model import Job

        job = Job(id="42", user="alice", cpus=512, gpus=8,
                  nodes=("n1", "n2", "n3"))
        deepest = "\n".join(
            self._frames(monkeypatch, capsys, job, [self.ROW, 0, 0])[3])
        assert "job total" in deepest
        assert "512 cores" in deepest
        assert "3 nodes" in deepest

    def test_stepping_out_of_a_job_returns_to_the_job_list(self, monkeypatch,
                                                          capsys):
        import nodetop.interactive as inter
        from nodetop.core.model import Job

        job = Job(id="42", user="alice", cpus=8, nodes=("n1",))
        frames = self._frames(monkeypatch, capsys, job,
                              [self.ROW, 0, 0, inter.Key.BACK, inter.Key.QUIT])
        assert len(frames) >= 5
        # Back at the job table: its facts line names the node, and the job's
        # own view does not.
        assert any("cores free" in ln for ln in frames[4])
        assert any("n1" in ln for ln in frames[4])


class TestJobTotalsAreNotPerNodeShares:
    """A per-node table has to show the share, not the job's total.

    `squeue` reports a job's counts over every node it holds, so a 42-node job
    read as **512 cores on a 48-core machine** -- a number the reader knows to
    be impossible, which discredits the whole column. It used to be marked
    `512 x42`, which explained nothing: "the cpu column doesn't make any sense.
    what do the column entries mean?"

    The share is now fetched (`Cluster.share_of`), and a job whose share cannot
    be established says so rather than substituting a total.
    """

    @staticmethod
    def _cluster(jobs, allocations=None):
        import dataclasses

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        node = Node(name="n1", state_raw="MIXED", cpus_total=48, cpus_alloc=40,
                    memory_mb=1000, gpus_total=4, gpus_alloc=2, queues=("q",))
        queue = Queue(name="q", node_names=("n1",), declared_nodes=1,
                      nodes=[node])
        cluster = Cluster(backend_name="synthetic", queue_term="partition",
                          nodes=[node], queues={"q": queue})
        return dataclasses.replace(
            cluster, _jobs=list(jobs),
            _allocations={(a.job, a.node): a for a in (allocations or ())})

    def _job_frame(self, monkeypatch, capsys, jobs, allocations=None):
        import nodetop.interactive as inter

        seen = {}
        # The funnel's own buckets sit in front of the partition rows: the
        # total, then "open to you". The partition is the entry after them.
        replies = iter([2, 0])

        def scripted(render, _count, **_kw):
            got = next(replies, inter.Key.QUIT)
            seen[len(seen)] = list(render(0))
            return got

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", scripted)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        cmd_status(self._cluster(jobs, allocations), _args(["status"]), PLAIN)
        capsys.readouterr()
        return "\n".join(seen[max(seen)])

    def test_a_single_node_job_needs_no_lookup(self, monkeypatch, capsys):
        # Its totals ARE its share, which is most jobs and the whole answer on
        # a backend that models a job as living on one machine.
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=40, gpus=2, nodes=("n1",))])
        assert " 40 " in out
        assert "x1" not in out

    def test_a_multi_node_job_shows_its_share_of_this_node(self, monkeypatch,
                                                           capsys):
        from nodetop.core.model import Allocation, Job

        out = self._job_frame(
            monkeypatch, capsys,
            [Job(id="1", user="u", cpus=512, gpus=8,
                 nodes=tuple(f"n{i}" for i in range(41)) + ("n1",))],
            [Allocation(job="1", node="n1", cpus=7, memory_mb=7168, gpus=1)])
        assert " 7 " in out          # cores here
        assert " 1 " in out          # accelerators here
        assert "512" not in out      # not the total
        assert "x42" not in out      # and not the marker nobody could read
        assert " 42 " in out         # the span, in its own column

    def test_a_share_that_cannot_be_established_says_so(self, monkeypatch,
                                                        capsys):
        # No allocation and more than one node: a total in a column that means
        # a share is the defect this class exists for.
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=512, gpus=8, nodes=("n0", "n1"))])
        assert "512" not in out
        assert "?" in out

    def test_the_memory_column_is_the_share_in_gb(self, monkeypatch, capsys):
        from nodetop.core.model import Allocation, Job

        out = self._job_frame(
            monkeypatch, capsys,
            [Job(id="1", user="u", cpus=8, nodes=("n0", "n1"))],
            [Allocation(job="1", node="n1", cpus=8, memory_mb=57600)])
        assert " 56 " in out         # 57600 MB
        assert "57600" not in out

    def test_a_job_holding_no_accelerator_shows_a_zero(self, monkeypatch,
                                                       capsys):
        # The node has accelerators and this job holds none of them, which is a
        # count. `·` was doing double duty for "none installed" and "none held".
        from nodetop.core.model import Allocation, Job

        out = self._job_frame(
            monkeypatch, capsys,
            [Job(id="1", user="u", cpus=8, gpus=0, nodes=("n1",))],
            [Allocation(job="1", node="n1", cpus=8, memory_mb=1024, gpus=0)])
        assert " 0 " in out

    def test_a_single_node_job_shows_one_under_nodes(self, monkeypatch, capsys):
        # `·` was standing in for the number 1: "what does . mean in the node
        # column? why this?" A count column holds counts.
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=8, nodes=("n1",)),
            Job(id="2", user="u", cpus=8, nodes=tuple(f"m{i}" for i in range(9))
                + ("n1",)),
        ])
        rows = [ln for ln in out.splitlines() if " u " in ln]
        assert len(rows) == 2
        assert PLAIN.g.sep not in "".join(rows), rows
        assert " 1 " in rows[0] or rows[0].rstrip().endswith(" 1")
        assert " 10 " in rows[1] or "10" in rows[1]

    def test_the_span_column_is_absent_when_nothing_spans(self, monkeypatch,
                                                          capsys):
        # A column of `·` costs width the job name needs.
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=8, nodes=("n1",))])
        assert "nodes" not in out

    def test_a_job_with_no_accelerators_shows_a_dash(self, monkeypatch, capsys):
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=8, gpus=0, nodes=("n1",))])
        assert PLAIN.g.sep in out

    def test_a_job_name_cannot_break_the_table(self, monkeypatch, capsys):
        # Job names are user-authored free text and land in a table cell.
        from nodetop.core.model import Job

        out = self._job_frame(monkeypatch, capsys, [
            Job(id="1", user="u", cpus=8, nodes=("n1",),
                name="evil\x1b[2J\rhidden\nsecond")])
        for ch in ("\x1b", "\r", "\n" + "second"):
            assert ch not in out.replace("\n", "", 0) or ch == "\n" + "second"
        assert "\x1b" not in out and "\r" not in out


class TestTheInteractiveFrameFitsTheScreen:
    """A frame taller than the window makes the redraw destroy the screen.

    The repaint moves the cursor up by the height of the previous frame, and 84
    partitions is a 93-line frame: on a 24-row terminal the cursor clamps at the
    top, the clear-to-end lands in the wrong place, and every keypress leaves
    another copy of the listing behind. Measured through a pty before the fix:
    252 rows on screen for 84 partitions.
    """

    @staticmethod
    def _cluster(count):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        for i in range(count):
            # Free cores strictly descending, so the ranked display order is
            # also name order and "row i" below can be named without reaching
            # into the command's own sort.
            name = f"p{i:03d}"
            node = Node(name=f"n{i:03d}", state_raw="MIXED", cpus_total=1000,
                        cpus_alloc=i, memory_mb=1000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node])
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _frames(self, monkeypatch, capsys, count, rows):
        import shutil

        import nodetop.interactive as inter

        seen = {}
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        monkeypatch.setattr(
            shutil, "get_terminal_size",
            lambda *_a, **_k: os.terminal_size((100, rows)))

        def capture(render, n, **_kw):
            seen["frames"] = [list(render(i)) for i in range(n)]
            return None

        monkeypatch.setattr(inter, "select", capture)
        cmd_status(self._cluster(count), _args(["status", "--all"]), PLAIN)
        capsys.readouterr()
        return seen["frames"]

    def test_every_frame_fits_the_terminal(self, monkeypatch, capsys):
        for frame in self._frames(monkeypatch, capsys, 60, rows=24):
            assert len(frame) <= 24

    def test_a_short_list_is_not_windowed(self, monkeypatch, capsys):
        # Windowing a list that already fits would hide rows for no reason.
        frames = self._frames(monkeypatch, capsys, 4, rows=40)
        assert all("above" not in "".join(f) and "below" not in "".join(f)
                   for f in frames)

    def test_it_says_which_slice_is_on_screen(self, monkeypatch, capsys):
        # Silently dropping rows would read as "this is all of them", which is
        # the same lie the funnel line exists to prevent. A count of what is
        # missing is not enough either -- "14 above" answers a question nobody
        # asked and raises one it does not answer. A position says where you are
        # and implies that moving will reach the rest.
        joined = "".join(self._frames(monkeypatch, capsys, 60, rows=24)[0])
        assert "of 60" in joined
        assert "above" not in joined and "below" not in joined

    def test_the_selection_stays_on_screen(self, monkeypatch, capsys):
        # The point of scrolling: the highlighted row must be in the window,
        # otherwise the cursor vanishes and the arrows appear to do nothing.
        #
        # The leading frames belong to the funnel's own buckets, which are not
        # partition rows; the partition frames follow them. Counted rather than
        # assumed -- the funnel has gained a term twice.
        every = self._frames(monkeypatch, capsys, 60, rows=24)
        funnel = sum(1 for f in every
                     if any(PLAIN.g.cursor in ln and "partition" in ln
                            for ln in f))
        for i, frame in enumerate(every[funnel:]):
            body = "\n".join(frame)
            assert f"p{i:03d} " in body, i

    def test_every_frame_is_the_same_size(self, monkeypatch, capsys):
        """The window does not move as the reader changes level.

        It used to size itself to its content, so stepping from an
        eighty-column overview into `3 down` shrank it to a third of the width
        and four rows -- "whatever we choose in the ui, the window should stay
        the same and the text and information getting displayed should
        dynamically get adjusted".
        """
        frames = self._frames(monkeypatch, capsys, 60, rows=24)
        sizes = {(len(f), width(f[0])) for f in frames}
        assert len(sizes) == 1, sizes

    def test_a_short_view_is_padded_not_shrunk(self, monkeypatch, capsys):
        # Four partitions do not fill a 24-row frame, and the border must not
        # close early to meet them.
        frames = self._frames(monkeypatch, capsys, 4, rows=24)
        for frame in frames:
            assert len(frame) == 23
            assert frame[-2].strip("│ ") == ""    # padding, then the border

    def test_headings_are_never_dropped_to_fit(self, monkeypatch, capsys):
        # They are the frame of reference for the row you are looking at.
        for frame in self._frames(monkeypatch, capsys, 60, rows=24):
            joined = "\n".join(frame)
            assert "partitions" in joined     # the funnel line
            assert "partition " in joined     # the column header


class TestZoomEdgeCases:
    """The new view's awkward inputs, which is where new code fails.

    None of these crashed; all of them said something wrong or useless, which is
    the harder class to notice.
    """

    @staticmethod
    def _cluster(nodes, **queue_kw):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Queue

        queue = Queue(name="q", node_names=tuple(n.name for n in nodes),
                      declared_nodes=len(nodes), nodes=nodes, **queue_kw)
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues={"q": queue})

    @staticmethod
    def _node(name, **kw):
        from nodetop.core.model import Node

        return Node(name=name, memory_mb=100, queues=("q",), **kw)

    def _out(self, cluster, capsys, argv=("zoom", "q")):
        cmd_zoom(cluster, _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    def test_one_node_is_not_one_nodes(self, capsys):
        # `plural` exists for exactly this and the new code walked past it.
        cluster = self._cluster([self._node("n1", state_raw="IDLE", cpus_total=8)])
        out = " ".join(self._out(cluster, capsys).split())
        assert "1 node " in out and "1 nodes" not in out

    def test_a_drained_nodes_meter_reads_empty(self, capsys):
        # A drained node reports its full complement free. Dimming that bar is
        # not enough -- dimming is a colour, and with colour off a full-length
        # bar beside "8/8" still reads as the roomiest row on the screen. The
        # meter measures room and there is none, which is what
        # `Queue.effective_free_cpus` has always said.
        cluster = self._cluster([self._node(
            "d1", state_raw="DOWN", cpus_total=8,
            conditions=frozenset({"DOWN"}))])
        row = [x for x in self._out(cluster, capsys).splitlines() if "d1" in x][0]
        assert PLAIN.g.blocks[-1] not in row      # no filled cell
        assert "8/8" in row                       # but the claim is still shown

    def test_a_node_with_no_memory_left_reads_empty_too(self, capsys):
        # 44 of 48 cores idle and every byte of memory allocated: the same
        # phantom capacity as a drained node, and the biggest number on the
        # screen. Real -- `wide` had 47 of these behind a headline of 2322
        # free cores.
        # `_node` gives the node 100 MB; this allocates all of it.
        cluster = self._cluster([self._node(
            "starved", state_raw="MIXED", cpus_total=48, cpus_alloc=4,
            memory_alloc_mb=100)])
        row = [x for x in self._out(cluster, capsys).splitlines()
               if "starved" in x][0]
        assert PLAIN.g.blocks[-1] not in row      # no filled cell
        assert "44/48" in row                     # the claim is still shown
        assert "0/0G" in row                      # and so is the reason

    def test_a_node_with_memory_to_spare_still_draws_its_meter(self, capsys):
        cluster = self._cluster([self._node(
            "roomy", state_raw="MIXED", cpus_total=48, cpus_alloc=4)])
        row = [x for x in self._out(cluster, capsys).splitlines()
               if "roomy" in x][0]
        assert PLAIN.g.blocks[-1] in row

    def test_a_node_with_no_accelerator_says_so_with_a_dash(self, capsys):
        # `·` is this tool's empty cell, and in a column headed "gpu free" it
        # answers a question the node is not being asked: "the . in the gpu
        # column is a very confusing thing ... putting a dot there means
        # nothing". A dash reads as not-applicable with colour off and in ASCII.
        cluster = self._cluster([
            self._node("cpu1", state_raw="IDLE", cpus_total=8),
            self._node("gpu1", state_raw="IDLE", cpus_total=8, gpus_total=4),
        ])
        rows = {ln.split()[1] if len(ln.split()) > 1 else "": ln
                for ln in self._out(cluster, capsys).splitlines()}
        line = next(ln for ln in rows.values() if "cpu1" in ln)
        assert PLAIN.g.dash in line
        assert next(ln for ln in rows.values() if "gpu1" in ln).count("4/4") == 1

    def test_the_accelerator_column_is_absent_when_nothing_has_one(self, capsys):
        # A column of dashes costs width the node name and reason need.
        cluster = self._cluster([self._node("cpu1", state_raw="IDLE",
                                            cpus_total=8)])
        out = self._out(cluster, capsys)
        assert "gpu" not in out
        assert PLAIN.g.dash not in out

    def test_each_free_column_says_which_resource_it_counts(self, capsys):
        # It was `cpu | free | mem free | gpu`, with one bare `free` doing the
        # work of three labels: "why so many frees? it looks weird and the
        # words are very unclear".
        cluster = self._cluster([self._node("gpu1", state_raw="IDLE",
                                            cpus_total=8, gpus_total=4)])
        out = self._out(cluster, capsys)
        for head in ("cpu free", "mem free", "gpu free"):
            assert head in out, head
        # And the meter follows the number it draws, as in the overview.
        row = next(ln for ln in out.splitlines() if "gpu1" in ln)
        assert row.index("8/8") < row.index(PLAIN.g.blocks[-1])

    def test_a_down_node_shows_why_it_is_down(self, capsys):
        """The reason, in full, and the state beside it.

        Opening a drained node used to give a four-line box saying "nothing
        running here" -- no state, no reason -- and the reason is exactly what
        the reader opened it for: "after hitting this one, nothing shows up,
        even people wanting to see the reason why this node is down". It is
        truncated in the listing, so this is the view that prints it whole.
        """
        import nodetop.interactive as inter

        reason = ("maintenance: replacing a failed accelerator and the fabric "
                  "card that came with it [root@2026-08-19T09:14:02]")
        # A healthy node beside it, so the partition itself stays usable and
        # the funnel keeps to its two terms -- otherwise entry 2 is the "down"
        # label rather than the partition row.
        cluster = self._cluster([
            self._node("ok1", state_raw="IDLE", cpus_total=8),
            self._node("d1", state_raw="DOWN*+DRAIN", cpus_total=8,
                       conditions=frozenset({"DOWN"}), unreachable=True,
                       reason=reason),
        ])
        frames: list[list[str]] = []
        # partition row, then the drained node -- unschedulable ones sort last.
        answers = iter([2, 1, 0])

        def scripted(render, _count, **_kw):
            got = next(answers, inter.Key.QUIT)
            frames.append(list(render(got if isinstance(got, int) else 0)))
            return got

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
            monkeypatch.setattr(inter, "select", scripted)
            monkeypatch.setattr(inter, "read_key",
                                lambda *_a, **_k: inter.Key.QUIT)
            cmd_status(cluster, _args(["status"]), PLAIN)
        finally:
            monkeypatch.undo()
        capsys.readouterr()
        # The node's own view, found by its facts line rather than by index:
        # stepping back out of it renders the listing again afterwards.
        opened = next("\n".join(f) for f in frames
                      if any(ln.lstrip("│ ").startswith("d1  ") for ln in f))
        assert "DOWN*+DRAIN" in opened
        # Whole, not truncated, and the stamp separated from the cause.
        assert "replacing a failed accelerator" in opened
        assert "fabric" in opened
        assert PLAIN.g.ellipsis not in opened
        assert "root" in opened
        assert "lost contact" in opened
        # And not the note that reads as "free".
        assert "nothing can start" in opened

    def test_a_node_with_no_room_is_not_marked_free(self, capsys):
        # Nothing running, no allocatable memory: the green mark is the one
        # thing on a row a reader trusts without reading the numbers.
        cluster = self._cluster([self._node(
            "hollow", state_raw="IDLE", cpus_total=48, memory_alloc_mb=100)])
        row = [x for x in self._out(cluster, capsys).splitlines()
               if "hollow" in x][0]
        assert PLAIN.g.ok not in row
        assert PLAIN.g.partial in row

    def test_a_routing_queue_gets_no_empty_node_listing(self, capsys):
        # The detail block above it already says "owns no nodes of its own".
        cluster = self._cluster([], forwards_to=("other",))
        out = self._out(cluster, capsys)
        assert "routing queue" in out
        assert "nothing to show" not in out
        assert "inside" not in out

    def test_a_filter_that_matches_nothing_says_which_filter(self, capsys):
        # "(nothing to show)" alone reads as "this queue is empty", which is a
        # different claim from "your filter excluded all of it".
        cluster = self._cluster([self._node("i1", state_raw="IDLE", cpus_total=8)])
        out = " ".join(self._out(cluster, capsys, ("zoom", "q", "--gpu")).split())
        assert "nothing matches --gpu" in out

    @pytest.mark.parametrize("argv", [
        ("zoom", "q"), ("zoom", "q", "--all"), ("zoom", "q", "--free"),
        ("zoom", "q", "--gpu"), ("zoom", "q", "--cpu"),
        ("zoom", "q", "-n", "0"), ("zoom", "q", "-n", "-5"),
        ("zoom", "q", "--json"),
    ])
    def test_no_combination_crashes_on_an_empty_queue(self, capsys, argv):
        assert cmd_zoom(self._cluster([]), _args(list(argv)), PLAIN) == 0


class TestQueuesLeadsWithWhatWorks:
    """`(q.usable, q.name)` sorted the broken ones to the top.

    False sorts before True, so this listing opened with the partitions that
    can start nothing -- the exact complaint that got the overview reordered and
    then got its DEAD block removed outright. A command whose rows are places
    you might submit to must not lead with the places you cannot.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        # `aaa-dead` sorts first by name and last by usefulness.
        for name, ok, alloc in (("aaa-dead", False, 0), ("mmm-small", True, 6),
                                ("zzz-big", True, 0)):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        cpus_alloc=alloc, memory_mb=1000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(
                name=name, node_names=(node.name,), declared_nodes=1,
                nodes=[node], enabled=ok,
                state_raw="UP" if ok else "DOWN")
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _out(self, capsys):
        cmd_queues(self._cluster(), _args(["queues", "--all"]), PLAIN)
        return capsys.readouterr().out

    def test_the_blocked_one_comes_last(self, capsys):
        out = self._out(capsys)
        assert out.index("zzz-big") < out.index("aaa-dead")
        assert out.index("mmm-small") < out.index("aaa-dead")

    def test_the_roomiest_usable_one_comes_first(self, capsys):
        out = self._out(capsys)
        assert out.index("zzz-big") < out.index("mmm-small")


class TestNodesLeadsWithRoom:
    """The listing is capped, so the order decides what the cap keeps.

    It was node-name order, which makes a capped window arbitrary: `--free`
    found 127 nodes with something free and showed the alphabetically first 20,
    the top row having 0 of 32 cores free while 107 roomier nodes sat behind
    `--all`. A cap is only defensible if what it keeps is what was wanted.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.hardware import AcceleratorSpec
        from nodetop.core.model import Node, Queue

        spec = AcceleratorSpec(model="A100", vendor="NVIDIA", arch="sm_80",
                               memory_gb=40)
        nodes = [
            # name order and room order are deliberately opposed
            Node(name="aaa-busy", state_raw="ALLOCATED", cpus_total=32,
                 cpus_alloc=32, memory_mb=1000, queues=("q",)),
            Node(name="mmm-roomy", state_raw="MIXED", cpus_total=32,
                 cpus_alloc=4, memory_mb=1000, queues=("q",)),
            Node(name="zzz-gpu", state_raw="MIXED", cpus_total=32,
                 cpus_alloc=30, memory_mb=1000, gpus_total=4, gpus_alloc=1,
                 accelerator=spec, queues=("q",)),
        ]
        queues = {"q": Queue(name="q", node_names=tuple(n.name for n in nodes),
                             declared_nodes=3, nodes=nodes)}
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _rows(self, capsys, argv=("nodes",)):
        cmd_nodes(self._cluster(), _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    def test_the_roomiest_node_is_listed_before_the_busiest(self, capsys):
        out = self._rows(capsys)
        assert out.index("mmm-roomy") < out.index("aaa-busy")

    def test_free_accelerators_outrank_free_cores(self, capsys):
        # A node with 3 free GPUs and 2 free cores beats one with 28 free cores
        # and no accelerator: those rows are read for the accelerator.
        out = self._rows(capsys)
        assert out.index("zzz-gpu") < out.index("mmm-roomy")

    def test_the_cap_keeps_the_roomiest(self, capsys):
        out = self._rows(capsys, ("nodes", "-n", "1"))
        assert "zzz-gpu" in out
        assert "aaa-busy" not in out

    def test_the_order_is_stable(self, capsys):
        first = self._rows(capsys)
        second = self._rows(capsys)
        assert first == second


class TestStatus:
    def test_text(self, cluster, capsys):
        assert cmd_status(cluster, _args(["status"]), PLAIN) == 0
        out = capsys.readouterr().out
        assert "partition" in out                # Slurm vocabulary, not "queue"
        assert "nodetop" in out

    def test_it_does_not_report_what_is_broken(self, cluster, capsys):
        # The overview answers "where can I run this" and nothing else. A DEAD
        # block naming the partitions that can start nothing, with their
        # phantom idle-node total, used to sit at the bottom of it -- and on a
        # healthy day it was the only red on the screen, about partitions the
        # reader was never going to use. It is off this view entirely now.
        cmd_status(cluster, _args(["status"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "DEAD" not in out
        assert "advertised" not in out

    def test_the_count_of_what_is_broken_is_still_there(self, cluster, capsys):
        # Dropping the block must not break the funnel's arithmetic: a reader
        # asking why 87 partitions became 5 still needs every term of it.
        cmd_status(cluster, _args(["status"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "down" in out

    def test_the_detail_is_one_command_away(self, cluster, capsys):
        # The condition for taking it off the overview: `queues` still names
        # the partition AND why it is dead, which is where someone looking for
        # a fault goes.
        from nodetop.cli import cmd_queues

        cmd_queues(cluster, _args(["queues", "--all"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "test" in out
        assert "QUEUE_DISABLED" in out

    def test_it_states_whether_entitlement_can_be_confirmed(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), PLAIN)
        # Compressed into the header: "dry-run ok" / "no dry-run".
        assert "no dry-run" not in capsys.readouterr().out

    def test_json(self, cluster, capsys):
        cmd_status(cluster, _args(["--json", "status"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert "test" in data["unusable_queues"]
        assert data["backend"] == "slurm"
        assert data["can_confirm_entitlement"] is True
        assert data["phantom_capacity"]["test"] >= 1


class TestQueues:
    def test_json(self, cluster, capsys):
        cmd_queues(cluster, _args(["--json", "queues", "--all"]), PLAIN)
        data = {q["name"]: q for q in json.loads(capsys.readouterr().out)}
        assert data["test"]["usable"] is False
        assert data["test"]["effective_free_nodes"] == 0
        assert {b["code"] for b in data["test"]["blockers"]} >= {
            "QUEUE_DISABLED", "NO_ACCOUNTS", "NO_QOS"
        }

    def test_the_effective_wall_limit_is_reported(self, cluster, capsys):
        cmd_queues(cluster, _args(["--json", "queues", "--all"]), PLAIN)
        data = {q["name"]: q for q in json.loads(capsys.readouterr().out)}
        # The partition says unlimited; the QOS is what actually bites.
        assert data["gn"]["max_walltime_queue"] == "unlimited"
        assert data["gn"]["max_walltime_effective"] == "2-00:00:00"

    def test_unresolved_nodes_are_reported(self, cluster, capsys):
        cmd_queues(cluster, _args(["--json", "queues", "-q", "test"]), PLAIN)
        data = json.loads(capsys.readouterr().out)[0]
        assert data["nodes_declared"] == 610
        assert data["nodes"] < 610

    def test_filter(self, cluster, capsys):
        cmd_queues(cluster, _args(["--json", "queues", "-p", "gn"]), PLAIN)
        assert [q["name"] for q in json.loads(capsys.readouterr().out)] == ["gn"]


class TestNodes:
    def test_inferred_memory_is_labelled(self, cluster, capsys):
        cmd_nodes(cluster, _args(["--json", "nodes", "--gpu", "--all"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        a100 = [r for r in rows if r["accelerator_model"] == "A100"]
        assert a100 and a100[0]["accelerator_memory_inferred"] is True

    def test_vendor_and_arch_are_exposed(self, cluster, capsys):
        cmd_nodes(cluster, _args(["--json", "nodes", "--gpu", "--all"]), PLAIN)
        rows = [r for r in json.loads(capsys.readouterr().out) if r["accelerator_model"]]
        assert rows[0]["accelerator_vendor"] == "NVIDIA"
        assert rows[0]["accelerator_arch"].startswith("sm_")

    def test_gpu_and_cpu_filters_are_disjoint(self, cluster, capsys):
        cmd_nodes(cluster, _args(["--json", "nodes", "--gpu", "--all"]), PLAIN)
        gpu = {r["name"] for r in json.loads(capsys.readouterr().out)}
        cmd_nodes(cluster, _args(["--json", "nodes", "--cpu", "--all"]), PLAIN)
        cpu = {r["name"] for r in json.loads(capsys.readouterr().out)}
        assert gpu and cpu and not (gpu & cpu)


class TestHealth:
    def test_json(self, cluster, capsys):
        cmd_health(cluster, _args(["--json", "health"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert any(n["name"] == "cn-0385" for n in data["unschedulable"])
        assert "cn" in data["unschedulable_nodelist"]

    def test_it_admits_what_the_keyword_scan_cannot_find(self, cluster, capsys):
        cmd_health(cluster, _args(["health"]), PLAIN)
        # The caveat survives, in three words rather than a clause: this scan
        # can only find impairment someone wrote into the reason field.
        out = " ".join(capsys.readouterr().out.split())
        assert "reason-field" in out


class TestWhere:
    def test_json_shape(self, cluster, capsys):
        cmd_where(cluster, _args(["--json", "where", "-g", "1", "--all"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        assert rows
        for row in rows:
            assert {"queue", "runnable_now", "reachable", "blockers",
                    "entitlement_unconfirmed", "submit_flags"} <= set(row)

    def test_the_dead_partition_is_excluded_by_default(self, cluster, capsys):
        cmd_where(cluster, _args(["--json", "where", "-g", "1"]), PLAIN)
        names = {r["queue"] for r in json.loads(capsys.readouterr().out)}
        assert "test" not in names

    def test_no_start_estimate_for_an_unreachable_queue(self, cluster, capsys):
        cmd_where(cluster, _args(["--json", "where", "-g", "1", "--all"]), PLAIN)
        rows = {r["queue"]: r for r in json.loads(capsys.readouterr().out)}
        assert rows["test"]["reachable"] is False
        assert rows["test"]["earliest_start"] is None

    def test_submit_flags_match_what_was_checked(self, cluster, capsys):
        cmd_where(cluster, _args(["--json", "where", "-g", "4", "-t", "2h"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        flags = " ".join(rows[0]["submit_flags"])
        assert "--gres=gpu:4" in flags and "--time=2h" in flags

    def test_exit_zero_when_somewhere_could_host_it(self, cluster):
        assert cmd_where(cluster, _args(["where", "-g", "1"]), PLAIN) == 0

    def test_exit_nonzero_when_no_hardware_could_ever_host_it(self, cluster):
        # Exit status answers "is there anywhere that could ever run this?",
        # so reachable-but-busy is success and wrong-hardware-everywhere is not.
        args = _args(["where", "-g", "8", "--gpu-mem", "999", "--needs", "fp8"])
        assert cmd_where(cluster, args, PLAIN) == 1


class TestCheck:
    def test_a_backend_without_a_dry_run_says_so_and_exits_2(self, capsys):
        from nodetop.backends.pbs import PbsBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        backend = PbsBackend(RecordedRunner({"pbsnodes": (0, '{"nodes":{}}', ""),
                                             "qstat": (0, "", "")}))
        cl = Cluster.load(backend, with_free_times=False)
        assert cmd_check(cl, _args(["check", "-g", "1"]), PLAIN) == 2
        assert "no verify-only submission" in capsys.readouterr().out

    def test_json_reports_the_absence(self, capsys):
        from nodetop.backends.lsf import LsfBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        backend = LsfBackend(RecordedRunner({"bhosts": (0, "", ""),
                                             "lshosts": (0, "", ""),
                                             "bqueues": (0, "", "")}))
        cl = Cluster.load(backend, with_free_times=False)
        cmd_check(cl, _args(["--json", "check"]), PLAIN)
        assert json.loads(capsys.readouterr().out)["can_probe"] is False


class TestExclude:
    def test_requires_a_selector(self, cluster):
        assert cmd_exclude(cluster, _args(["exclude"]), PLAIN) == 2

    def test_gpu_nodes_emit_a_collapsed_nodelist(self, cluster, capsys):
        cmd_exclude(cluster, _args(["exclude", "--gpu-nodes"]), PLAIN)
        out = capsys.readouterr().out.strip()
        # Decided by resource count, so a bigmem node must not appear.
        assert "bigmem" not in out
        assert out

    def test_json(self, cluster, capsys):
        cmd_exclude(cluster, _args(["--json", "exclude", "--unschedulable"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert data["count"] >= 1
        assert "cn-0385" in data["nodes"]


class TestShapeFlags:
    def test_exclusions_are_expanded_then_collapsed(self):
        from nodetop.cli import _shape_from_args

        shape = _shape_from_args(_args(["where", "--exclude", "n-[1-3]"]))
        assert shape.exclude == ("n-1", "n-2", "n-3")

    def test_tolerations(self):
        from nodetop.cli import _shape_from_args

        shape = _shape_from_args(_args(["where", "--tolerates", "a=b:NoSchedule"]))
        assert shape.tolerates == ("a=b:NoSchedule",)

    def test_walltime_accepts_friendly_forms(self):
        from nodetop.cli import _shape_from_args

        assert _shape_from_args(_args(["where", "-t", "90m"])).walltime_seconds == 5400
        assert JobShape(walltime="36h").walltime_seconds == 129600


class TestAccelerators:
    """The one question no scheduler can answer."""

    def test_json_inventory(self, cluster, capsys):
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert data["accelerators_installed"] > 0
        assert "A100" in data["models"]
        assert data["models"]["A100"]["vendor"] == "NVIDIA"
        assert data["models"]["A100"]["arch"] == "sm_80"

    def test_inferred_memory_is_flagged_per_model(self, cluster, capsys):
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert data["models"]["A100"]["memory_inferred"] is True

    def test_fp8_reach_is_a_strict_subset_of_bf16_reach(self, cluster, capsys):
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        reach = json.loads(capsys.readouterr().out)["capability_reach"]
        # Every fp8 part (sm_89+) also does bf16 (sm_80+), and the fixture has
        # both A100s and an H100, so the subset must be strict.
        assert 0 < reach["fp8"]["installed"] < reach["bf16"]["installed"]

    def test_hardware_behind_a_drained_node_is_not_reach(self, cluster, capsys):
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        reach = json.loads(capsys.readouterr().out)["capability_reach"]
        # The only fp8-capable node in the fixture is DOWN+DRAIN, so the
        # capability is installed but has zero reach right now. Counting it as
        # free would promise hardware the scheduler will not hand out.
        assert reach["fp8"]["installed"] > 0
        assert reach["fp8"]["free"] == 0

    def test_reach_never_exceeds_what_is_installed(self, cluster, capsys):
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        installed = data["accelerators_installed"]
        for cap, counts in data["capability_reach"].items():
            assert counts["installed"] <= installed, cap
            assert counts["free"] <= counts["installed"], cap

    def test_free_counts_ignore_unschedulable_nodes(self, cluster, capsys):
        # Hardware behind a drained node is not reach.
        from nodetop.cli import cmd_accelerators

        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        for model in data["models"].values():
            assert model["free"] <= model["installed"]

    def test_alias(self):
        assert _args(["accel"]).command == "accel"


class TestQueuesPresentation:
    """A block per queue is right for one and unreadable for eighty-seven."""

    def test_the_table_is_the_default(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues"]), PLAIN)
        out = capsys.readouterr().out
        assert "blocked by" in out          # the table header
        # "maxtime" is no longer a discriminator: the table's own header row
        # carries it now that headers are lowercase. The per-queue block is
        # identified by its "name [STATE]" heading, which the table has no
        # equivalent of.
        assert "[UP]" not in out

    def test_naming_a_queue_implies_detail(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues", "-q", "test"]), PLAIN)
        out = capsys.readouterr().out
        assert "maxtime" in out
        assert "BLOCKED BY" not in out

    def test_detail_can_be_asked_for_explicitly(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues", "--detail"]), PLAIN)
        assert "maxtime" in capsys.readouterr().out

    def test_the_table_stays_compact_on_a_large_cluster(self, cluster, capsys):
        # The block form printed 619 lines for 87 partitions on the reference
        # cluster; the table has to stay proportional to the queue count.
        cmd_queues(cluster, _args(["queues"]), PLAIN)
        lines = capsys.readouterr().out.strip().splitlines()
        # header + rule + one row per queue + section + footnote, and nothing more.
        assert len(lines) <= len(cluster.queues) + 7

    def test_the_table_points_at_the_deeper_view(self, cluster, capsys):
        cmd_queues(cluster, _args(["queues"]), PLAIN)
        assert "zoom" in " ".join(capsys.readouterr().out.split())

    def test_json_is_unchanged_by_the_presentation_switch(self, cluster, capsys):
        cmd_queues(cluster, _args(["--json", "queues", "--all"]), PLAIN)
        plain = json.loads(capsys.readouterr().out)
        # Same flags on both sides: the claim is that --detail does not change
        # the JSON, so an entitlement filter applied to one and not the other
        # is comparing two different questions.
        cmd_queues(cluster, _args(["--json", "queues", "--detail", "--all"]), PLAIN)
        detailed = json.loads(capsys.readouterr().out)
        assert plain == detailed


class TestNodesPresentation:
    def test_the_header_states_how_much_is_shown(self, cluster, capsys):
        from nodetop.cli import cmd_nodes

        cmd_nodes(cluster, _args(["nodes", "--gpu"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "with GPUs" in out
        assert " of " in out            # "N of TOTAL" when filtered

    def test_an_unfiltered_listing_says_all(self, cluster, capsys):
        from nodetop.cli import cmd_nodes

        cmd_nodes(cluster, _args(["nodes", "--all"]), PLAIN)
        assert "all " in " ".join(capsys.readouterr().out.split())

    def test_a_small_listing_gets_no_filter_hint(self, cluster, capsys):
        # The fixture cluster is nine nodes; the hint would be noise.
        from nodetop.cli import cmd_nodes

        cmd_nodes(cluster, _args(["nodes"]), PLAIN)
        assert "narrow this listing" not in capsys.readouterr().out

    def test_json_is_unaffected_by_the_header(self, cluster, capsys):
        from nodetop.cli import cmd_nodes

        cmd_nodes(cluster, _args(["--json", "nodes"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        assert isinstance(rows, list) and rows


class TestRoutingQueuePresentation:
    def _routing_cluster(self):
        import json as _json

        from nodetop.backends.pbs import PbsBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        nodes = _json.dumps({"nodes": {"n1": {
            "state": "free",
            "resources_available": {"ncpus": 8, "Qlist": "execq"},
            "resources_assigned": {}}}})
        qstat = ("Queue: execq\n    queue_type = Execution\n    enabled = True\n"
                 "    started = True\n\nQueue: routeq\n    queue_type = Route\n"
                 "    enabled = True\n    started = True\n"
                 "    route_destinations = execq\n")
        return Cluster.load(
            PbsBackend(RecordedRunner({
                "pbsnodes -a -F json": (0, nodes, ""),
                "qstat -Qf": (0, qstat, ""),
                "qstat -f": (0, "", ""),
            })),
            with_free_times=False,
        )

    def test_the_table_shows_the_destinations(self, capsys):
        cmd_queues(self._routing_cluster(), _args(["queues"]), PLAIN)
        out = capsys.readouterr().out
        assert "execq" in out
        assert "->" in out or "→" in out

    def test_the_detail_view_says_it_owns_no_nodes(self, capsys):
        cmd_queues(self._routing_cluster(), _args(["queues", "-q", "routeq"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "routing queue" in out
        assert "owns no nodes" in out
        assert "forwards to execq" in out

    def test_json_exposes_the_routing(self, capsys):
        cmd_queues(self._routing_cluster(), _args(["--json", "queues"]), PLAIN)
        data = {q["name"]: q for q in json.loads(capsys.readouterr().out)}
        assert data["routeq"]["routes"] is True
        assert data["routeq"]["forwards_to"] == ["execq"]
        assert data["execq"]["routes"] is False


class TestStatusIdentity:
    #: A username the test controls.
    #:
    #: The assertion used to be the literal name of whoever wrote it, which
    #: passes on their machine and nowhere else -- CI runs as `runner` and went
    #: red on the first push. The name comes from the environment rather than
    #: from the recorded fixture, so the environment is what has to be pinned.
    USER = "testuser"

    @pytest.fixture(autouse=True)
    def _fixed_user(self, monkeypatch):
        monkeypatch.setenv("USER", self.USER)

    def _templated(self):
        from nodetop.backends.slurm import SlurmBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        return Cluster.load(
            SlurmBackend(RecordedRunner({
                "scontrol show node": (
                    0, "NodeName=n1 State=IDLE CPUTot=8 Partitions=p\n", ""),
                "scontrol show partition": (
                    0, "PartitionName=p\n   State=UP AllowGroups=ALL\n"
                       "   Nodes=n1\n   TotalNodes=1\n", ""),
                "show qos": (0, "", ""),
                # Two accounts, one identical menu.
                "show assoc": (0, "acct-a||x,y\nacct-b||x,y\n", ""),
                "squeue": (0, "", ""),
            })),
            with_free_times=False,
        )

    def test_the_header_names_who_you_are(self, capsys):
        cmd_status(self._templated(), _args(["status"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        # The raw account count left the header: now that access is filtered
        # and the hidden counts are reported, "34 accounts" is context rather
        # than an answer. The username stays -- it says whose access this is.
        assert self.USER in out

    def test_templated_entitlements_are_called_out_once(self, capsys):
        # A property of the cluster, not of one placement -- but context rather
        # than headline, so it sits at the bottom in one line.
        # Only under --declared: with the dry-run running, which partitions you
        # may use has been settled by the control plane, so the state of the
        # association table is no longer the reader's problem.
        cmd_status(self._templated(), _args(["status", "--declared"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "TEMPLATED" in out
        assert out.count("TEMPLATED") == 1

    def test_a_cluster_with_distinct_entitlements_gets_no_warning(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), PLAIN)
        assert "TEMPLATED" not in capsys.readouterr().out

    def test_json_exposes_the_identity_summary(self, capsys):
        cmd_status(self._templated(), _args(["--json", "status"]), PLAIN)
        data = json.loads(capsys.readouterr().out)["identity"]
        assert data["accounts"] == 2
        assert data["entitlements_look_templated"] is True
        assert data["user"]

    def test_json_identity_is_null_when_unavailable(self, capsys):
        from nodetop.core.cluster import Cluster

        cmd_status(Cluster(), _args(["--json", "status"]), PLAIN)
        assert json.loads(capsys.readouterr().out)["identity"] is None


class TestStatusJsonSaysWhatThePanelSays:
    """`--json` used to answer a different question than the panel.

    It returned `cluster.summary()` and nothing else: the panel said "222 of
    358 GPUs, 53 free" -- your partitions -- while the JSON reported 358 and
    126, counting accelerators in partitions the account cannot submit to, and
    carried neither the funnel nor a single partition row.
    """

    def _json(self, cluster, capsys, argv=("status", "--json")):
        assert cmd_status(cluster, _args(list(argv)), PLAIN) == 0
        return json.loads(capsys.readouterr().out)

    def test_the_funnel_terms_sum_to_the_partition_count(self, cluster, capsys):
        d = self._json(cluster, capsys)
        f = d["funnel"]
        assert f["total"] == d["queues"]
        assert sum(v for k, v in f.items()
                   if k not in {"total", "unconfirmed"}) == f["total"]

    def test_the_listed_rows_are_the_ones_the_funnel_shows(self, cluster, capsys):
        d = self._json(cluster, capsys)
        assert len(d["listed"]) == d["funnel"]["shown"]

    def test_every_other_partition_is_named_with_its_reason(self, cluster, capsys):
        d = self._json(cluster, capsys)
        assert len(d["listed"]) + len(d["excluded"]) == d["funnel"]["total"]
        assert all(e["reason"] for e in d["excluded"])

    def test_the_scope_is_yours_not_the_whole_cluster(self, cluster, capsys):
        # Both are carried, and they are labelled: `accelerators_total` at the
        # top level is the cluster, `yours` is the population the panel counts.
        d = self._json(cluster, capsys)
        assert d["yours"]["accelerators_total"] <= d["accelerators_total"]
        assert d["yours"]["nodes"] <= d["nodes"]

    def test_the_allocatable_core_count_is_carried_too(self, cluster, capsys):
        # The number the panel meters, which no JSON field used to hold.
        y = self._json(cluster, capsys)["yours"]
        assert y["effective_free_cpus"] <= y["cpus_free_advertised"]

    def test_an_empty_cluster_still_emits_the_whole_schema(self, capsys):
        from nodetop.core.cluster import Cluster

        d = self._json(Cluster(), capsys)
        for key in ("yours", "funnel", "listed", "excluded"):
            assert key in d, key
        assert d["funnel"]["shown"] == 0


class TestStatusIsSelfLimiting:
    """The dashboard's length must not track the size of the cluster.

    On a 607-node, 87-partition cluster the overview grew a row per partition,
    which is what turned it into a listing to scroll rather than a screen to
    read.

    Built synthetically rather than from the recorded fixture: a row needs a
    queue with genuinely free capacity, and every usable queue in the recording
    is full.  (Cloning the fixture's queues produced 62 usable queues and zero
    rows -- correct behaviour, useless test.)
    """

    #: Row names are short on purpose: the table clips a long name to an
    #: ellipsis, so a marker that has to survive clipping must be tiny.
    MARK = "zz"

    @classmethod
    def _cluster(cls, n):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        for i in range(n):
            name = f"{cls.MARK}{i}"
            nodes.append(Node(name=f"n{i}", state_raw="IDLE", cpus_total=8,
                              memory_mb=16000, queues=(name,)))
            # `nodes` is the resolved list Cluster.load wires up; a
            # hand-built Cluster has to supply it directly.
            queues[name] = Queue(name=name, node_names=(f"n{i}",),
                                 declared_nodes=1, nodes=[nodes[-1]])
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _rows(self, cluster, argv, capsys):
        cmd_status(cluster, _args(argv), PLAIN)
        return [ln for ln in capsys.readouterr().out.splitlines() if self.MARK in ln]

    def test_the_capacity_table_is_capped(self, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        rows = self._rows(self._cluster(STATUS_ROWS * 3), ["status"], capsys)
        assert 0 < len(rows) <= STATUS_ROWS

    def test_the_cap_says_how_much_it_withheld(self, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        cmd_status(self._cluster(STATUS_ROWS * 3), _args(["status"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        # Silently showing 12 of 36 reads as "there are 12".
        assert "--all" in out
        assert "more" in out

    def test_all_lifts_the_cap(self, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        cluster = self._cluster(STATUS_ROWS * 3)
        capped = self._rows(cluster, ["status"], capsys)
        full = self._rows(cluster, ["status", "--all"], capsys)
        assert len(full) > len(capped)
        assert len(full) >= STATUS_ROWS * 3

    def test_a_cluster_under_the_cap_shows_everything(self, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        rows = self._rows(self._cluster(3), ["status"], capsys)
        assert len(rows) == 3

    def test_a_cluster_under_the_cap_gets_no_pointer(self, monkeypatch, capsys):
        # The hint is about what was withheld, so with nothing withheld it is
        # noise -- which is the whole complaint the compaction answered.
        monkeypatch.setenv("COLUMNS", "200")
        cmd_status(self._cluster(3), _args(["status"]), PLAIN)
        out = " ".join(capsys.readouterr().out.split())
        assert "more" not in out
        assert "the other" not in out

    def test_the_cap_is_a_named_constant_not_a_literal(self):
        # It is referred to in the hint text, so it must be one value.
        assert isinstance(STATUS_ROWS, int) and STATUS_ROWS > 0


class TestHealthGroupsByReasonNotByTimestamp:
    """One maintenance window is one finding.

    Slurm stamps every drain reason with ``[who@when]``. Keying the grouping on
    the raw string split a single window into a row per second the operator
    spent typing: on a live cluster, 52 nodes drained for "maintenance" showed
    up as five separate rows whose only difference was a timestamp -- and the
    actual finding ("52 nodes have been out for five weeks") was invisible.
    """

    @staticmethod
    def _cluster(reasons):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node

        nodes = [
            Node(name=f"n{i}", state_raw="DOWN", conditions=frozenset({"DOWN"}),
                 cpus_total=8, reason=r)
            for i, r in enumerate(reasons)
        ]
        return Cluster(backend_name="synthetic", nodes=nodes)

    def _out(self, reasons, capsys):
        cmd_health(self._cluster(reasons), _args(["health"]), PLAIN)
        return capsys.readouterr().out

    def test_the_same_reason_with_different_stamps_is_one_row(self, capsys):
        out = self._out([
            "maintenance [root@2026-07-16T08:51:31]",
            "maintenance [root@2026-07-16T08:50:59]",
            "maintenance [admin@2026-08-01T00:00:00]",
        ], capsys)
        assert out.count("maintenance") == 1
        assert "  3  maintenance" in out

    def test_genuinely_different_reasons_stay_separate(self, capsys):
        # Collapsing must not go so far as to merge distinct causes.
        out = self._out([
            "maintenance [root@2026-07-16T08:51:31]",
            "maintenance: hardware issue [root@2026-08-20T18:55:05]",
            "planned [root@2026-07-16T14:29:28]",
        ], capsys)
        assert "maintenance: hardware issue" in out
        assert "planned" in out
        assert "  1  maintenance: hardware issue" in out

    def test_no_timestamp_appears_in_a_group_label(self, capsys):
        out = self._out(["maintenance [root@2026-07-16T08:51:31]"], capsys)
        assert "2026-07-16T08:51:31" not in out

    def test_the_oldest_stamp_becomes_the_group_age(self, capsys):
        # The stamp is not thrown away -- how long they have been out is the
        # part worth knowing, and the oldest is the honest answer for a group.
        out = self._out([
            "maintenance [root@2020-01-01T00:00:00]",
            "maintenance [root@2026-08-01T00:00:00]",
        ], capsys)
        assert "for " in out
        assert "d" in out.split("for ")[1][:12]  # an age in days

    def test_an_unstamped_reason_gets_no_age(self, capsys):
        out = self._out(["Not responding"], capsys)
        assert "Not responding" in out
        assert "for " not in out

    def test_a_node_with_no_reason_falls_back_to_its_state(self, capsys):
        out = self._out([""], capsys)
        assert "state DOWN" in out


class TestHealthTextAndJsonAgree:
    """The two views must not disagree about what one cause is.

    The text view groups by parsed reason; the JSON view exposes the same parse
    so a script can reach the same grouping. If only one side parsed, a
    dashboard built on the JSON would report five maintenance windows where the
    terminal reported one.
    """

    REASONS = [
        "maintenance [root@2026-07-16T08:51:31]",
        "maintenance [root@2026-07-16T08:50:59]",
        "planned [root@2026-07-16T14:29:28]",
        "Not responding",
    ]

    def _cluster(self):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node

        return Cluster(backend_name="synthetic", nodes=[
            Node(name=f"n{i}", state_raw="DOWN", conditions=frozenset({"DOWN"}),
                 cpus_total=8, reason=r)
            for i, r in enumerate(self.REASONS)
        ])

    def test_the_json_carries_the_parsed_reason(self, capsys):
        cmd_health(self._cluster(), _args(["--json", "health"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)["unschedulable"]
        assert {r["reason_text"] for r in rows} == {
            "maintenance", "planned", "Not responding"}

    def test_the_verbatim_reason_is_still_there(self, capsys):
        # Parsing is additive: a consumer that wants the raw string keeps it.
        cmd_health(self._cluster(), _args(["--json", "health"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)["unschedulable"]
        assert any(r["reason"] == self.REASONS[0] for r in rows)

    def test_an_unstamped_reason_has_no_who_or_when(self, capsys):
        cmd_health(self._cluster(), _args(["--json", "health"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)["unschedulable"]
        bare = next(r for r in rows if r["reason_text"] == "Not responding")
        assert bare["reason_set_by"] is None
        assert bare["reason_set_at"] is None

    def test_both_views_count_each_cause_the_same(self, capsys):
        from collections import Counter

        cmd_health(self._cluster(), _args(["--json", "health"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)["unschedulable"]
        counts = Counter(r["reason_text"] for r in rows)

        cmd_health(self._cluster(), _args(["health"]), PLAIN)
        text = capsys.readouterr().out
        for cause, n in counts.items():
            assert f"{n}  {cause}" in text, f"{cause}={n} missing from the text view"


class TestWaitsAreMeasuredFromTheDataClock:
    """START is relative to when the data was taken, not when it is read.

    Node free times are absolute instants. Comparing them against
    ``datetime.now()`` is right for a live query and wrong for a replay by
    exactly the snapshot's age: a node recorded as free in three hours reads as
    "overdue" the moment the recording is older than that. The estimate then
    looks authoritative while being nonsense.
    """

    @staticmethod
    def _cluster(taken_ago_days, free_in_hours):
        from datetime import datetime, timedelta

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        taken = datetime.now() - timedelta(days=taken_ago_days)
        node = Node(name="n0", state_raw="ALLOCATED", cpus_total=8,
                    cpus_alloc=8, memory_mb=16000, queues=("q",))
        queue = Queue(name="q", node_names=("n0",), declared_nodes=1,
                      nodes=[node])
        return Cluster(
            backend_name="synthetic", nodes=[node], queues={"q": queue},
            node_free_times={"n0": taken + timedelta(hours=free_in_hours)},
            taken_at=taken,
        )

    def _start_cell(self, cluster, capsys):
        cmd_where(cluster, _args(["where", "-c", "1", "--all"]), PLAIN)
        out = capsys.readouterr().out
        row = next(ln for ln in out.splitlines() if " q " in f" {ln.strip()} ")
        return row

    def test_a_week_old_snapshot_does_not_report_overdue(self, capsys):
        # Free three hours after capture, read a week later. Against now() this
        # is ~7 days overdue; against the capture clock it is "3h".
        row = self._start_cell(self._cluster(7, 3), capsys)
        assert "overdue" not in row
        assert "3h" in row

    def test_a_live_cluster_is_unaffected(self, capsys):
        row = self._start_cell(self._cluster(0, 3), capsys)
        assert "3h" in row

    def test_a_genuinely_past_free_time_still_reads_overdue(self, capsys):
        # Measured from the right clock, a free time before the capture really
        # is overdue -- the fix must not paper that over.
        row = self._start_cell(self._cluster(7, -5), capsys)
        assert "overdue" in row


class TestStatusLeadsWithWhatYouCanUse:
    """Order and grouping, both of which were wrong.

    The dashboard opened with "unusable partitions", so the first thing on
    screen was always a list of things that do not work. And it ranked every
    partition together by free capacity, which on a real cluster put eleven
    group-owned partitions into the top twelve rows -- so the headline answer
    to "where can I go" was almost entirely places the reader cannot go.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        # One shared partition with modest capacity...
        nodes.append(Node(name="s1", state_raw="IDLE", cpus_total=8,
                          memory_mb=16000, gpus_total=2, queues=("openq",)))
        queues["openq"] = Queue(name="openq", node_names=("s1",),
                                 declared_nodes=1, nodes=[nodes[-1]])
        # ...and a private one with far more, so a capacity-only sort would
        # put it first.
        # Six of them: the brief view caps the private list at four, so a
        # single private partition made the brief and --all views identical.
        for i in range(6):
            name = "pi-rich" if i == 0 else f"pi-other{i}"
            nodes.append(Node(name=f"p{i}", state_raw="IDLE", cpus_total=64,
                              memory_mb=256000, gpus_total=8 - i,
                              queues=(name,)))
            queues[name] = Queue(name=name, node_names=(f"p{i}",),
                                 declared_nodes=1, nodes=[nodes[-1]],
                                 allow_accounts=(name,))
        # ...and one that can start nothing but advertises idle nodes.
        nodes.append(Node(name="d1", state_raw="IDLE", cpus_total=8,
                          memory_mb=16000, queues=("broken",)))
        queues["broken"] = Queue(name="broken", node_names=("d1",),
                                 declared_nodes=1, nodes=[nodes[-1]],
                                 enabled=False, state_raw="DOWN")
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _out(self, capsys, argv=("status",)):
        cmd_status(self._cluster(), _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    def test_the_shared_partition_comes_before_the_private_one(self, capsys):
        # Marker names chosen not to collide with the headings: the first
        # version used "shared", which also appears in the section note, so
        # index() found the heading and a mutation merging the two tables
        # survived the assertion.
        out = self._out(capsys)
        assert "openq" in out and "pi-rich" in out
        assert out.index("openq") < out.index("pi-rich")

    def test_the_private_partition_is_labelled_as_reserved(self, capsys):
        out = " ".join(self._out(capsys).split())
        assert "GROUP-ONLY" in out
        # And it is not in the open table: the heading must not be the only
        # thing separating them.
        head, _, tail = out.partition("GROUP-ONLY")
        assert "pi-rich" not in head
        assert "openq" in head

    def test_a_capacity_only_sort_would_have_ranked_it_first(self, capsys):
        # Guards the guard: if the fixture stopped exercising the split, the
        # ordering assertion above would pass trivially.
        cluster = self._cluster()
        by_capacity = sorted(cluster.queues.values(),
                             key=lambda q: -q.effective_free_gpus)
        assert by_capacity[0].name == "pi-rich"

    def test_the_failure_is_not_on_this_view_at_all(self, capsys):
        # Demoted to a count, then off the view. "Failures come last" was the
        # half-measure: last is still on screen, every run, in red.
        out = " ".join(self._out(capsys).split())
        assert "broken" not in out
        assert "DEAD" not in out

    def test_it_is_counted_rather_than_listed(self, capsys):
        out = " ".join(self._out(capsys).split())
        assert "1 down" in out

    def test_the_partitions_you_can_use_are_what_is_listed(self, capsys):
        out = self._out(capsys)
        assert "openq" in out

    def test_all_shows_the_private_ones_in_full(self, capsys):
        # The brief view caps the private list, so the fixture needs more
        # private partitions than the cap for --all to have anything to add.
        # With one, the two views were byte-identical and this passed on
        # nothing.
        brief = self._out(capsys)
        full = self._out(capsys, ("status", "--all"))
        assert "more" in brief
        assert "more" not in full
        # Rows, not characters. The brief view carries a longer funnel line --
        # it has exclusions to account for and --all has none -- so total output
        # length is not a proxy for how many partitions were listed.
        assert full.count("pi-other") > brief.count("pi-other")


class TestStatusIsNotFramedAroundGpus:
    """Nodes are the spine; accelerators are a column.

    The overview used to lead with a GPU fraction, rank by free GPUs, and give
    its meter to GPU share. On the cluster it was built against that is 91 of
    607 nodes -- so five of seven shared partitions drew an empty bar and a
    dash, which reads as missing data rather than as a CPU partition, and the
    ranking sorted the cluster by a property 85% of it does not have.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        # A big GPU partition with almost nothing free, and a small CPU one
        # entirely free. Ranking by free GPUs puts the first one on top; ranking
        # by free nodes puts the second one there, which is where the room is.
        specs = [("bigq", 8, 4), ("cpuq", 3, 0)]
        for name, count, gpus in specs:
            mine = []
            for i in range(count):
                busy = name == "bigq" and i > 0
                mine.append(Node(
                    name=f"{name}{i}", state_raw="ALLOCATED" if busy else "IDLE",
                    cpus_total=8, cpus_alloc=8 if busy else 0,
                    memory_mb=16000, gpus_total=gpus,
                    gpus_alloc=gpus if busy else 0, queues=(name,)))
            nodes += mine
            queues[name] = Queue(name=name, node_names=tuple(n.name for n in mine),
                                 declared_nodes=count, nodes=mine)
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _out(self, capsys):
        cmd_status(self._cluster(), _args(["status"]), PLAIN)
        return capsys.readouterr().out

    def test_the_header_leads_with_nodes(self, capsys):
        out = self._out(capsys)
        head = out.splitlines()[1]
        assert "nodes" in head and " up" in head
        assert head.index("nodes") < head.index("GPU")

    def test_every_row_has_a_meaningful_meter(self, capsys):
        # The point of the change: the bar measures free NODES, so a wholly
        # free CPU partition draws a full one. Asserting only that some meter
        # character is present is vacuous -- a GPU-share meter on a CPU
        # partition is all trough, and "░" is still a meter character.
        out = self._out(capsys)
        cpu = next(ln for ln in out.splitlines() if "cpuq" in ln)
        assert "█" * 10 in cpu, "a fully free partition should read as full"

        gpu = next(ln for ln in out.splitlines() if "bigq" in ln)
        # 1 of 8 nodes free: mostly trough, but not empty.
        assert "░" in gpu
        assert "█" * 10 not in gpu

    def test_a_cpu_partition_says_the_question_does_not_arise(self, capsys):
        # A dash, and not a blank. The column is headed `gpu free`, so it asks
        # a question, and a partition with no accelerators is not answering it
        # -- the same reading the node table uses. This assertion used to
        # require the blank, from before the column had a name.
        out = self._out(capsys)
        row = next(ln for ln in out.splitlines() if "cpuq" in ln)
        assert "—" in row
        assert "/0" not in row       # and never a fraction over nothing

    def test_ranking_is_by_free_nodes_not_free_gpus(self, capsys):
        out = self._out(capsys)
        assert out.index("cpuq") < out.index("bigq")

    def test_a_gpu_ranking_would_have_inverted_that(self):
        # Guards the guard.
        cluster = self._cluster()
        by_gpu = sorted(cluster.queues.values(), key=lambda q: -q.effective_free_gpus)
        assert by_gpu[0].name == "bigq"

    def test_the_gpu_column_still_appears_where_there_are_gpus(self, capsys):
        out = self._out(capsys)
        row = next(ln for ln in out.splitlines() if "bigq" in ln)
        # One cell, `free/total`, under a header that names the numerator.
        assert "4/32" in row

    def test_every_fraction_sits_under_a_header_that_names_it(self, capsys):
        """`free/total` in one cell, and the header says which side is free.

        This assertion used to be the opposite: no cell could hold a fraction
        at all, because `4/4` cannot say which number is the free one -- "4/4
        100%" was once read as meaning all four are BUSY. Splitting them fixed
        that and created a worse one: two columns headed plainly `free`, one
        for cores and one for accelerators, so the word covered two resources
        at once. "why free appears twice on the column title? what does it
        mean?" The header carries the answer instead, and every fraction in the
        table is under one.
        """
        import re

        out = self._out(capsys)
        header = next(ln for ln in out.splitlines() if "cores free" in ln)
        # No COLUMN is headed by a bare `free`: every one names its resource.
        cells = [c for c in re.split(r"\s{2,}", header.strip("│ ")) if c]
        assert "free" not in cells, cells
        assert "share" not in cells
        for want in ("nodes idle", "cores free", "gpu free"):
            assert want in cells, (want, cells)
        # And the numbers under them are fractions, not lone counts.
        rows = [ln for ln in out.splitlines()
                if any(q in ln for q in ("bigq", "cpuq"))]
        assert rows
        for row in rows:
            fractions = [tok for tok in row.split()
                         if re.fullmatch(r"\d+/\d+", tok)]
            # `nodes idle` and `cores free` always; `gpu free` only where there
            # are accelerators to count.
            assert 2 <= len(fractions) <= 3, (row, fractions)


class TestQueuesIsNotFramedAroundGpus:
    """Same defect as the overview had, in the view with the most rows.

    `queues` metered GPU share, so on a cluster where 70 of 87 partitions have
    no accelerator, 70 rows drew an empty bar -- while `up`, the quantity every
    partition has, got no meter at all.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        for name, gpus in (("gpuq", 4), ("cpuq", 0)):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=16000, gpus_total=gpus, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node])
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def _row(self, capsys, name):
        cmd_queues(self._cluster(), _args(["queues"]), PLAIN)
        out = capsys.readouterr().out
        return next(ln for ln in out.splitlines() if name in ln)

    def test_a_cpu_partition_gets_a_full_meter_when_free(self, capsys):
        # It is entirely free; the bar should say so rather than sit empty
        # because the partition has no GPUs.
        assert "█" * 8 in self._row(capsys, "cpuq")

    def test_a_cpu_partition_has_no_gpu_figure(self, capsys):
        row = self._row(capsys, "cpuq")
        assert "/0" not in row

    def test_a_gpu_partition_still_reports_its_gpus(self, capsys):
        assert "4/4" in self._row(capsys, "gpuq")

    def test_the_free_column_does_not_repeat_the_denominator(self, capsys):
        # `up 1/1` already carries the total.
        row = self._row(capsys, "cpuq")
        assert row.count("1/1") == 1


class TestNodesIsCappedAndMetered:
    """The last unbounded dump, and the last table without a meter.

    `nodetop nodes` answered "how are my 607 nodes doing" with 607 rows, which
    is not an answer -- it is the raw data again. And it was the only table left
    with no meter, so scanning it meant reading `0/32` against `24/32` as text.
    """

    @staticmethod
    def _cluster(count=50):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        # n000 wholly free, n001 wholly busy, the rest in between. The first
        # version used `(i * 4) % 32`, which never reaches 32 -- so there was no
        # fully-allocated node to assert an empty meter against.
        nodes = [
            Node(name=f"n{i:03d}", state_raw="MIXED", cpus_total=32,
                 cpus_alloc=0 if i == 0 else 32 if i == 1 else (i * 4) % 32,
                 memory_mb=64000, queues=("q",))
            for i in range(count)
        ]
        queue = Queue(name="q", node_names=tuple(n.name for n in nodes),
                      declared_nodes=count, nodes=nodes)
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues={"q": queue})

    def _rows(self, capsys, argv=("nodes",)):
        from nodetop.cli import cmd_nodes

        cmd_nodes(self._cluster(), _args(list(argv)), PLAIN)
        out = capsys.readouterr().out
        return out, [ln for ln in out.splitlines() if " n0" in ln]

    def test_it_caps_by_default(self, capsys):
        _out, rows = self._rows(capsys)
        assert 0 < len(rows) <= 20

    def test_it_says_how_many_it_withheld(self, capsys):
        out, _rows = self._rows(capsys)
        assert "more" in out
        assert "--all" in out

    def test_all_lifts_the_cap(self, capsys):
        _out, rows = self._rows(capsys, ("nodes", "--all"))
        assert len(rows) == 50

    def test_top_sets_the_cap(self, capsys):
        _out, rows = self._rows(capsys, ("nodes", "-n", "5"))
        assert len(rows) == 5

    def test_a_cluster_under_the_cap_gets_no_pointer(self, capsys):
        from nodetop.cli import cmd_nodes

        cmd_nodes(self._cluster(3), _args(["nodes"]), PLAIN)
        out = capsys.readouterr().out
        assert "more" not in out

    def test_every_row_has_a_cpu_meter(self, capsys):
        _out, rows = self._rows(capsys)
        for row in rows:
            assert "█" in row or "░" in row

    def test_the_meter_tracks_cpu_availability(self, capsys):
        # n000 has every core free, so its meter is full; n008 has 0 free.
        _out, rows = self._rows(capsys, ("nodes", "--all"))
        full = next(r for r in rows if "n000" in r)
        assert "█" * 8 in full
        empty = next(r for r in rows if "n001" in r)
        assert "█" not in empty
        assert "░" * 8 in empty


class TestStatusFiltersToWhatWillTakeTheJob:
    """Two filters, both on by default, measured against the control plane.

    The allowlist filter is instant and had no false negatives on this cluster:
    65 partitions with room down to 12, every drop confirmed ACCOUNT_MISMATCH.
    It is nowhere near sufficient though -- of the 12 it keeps, a dry-run
    accepts 3. The association table lists this user in `ssd`,
    `pi-okafor`, `pi-tanaka` and three more, and the submit plugin rejects
    all of them with "Invalid membership". So the dry-run runs by default too,
    and it is cheap because the allowlist filter goes first.
    """

    @staticmethod
    def _cluster(*, accepts, allowlists=None):
        import dataclasses

        from nodetop.core.cluster import Cluster
        from nodetop.core.model import (
            BackendCapabilities,
            Identity,
            Node,
            Queue,
            Verdict,
            VerdictCategory,
        )

        allowlists = allowlists or {}
        nodes, queues = [], {}
        for name in ("yes-q", "refused-q", "notmine-q"):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=16000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node],
                                 allow_accounts=allowlists.get(name, ()))

        class _Backend:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub --test-only")

            def probe(self, queue, shape, account=None):
                ok = queue in accepts
                return Verdict(
                    queue=queue, account=account, allowed=ok,
                    category=VerdictCategory.OK if ok
                    else VerdictCategory.NOT_ENTITLED,
                    reason="ok" if ok else "no")

            def submit_flags(self, queue, shape):
                return []

        return dataclasses.replace(
            Cluster(backend_name="synthetic", queue_term="partition",
                    nodes=nodes, queues=queues,
                    identity=Identity(user="me", accounts=("mine",),
                                      qos=("q",))),
            capabilities=_Backend().capabilities(), _backend=_Backend())

    ALLOWLISTS = {"notmine-q": ("someone-else",)}

    def _out(self, capsys, argv=("status",), accepts=("yes-q", "notmine-q")):
        cmd_status(self._cluster(accepts=accepts, allowlists=self.ALLOWLISTS),
                   _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    def test_an_allowlist_you_are_not_on_is_hidden(self, capsys):
        # Instant, no dry-run needed: plain set intersection. This is what
        # `jfkfloor2` (4 accounts) and `voltron` (5) needed and the old
        # "<=2 accounts" heuristic never gave them.
        assert "notmine-q" not in self._out(capsys)

    def test_a_partition_the_dry_run_refuses_is_hidden(self, capsys):
        assert "refused-q" not in self._out(capsys)

    def test_the_accepting_one_survives(self, capsys):
        assert "yes-q" in self._out(capsys)

    def test_both_filters_are_counted_separately(self, capsys):
        out = " ".join(self._out(capsys).split())
        assert "1 no access" in out
        assert "1 refused" in out
        # And the two are not conflated into one number: the reader can tell an
        # allowlist that excludes them from a dry-run that refused them, which
        # are different problems with different remedies.
        assert "2 refused" not in out

    def test_the_funnel_accounts_for_every_partition(self, capsys):
        # The one question this view has been asked twice: "it says 87
        # partitions, why am I looking at five rows?" The line has to answer it
        # by arithmetic -- shown + every exclusion == the total -- because a
        # total that does not reconcile is what prompted the question.
        out = " ".join(self._out(capsys).split())
        assert "3 partitions" in out
        assert "1 open to you" in out
        assert "1 no access" in out
        assert "1 refused" in out

    def test_declared_skips_only_the_dry_run(self, capsys):
        out = self._out(capsys, ("status", "--declared"))
        assert "refused-q" in out       # the dry-run did not run
        assert "notmine-q" not in out   # the allowlist filter still did

    def test_declared_says_it_is_trusting_the_allowlists(self, capsys):
        out = " ".join(self._out(capsys, ("status", "--declared")).split())
        assert "DECLARED ONLY" in out
        assert "over-report" in out

    def test_all_shows_everything_unfiltered(self, capsys):
        out = self._out(capsys, ("status", "--all"))
        for name in ("yes-q", "refused-q", "notmine-q"):
            assert name in out

    def test_everything_refusing_is_not_an_error(self, capsys):
        # An honest empty answer: nothing here will take the job.
        out = self._out(capsys, ("status",), accepts=())
        assert "yes-q" not in out
        joined = " ".join(out.split())
        # An honest empty answer says so in the funnel rather than by going
        # blank: nothing open to you, and every partition accounted for.
        assert "0 open to you" in joined
        assert "refused" in joined


class TestEveryListingFiltersByAccess:
    """One helper, four commands, so this cannot diverge again.

    The entitlement logic lived in `evaluate` and only `where` called it. So
    `status` ranked partitions the caller could not submit to, and `queues`,
    `nodes` and `accelerators` reported cluster-wide totals as though they were
    the caller's: 84 usable partitions where 19 are on the allowlist, 607 nodes
    where 330 are inside those 19, 358 accelerators where 230 are.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Identity, Node, Queue

        nodes, queues = [], {}
        for name, allow in (("mine", ("mine",)), ("theirs", ("someone-else",))):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=16000, gpus_total=4, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node],
                                 allow_accounts=allow)
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues,
                       identity=Identity(user="me", accounts=("mine",),
                                         qos=("q",)))

    def _run(self, capsys, fn, argv):
        fn(self._cluster(), _args(list(argv)), PLAIN)
        return capsys.readouterr().out

    @pytest.mark.parametrize("fn_name,argv", [
        ("cmd_status", ["status"]),
        ("cmd_queues", ["queues"]),
        ("cmd_nodes", ["nodes"]),
        ("cmd_accelerators", ["accelerators"]),
    ])
    def test_it_hides_what_you_are_not_entitled_to(self, capsys, fn_name, argv):
        import nodetop.cli as cli

        out = self._run(capsys, getattr(cli, fn_name), argv)
        assert "n-theirs" not in out and "theirs" not in out

    @pytest.mark.parametrize("fn_name,argv", [
        ("cmd_queues", ["queues", "--all"]),
        ("cmd_nodes", ["nodes", "--all"]),
    ])
    def test_all_turns_the_filter_off(self, capsys, fn_name, argv):
        import nodetop.cli as cli

        assert "theirs" in self._run(capsys, getattr(cli, fn_name), argv)

    def test_all_turns_it_off_for_the_gpu_inventory_too(self, capsys):
        # Asserted on the count, not on a queue name: this view lists models.
        import nodetop.cli as cli

        out = " ".join(self._run(capsys, cli.cmd_accelerators,
                                 ["accelerators", "--all"]).split())
        assert "8 GPUs" in out
        assert "of 8 on the cluster" not in out  # no context needed: it IS all

    @pytest.mark.parametrize("fn_name,argv", [
        ("cmd_queues", ["queues"]),
        ("cmd_nodes", ["nodes"]),
    ])
    def test_it_says_how_much_it_hid(self, capsys, fn_name, argv):
        import nodetop.cli as cli

        out = " ".join(self._run(capsys, getattr(cli, fn_name), argv).split())
        assert "not on your allowlist" in out

    def test_the_gpu_inventory_keeps_the_cluster_figure_as_context(self, capsys):
        import nodetop.cli as cli

        out = " ".join(self._run(capsys, cli.cmd_accelerators,
                                 ["accelerators"]).split())
        assert "4 GPUs of 8 on the cluster" in out

    def test_filtering_to_nothing_falls_back_to_unfiltered(self, capsys):
        # A blank screen means the entitlement data is unusable, not that the
        # caller may use nothing -- and it hides the very data that explains it.
        import dataclasses

        import nodetop.cli as cli
        from nodetop.core.model import Identity

        cluster = dataclasses.replace(
            self._cluster(),
            identity=Identity(user="me", accounts=("nobody",), qos=("q",)))
        cli.cmd_queues(cluster, _args(["queues"]), PLAIN)
        out = capsys.readouterr().out
        assert "mine" in out and "theirs" in out
        assert "not on your allowlist" not in out


class TestAFullPartitionYouCanUseIsStillListed:
    """Access is the filter; current occupancy is a column.

    The overview dropped anything with no free capacity *before* applying the
    access filter, so a partition this account can submit to vanished because
    it happened to be busy at the instant of the query. Measured on the live
    cluster: 5 partitions accept a dry-run, 2 of them were full, and the screen
    said 3. That understates access, not capacity -- "where can I run this"
    includes "where can I queue".
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Identity, Node, Queue

        nodes, queues = [], {}
        # `roomy` has a free node, `full` has none -- both are yours.
        for name, alloc in (("roomy", 0), ("full", 8)):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        cpus_alloc=alloc, memory_mb=16000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node],
                                 allow_accounts=("mine",))
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues,
                       identity=Identity(user="me", accounts=("mine",),
                                         qos=("q",)))

    def _out(self, capsys):
        cmd_status(self._cluster(), _args(["status"]), PLAIN)
        return capsys.readouterr().out

    def test_the_full_one_is_listed(self, capsys):
        assert "full" in self._out(capsys)

    def test_the_roomy_one_is_too(self, capsys):
        assert "roomy" in self._out(capsys)

    def test_the_full_one_reports_zero_free(self, capsys):
        # `0/8`, not a lone `0`: the columns hold `free/total` now, so the
        # denominator says how much is there to be full of.
        row = next(ln for ln in self._out(capsys).splitlines() if "full" in ln)
        assert "0/8" in row

    def test_the_roomy_one_comes_first(self, capsys):
        # Still ordered by where the room is; the full one is not promoted.
        out = self._out(capsys)
        assert out.index("roomy") < out.index("full")

    def test_it_is_not_counted_as_hidden(self, capsys):
        # Nothing was filtered here, so there is nothing to disclose.
        out = " ".join(self._out(capsys).split())
        assert "not on your allowlist" not in out


class TestFreeMeansReachableAndFree:
    """Phantom capacity in the summary of a tool written to catch it.

    `Queue.effective_free_gpus` has always returned 0 for an unusable queue.
    `Cluster.summary` did not: it counted every *schedulable* node's free
    resources, so an idle four-GPU node whose only partition was DOWN read as
    four free accelerators. `accelerators` computed its own totals the same way.
    """

    @staticmethod
    def _cluster(*, live_gpus=2, dead_gpus=4):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        nodes, queues = [], {}
        for name, gpus, dead in (("live", live_gpus, False),
                                 ("dead", dead_gpus, True)):
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=16000, gpus_total=gpus, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(
                name=name, node_names=(node.name,), declared_nodes=1,
                nodes=[node],
                **({"state_raw": "DOWN", "enabled": False} if dead else {}))
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues)

    def test_the_dead_queue_agrees_it_has_nothing_free(self):
        cluster = self._cluster()
        assert cluster.queues["dead"].effective_free_gpus == 0

    def test_the_summary_agrees_too(self):
        # It did not: this is the bug.
        assert self._cluster().summary()["accelerators_free"] == 2

    def test_a_wholly_dead_cluster_reports_nothing_free(self):
        cluster = self._cluster(live_gpus=0)
        assert cluster.summary()["accelerators_free"] == 0

    def test_the_installed_total_still_counts_everything(self):
        # Reachability changes what is *free*, never what exists.
        assert self._cluster().summary()["accelerators_total"] == 6

    def test_reachable_nodes_excludes_a_node_in_no_queue(self):
        # Nothing can be submitted to it, so its capacity is not capacity.
        import dataclasses

        from nodetop.core.model import Node

        cluster = self._cluster()
        orphan = Node(name="orphan", state_raw="IDLE", cpus_total=8,
                      memory_mb=16000, gpus_total=8, queues=())
        cluster = dataclasses.replace(cluster, nodes=[*cluster.nodes, orphan])
        assert "orphan" not in {n.name for n in cluster.reachable_nodes()}
        assert cluster.summary()["accelerators_free"] == 2

    def test_the_inventory_view_counts_the_same_way(self, capsys):
        from nodetop.cli import cmd_accelerators

        # Unconditional: the first version guarded the assertion on a key name
        # it had guessed at, so a wrong guess made the test pass on nothing.
        cmd_accelerators(self._cluster(),
                         _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert data["accelerators_installed"] == 6
        assert sum(m["free"] for m in data["models"].values()) == 2


class TestEachFunnelLabelIsItsOwnTarget:
    """"i want to hit enter myself to each of the labels."

    The counts on the funnel line -- `65 no access`, `11 refused`, `3 down` --
    are each a set of partitions the reader can be shown. They share one body
    row, so the cursor cannot tell them apart by position: the selected term is
    the one drawn in the accent colour, and entering it opens that reason's
    partitions rather than all of them.
    """

    @staticmethod
    def _cluster():
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Identity, Node, Queue

        nodes, queues = [], {}
        # one usable, one the account is not on, one dead
        specs = (("open", (), True), ("private", ("someone-else",), True),
                 ("broken", (), False))
        for name, allow, alive in specs:
            node = Node(name=f"n-{name}", state_raw="IDLE", cpus_total=8,
                        memory_mb=1000, queues=(name,))
            nodes.append(node)
            queues[name] = Queue(name=name, node_names=(node.name,),
                                 declared_nodes=1, nodes=[node],
                                 allow_accounts=allow, enabled=alive,
                                 state_raw="UP" if alive else "DOWN")
        return Cluster(backend_name="synthetic", queue_term="partition",
                       nodes=nodes, queues=queues,
                       identity=Identity(user="me", accounts=("mine",),
                                         qos=("q",)))

    def _walk(self, monkeypatch, capsys, replies):
        import nodetop.interactive as inter

        frames: list[list[str]] = []
        answers = iter(replies)

        def scripted(render, _count, **_kw):
            got = next(answers, inter.Key.QUIT)
            frames.append(list(render(got if isinstance(got, int) else 0)))
            return got

        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "select", scripted)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        assert cmd_status(self._cluster(), _args(["status"]), PLAIN) == 0
        capsys.readouterr()
        return frames

    def test_the_labels_are_selectable_alongside_the_rows(self, monkeypatch,
                                                         capsys):
        import nodetop.interactive as inter

        counts = []
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        monkeypatch.setattr(inter, "select",
                            lambda _r, n, **_k: counts.append(n) or inter.Key.QUIT)
        cmd_status(self._cluster(), _args(["status"]), PLAIN)
        capsys.readouterr()
        # one usable partition to list, plus a label for each exclusion reason
        assert counts and counts[0] > 1

    def test_entering_a_label_shows_only_that_reason(self, monkeypatch, capsys):
        # Entry 0 is the partition total, 1 is "open to you", so the first
        # exclusion reason is 2.
        frames = self._walk(monkeypatch, capsys, [2])
        opened = "\n".join(frames[1])
        # The header names the reason, and every row shares it.
        reasons = {"no access", "down", "no nodes", "refused"}
        named = [r for r in reasons if r in opened]
        assert named, opened
        assert opened.count(named[0]) >= 2

    def test_the_selected_label_is_marked_without_moving_the_line(self,
                                                                 monkeypatch,
                                                                 capsys):
        # Several targets share one row, so the row must not shift as the cursor
        # moves between them -- only the accent moves.
        first = self._walk(monkeypatch, capsys, [0])[0]
        line_a = [ln for ln in first if "open to you" in ln][0]
        second = self._walk(monkeypatch, capsys, [1])[0]
        line_b = [ln for ln in second if "open to you" in ln][0]
        assert width(line_a) == width(line_b)

    def test_the_cursor_moves_to_the_selected_label(self, monkeypatch, capsys):
        """The glyph points at the term, not at the left margin.

        The row cursor was pinned to column 0 of this line while the selection
        moved between terms by colour alone, so it pointed at the partition
        total whatever was chosen and Right read as having done nothing:
        "when pressing the right arrow, it doesn't land at ' 8 open to you'".
        """
        def pointed_at(entry: int) -> str:
            line = [ln for ln in self._walk(monkeypatch, capsys, [entry])[0]
                    if "open to you" in ln][0]
            # One cursor on the row, not one per candidate and not two.
            assert line.count(PLAIN.g.cursor) == 1, line
            return line.split(PLAIN.g.cursor)[1].strip()

        # The total leads the line and is a target like the rest of them:
        # "why can't we select the 87 partitions?"
        assert pointed_at(0).startswith("3 partitions")
        assert pointed_at(1).startswith("1 open to you")
        assert not pointed_at(2).startswith("1 open to you")

    def test_the_total_opens_every_partition_with_its_reason(self, monkeypatch,
                                                            capsys):
        # The one view where the funnel's arithmetic can be checked rather than
        # trusted: its terms, itemised.
        opened = "\n".join(self._walk(monkeypatch, capsys, [0])[1])
        assert "3 partitions" in opened
        for name in ("open", "private", "broken"):
            assert name in opened, name
        # And each carries the word that put it there.
        assert "open" in opened and "no access" in opened and "down" in opened

    def test_a_cluster_with_nothing_excluded_has_no_labels(self, monkeypatch,
                                                           capsys):
        import nodetop.interactive as inter
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        node = Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=1000,
                    queues=("q",))
        cluster = Cluster(backend_name="synthetic", queue_term="partition",
                          nodes=[node],
                          queues={"q": Queue(name="q", node_names=("n1",),
                                             declared_nodes=1, nodes=[node])})
        counts = []
        monkeypatch.setattr(inter, "supported", lambda *_a, **_k: True)
        monkeypatch.setattr(inter, "read_key", lambda *_a, **_k: inter.Key.QUIT)
        monkeypatch.setattr(inter, "select",
                            lambda _r, n, **_k: counts.append(n) or inter.Key.QUIT)
        cmd_status(cluster, _args(["status"]), PLAIN)
        capsys.readouterr()
        # The partition, plus the funnel's own "open to you" bucket. No
        # exclusion buckets, because nothing is excluded -- which is the point.
        # The total, "open to you", and the single partition row. No exclusion
        # terms, because nothing is excluded.
        assert counts == [3]
