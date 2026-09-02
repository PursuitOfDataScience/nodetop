"""The check command: what it asks, what it reports, what it exits with."""

from __future__ import annotations

import json

from nodetop.cli import build_parser, cmd_check, cmd_where
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _prose(text: str) -> str:
    return " ".join(text.split())


class TestExitStatus:
    """`nodetop check ... && sbatch ...` has to behave."""

    def test_accepted_exits_zero(self, accepting_cluster, capsys):
        assert cmd_check(accepting_cluster, _args(["check", "-g", "1"]), PLAIN) == 0
        capsys.readouterr()

    def test_total_refusal_exits_one(self, refusing_cluster, capsys):
        # Returning zero here would wave the caller straight through a refusal.
        assert cmd_check(refusing_cluster, _args(["check", "-g", "1"]), PLAIN) == 1
        capsys.readouterr()

    def test_json_uses_the_same_status(self, refusing_cluster, capsys):
        assert cmd_check(refusing_cluster, _args(["--json", "check", "-g", "1"]),
                         PLAIN) == 1
        capsys.readouterr()

    def test_no_dry_run_available_exits_two(self, capsys):
        from nodetop.backends.lsf import LsfBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        backend = LsfBackend(RecordedRunner({
            "bhosts": (0, "", ""), "lshosts": (0, "", ""), "bqueues": (0, "", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        # Distinct from "refused": nothing was asked, so nothing was learned.
        assert cmd_check(cluster, _args(["check"]), PLAIN) == 2
        assert "no verify-only" in _prose(capsys.readouterr().out)


class TestReporting:
    def test_the_effective_qos_is_reported_not_the_requested_one(
        self, accepting_cluster, capsys
    ):
        cmd_check(accepting_cluster, _args(["--json", "check", "-g", "1"]), PLAIN)
        data = json.loads(capsys.readouterr().out)["queues"]
        # The site auto-promoted alpha -> alpha-prio.
        assert any(v["effective_qos"] == "alpha-prio" for v in data.values())

    def test_a_predicted_start_is_carried_through(self, accepting_cluster, capsys):
        cmd_check(accepting_cluster, _args(["--json", "check", "-g", "1"]), PLAIN)
        data = json.loads(capsys.readouterr().out)["queues"]
        assert any(v["predicted_start"] for v in data.values())

    def test_the_refusal_category_is_shown(self, refusing_cluster, capsys):
        cmd_check(refusing_cluster, _args(["check", "-g", "1"]), PLAIN)
        assert "NOT_ENTITLED" in capsys.readouterr().out

    def test_a_filter_scheduler_disagreement_is_called_out(
        self, disagreeing_cluster, capsys
    ):
        cmd_check(disagreeing_cluster, _args(["check", "-g", "1"]), PLAIN)
        out = _prose(capsys.readouterr().out)
        assert "disagree" in out
        assert "opposite of the truth" in out

    def test_the_notes_say_what_the_check_cannot_cover(self, accepting_cluster, capsys):
        cmd_check(accepting_cluster, _args(["check", "-g", "1"]), PLAIN)
        # sbatch --test-only does not evaluate QOS ceilings; saying so is the
        # difference between a verdict and a false reassurance.
        assert "not covered" in _prose(capsys.readouterr().out)

    def test_capability_flags_are_declared_as_taking_no_part(
        self, accepting_cluster, capsys
    ):
        # check asks the control plane, and no batch system can express a dtype
        # or an HBM size -- so those flags cannot participate. Accepting them
        # silently would answer a different question than the one typed.
        cmd_check(accepting_cluster,
                  _args(["check", "-g", "1", "--needs", "fp8", "--gpu-mem", "40"]),
                  PLAIN)
        out = _prose(capsys.readouterr().out)
        assert "--needs fp8" in out
        assert "--gpu-mem 40" in out
        assert "took no part in this check" in out

    def test_no_such_note_when_the_flags_are_absent(self, accepting_cluster, capsys):
        cmd_check(accepting_cluster, _args(["check", "-g", "1"]), PLAIN)
        assert "took no part" not in _prose(capsys.readouterr().out)

    def test_nothing_to_check_is_stated_not_left_blank(self, cpu_only_cluster, capsys):
        # A GPU shape on a cluster with no accelerators has no candidate queue.
        rc = cmd_check(cpu_only_cluster, _args(["check", "-g", "4"]), PLAIN)
        out = _prose(capsys.readouterr().out)
        assert rc == 1
        assert "nothing to check" in out

    def test_an_explicit_queue_list_is_honoured(self, accepting_cluster, capsys):
        cmd_check(accepting_cluster,
                  _args(["--json", "check", "-q", "beta", "-g", "1"]), PLAIN)
        assert set(json.loads(capsys.readouterr().out)["queues"]) == {"beta"}


class TestWhereWithProbe:
    def test_a_probe_refusal_makes_the_placement_unreachable(
        self, refusing_cluster, capsys
    ):
        cmd_where(refusing_cluster,
                  _args(["--json", "where", "-g", "1", "--all"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        assert rows
        assert all(not r["reachable"] for r in rows)

    def test_a_probe_acceptance_marks_the_placement_confirmed(
        self, accepting_cluster, capsys
    ):
        cmd_where(accepting_cluster,
                  _args(["--json", "where", "-g", "1"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        assert any(r["confirmed"] for r in rows)
        assert all(not r["entitlement_unconfirmed"] for r in rows)


class TestAccessColumn:
    """The ACCESS column earns its width only when it varies."""

    def test_it_is_omitted_when_no_probe_was_run(self, accepting_cluster, capsys):
        # The same word on all nineteen rows is noise, not information -- and
        # the column's absence is now the whole statement. There used to be a
        # sentence underneath saying "access UNCHECKED, add --check to
        # confirm"; that repeated what the column would have said and was
        # printed whether or not anything had been withheld.
        # --declared: probing is the default now, so "no probe was run" has to
        # be asked for.
        cmd_where(accepting_cluster, _args(["where", "-g", "1", "--declared"]),
                  PLAIN)
        out = capsys.readouterr().out
        assert "access" not in out
        assert "unchecked" not in out.lower()

    def test_it_appears_once_a_probe_has_answered(self, accepting_cluster, capsys):
        cmd_where(accepting_cluster, _args(["where", "-g", "1"]), PLAIN)
        out = capsys.readouterr().out
        assert "access" in out
        assert "confirmed" in out

    def test_the_column_says_unchecked_when_it_is_shown(self, capsys):
        # Shown because a group-owned queue makes it vary -- and then the word
        # in the cell is the whole message. No accompanying sentence.
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Node, Queue

        node = Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                    gpus_total=4, queues=("mine", "theirs"))
        from nodetop.core.model import BackendCapabilities

        cluster = Cluster(
            backend_name="slurm", queue_term="partition", nodes=[node],
            capabilities=BackendCapabilities(
                probe=True, probe_supported=True,
                probe_command="sbatch --test-only"),
            queues={
                "mine": Queue(name="mine", node_names=("n1",), declared_nodes=1,
                              nodes=[node]),
                "theirs": Queue(name="theirs", node_names=("n1",),
                                declared_nodes=1, nodes=[node],
                                allow_accounts=("pi-them",)),
            },
        )
        cmd_where(cluster, _args(["where", "-g", "1"]), PLAIN)
        out = capsys.readouterr().out
        assert "access" in out
        assert "unchecked" in out
        assert "group-only" in out
        assert "add --check" not in out   # no sentence restating the column

    def test_a_backend_with_no_dry_run_says_declared_instead(self, capsys):
        from nodetop.backends.pbs import PbsBackend
        from nodetop.core.cluster import Cluster
        from nodetop.runner import RecordedRunner

        backend = PbsBackend(RecordedRunner({
            "pbsnodes": (0, '{"nodes":{"n1":{"state":"free","resources_available":'
                            '{"ncpus":8,"Qlist":"q"},"resources_assigned":{}}}}', ""),
            "qstat": (0, "Queue: q\n    enabled = True\n    started = True\n", ""),
        }))
        cluster = Cluster.load(backend, with_free_times=False)
        cmd_where(cluster, _args(["where", "-c", "2"]), PLAIN)
        out = _prose(capsys.readouterr().out)
        # Not "unchecked": there is nothing to check, which is a different fact.
        assert "DECLARED, not confirmed" in out
        assert "--check" not in out


class TestJsonCarriesTheCaveats:
    """A caveat only in the text is a caveat a script never learns."""

    def _json(self, cluster, capsys, argv):
        cmd_check(cluster, _args(argv), PLAIN)
        return json.loads(capsys.readouterr().out)

    def test_the_uncheckable_flags_are_listed(self, accepting_cluster, capsys):
        data = self._json(accepting_cluster, capsys,
                          ["--json", "check", "-g", "1", "--needs", "fp8"])
        assert any("took no part" in n for n in data["not_covered"])

    def test_the_backend_caveats_are_listed(self, accepting_cluster, capsys):
        data = self._json(accepting_cluster, capsys, ["--json", "check", "-g", "1"])
        # sbatch --test-only does not evaluate QOS ceilings; a script deciding
        # whether to submit needs to know that.
        assert any("QOS ceilings" in n for n in data["not_covered"])

    def test_a_filter_scheduler_disagreement_is_flagged(
        self, disagreeing_cluster, capsys
    ):
        data = self._json(disagreeing_cluster, capsys, ["--json", "check", "-g", "1"])
        assert data["filter_scheduler_disagreements"]

    def test_the_counts_are_present(self, accepting_cluster, capsys):
        data = self._json(accepting_cluster, capsys, ["--json", "check", "-g", "1"])
        assert data["asked"] >= 1
        assert data["accepted"] == data["asked"]

    def test_text_and_json_report_the_same_caveats(self, accepting_cluster, capsys):
        cmd_check(accepting_cluster,
                  _args(["check", "-g", "1", "--needs", "fp8"]), PLAIN)
        text = " ".join(capsys.readouterr().out.split())
        data = self._json(accepting_cluster, capsys,
                          ["--json", "check", "-g", "1", "--needs", "fp8"])
        for note in data["not_covered"]:
            # Same source, so every JSON note must appear in the prose.
            assert note.split(":")[0] in text


class TestWhyEntitlementIsUnconfirmed:
    """Three different reasons, three different remedies.

    Asserting "this system has no dry-run" when the truth is "its client is not
    installed" or "this is a recording" is a false statement about the
    scheduler. The function handled the recording case only, so on a host
    without `kubectl` it announced that *Kubernetes* has no verify-only
    submission -- the one backend here whose server-side dry-run runs real
    admission including ResourceQuota, and which says exactly that in its own
    capability notes.
    """

    @staticmethod
    def _cluster(*, backend="slurm", supported=False, here=False, replayed=False,
                 command=""):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import BackendCapabilities

        return Cluster(
            backend_name=backend,
            replayed=replayed,
            capabilities=BackendCapabilities(
                probe=here, probe_supported=supported, probe_command=command),
        )

    def _why(self, **kw):
        from nodetop.cli import _why_no_probe

        return _why_no_probe(self._cluster(**kw))

    def test_a_system_with_no_dry_run_says_so(self):
        why = self._why(backend="pbs", supported=False)
        assert "no verify-only submission" in why
        assert "pbs" in why

    def test_a_missing_client_is_not_reported_as_a_missing_feature(self):
        why = self._why(backend="kubernetes", supported=True, here=False,
                        command="kubectl auth can-i + --dry-run=server")
        assert "no verify-only submission" not in why
        assert "can dry-run" in why
        assert "kubectl is not on PATH" in why

    def test_it_names_the_binary_not_the_whole_command(self):
        why = self._why(backend="slurm", supported=True, here=False,
                        command="sbatch --test-only")
        assert "sbatch is not on PATH" in why

    def test_a_recording_says_it_is_a_recording(self):
        # Outranks both: a replay cannot dry-run whatever the client situation.
        why = self._why(backend="slurm", supported=True, here=True,
                        replayed=True, command="sbatch --test-only")
        assert "replayed snapshot" in why
        assert "not on PATH" not in why

    def test_a_command_with_no_text_still_yields_a_sentence(self):
        why = self._why(backend="sge", supported=True, here=False, command="")
        assert "sge" in why
        assert why.strip()

    def test_no_capabilities_at_all_falls_back_safely(self):
        from nodetop.cli import _why_no_probe
        from nodetop.core.cluster import Cluster

        why = _why_no_probe(Cluster(backend_name="mystery"))
        assert "mystery" in why


class TestGroupOnlyIsMarkedInWhere:
    """`where` must not report RUN NOW for somebody else's hardware.

    Measured on a live cluster: `where -g 1` listed five partitions and called
    four of them RUN NOW. A dry-run then refused all but one. Three of the four
    were group-private and are now marked; the fourth is the honest limit of a
    structural reading -- see TestTheHeuristicHasAKnownBlindSpot below.
    """

    #: Deliberately adversarial. `Placement.score` orders by
    #: `-nodes_available` and then by NAME, so the private partition is given
    #: both more nodes and an alphabetically earlier name: without the
    #: shared-first nudge it ranks FIRST. The previous fixture gave both one
    #: node and called them "openq" / "pi-theirs", where the alphabet alone
    #: produced the wanted order -- so deleting the sort survived the test.
    SHARED = "zopen"
    PRIVATE = "agroup"

    @classmethod
    def _cluster(cls):
        from nodetop.core.cluster import Cluster
        from nodetop.core.model import Identity, Node, Queue

        nodes, queues = [], {}
        for name, allow, count in ((cls.SHARED, (), 1),
                                   (cls.PRIVATE, ("pi-theirs",), 3)):
            mine = [
                Node(name=f"n-{name}-{i}", state_raw="IDLE", cpus_total=16,
                     memory_mb=64000, gpus_total=4, queues=(name,))
                for i in range(count)
            ]
            nodes += mine
            queues[name] = Queue(
                name=name, node_names=tuple(n.name for n in mine),
                declared_nodes=count, nodes=mine, allow_accounts=allow)
        return Cluster(
            backend_name="synthetic", queue_term="partition",
            nodes=nodes, queues=queues,
            identity=Identity(user="me", accounts=("a", "pi-theirs"), qos=("q",)),
        )

    def _out(self, capsys, extra=()):
        cmd_where(self._cluster(),
                  _args(["where", "-g", "1", *extra]), PLAIN)
        return capsys.readouterr().out

    def test_the_access_column_appears_without_a_probe(self, capsys):
        # It used to appear only when probed, on the grounds that one repeated
        # word is noise. Group ownership makes the column vary, which is the
        # case that matters.
        assert "access" in self._out(capsys)

    def test_the_group_partition_is_marked(self, capsys):
        # On its ROW, not merely somewhere in the output: the legend also
        # contains the phrase, so a whole-output assertion passed even with the
        # cell removed.
        out = self._out(capsys)
        row = next(ln for ln in out.splitlines()
                   if self.PRIVATE in ln and "allows 1-2" not in ln)
        assert "group-only" in row

    def test_the_shared_partition_is_not(self, capsys):
        out = self._out(capsys)
        shared_row = next(ln for ln in out.splitlines() if self.SHARED in ln)
        assert "group-only" not in shared_row

    def test_the_marker_is_explained(self, capsys):
        out = " ".join(self._out(capsys).split())
        assert "group-only" in out  # the marker itself; no glossary line any more

    def test_the_shared_partition_is_listed_first(self, capsys):
        # Both can run the job now; only one of them can be submitted to. The
        # private one has four times the free GPUs, so a capacity-only order
        # puts it first -- which is what this asserts against.
        out = self._out(capsys)
        assert out.index(self.SHARED) < out.index(self.PRIVATE)

    def test_the_default_ranking_would_have_put_the_private_one_first(self):
        # Guards the guard: without this, the assertion above could hold purely
        # because the default order already agreed, and removing the nudge
        # would go unnoticed.
        from nodetop.core.fit import rank
        from nodetop.core.model import JobShape

        cluster = self._cluster()
        ordered = rank(cluster, JobShape(nodes=1, gpus_per_node=1))
        assert ordered[0].queue == self.PRIVATE

    def test_a_confirmed_verdict_overrides_the_guess(self, capsys):
        # If a probe says you ARE in the group, the partition is not
        # second-class: the guess must not outrank the control plane, and the
        # private partition's larger capacity should then win.
        import dataclasses

        from nodetop.core.model import (
            BackendCapabilities,
            Verdict,
            VerdictCategory,
        )

        class _Accepting:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub --test-only")

            def probe(self, queue, shape, account=None):
                return Verdict(queue=queue, account=account, allowed=True,
                               category=VerdictCategory.OK, reason="ok")

            def submit_flags(self, queue, shape):
                return []

        cluster = dataclasses.replace(
            self._cluster(),
            capabilities=_Accepting().capabilities(),
            _backend=_Accepting(),
        )
        cmd_where(cluster, _args(["where", "-g", "1"]), PLAIN)
        out = capsys.readouterr().out
        assert "confirmed" in out
        assert "group-only" not in out
        # And the nudge stops applying: with the control plane saying yes to
        # both, the private partition's three nodes win and rank()'s own order
        # stands. Asserting this is what makes the override testable at all --
        # the marker disappearing is decided by the verdict branch, not by the
        # ordering key.
        assert out.index(self.PRIVATE) < out.index(self.SHARED)


class TestTheHeuristicHasAKnownBlindSpot:
    """A queue can declare openness on every axis and still refuse.

    Measured: the live `gpu` partition has an empty account allowlist and a QOS
    allowlist that intersects the caller's, so nothing structural marks it --
    and the dry-run returns NOT_ENTITLED anyway, because the accounting
    database claims the same 92 QOS entries for all 34 of the caller's
    accounts. This is recorded as a test so the heuristic is never mistaken for
    a substitute for `--check`.
    """

    @staticmethod
    def _open_but_refusing():
        from nodetop.core.model import Queue

        return Queue(name="gpu", allow_accounts=(), allow_qos=("gpu", "debug"))

    def test_such_a_queue_is_not_flagged(self):
        assert self._open_but_refusing().is_dedicated is False

    def test_and_that_is_the_correct_structural_answer(self):
        # Not a bug to fix by widening the heuristic: the queue really is open
        # by declaration. The lie is in the association dump, and only the
        # control plane can expose it.
        q = self._open_but_refusing()
        assert not q.allow_accounts
        assert q.allow_qos


class TestUnknownCapabilitiesAreNotReportedAsAbsent:
    def test_it_says_unknown_rather_than_none(self):
        from nodetop.cli import _why_no_probe
        from nodetop.core.cluster import Cluster

        why = _why_no_probe(Cluster(backend_name="slurm"))
        # Slurm HAS a dry-run; saying it does not, because we failed to read
        # the capability block, is the same false claim by another route.
        assert "no verify-only submission" not in why
        assert "could not be read" in why


class TestUnansweredIsNotRefused:
    """Nine sites read `verdict.allowed` and only three checked `durable`.

    Found the first three by accident, the rest by grepping every reader. These
    are the remaining ones that carried a consequence: the ordering key, the
    ACCESS cell's colour, the detail tree's tag, and `check`'s accepted count.
    """

    @staticmethod
    def _place(category):
        from nodetop.core.capacity import Capacity
        from nodetop.core.fit import Placement
        from nodetop.core.model import JobShape, Verdict

        return Placement(
            queue="q", shape=JobShape(nodes=1),
            capacity=Capacity(considered=1, required_nodes=1,
                              hardware_nodes=("n",), capable_nodes=("n",)),
            verdict=Verdict(queue="q", allowed=False, category=category,
                            reason="x"),
        )

    def test_rank_already_puts_the_confirmed_one_first(self, capsys):
        # Not a `_second_class` test: `Placement.score` orders confirmed ahead
        # of unconfirmed, which is why re-checking refusals in the ordering key
        # was redundant. `big` has four times the capacity and an unanswered
        # probe; `small` is confirmed and still leads.
        cmd_where(self._mixed_cluster(), _args(["where", "-c", "1"]), PLAIN)
        out = capsys.readouterr().out
        assert out.index("small") < out.index("big")

    def test_an_unanswered_partition_is_still_listed(self, capsys):
        # It is not filtered out, because nothing was established about it.
        cmd_where(self._mixed_cluster(), _args(["where", "-c", "1"]), PLAIN)
        assert "big" in capsys.readouterr().out

    @staticmethod
    def _mixed_cluster():
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

        nodes, queues = [], {}
        for name, count in (("big", 4), ("small", 1)):
            mine = [Node(name=f"{name}{i}", state_raw="IDLE", cpus_total=8,
                         memory_mb=16000, queues=(name,)) for i in range(count)]
            nodes += mine
            queues[name] = Queue(name=name, declared_nodes=count, nodes=mine,
                                 node_names=tuple(n.name for n in mine))

        class _Mixed:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub")

            def probe(self, q, shape, account=None):
                down = q == "big"
                return Verdict(
                    queue=q, allowed=not down,
                    category=(VerdictCategory.CONTROL_PLANE_DOWN if down
                              else VerdictCategory.OK),
                    reason="x")

            def submit_flags(self, q, shape):
                return []

        return dataclasses.replace(
            Cluster(backend_name="synthetic", queue_term="partition",
                    nodes=nodes, queues=queues,
                    identity=Identity(user="me", accounts=("mine",),
                                      qos=("x",))),
            capabilities=_Mixed().capabilities(), _backend=_Mixed())

    def test_the_access_cell_warns_rather_than_denying(self, capsys):
        # Rendered, not re-derived: the first version of this test rebuilt the
        # same conditional it was meant to be checking, so it would have passed
        # against any implementation including the broken one.
        from nodetop.core.model import VerdictCategory, category_label
        from nodetop.render import Glyphs, Style

        colour = Style(depth=24, glyphs=Glyphs())
        cmd_where(self._downed_cluster(), _args(["where", "-g", "1"]), colour)
        transient = capsys.readouterr().out

        # The DISPLAY form of the category, and not the wire form: the cell sits
        # next to `confirmed` and `unchecked`, and painting the enum member
        # through put two vocabularies in one column.
        label = category_label(VerdictCategory.CONTROL_PLANE_DOWN)
        assert label in transient
        assert "CONTROL_PLANE_DOWN" not in transient
        # The warn colour, not the bad one: red on an outage reads as "denied".
        warn_seq = colour.warn("x")[: colour.warn("x").index("x")]
        bad_seq = colour.bad("x")[: colour.bad("x").index("x")]
        assert warn_seq != bad_seq
        row = next(ln for ln in transient.splitlines() if label in ln)
        assert warn_seq in row
        assert row.index(warn_seq) < row.index(label)

    def test_check_counts_unanswered_apart_from_refused(self, capsys):
        # "1 of 3 accepted" hides that one of the other two was never asked.
        cmd_check(self._downed_cluster(),
                  _args(["--json", "check", "-q", "alpha", "-g", "1"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert "unanswered" in data
        assert data["unanswered"] == 1
        assert data["accepted"] == 0

    def test_the_exit_status_is_still_strict(self, capsys):
        # Not accepted is not accepted: an unanswered probe must not wave a
        # `nodetop check && sbatch` caller through.
        rc = cmd_check(self._downed_cluster(),
                       _args(["check", "-q", "alpha", "-g", "1"]), PLAIN)
        capsys.readouterr()
        assert rc == 1

    @staticmethod
    def _downed_cluster():
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

        node = Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=16000,
                    gpus_total=4, queues=("alpha",))
        queue = Queue(name="alpha", node_names=("n1",), declared_nodes=1,
                      nodes=[node])

        class _Down:
            name = "synthetic"
            queue_term = "partition"

            def capabilities(self):
                return BackendCapabilities(probe=True, probe_supported=True,
                                           probe_command="stub")

            def probe(self, q, shape, account=None):
                return Verdict(queue=q, allowed=False,
                               category=VerdictCategory.CONTROL_PLANE_DOWN,
                               reason="unreachable")

            def submit_flags(self, q, shape):
                return []

        return dataclasses.replace(
            Cluster(backend_name="synthetic", queue_term="partition",
                    nodes=[node], queues={"alpha": queue},
                    identity=Identity(user="me", accounts=("mine",),
                                      qos=("x",))),
            capabilities=_Down().capabilities(), _backend=_Down())
