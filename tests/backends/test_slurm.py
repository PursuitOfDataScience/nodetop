"""Slurm -- the reference backend, validated against a live cluster."""

from __future__ import annotations

import pytest
from fixtures.slurm import probe_outputs as po

from nodetop.backends.slurm import SlurmBackend, parse_probe, parse_slurm_duration
from nodetop.core.model import JobShape, VerdictCategory
from nodetop.runner import RecordedRunner

B1 = "beagle3-0001"
BIGMEM = "beagle3-bigmem1"
DEGRADED = "midway3-0385"


def _parse(text, queue="p", account="a", rc=1):
    # sbatch writes its verdict to stderr and exits non-zero even when the
    # answer is informative, so that is the realistic call shape.
    return parse_probe(queue, account, rc, "", text)


class TestSlurmDuration:
    @pytest.mark.parametrize("text,seconds", [
        ("60", 3600),          # bare number is minutes
        ("2:00", 120),         # MM:SS, not HH:MM
        ("1:00:00", 3600),
        ("2-00:00:00", 172800),
        ("2-12", 216000),      # D-HH
        ("UNLIMITED", None),
    ])
    def test_grammar(self, text, seconds):
        assert parse_slurm_duration(text) == seconds


class TestNodes:
    def test_all_records(self, slurm_backend):
        assert len(slurm_backend.load_nodes()) == 9

    def test_fields(self, slurm_backend):
        n = {x.name: x for x in slurm_backend.load_nodes()}[B1]
        assert (n.cpus_total, n.cpus_alloc) == (32, 32)
        assert n.memory_mb == 256000
        assert n.gpus_total == 4
        assert n.accelerator.model == "A100"

    def test_compound_state_is_decomposed(self, slurm_backend):
        # "DOWN*+DRAIN" is a base state, a not-responding marker and a flag
        # fused into one token.
        n = {x.name: x for x in slurm_backend.load_nodes()}[DEGRADED]
        assert n.state_raw == "DOWN*+DRAIN"
        assert n.conditions == frozenset({"DOWN", "DRAIN"})
        assert n.unreachable is True
        assert n.schedulable is False

    def test_reason_with_spaces_is_not_truncated(self, slurm_backend):
        n = {x.name: x for x in slurm_backend.load_nodes()}[DEGRADED]
        assert n.reason.startswith("maintenance: hardware issue")

    def test_accelerator_ness_is_from_gres_not_hostname(self, slurm_backend):
        # This host sits among 44 accelerator nodes and has none.
        n = {x.name: x for x in slurm_backend.load_nodes()}[BIGMEM]
        assert n.name.startswith("beagle3")
        assert n.is_gpu_node is False


class TestPartitions:
    def _queues(self, slurm_backend):
        return {q.name: q for q in slurm_backend.load_queues()}

    def test_hidden_partitions_are_included(self, slurm_backend):
        # A hidden partition is exactly the kind that has quietly been taken
        # out of service, so omitting it hides the diagnosis.
        assert self._queues(slurm_backend)["test"].hidden is True

    def test_all_four_kill_switches(self, slurm_backend):
        q = self._queues(slurm_backend)["test"]
        codes = {b.code for b in q.structural_blockers()}
        assert "QUEUE_DISABLED" in codes
        assert "NO_ACCOUNTS" in codes
        assert "NO_QOS" in codes
        assert q.hidden is True
        assert q.usable is False

    def test_the_dead_partition_still_declares_its_nodes(self, slurm_backend):
        assert self._queues(slurm_backend)["test"].declared_nodes == 610

    def test_bracket_notation_in_an_unusual_position(self, slurm_backend):
        # Nodes=beagle3-00[01-44]
        assert len(self._queues(slurm_backend)["beagle3"].node_names) == 44

    def test_allow_accounts(self, slurm_backend):
        assert "rcc-staff" in self._queues(slurm_backend)["beagle3"].allow_accounts

    def test_drain_state_accepts_but_does_not_start(self):
        # Slurm has one switch where PBS has two; DRAIN is the
        # accepts-but-never-starts case.
        b = SlurmBackend(RecordedRunner({"scontrol show partition": (
            0, "PartitionName=d\n   State=DRAIN AllowGroups=ALL\n   TotalNodes=1\n", ""
        )}))
        q = b.load_queues()[0]
        assert (q.enabled, q.started) == (True, False)


class TestQos:
    def test_ceilings(self, slurm_backend):
        limits = slurm_backend.load_limits()["beagle3"]
        assert limits.max_walltime_seconds == 172800
        assert limits.per_job == {"gpu": 4}
        assert limits.per_user == {"cpu": 256, "gpu": 32, "node": 8}

    def test_empty_columns_do_not_raise(self, slurm_backend):
        # "test|7-00:00:00|||||100000000|" -- five empty fields in a row.
        limits = slurm_backend.load_limits()["test"]
        assert limits.max_walltime_seconds == 604800
        assert limits.per_job == {}

    def test_the_ceiling_the_dry_run_misses(self, slurm_backend):
        # sbatch --test-only returns PASSED with a start time for this shape,
        # and it then pends indefinitely.
        limits = slurm_backend.load_limits()["beagle3"]
        got = limits.blockers(JobShape(nodes=3, gpus_per_node=4, walltime="1h"))
        assert "MAX_GPU_JOB" in {b.code for b in got}


class TestProbeAccepted:
    def test_allowed(self):
        v = _parse(po.ACCEPTED, rc=0)
        assert v.allowed is True
        assert v.category == VerdictCategory.OK

    def test_predicted_start_and_nodes(self):
        v = _parse(po.ACCEPTED, rc=0)
        assert v.predicted_start.isoformat() == "2026-08-21T17:00:12"
        assert v.predicted_nodes == ("beagle3-0006",)

    def test_effective_qos_is_what_the_controller_chose(self):
        # Asked for "beagle3"; the site auto-promoted to "beagle3-prio".
        # Checking ceilings against the requested name checks the wrong ones.
        v = _parse(po.ACCEPTED, rc=0)
        assert v.effective_qos == "beagle3-prio"
        assert v.effective_account == "rcc-staff"

    def test_shared_partition_preamble(self):
        v = _parse(po.SHARED_PARTITION, rc=0)
        assert v.allowed is True
        assert v.predicted_nodes == ("midway3-0278",)


class TestTwoLayerDisagreement:
    """The case a single-source check gets exactly backwards."""

    def test_filter_passed_but_core_refused_is_not_allowed(self):
        v = _parse(po.PLUGIN_PASSED_CORE_REFUSED)
        assert v.filter_verdict == "PASSED"
        assert v.allowed is False
        assert v.category == VerdictCategory.ACCOUNT_MISMATCH

    def test_the_disagreement_is_surfaced(self):
        v = _parse(po.PLUGIN_PASSED_CORE_REFUSED)
        assert "site submit filter reported PASSED" in v.reason

    def test_filter_rejection_is_classified_from_its_own_reason(self):
        v = _parse(po.PLUGIN_REJECTED)
        assert v.filter_verdict == "REJECTED"
        assert v.category == VerdictCategory.NOT_ENTITLED
        assert "pi-pedramh" in v.reason


class TestProbeTaxonomy:
    @pytest.mark.parametrize("text,category", [
        (po.NO_ACCOUNT, VerdictCategory.NO_ACCOUNT),
        (po.SHAPE_TOO_BIG, VerdictCategory.SHAPE_UNAVAILABLE),
        (po.CONTROLLER_IO_ERROR, VerdictCategory.CONTROL_PLANE_DOWN),
        (po.STOCK_QOS_VIOLATION, VerdictCategory.QUOTA_EXCEEDED),
        (po.STOCK_BAD_PARTITION, VerdictCategory.UNKNOWN_QUEUE),
        (po.STOCK_NODE_CONFIG, VerdictCategory.SHAPE_UNAVAILABLE),
        (po.STOCK_TIME_LIMIT, VerdictCategory.TIME_LIMIT),
    ])
    def test_categories(self, text, category):
        assert _parse(text).category == category

    def test_a_controller_outage_is_not_a_statement_about_access(self):
        # A controller that cannot write job scripts fails EVERY submission,
        # including a bare --wrap=hostname.
        v = _parse(po.CONTROLLER_IO_ERROR)
        assert v.allowed is False
        assert v.durable is False

    def test_membership_refusal_is_durable(self):
        assert _parse(po.PLUGIN_REJECTED).durable is True

    def test_unknown_output_is_labelled_unknown_not_guessed(self):
        assert _parse("something nobody has seen").category == VerdictCategory.UNKNOWN


class TestProbeInvocation:
    def test_test_only_is_hard_coded(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        SlurmBackend(runner).probe("beagle3", JobShape(gpus_per_node=1), "acct")
        cmd = runner.calls[0]
        assert cmd[0] == "sbatch"
        assert "--test-only" in cmd

    def test_shape_becomes_sbatch_flags(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        shape = JobShape(nodes=2, gpus_per_node=4, cpus_per_task=8,
                         memory_gb=64, walltime="2-00:00:00")
        SlurmBackend(runner).probe("beagle3", shape, "acct")
        joined = " ".join(runner.calls[0])
        for flag in ["--nodes=2", "--gres=gpu:4", "--cpus-per-task=8",
                     "--time=2-00:00:00", "--account=acct", "--partition=beagle3"]:
            assert flag in joined

    def test_the_probe_account_wins_over_the_shape_account(self):
        runner = RecordedRunner({"sbatch": (0, "", po.ACCEPTED)})
        SlurmBackend(runner).probe("beagle3", JobShape(account="from-shape"), "from-probe")
        joined = " ".join(runner.calls[0])
        assert "--account=from-probe" in joined
        assert "--account=from-shape" not in joined

    def test_a_runner_failure_becomes_a_finding_not_a_crash(self):
        v = SlurmBackend(RecordedRunner({})).probe("beagle3", JobShape())
        assert v.category == VerdictCategory.CONTROL_PLANE_DOWN
        assert v.durable is False


class TestNodelist:
    def test_collapsed_bracket_notation(self, slurm_backend):
        names = ["midway3-0298", "midway3-0377", "midway3-0378", "midway3-0423"]
        assert slurm_backend.format_nodelist(names) == "midway3-[0298,0377-0378,0423]"


class TestTresMemoryScale:
    """Slurm TRES memory defaults to MB, with an optional suffix."""

    @pytest.mark.parametrize("text,mb", [
        ("mem=500K", 0),          # ~0.49 MB
        ("mem=2048K", 2),
        ("mem=500M", 500),
        ("mem=4G", 4096),
        ("mem=1T", 1024 ** 2),
        ("mem=100", 100),         # no suffix means MB
    ])
    def test_scales(self, text, mb):
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map(text)["mem_mb"] == mb

    def test_a_kilobyte_ceiling_is_not_read_as_megabytes(self):
        # An integer 1 // 1024 is 0, and dodging that with `or 1` makes K equal
        # M -- so a mem=500K ceiling reads 1024x too large, and an over-limit
        # job is never flagged.
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map("mem=500K")["mem_mb"] < _parse_tres_map("mem=500M")["mem_mb"]

    def test_an_over_limit_job_is_flagged_against_a_kilobyte_ceiling(self):
        from nodetop.backends.slurm import _parse_tres_map
        from nodetop.core.model import JobShape, Limits

        limits = Limits(name="q", per_job=_parse_tres_map("mem=500K"))
        codes = {b.code for b in limits.blockers(JobShape(memory_gb=1))}
        assert "MAX_MEM_MB_JOB" in codes

    def test_other_resources_are_unaffected(self):
        from nodetop.backends.slurm import _parse_tres_map

        assert _parse_tres_map("cpu=8,mem=2G,gres/gpu=4") == {
            "cpu": 8, "mem_mb": 2048, "gpu": 4
        }


class TestIdentityQuery:
    """The argv shape matters, and getting it wrong fails silently."""

    def test_where_and_the_condition_are_separate_arguments(self):
        # Passing "where user=X" as one element makes sacctmgr answer
        # `Unknown condition`, so the identity comes back empty -- and an
        # empty identity silently disables every account and QOS access check,
        # because those are tri-state and read "nothing to compare against" as
        # "no verdict". Nothing errors; a whole analysis layer just stops.
        runner = RecordedRunner({"sacctmgr": (0, "acct||beagle3\n", "")})
        SlurmBackend(runner).load_identity()
        argv = next(c for c in runner.calls if c[0] == "sacctmgr")
        assert "where" in argv
        assert any(a.startswith("user=") for a in argv)
        assert not any(a.startswith("where user=") for a in argv)

    def test_the_query_asks_for_the_fields_it_parses(self):
        runner = RecordedRunner({"sacctmgr": (0, "", "")})
        SlurmBackend(runner).load_identity()
        argv = next(c for c in runner.calls if c[0] == "sacctmgr")
        assert "format=Account,Partition,QOS" in argv

    def test_a_rejected_query_is_raised_not_swallowed(self):
        # It used to return an empty identity, and that is indistinguishable
        # from a user who genuinely holds no associations. The account and QOS
        # checks downstream are tri-state and read "nothing to compare against"
        # as "no verdict", so a failed query silently disabled all of them.
        # Measured consequence: with `sacctmgr` down, all 34 accounts vanished,
        # every probe ran with no --account, and the overview reported
        # "0 open to you, 83 refused" -- total loss of access, asserted
        # confidently, during a database hiccup.
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({"sacctmgr": (1, "", "Unknown condition")})
        with pytest.raises(CommandError):
            SlurmBackend(runner).load_identity()

    def test_the_cluster_records_it_rather_than_crashing(self):
        # Raising is only correct because Cluster.load catches it: the report
        # still renders, `identity` is None, and the entitlement filter already
        # treats that as "cannot filter" rather than "entitled to nothing".
        from nodetop.core.cluster import Cluster

        runner = RecordedRunner({"sacctmgr": (1, "", "Unknown condition")})
        cluster = Cluster.load(SlurmBackend(runner))
        assert cluster.identity is None
        assert "identity" in cluster.errors

    def test_a_real_association_dump_is_parsed(self):
        # Two accounts, an identical queue list: the templated-entitlements
        # signal this backend exists to surface.
        dump = (
            "acct-a||aaz,beagle3,caslake\n"
            "acct-b||aaz,beagle3,caslake\n"
        )
        ident = SlurmBackend(RecordedRunner({"sacctmgr": (0, dump, "")})).load_identity()
        assert set(ident.accounts) == {"acct-a", "acct-b"}
        assert "beagle3" in ident.qos
        assert ident.entitlements_look_templated is True


class TestLocalGroupLookupIsAllOrNothing:
    """A half-collected group list is worse than none at all.

    The membership check downstream is tri-state: an EMPTY set reads as "cannot
    tell" and yields no verdict, while a NON-EMPTY one is taken as
    authoritative. So a lookup that gathered the supplementary groups and then
    failed on the primary had the tool assert "none of your groups are permitted
    here" from a list it knew to be incomplete -- a false denial, which hides a
    queue the caller can use.
    """

    def test_a_failure_on_the_primary_group_yields_nothing(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.slurm import _unix_groups

        class _G:
            gr_name = "supplementary"
            gr_mem = ["testuser"]

        monkeypatch.setattr(grp_mod, "getgrall", lambda: [_G()])
        import pwd as pwd_mod
        monkeypatch.setattr(pwd_mod, "getpwnam",
                            lambda _u: (_ for _ in ()).throw(KeyError("no such user")))
        # Not ("supplementary",): that would be a verdict from a partial read.
        assert _unix_groups("testuser") == ()

    def test_a_complete_lookup_returns_every_group(self, monkeypatch):
        import grp as grp_mod
        import pwd as pwd_mod

        from nodetop.backends.slurm import _unix_groups

        class _G:
            def __init__(self, name, mem):
                self.gr_name, self.gr_mem = name, mem

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: [_G("sup", ["testuser"]), _G("other", ["nobody"])])
        monkeypatch.setattr(pwd_mod, "getpwnam", lambda _u: type("P", (), {"pw_gid": 7})())
        monkeypatch.setattr(grp_mod, "getgrgid",
                            lambda _g: type("G", (), {"gr_name": "primary"})())
        assert _unix_groups("testuser") == ("primary", "sup")


class TestTheExitStatusOutranksTheText:
    """`sbatch --test-only` exits 0 if and only if the job would be accepted.

    That makes the return code the one authoritative, wording-independent
    signal, and `filter_verdict == "PASSED"` used to be sufficient for
    `allowed=True` on its own -- so a run that exited 1 while its site plugin
    printed PASSED was reported as allowed. It inverts the reliability ordering
    this module is built on: the filter's PASSED is exactly the claim documented
    as generous fiction.

    It was reachable because the core-refusal branch recognises only two message
    prefixes (`allocation failure:`, `Batch job submission failed:`). Any other
    refusal wording -- Slurm has many, and site plugins add their own -- fell
    through and was granted.
    """

    def test_a_nonzero_exit_is_not_an_acceptance(self):
        v = parse_probe("q", "a", 1, "sbatch: error: Verification: ***PASSED***\n", "")
        assert not v.allowed

    def test_it_is_unsettled_rather_than_a_durable_refusal(self):
        # We know it failed; we do not know why. Claiming a durable refusal
        # would assert more than the output established, and that is what gets a
        # usable partition hidden.
        v = parse_probe("q", "a", 1, "sbatch: error: Verification: ***PASSED***\n", "")
        assert not v.durable

    def test_a_predicted_start_does_not_outvote_a_failure(self):
        # Contradictory output -- a start time printed, then a non-zero exit.
        # The conservative reading is the one this module commits to everywhere.
        out = ("sbatch: Verification: ***PASSED***\n"
               "sbatch: Job 1 to start at 2026-08-23T10:00:00 using 1 processors\n")
        assert not parse_probe("q", "a", 1, out, "").allowed

    def test_a_genuine_acceptance_is_unaffected(self):
        out = ("sbatch: Verification: ***PASSED***\n"
               "sbatch: Job 1 to start at 2026-08-23T10:00:00 using 1 processors\n")
        v = parse_probe("q", "a", 0, out, "")
        assert v.allowed and v.confirmed
        assert v.predicted_start is not None

    def test_a_recognised_refusal_keeps_its_specific_category(self):
        # The rc veto sits after the core-refusal branch so a known message
        # still supplies the better category and reason.
        v = parse_probe(
            "q", "a", 1, "sbatch: error: Verification: ***PASSED***\n",
            "sbatch: error: Batch job submission failed: Invalid account or "
            "account/partition combination specified\n")
        assert not v.allowed and v.durable
        assert v.category == VerdictCategory.ACCOUNT_MISMATCH
        assert "site submit filter reported PASSED" in v.reason


class TestAnExhaustedAllocationIsNotADeniedOne:
    """Different problem, opposite remedy.

    Sites reject an empty allocation through the same channel as a permission
    failure -- the observed pair is `Reason: No sufficient SU allocations for a
    shared partition` with `allocation failure: Access/permission denied` on
    stderr -- so it fell through to ACCESS_DENIED and read as "you are not
    allowed here". The answer to one is to ask for service units; the answer to
    the other is to ask for access.
    """

    def test_su_exhaustion_is_a_quota(self):
        v = parse_probe(
            "caslake", "acct", 1,
            "sbatch: error: Verification: ***REJECTED***\n"
            "sbatch: error: Reason: No sufficient SU allocations for a shared partition\n",
            "allocation failure: Access/permission denied\n")
        assert v.category == VerdictCategory.QUOTA_EXCEEDED
        assert v.durable and not v.allowed
        # And the operator's own wording survives, since that is what says why.
        assert "SU allocations" in v.reason

    def test_a_plain_permission_denial_is_still_access_denied(self):
        # The generic pattern must keep working; the quota entry sits before it
        # in a first-match-wins table, which is the whole mechanism.
        v = parse_probe("q", "a", 1, "",
                        "sbatch: error: Batch job submission failed: "
                        "Access/permission denied\n")
        assert v.category == VerdictCategory.ACCESS_DENIED


class TestEitherOutputShapeParsesTheSame:
    """`scontrol` emits one record per line or one field per line, one flag apart.

    This backend asks for `--oneliner` when listing nodes and not when listing
    partitions, and each parser understood only the shape its own command
    happened to produce. Both failed silently on the other, in opposite
    directions:

    * partitions were split on blank lines, so oneliner input became a *single*
      record whose field dictionary kept only the last value for each key --
      2000 partitions collapsing to 1, with nothing reported;
    * nodes were read one per line, so multi-line input gave a record per line
      and every node came back with 0 CPUs and no state: a cluster that appears
      to own no resources at all.

    Neither is reachable while the argv here is fixed, which is precisely why it
    would go unnoticed. The way in is a replayed snapshot recorded where
    `scontrol` behaved differently, a site wrapper, or a version whose output
    changed shape -- and a parser that cannot tell "no records" from "one merged
    record" has no way to complain.
    """

    NODES_ONELINE = (
        "NodeName=n1 CPUAlloc=4 CPUTot=8 State=MIXED RealMemory=1000 Gres=gpu:2\n"
        "NodeName=n2 CPUAlloc=0 CPUTot=8 State=IDLE RealMemory=1000 Gres=gpu:2\n"
    )
    NODES_MULTILINE = (
        "NodeName=n1 CPUAlloc=4 CPUTot=8\n"
        "   State=MIXED RealMemory=1000\n"
        "   Gres=gpu:2\n"
        "\n"
        "NodeName=n2 CPUAlloc=0 CPUTot=8\n"
        "   State=IDLE RealMemory=1000\n"
        "   Gres=gpu:2\n"
    )

    def test_nodes_parse_the_same_either_way(self):
        b = SlurmBackend(RecordedRunner({}))
        a = [(n.name, n.cpus_total, n.cpus_free, n.gpus_total, n.state_raw)
             for n in b.parse_nodes(self.NODES_ONELINE)]
        c = [(n.name, n.cpus_total, n.cpus_free, n.gpus_total, n.state_raw)
             for n in b.parse_nodes(self.NODES_MULTILINE)]
        assert a == c
        assert a == [("n1", 8, 4, 2, "MIXED"), ("n2", 8, 8, 2, "IDLE")]

    def test_a_multiline_node_record_is_not_truncated(self):
        # The specific old failure: fields after the first line were dropped, so
        # every node read as 0 CPUs with no state.
        nodes = SlurmBackend(RecordedRunner({})).parse_nodes(self.NODES_MULTILINE)
        assert all(n.cpus_total == 8 and n.state_raw for n in nodes)

    def test_oneliner_partitions_do_not_collapse_into_one(self):
        text = "\n".join(
            f"PartitionName=p{i} AllowGroups=ALL AllowAccounts=ALL "
            f"State=UP MaxTime=UNLIMITED Nodes=n-{i}" for i in range(5))
        queues = SlurmBackend(RecordedRunner({})).parse_queues(text)
        assert [q.name for q in queues] == [f"p{i}" for i in range(5)]

    def test_the_real_multiline_fixture_still_parses(self, slurm_backend):
        # The format actually in use must be unaffected by the hardening.
        assert len(slurm_backend.load_queues()) == 3

    def test_the_header_keyword_inside_free_text_does_not_split_a_record(self):
        # A node's Reason is operator-authored, so it can contain anything. The
        # header is required at a line start for this reason.
        text = ("NodeName=n1 CPUTot=8 State=DOWN RealMemory=1000 "
                "Reason=replacing NodeName=n2 per ticket\n")
        nodes = SlurmBackend(RecordedRunner({})).parse_nodes(text)
        assert [n.name for n in nodes] == ["n1"]

    def test_empty_and_junk_input_yield_nothing_rather_than_a_phantom(self):
        b = SlurmBackend(RecordedRunner({}))
        for junk in ("", "\n\n", "no records here", "   \n   "):
            assert b.parse_nodes(junk) == []
            assert b.parse_queues(junk) == []
