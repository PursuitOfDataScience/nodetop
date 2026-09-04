"""The eight findings from the first cross-cluster evaluation (NT-1 .. NT-8).

Filed against 0.5.0 by installing it on a 1,614-node Slurm 23.02 cluster it was
not developed on -- CentOS 7.9, Python 3.14, cgroup v1, 84 partitions, 62 with
an association and ~20 the control plane refuses.  Seven of the eight are gaps
in a mechanism that already existed, which is why each test below asserts the
mechanism rather than a string: the labels will be reworded, the arithmetic must
not come back.

One shape dominates the set and is worth naming once: **"I could not ask" was
rendered as "the answer is no problem."**  NT-1 is the worst instance, because
the field that should carry the hedge exists, is read by consumers, and was set
to ``false``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta

import pytest

from nodetop.cli import build_parser, cmd_accelerators, cmd_health, cmd_where
from nodetop.core.cluster import Cluster
from nodetop.core.hardware import identify_accelerator, name_accelerator
from nodetop.core.model import (
    CATEGORY_LABELS,
    BackendCapabilities,
    Identity,
    JobShape,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
    category_label,
)
from nodetop.render import Glyphs, Style
from nodetop.runner import Runner

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# NT-1 -- a replay has no dry-run, and said so nowhere a consumer could see
# ---------------------------------------------------------------------------
class TestReplayDoesNotInventEntitlement:
    """``snapshot`` cannot record a dry-run, and must not imply one.

    Measured on the same cluster in the same second: ``where`` returned 32 rows
    live and **53** off a snapshot under a minute old, because every one of the
    21 partitions the live run had *measured* as refusing this user came back
    open.  ``status`` showed the arithmetic -- live ``30 open (25 unconfirmed) +
    21 refused``, replay ``51 open`` -- and 30 + 21 = 51.
    """

    def test_a_replayed_placement_is_marked_unconfirmed(self, replayed_cluster):
        from nodetop.core.fit import rank

        places = rank(replayed_cluster, JobShape(cpus_per_task=1), use_probe=True)
        assert places
        # The field whose whole job is to carry this. It used to be read off the
        # recorded capabilities, and a recording restores `probe=True` from the
        # live run it was taken on.
        assert all(p.entitlement_unconfirmed for p in places)
        assert all(p.verdict is None for p in places)
        assert not any(p.confirmed for p in places)

    def test_the_replay_asks_the_backend_nothing(self, replayed_cluster):
        from nodetop.core.fit import rank

        asked = []
        backend = replayed_cluster._backend
        original = backend.probe

        def _spy(*a, **k):  # pragma: no cover - must not be reached
            asked.append(a)
            return original(*a, **k)

        backend.probe = _spy  # type: ignore[method-assign]
        try:
            rank(replayed_cluster, JobShape(cpus_per_task=1), use_probe=True)
        finally:
            backend.probe = original  # type: ignore[method-assign]
        # Not merely harmless: the probe loop used to run, spend a decrement of
        # the shared budget per candidate, and get None back from
        # `Cluster.probe` before the call ever left the process.
        assert asked == []

    def test_the_json_says_why_it_is_unconfirmed(self, replayed_cluster, capsys):
        cmd_where(replayed_cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        assert rows
        for row in rows:
            assert row["entitlement_unconfirmed"] is True
            assert row["confirmed"] is False
            # Three states are representable and the replay used to report
            # none of them: `confirmed` and `entitlement_unconfirmed` were both
            # false for every queue, so `build` -- which this user can genuinely
            # submit to -- was indistinguishable from `dali`, which refuses them.
            assert row["entitlement_source"] == "declared"

    def test_submit_flags_do_not_come_with_an_empty_caveat_list(
        self,
        replayed_cluster,
        capsys,
    ):
        cmd_where(replayed_cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        rows = json.loads(capsys.readouterr().out)
        for row in rows:
            # `submit_flags` is the part a reader copies, and it arrived beside
            # `blockers: []`, `caveats: []` and a hedge field set to false.
            assert row["submit_flags"]
            assert any("DECLARED" in c for c in row["caveats"]), row["caveats"]

    def test_the_funnel_carries_the_qualifier(self, replayed_cluster, capsys):
        from nodetop.cli import cmd_status

        cmd_status(replayed_cluster, _args(["--json", "status"]), PLAIN)
        funnel = json.loads(capsys.readouterr().out)["funnel"]
        # `shown` with no `unconfirmed` beside it read MORE confident than the
        # live line it was a recording of.
        assert funnel["unconfirmed"] == funnel["shown"]

    def test_the_printed_funnel_says_it_too(self, replayed_cluster, capsys):
        from nodetop.cli import cmd_status

        cmd_status(replayed_cluster, _args(["status"]), PLAIN)
        out = capsys.readouterr().out
        assert "unconfirmed" in out


# ---------------------------------------------------------------------------
# NT-2 -- 232 of 384 GPUs rendered as UNKNOWN, and the scheduler had named them
# ---------------------------------------------------------------------------
class TestGpuVocabulary:
    """Every model token this cluster names in node features, verbatim.

    ``UNKNOWN`` was the LARGEST group on the cluster -- 232 GPUs, more than the
    V100s (122), K80s (20) and P100s (10) put together -- and three of the eight
    misses are Tesla datacentre parts from the same product line as the K80 the
    table already had.
    """

    #: (feature string as `sinfo -h -N -o "%G|%f"` printed it, expected model)
    FEATURES = [
        ("lc,gold-6148,192GB,rtx2080ti,gpu,l16b", "RTX2080TI"),
        ("lc,e5-2620v4,64G,gtx1080,gpu,noib", "GTX1080"),
        ("tc,e5-2670,32G,ib,k20m,gpu,ibspine-g20", "K20M"),
        ("tc,gold-6148,192GB,v100,gpu,d20b", "V100"),
        ("tc,e5-2670,32G,ib,k40m,gpu", "K40M"),
        ("tc,e5-2670,32G,ib,m2090,gpu", "M2090"),
        ("lc,gtxtitanx,gpu", "GTXTITANX"),
        ("lc,gtx780,gpu", "GTX780"),
        # A hyphen, so whatever matcher is used has to normalise separators.
        ("lc,titan-v,gpu", "TITANV"),
    ]

    @pytest.mark.parametrize("features,model", FEATURES)
    def test_the_model_is_identified(self, features, model):
        spec = identify_accelerator(None, features)
        assert spec is not None, f"{features!r} still unidentified"
        assert spec.model == model

    @pytest.mark.parametrize("features,model", FEATURES)
    def test_none_of_them_claims_a_capability_it_lacks(self, features, model):
        # Every one is pre-Ampere, so this is the answer rather than a guess --
        # and it is the reason a wrong-but-confident label would have been worse
        # than the shrug it replaces.
        spec = identify_accelerator(None, features)
        assert spec is not None
        if spec.sm is not None and spec.sm < 80:
            assert not spec.bf16
            assert not spec.tf32
            assert not spec.fp8

    def test_separators_do_not_change_the_answer(self):
        got = {
            identify_accelerator(None, f"gpu,{t}").model  # type: ignore[union-attr]
            for t in ("titan-v", "titanv", "TITAN_V", "Titan-V")
        }
        assert got == {"TITANV"}

    def test_an_unnamed_card_is_called_what_the_scheduler_called_it(self):
        # The raw string is strictly more useful than a shrug: `rtx2080ti` tells
        # the reader everything, `UNKNOWN` tells them nothing AND hides the fact
        # that the scheduler did say.
        node = Node(name="n", gpus_total=3, accelerator=None, accelerator_label="quadro-p5000")
        assert node.accelerator_name == "quadro-p5000"
        assert node.accelerator_identified is False

    def test_naming_a_card_claims_nothing_about_it(self):
        from nodetop.core.hardware import supports

        node = Node(name="n", gpus_total=1, accelerator=None, accelerator_label="quadro-p5000")
        assert node.accelerator is None
        # Tri-state intact: "we cannot identify this" must not collapse into
        # "this cannot do the job", which is the only thing that justifies
        # excluding a node.
        assert supports(node.accelerator, "bf16") is None

    def test_a_fabric_label_is_not_mistaken_for_a_gpu(self):
        # `l16b`, `d20b` and `ibspine-g20` are switch labels and sit right next
        # to the GPU token on this cluster. A loose shape match takes `l16b`.
        assert name_accelerator(None, "lc,e5-2670,32G,ib,d20b,l16b,noib") is None
        assert name_accelerator(None, "tc,ib,ibspine-g20,noib") is None

    def test_a_named_family_beats_a_bare_part_number(self):
        assert name_accelerator(None, "lc,gold-6148,192GB,rtx2080ti,gpu,l16b") == ("rtx2080ti")

    @pytest.mark.parametrize(
        "labels",
        [
            "nvidia_driver_535,gpu",
            "nvidia_driver_535",
            "nvidia-driver-470",
            "NVIDIA_Driver_550.54",
            "gpu,nvidia_firmware_92",
        ],
    )
    def test_a_driver_version_is_not_a_card(self, labels):
        """It has a vendor prefix, so requiring one does not exclude it.

        `nvidia_driver_535` fits `_GPU_NAME_SHAPED` exactly once `_normalise` has
        taken the underscores out -- an `nvidia` prefix and twelve or fewer
        alphanumerics -- so it became a model row in `nodetop accelerators`. A
        driver version is a real fact about the node and it is not what the card
        is called.
        """
        assert name_accelerator(None, labels) is None

    def test_a_real_card_beside_a_driver_label_still_wins(self):
        # The consequential ordering: the driver label came FIRST in the list and
        # both are "named family" tokens, so it beat the card to the answer.
        assert name_accelerator(None, "nvidia_driver_535,tesla_v100") == "tesla_v100"
        assert name_accelerator(None, "nvidia-driver-470,rtx2080ti") == "rtx2080ti"

    def test_a_typed_resource_is_unaffected(self):
        # The scheduler has already said this field is a GPU type, so it needs no
        # shape test and must not acquire one.
        assert name_accelerator("gpu:a100:4", "nvidia_driver_535") == "a100"

    def test_the_capability_rows_divide_the_identified_population(self, capsys):
        cluster = _cluster_with_one_unidentified_gpu()
        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        data = json.loads(capsys.readouterr().out)
        assert data["accelerators_installed"] == 8
        assert data["accelerators_unidentifiable"] == 4
        assert data["accelerators_identified"] == 4
        for cap in data["capability_reach"].values():
            # `bf16 0/384` off a denominator that was 60% unknown read as a
            # measurement. It was right on that cluster by luck -- Fermi through
            # Turing have none of these -- and one A100 behind an unrecognised
            # token makes it false while looking identical.
            assert cap["of_identified"] == 4
            assert cap["unidentified"] == 4

    def test_the_unidentified_group_is_named_and_flagged(self, capsys):
        cluster = _cluster_with_one_unidentified_gpu()
        cmd_accelerators(cluster, _args(["--json", "accelerators", "--all"]), PLAIN)
        models = json.loads(capsys.readouterr().out)["models"]
        assert models["A100"]["identified"] is True
        # Named after the label, and carrying a non-magic discriminator: testing
        # `model == "UNKNOWN"` was the only way to tell the two apart.
        assert "quadro-p5000" in models
        assert models["quadro-p5000"]["identified"] is False
        assert models["quadro-p5000"]["scheduler_label"] == "quadro-p5000"

    def test_the_disclosure_line_prints_the_count_once(self, capsys):
        cluster = _cluster_with_one_unidentified_gpu()
        cmd_accelerators(cluster, _args(["accelerators", "--all"]), PLAIN)
        out = capsys.readouterr().out
        note = next(ln for ln in out.splitlines() if "capability row" in ln)
        # "4 4 accelerators": a count interpolated into a string that already
        # contains it, because `plural()` writes the number itself.
        assert "4 4" not in note
        assert note.count("4") >= 1


def _cluster_with_one_unidentified_gpu() -> Cluster:
    """Four identified GPUs and four the vocabulary cannot place."""
    known = Node(
        name="k1",
        state_raw="IDLE",
        cpus_total=8,
        memory_mb=16000,
        gpus_total=4,
        queues=("gpu",),
        accelerator=identify_accelerator(None, "gpu,a100"),
        accelerator_label="a100",
    )
    unknown = Node(
        name="u1",
        state_raw="IDLE",
        cpus_total=8,
        memory_mb=16000,
        gpus_total=4,
        queues=("gpu",),
        accelerator=None,
        accelerator_label="quadro-p5000",
    )
    nodes = [known, unknown]
    return Cluster(
        backend_name="slurm",
        queue_term="partition",
        nodes=nodes,
        queues={"gpu": Queue(name="gpu", node_names=("k1", "u1"), declared_nodes=2, nodes=nodes)},
    )


# ---------------------------------------------------------------------------
# NT-3 -- a group labelled with its oldest node's age
# ---------------------------------------------------------------------------
class TestHealthReportsTheNewestFailureInAGroup:
    """854 nodes, oldest 809 days, newest four minutes -- reported as 809 days.

    The other reason-groups on that cluster had **zero** spread, because they
    were bulk-drained in single administrative actions.  It only shows on the
    group that accumulates over time, and that is the group holding 854 of the
    1,019 unschedulable nodes.
    """

    def _health(self, reasons: list[str], capsys) -> str:
        nodes = [
            Node(name=f"n{i}", state_raw="DOWN", conditions=frozenset({"DOWN"}), reason=r)
            for i, r in enumerate(reasons)
        ]
        cluster = Cluster(backend_name="slurm", queue_term="partition", nodes=nodes, queues={})
        cmd_health(cluster, _args(["health"]), PLAIN)
        return capsys.readouterr().out

    def test_a_fresh_failure_is_not_hidden_inside_a_graveyard(self, capsys):
        now = datetime.now()
        old = (now - timedelta(days=809)).strftime("%Y-%m-%dT%H:%M:%S")
        fresh = (now - timedelta(minutes=4)).strftime("%Y-%m-%dT%H:%M:%S")
        out = self._health(
            [
                f"Not responding [root@{old}]",
                f"Not responding [root@{fresh}]",
            ],
            capsys,
        )
        # "down for 809 days" reads as decommissioned hardware to ignore; "down
        # for 4 minutes" reads as a live incident. Collapsing the second into
        # the first is what this view exists not to do.
        assert "newest" in out
        assert "4m" in out
        assert "809d" in out

    def test_a_group_drained_at_once_still_reads_as_one_age(self, capsys):
        stamp = (datetime.now() - timedelta(days=41)).strftime("%Y-%m-%dT%H:%M:%S")
        out = self._health([f"maintenance [root@{stamp}]"] * 3, capsys)
        # No spread, so no range: a line that never changes an answer is a line
        # in the way.
        assert "for 41d" in out
        assert "newest" not in out

    def test_a_partly_stamped_group_says_how_many_it_measured(self, capsys):
        stamp = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
        out = self._health([f"maintenance [root@{stamp}]", "maintenance"], capsys)
        assert "1/2 stamped" in out


# ---------------------------------------------------------------------------
# NT-4 -- Slurm's own torque wrappers satisfied the PBS client probe
# ---------------------------------------------------------------------------
class TestPbsIsNotDetectedFromSlurmsWrappers:
    """``contribs/torque`` ships with Slurm and is installed at a great many
    sites, so this is the common case for the backend most likely to be
    misdetected -- not a quirk of one cluster.

    And it did not fail visibly.  ``pbsnodes`` IS the shim and really does
    enumerate every node, so ``--backend pbs`` produced correct denominators --
    1614 nodes, 422 GPUs, matching the slurm backend exactly -- with ``0 up`` on
    a cluster with 589 up, and **exit 0**.  A backend whose client is genuinely
    absent exits 3; a backend that is misdetected exited 0.
    """

    def _fake_path(self, monkeypatch, prefix: str, binaries: list[str]) -> None:
        """Pretend PATH holds exactly ``binaries``, under ``prefix``.

        `nodetop.backends.pbs`'s own `which`/`resolve` are replaced rather than
        `$PATH`, because an autouse fixture claims every scheduler client is
        installed on every host -- deliberately, and its docstring says a test
        needing one to be ABSENT should patch narrowly, here.
        """
        present = set(binaries)
        monkeypatch.setattr("nodetop.backends.pbs.which", lambda b: b in present)
        monkeypatch.setattr(
            "nodetop.backends.pbs.resolve",
            lambda b: f"/software/{prefix}/bin/{b}" if b in present else None,
        )

    def test_the_wrappers_do_not_detect_as_pbs(self, monkeypatch):
        from nodetop.backends.pbs import PbsBackend

        # Verbatim from the cluster: `readlink -f $(command -v qstat)` gave
        # /software/slurm-23.02-el7-x86_64/bin/qstat, and the version lives in
        # the directory name, which is how these are laid out.
        self._fake_path(
            monkeypatch, "slurm-23.02-el7-x86_64", ["qstat", "qsub", "pbsnodes", "qdel"]
        )
        assert PbsBackend.wrapped_by() == "slurm"
        assert PbsBackend.detect() is False

    def test_a_real_pbs_still_detects(self, monkeypatch):
        from nodetop.backends.pbs import PbsBackend

        # `qmgr` and `pbs_server` have no Slurm equivalent and no Slurm package
        # installs them, so either one settles it on its own.
        self._fake_path(monkeypatch, "pbs/22.05", ["qstat", "qsub", "pbsnodes", "qmgr"])
        assert PbsBackend.wrapped_by() is None
        assert PbsBackend.detect() is True

    def test_a_pbs_under_a_path_that_merely_mentions_slurm_still_detects(
        self,
        monkeypatch,
    ):
        """A path COMPONENT, not a substring.

        `/opt/slurmy-site/pbs/bin` is not a Slurm install prefix, and refusing to
        detect there would be the same class of error in the other direction.

        **`pbs_server` is deliberately absent from this PATH.** With it present
        the `_PBS_ONLY` short-circuit at the top of `wrapped_by` returns before
        the path-component logic runs at all, so the test passed against the
        pre-fix code and asserted nothing about the thing it names -- which is
        how it was written the first time.
        """
        from nodetop.backends.pbs import PbsBackend

        # `slurmy` contains "slurm" and is not "slurm": neither a whole
        # component nor a `slurm-<version>` prefix.
        self._fake_path(monkeypatch, "slurmy-site/pbs", ["qstat", "qsub", "pbsnodes", "qdel"])
        assert PbsBackend.wrapped_by() is None
        assert PbsBackend.detect() is True

    @pytest.mark.parametrize(
        "prefix,expected",
        [
            # The two forms that ARE an install prefix, so the substring test
            # above has something to be distinguished from.
            ("slurm/bin", "slurm"),
            ("slurm-23.02-el7-x86_64", "slurm"),
            # And the ones that are not.
            ("slurmy-site/pbs", None),
            ("pbs/22.05", None),
            # A component starting `slurm-` but not versioned. This row used to
            # record the opposite, because `startswith("slurm-")` matched any
            # such name -- so a genuine client-only PBS installed under a
            # directory called `slurm-logs` was reported as somebody else's
            # wrapper and not detected at all. The rule now requires a digit
            # after the dash, which is what "install prefix" actually means.
            ("data/slurm-logs/pbs", None),
            ("var/slurm-data/pbs", None),
        ],
    )
    def test_only_a_whole_component_names_a_wrapper(
        self,
        monkeypatch,
        prefix,
        expected,
    ):
        from nodetop.backends.pbs import PbsBackend

        # No `_PBS_ONLY` binary, so the path logic is what decides.
        self._fake_path(monkeypatch, prefix, ["qstat", "qsub", "pbsnodes", "qdel"])
        assert PbsBackend.wrapped_by() == expected

    def test_an_absent_client_is_not_reported_as_a_wrapper(self, monkeypatch):
        from nodetop.backends.pbs import PbsBackend

        self._fake_path(monkeypatch, "empty", [])
        assert PbsBackend.detect() is False
        assert PbsBackend.wrapped_by() is None

    def test_a_cluster_with_no_queues_exits_nonzero(self):
        from nodetop.cli import _reject_broken_snapshot

        # Nodes but no queues, and the queue query is the one that failed. Every
        # view is then a confidently-shaped nothing: `0 queues - 0 open to you`.
        cluster = Cluster(
            backend_name="pbs",
            queue_term="queue",
            nodes=[Node(name="n1", state_raw="free", cpus_total=8)],
            queues={},
            errors={"queues": "CommandError: qstat -Qf exited 2: Unknown option"},
        )
        assert _reject_broken_snapshot(cluster, "status") == 3

    def test_a_partial_failure_with_queues_intact_still_returns_zero(self):
        # CONTROL: behaviour that must not change. Passes pre-fix, deliberately.
        from nodetop.cli import _reject_broken_snapshot

        node = Node(name="n1", state_raw="free", cpus_total=8)
        cluster = Cluster(
            backend_name="slurm",
            queue_term="partition",
            nodes=[node],
            queues={"q": Queue(name="q", node_names=("n1",), nodes=[node])},
            errors={"limits": "CommandError: sacctmgr exited 1"},
        )
        assert _reject_broken_snapshot(cluster, "status") == 0


# ---------------------------------------------------------------------------
# NT-5 -- the access column printed raw enum members beside prose
# ---------------------------------------------------------------------------
class TestCategoriesHaveADisplayForm:
    """``ACCOUNTS_UNTRIED`` and ``UNKNOWN`` surfaced in a column whose other
    values were ``unchecked``, ``confirmed`` and ``refused``.

    The categories themselves are a wire vocabulary -- they go into ``--json``
    and are compared in code -- so they stay SCREAMING_SNAKE and get a display
    mapping instead of being collapsed into "refused", which is what would have
    lost the distinction ``ACCOUNTS_UNTRIED`` exists to make.
    """

    def test_every_category_has_a_label(self):
        members = {
            v
            for k, v in vars(VerdictCategory).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        assert members
        missing = members - set(CATEGORY_LABELS)
        assert not missing, f"no display label for {sorted(missing)}"

    def test_no_label_is_screaming_snake(self):
        for name, label in CATEGORY_LABELS.items():
            assert label == label.lower(), name
            assert "_" not in label, name

    def test_an_unrecognised_category_still_reads_as_prose(self):
        # A category this table has not caught up with must not fall back to
        # the raw token, which would reintroduce exactly the leak.
        assert category_label("SOME_NEW_THING") == "some new thing"
        assert category_label("") == "unknown"

    def test_the_help_examples_name_no_foreign_partition(self):
        from nodetop.cli import EXAMPLES

        # `nodetop zoom gn` named a partition from the development cluster, on a
        # line whose padding was written for a longer name. Same pattern already
        # filed against two sibling packages.
        assert "zoom gn" not in EXAMPLES
        for line in EXAMPLES.splitlines():
            if line.strip().startswith("nodetop ") and "  " in line.strip():
                command, _, description = line.strip().partition("  ")
                if description.strip():
                    assert line.index(description.strip()) >= 37, line


# ---------------------------------------------------------------------------
# NT-6 -- a no-year strptime that Python 3.15 breaks, one way silently
# ---------------------------------------------------------------------------
class TestNoYearTimestampsSupplyTheYear:
    """The ``1900`` default is what CPython is removing.

    Both announced outcomes broke the old ``got.year == 1900`` sentinel, and one
    broke it silently: a raise is swallowed by the surrounding
    ``except ValueError: continue`` and the branch stops matching, while a
    different default year no longer trips the sentinel and feeds a wrong year
    into every duration computed from it.  ``requires-python`` has no upper
    bound, so 3.15 will install this.
    """

    def test_no_deprecation_warning_is_emitted(self, recwarn):
        import warnings

        from nodetop.core.duration import parse_timestamp

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # The reachable path today: a snapshot whose `captured_at` is not
            # isoformat falls through to the format list, and the warning is
            # attributed to __main__ where DeprecationWarning shows by default.
            got = parse_timestamp("Aug 26 10:40")
        assert got is not None
        assert got.year == datetime.now().year

    def test_no_year_less_format_ever_reaches_strptime(self, monkeypatch):
        """What the fix actually changed.

        ``got.year != 1900`` passed against the pre-fix code too -- that code
        also substituted the current year, by *detecting* the 1900 default
        afterwards. The change is that ``strptime`` is never handed a year-less
        format at all, so neither announced 3.15 behaviour (raise, or a
        different default year) can reach it: there is no sentinel left to be
        wrong about.

        Asserted on the calls rather than on the source, so it pins the property
        and not the spelling. Pre-fix this fails: ``"%b %d %H:%M"`` went to
        ``strptime`` bare.
        """
        from nodetop.core import duration as duration_mod

        seen: list[str] = []
        real = duration_mod.datetime

        class _Recording(real):  # type: ignore[misc, valid-type]
            @classmethod
            def strptime(cls, text, fmt):  # type: ignore[override]
                seen.append(fmt)
                return real.strptime(text, fmt)

        monkeypatch.setattr(duration_mod, "datetime", _Recording)
        assert duration_mod.parse_timestamp("Jan 02 03:04") is not None
        assert seen, "nothing reached strptime, so this asserts nothing"
        assert all("%Y" in fmt for fmt in seen), seen

    def test_a_corrupt_captured_at_still_does_not_crash(self):
        # CONTROL: behaviour that must not change. Passes pre-fix, deliberately.
        from nodetop.core.duration import parse_timestamp

        assert parse_timestamp("not a timestamp at all") is None

    def test_a_leap_day_parses_in_a_leap_year(self):
        from nodetop.core.duration import parse_timestamp

        # The other case the CPython warning names: Feb 29 is unparsable
        # against 1900, so the old code returned None for it every time. It now
        # depends only on whether the assumed year is a leap year, which is the
        # honest answer for a scheduler's near-term estimate.
        year = datetime.now().year
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        got = parse_timestamp("Feb 29 10:40")
        assert (got is not None) is leap


# ---------------------------------------------------------------------------
# NT-7 -- the sdist shipped the test and not the tool it drives
# ---------------------------------------------------------------------------
class TestTheSdistCanRunItsOwnSuite:
    def test_the_pyz_builder_is_in_the_sdist_manifest(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        manifest = (root / "MANIFEST.in").read_text()
        # All 7 failures in the suite were one file: `tests/test_pyz.py` runs
        # `tools/build_pyz.py`, and only the former was packaged. Same shape as
        # a sibling package's absent `conftest.py`, which made a suite collect
        # zero tests and report success.
        assert "tools" in manifest

    def test_the_builder_is_found_relative_to_the_tree(self):
        from tests import test_pyz

        # Not relative to the working directory: pytest runs from wherever the
        # reader is standing.
        assert test_pyz.TOOL.is_absolute()


# ---------------------------------------------------------------------------
# NT-8 -- the probe omitted --qos, so 16 partitions were "unconfirmable"
# ---------------------------------------------------------------------------
class _Spy(Runner):
    """Records every command and accepts every dry-run."""

    def __init__(self) -> None:
        self.cmds: list[list[str]] = []

    def run(self, cmd, timeout: float = 30.0) -> str:  # pragma: no cover
        self.cmds.append(list(cmd))
        return ""

    def run_full(self, cmd, timeout: float = 30.0):
        self.cmds.append(list(cmd))
        return (
            0,
            "sbatch: Job 1 to start at 2026-08-26T12:00:00 using 1 processors on nodes dali001",
            "",
        )


def _midway2ish(assoc: str) -> tuple[Cluster, _Spy]:
    """A cluster whose associations name a QOS and no default.

    Verbatim shape from the cluster this was found on::

        rcc-staff|dali|stafftest|
        rcc-staff|cobey|stafftest|

    An allowed QOS, an EMPTY ``DefaultQOS`` -- so there is nothing for the
    scheduler to fall back on and a probe naming no QOS is refused outright.
    """
    from nodetop.backends.slurm import SlurmBackend

    spy = _Spy()
    backend = SlurmBackend(spy)
    ident = backend.parse_identity(assoc, "me")
    node = Node(name="dali001", state_raw="IDLE", cpus_total=8, memory_mb=16000, queues=("dali",))
    cluster = dataclasses.replace(
        Cluster(
            backend_name="slurm",
            queue_term="partition",
            nodes=[node],
            queues={
                "dali": Queue(name="dali", node_names=("dali001",), declared_nodes=1, nodes=[node])
            },
            identity=ident,
        ),
        capabilities=BackendCapabilities(
            probe=True, probe_supported=True, probe_command="sbatch --test-only"
        ),
        _backend=backend,
    )
    return cluster, spy


class TestTheProbeNamesTheQosTheAssociationGives:
    """``Invalid qos specification`` on 16 of 30 partitions, reported as
    unconfirmable -- using a QOS nodetop had already read in its first query.

    ``sacctmgr -nP show assoc ... format=Account,Partition,QOS,DefaultQOS`` asks
    for both, so the distinction between "an allowed QOS" and "a default that
    will be applied for you" was already modelled.  Only the probe never named
    one.  Five for five on partitions reported as unconfirmable, naming the QOS
    flipped them to *starts now*.
    """

    ASSOC = "rcc-staff|dali|stafftest|\nrcc-staff|cobey|stafftest|\n"

    def test_the_dry_run_carries_the_qos(self):
        cluster, spy = _midway2ish(self.ASSOC)
        verdict = cluster.probe("dali", JobShape(cpus_per_task=1), "rcc-staff")
        assert verdict is not None and verdict.allowed
        assert "--qos=stafftest" in spy.cmds[-1]

    def test_the_flags_shown_to_the_reader_carry_it_too(self):
        cluster, _ = _midway2ish(self.ASSOC)
        flags = cluster.submit_flags("dali", JobShape(cpus_per_task=1))
        # A probe that succeeds only because of a flag the reader is not shown
        # is its own kind of wrong answer: on this cluster there is no default
        # QOS, so a submission without it is refused.
        assert "--qos=stafftest" in flags

    def test_an_explicit_qos_wins(self):
        cluster, spy = _midway2ish(self.ASSOC)
        cluster.probe("dali", JobShape(cpus_per_task=1, qos="mine"), "rcc-staff")
        assert "--qos=mine" in spy.cmds[-1]
        assert "--qos=stafftest" not in spy.cmds[-1]

    def test_a_default_qos_is_named(self):
        cluster, spy = _midway2ish("pi-x|dali|normal,high|normal\n")
        cluster.probe("dali", JobShape(cpus_per_task=1), "pi-x")
        assert "--qos=normal" in spy.cmds[-1]

    def test_several_qos_and_no_default_names_none(self):
        # Naming one would be a guess, and a wrong --qos turns an answerable
        # question into a refusal -- the exact failure this removes.
        cluster, spy = _midway2ish("pi-x|dali|normal,high|\n")
        cluster.probe("dali", JobShape(cpus_per_task=1), "pi-x")
        assert not any(f.startswith("--qos=") for f in spy.cmds[-1])

    def test_a_templated_association_names_none(self):
        # One real cluster puts the PARTITION list in the QOS column and leaves
        # DefaultQOS empty for all 34 accounts. There is no QOS to name there,
        # and inventing one from that list would be nonsense.
        cluster, spy = _midway2ish(
            "acct-a||caslake,amd,beagle3,gpu,bigmem|\nacct-b||caslake,amd,beagle3,gpu,bigmem|\n"
        )
        cluster.probe("dali", JobShape(cpus_per_task=1), "acct-a")
        assert not any(f.startswith("--qos=") for f in spy.cmds[-1])

    def test_a_per_partition_row_beats_the_blanket_one(self):
        cluster, spy = _midway2ish("pi-x||general|general\npi-x|dali|stafftest|\n")
        cluster.probe("dali", JobShape(cpus_per_task=1), "pi-x")
        assert "--qos=stafftest" in spy.cmds[-1]

    def test_the_flag_exists_so_the_reader_can_settle_it(self):
        # `ACCOUNTS_UNTRIED` says "name one with -A to settle it"; INVALID_QOS
        # had no caveat, no hint and no flag -- a dead end.
        for command in ("where", "check"):
            args = _args([command, "--qos", "mine"])
            assert args.qos == "mine"

    def test_an_envelope_refusal_says_which_flag_settles_it(self):
        from nodetop.core.fit import evaluate

        node = Node(name="n1", state_raw="IDLE", cpus_total=8, memory_mb=16000, queues=("q",))
        queue = Queue(name="q", node_names=("n1",), declared_nodes=1, nodes=[node])

        class _Refuses:
            def probe(self, q, shape, account=None):
                return Verdict(
                    queue=q,
                    account=account,
                    allowed=False,
                    category=VerdictCategory.INVALID_QOS,
                    reason="Invalid qos specification",
                )

            def capabilities(self):
                return BackendCapabilities(
                    probe=True, probe_supported=True, probe_command="sbatch --test-only"
                )

            def submit_flags(self, q, shape):
                return []

        backend = _Refuses()
        cluster = dataclasses.replace(
            Cluster(
                backend_name="slurm",
                queue_term="partition",
                nodes=[node],
                queues={"q": queue},
                identity=Identity(user="me", accounts=("a",)),
            ),
            capabilities=backend.capabilities(),
            _backend=backend,
        )
        place = evaluate(cluster, JobShape(cpus_per_task=1), queue, use_probe=True, accounts=["a"])
        assert any("--qos" in c for c in place.caveats), place.caveats


class TestAnUnaskedQueueSaysWhyItWasNotAsked:
    """``unchecked`` is the one word that is not true of a queue the budget
    could not reach, and it points the reader away from the fix.

    With 84 partitions and 6 accounts an exhaustive sweep is 504 dry-runs
    against a ceiling of 150, so this is the ordinary case on a cluster of that
    size rather than an edge one -- and the remedy is a flag away.
    """

    def _cluster(self, monkeypatch, budget_total: int):
        """Two queues and a budget that only reaches one of them.

        Deliberately MIXED: with nothing answered anywhere the access column is
        dropped as invariant, which is the documented design and not what this
        finding is about.  The bug is a row that says `unchecked` beside a row
        that says `confirmed`.
        """
        import nodetop.core.fit as fit

        nodes = [
            Node(name=f"n{i}", state_raw="IDLE", cpus_total=8, memory_mb=16000, queues=(f"q{i}",))
            for i in range(2)
        ]
        queues = {
            f"q{i}": Queue(name=f"q{i}", node_names=(f"n{i}",), declared_nodes=1, nodes=[nodes[i]])
            for i in range(2)
        }
        caps = BackendCapabilities(
            probe=True, probe_supported=True, probe_command="sbatch --test-only"
        )

        class _Accepts:
            def probe(self, q, shape, account=None):
                return Verdict(
                    queue=q, account=account, allowed=True, category=VerdictCategory.OK, reason="ok"
                )

            def capabilities(self):
                return caps

            def submit_flags(self, q, shape):
                return [f"--partition={q}"]

            def format_nodelist(self, names):
                return ",".join(sorted(names))

        backend = _Accepts()
        cluster = dataclasses.replace(
            Cluster(
                backend_name="slurm",
                queue_term="partition",
                nodes=nodes,
                queues=queues,
                identity=Identity(user="me", accounts=("a",)),
            ),
            capabilities=caps,
            _backend=backend,
        )
        original = fit.ProbeBudget

        def _capped(*_a, **_k):
            return original(total=budget_total)

        monkeypatch.setattr(fit, "ProbeBudget", _capped)
        return cluster

    def test_it_records_that_it_was_asked_for_and_not_asked(self, monkeypatch):
        from nodetop.core.fit import rank

        cluster = self._cluster(monkeypatch, budget_total=1)
        places = {
            p.queue: p
            for p in rank(cluster, JobShape(cpus_per_task=1), use_probe=True, accounts=["a"])
        }
        spent = [p for p in places.values() if p.probes == (0, 1)]
        assert spent, {q: p.probes for q, p in places.items()}
        assert all(p.entitlement_unconfirmed for p in spent)

    def test_the_caveat_names_the_budget_and_the_remedy(self, monkeypatch):
        from nodetop.core.fit import rank

        cluster = self._cluster(monkeypatch, budget_total=1)
        places = rank(cluster, JobShape(cpus_per_task=1), use_probe=True, accounts=["a"])
        spent = next(p for p in places if p.probes == (0, 1))
        # No silent caps: a bounded sweep that does not say what it dropped
        # reads as "covered everything".
        assert any("budget" in c for c in spent.caveats), spent.caveats
        assert any("-q" in c for c in spent.caveats), spent.caveats

    def test_the_json_distinguishes_it_from_declared(self, monkeypatch, capsys):
        cluster = self._cluster(monkeypatch, budget_total=1)
        cmd_where(cluster, _args(["--json", "where", "-c", "1"]), PLAIN)
        rows = {r["queue"]: r for r in json.loads(capsys.readouterr().out)}
        sources = {r["entitlement_source"] for r in rows.values()}
        # Both halves present, and named apart: "declared" is "there was no
        # dry-run to run", which is a different fact with a different remedy.
        assert "probe budget spent" in sources
        assert "confirmed" in sources
        assert "declared" not in sources

    def test_the_column_does_not_call_it_unchecked(self, monkeypatch, capsys):
        cluster = self._cluster(monkeypatch, budget_total=1)
        cmd_where(cluster, _args(["where", "-c", "1"]), PLAIN)
        out = capsys.readouterr().out
        assert "budget spent" in out
        assert "unchecked" not in out


class TestReviewFollowUps:
    """Three points raised on the round-1 solutions, 2026-08-27."""

    def test_a_distro_packaged_wrapper_is_still_recognised(self):
        """NT-4: the first two signals are properties of PACKAGING.

        The marker-binary rule and the install-prefix rule both hold on a site
        that installs Slurm under a versioned `/software` prefix -- and the
        distro `slurm-torque` RPM defeats both: the shims land in `/usr/bin`, so
        there is no `slurm-<version>` component, and it still ships none of the
        five PBS-only binaries. That is the ordinary case on any RPM/DEB Slurm
        cluster, not an exotic one.
        """
        import nodetop.backends.pbs as pbsmod
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        shim = {"qstat", "qsub", "pbsnodes", "qdel"}
        pbsmod.which = lambda b: b in shim
        pbsmod.resolve = lambda b: f"/usr/bin/{b}" if b in shim else None
        slurm_says = RecordedRunner(
            {
                "pbsnodes": (0, "n1\n     state = free\n     slurmstate=idle\n", ""),
            }
        )
        assert PbsBackend.wrapped_by(slurm_says) == "slurm"

        # A real PBS under the same path, with no marker in its output.
        real = RecordedRunner({"pbsnodes": (0, "n1\n     state = free\n     np = 40\n", "")})
        assert PbsBackend.wrapped_by(real) is None

        # And a probe that cannot answer says nothing either way.
        broken = RecordedRunner({"pbsnodes": (1, "", "cannot connect")})
        assert PbsBackend.wrapped_by(broken) is None

    def test_a_rejected_qos_is_not_called_undecided(self):
        """NT-5: the control plane decided; the label said it had not.

        `Invalid qos specification` is a decision. A row here means the QOS was
        rejected -- whether nodetop named one from the association or `qos_for`
        declined and Slurm found no default to apply -- and both are actionable,
        where "undecided" reads as nodetop not having got round to it.
        """
        from nodetop.core.model import CATEGORY_LABELS, VerdictCategory, category_label

        assert category_label(VerdictCategory.INVALID_QOS) == "qos rejected"
        assert "undecided" not in " ".join(CATEGORY_LABELS.values())

    @pytest.mark.parametrize("offset_days", [1, 40, 200, 300])
    def test_a_no_year_stamp_is_never_read_as_the_future(self, offset_days):
        """NT-6: the assumed year is wrong across a New Year boundary.

        `Dec 31 23:50` parsed on 2 January became a date eleven months and thirty
        days AHEAD -- and this is LSF's short form, used for exactly the recent
        events where the wrap matters. The classic syslog-timestamp bug; it only
        surfaces in the first days of January, which is why it survives review.
        """
        from datetime import timedelta

        from nodetop.core.duration import parse_timestamp

        now = datetime.now()
        stamp = (now - timedelta(days=offset_days)).strftime("%b %d %H:%M")
        got = parse_timestamp(stamp)
        assert got is not None, stamp
        assert got - now <= timedelta(days=1), f"{stamp!r} parsed as {got}, which is in the future"

    def test_a_stamp_slightly_ahead_of_the_clock_is_left_alone(self):
        """The control: a day of slack, not zero.

        These are scheduler estimates and a start time minutes ahead of the local
        clock is ordinary -- the same clock-skew allowance `fit` already makes.
        Rolling those back a year would be a new wrong answer.
        """
        from datetime import timedelta

        from nodetop.core.duration import parse_timestamp

        now = datetime.now()
        soon = now + timedelta(hours=2)
        got = parse_timestamp(soon.strftime("%b %d %H:%M"))
        assert got is not None
        assert got.year == soon.year, got
