"""Every backend, held to the same query discipline.

Two properties are enforced here, and both are correctness rather than
efficiency:

* **No command is issued twice.**  A report is supposed to describe one
  instant.  Fetching the same source again for a second consumer means two
  different instants can appear in the same output, which is exactly what
  taking a single snapshot is meant to prevent.
* **No argv element smuggles a separator.**  Passing ``"where user=alice"`` as
  one element instead of two made ``sacctmgr`` answer *Unknown condition*,
  which returned an empty identity and silently disabled every account and QOS
  access check.  Nothing errored; a whole analysis layer just stopped.
"""

from __future__ import annotations

import contextlib
from collections import Counter

import pytest

from nodetop import backends
from nodetop.core.cluster import Cluster
from nodetop.core.model import JobShape

BACKENDS = backends.names()


class TestNothingIsAskedForInTheSecondWave:
    """A query issued from inside another query's handler arrives late.

    `Cluster.load` fires its queries together, but `scontrol show config` was
    not one of them: `load_nodes` reached for it partway through its own parse,
    so it left ~70 ms after the first wave. During one bad spell on a live
    controller that straggler took **2.0s on two runs in five** while the four
    first-wave queries stayed normal; asked on its own it is 17 ms, twenty out
    of twenty.

    The honest limit of that: an interleaved A/B of both versions later found no
    stalls in either, so the controller had calmed and the clean runs after the
    change do not prove it caused them. What this test pins is the staging
    itself, which is right either way -- one wave rather than one wave and a
    straggler -- and that warming costs no extra query.
    """

    def test_the_load_warms_every_backend(self, monkeypatch):
        called = []

        class Warms(Recorder):
            pass

        for name in BACKENDS:
            backend = backends.get(name, runner=Recorder())
            monkeypatch.setattr(type(backend), "warm",
                                lambda _self, n=name: called.append(n),
                                raising=False)
            with contextlib.suppress(Exception):
                Cluster.load(backend)
        assert sorted(called) == sorted(BACKENDS), called

    def test_slurm_asks_for_its_config_exactly_once(self, slurm_backend):
        # Warming it and then parsing nodes must not be two readings of one
        # file: the cache is what makes `warm` a scheduling change rather than
        # an extra query.
        slurm_backend.warm()
        before = sum(1 for c in slurm_backend.runner.calls
                     if "config" in " ".join(c))
        slurm_backend.load_nodes()
        slurm_backend.load_identity()
        after = sum(1 for c in slurm_backend.runner.calls
                    if "config" in " ".join(c))
        assert before == 1 and after == 1, slurm_backend.runner.calls

    def test_a_backend_without_one_is_not_a_problem(self):
        # `warm` is optional; most backends have nothing that needs it.
        backend = backends.get("lsf", runner=Recorder())
        Cluster.load(backend)      # must not raise


class Recorder:
    """A runner that answers nothing and remembers everything."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run_full(self, cmd, timeout: float = 30) -> tuple[int, str, str]:
        self.calls.append(list(cmd))
        return (0, "", "")

    def run(self, cmd, timeout: float = 30) -> str:
        self.calls.append(list(cmd))
        return ""

    def ok(self, cmd, timeout: float = 30) -> bool:
        self.calls.append(list(cmd))
        return True


def _exercise(name: str) -> Recorder:
    rec = Recorder()
    backend = backends.get(name, rec)
    # A blank runner is allowed to make parsing give up; what is under test is
    # which commands were issued, not whether they returned anything usable.
    with contextlib.suppress(Exception):
        Cluster.load(backend, with_free_times=True)
    with contextlib.suppress(Exception):
        backend.probe("somequeue", JobShape(gpus_per_node=1), "someaccount")
    return rec


@pytest.mark.parametrize("name", BACKENDS)
class TestQueryDiscipline:
    def test_no_command_is_issued_twice(self, name):
        rec = _exercise(name)
        joined = [" ".join(c) for c in rec.calls]
        dupes = {k: v for k, v in Counter(joined).items() if v > 1}
        assert not dupes, f"{name} repeats: {dupes}"

    def test_no_argv_element_smuggles_a_separator(self, name):
        rec = _exercise(name)
        for argv in rec.calls:
            # bash -c legitimately carries a whole script; nothing else should
            # carry a space in a non-flag position.
            if argv[:2] == ["bash", "-c"]:
                continue
            for element in argv[1:]:
                if element.startswith("-"):
                    continue
                assert " " not in element, (
                    f"{name} passes {element!r} as one argument; a separator "
                    f"inside an argv element is almost always a missed split"
                )

    def test_a_snapshot_stays_small(self, name):
        # A question should not cost dozens of round trips. This is a ceiling,
        # not a target: if it needs raising, raise it deliberately.
        rec = _exercise(name)
        assert len(rec.calls) <= 12, f"{name} issued {len(rec.calls)} commands"

    def test_every_command_starts_with_a_program_name(self, name):
        rec = _exercise(name)
        for argv in rec.calls:
            assert argv and argv[0] and not argv[0].startswith("-")


class TestKnownArgvShapes:
    """Spot checks on the shapes that have actually gone wrong."""

    def test_slurm_separates_where_from_its_condition(self):
        rec = _exercise("slurm")
        sacctmgr = [c for c in rec.calls if c[0] == "sacctmgr"]
        assert sacctmgr
        for argv in sacctmgr:
            assert not any(a.startswith("where ") for a in argv)

    def test_the_squeue_format_is_a_single_argument(self):
        # "%N|%e" is a format string, not a shell pipe: it has to arrive whole.
        rec = _exercise("slurm")
        squeue = [c for c in rec.calls if c[0] == "squeue"]
        assert squeue
        for argv in squeue:
            if "-o" in argv:
                fmt = argv[argv.index("-o") + 1]
                assert "|" in fmt

    def test_slurm_asks_for_nodes_and_partitions_at_the_same_visibility(self):
        # Slurm applies the hidden-partition filter to BOTH lists, so asking
        # for one with `--all` and the other without makes the two disagree
        # about what the cluster contains: the partition list named a member
        # the node list did not mention, and that partition reported "1 node
        # claimed but unresolved" with no capacity. Measured on a live cluster:
        # 31 nodes plain, 32 with the flag.
        rec = _exercise("slurm")
        for what in ("node", "partition"):
            argv = next(c for c in rec.calls
                        if c[:3] == ["scontrol", "show", what])
            assert "--all" in argv, f"{what} query is narrower: {argv}"

    def test_every_dry_run_carries_its_no_op_flag(self):
        # The flag that keeps a probe read-only must never be optional.
        expected = {
            "slurm": "--test-only",
            "sge": "-w",
            "kubernetes": "--dry-run=server",
        }
        for name, flag in expected.items():
            rec = _exercise(name)
            probes = [
                c for c in rec.calls
                if c[0] in {"sbatch", "qsub"} or "run" in c
            ]
            if not probes:
                continue
            assert any(flag in c for c in probes), f"{name} probe lacks {flag}"


@pytest.mark.parametrize("name", BACKENDS)
class TestCapabilitiesMatchBehaviour:
    """A backend's declaration about itself has to be true.

    `BackendCapabilities` is what the reporting layer trusts when it decides
    whether to say "confirmed" or "declared". A backend that claims it cannot
    dry-run and then produces a verdict — or claims it can and returns nothing —
    makes that decision on false information.
    """

    def _backend(self, name):
        return backends.get(name, Recorder())

    def test_probe_matches_the_claim(self, name):
        backend = self._backend(name)
        claimed = backend.capabilities().probe
        got = backend.probe("q", JobShape(gpus_per_node=1), "acct") is not None
        assert got == claimed, (
            f"{name} claims probe={claimed} but "
            f"{'produced' if got else 'produced no'} verdict"
        )

    def test_a_backend_claiming_no_limits_returns_none(self, name):
        backend = self._backend(name)
        if backend.capabilities().limits:
            return
        with contextlib.suppress(Exception):
            assert backend.load_limits() == {}

    def test_a_backend_claiming_no_free_times_returns_none(self, name):
        backend = self._backend(name)
        if backend.capabilities().free_times:
            return
        with contextlib.suppress(Exception):
            assert backend.load_node_free_times() == {}

    def test_a_backend_claiming_no_identity_reports_no_associations(self, name):
        backend = self._backend(name)
        if backend.capabilities().identity:
            return
        with contextlib.suppress(Exception):
            ident = backend.load_identity()
            # A username is always knowable locally; associations are not.
            assert not ident.accounts
            assert not ident.qos

    def test_a_probe_command_is_named_whenever_probing_is_claimed(self, name):
        caps = self._backend(name).capabilities()
        if caps.probe:
            assert caps.probe_command, f"{name} claims probe but names no command"
        # And a backend that cannot probe must explain why.
        else:
            assert caps.notes, f"{name} cannot probe and says nothing about it"
