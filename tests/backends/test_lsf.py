"""IBM Spectrum LSF."""

from __future__ import annotations

import pytest

from nodetop.backends.lsf import LsfBackend
from nodetop.core.model import JobShape
from nodetop.runner import RecordedRunner


def _nodes(lsf_backend):
    return {n.name: n for n in lsf_backend.load_nodes()}


def _queues(lsf_backend):
    return {q.name: q for q in lsf_backend.load_queues()}


class TestNodes:
    def test_all_hosts(self, lsf_backend):
        assert len(lsf_backend.load_nodes()) == 6

    def test_hardware_comes_from_lshosts(self, lsf_backend):
        n = _nodes(lsf_backend)["gpu-01"]
        assert n.cpus_total == 40
        assert n.memory_mb == 512 * 1024

    def test_accelerator_model_from_the_gpu_table(self, lsf_backend):
        assert _nodes(lsf_backend)["gpu-01"].accelerator.model == "A100"
        assert _nodes(lsf_backend)["gpu-02"].accelerator.model == "V100"

    def test_gpu_occupancy(self, lsf_backend):
        # gpu-02's two cards each hold 30G, so neither is free.
        n = _nodes(lsf_backend)["gpu-02"]
        assert (n.gpus_total, n.gpus_free) == (2, 0)

    def test_closed_adm_is_unschedulable(self, lsf_backend):
        assert _nodes(lsf_backend)["gpu-03"].schedulable is False

    def test_unavail_is_unreachable(self, lsf_backend):
        n = _nodes(lsf_backend)["gpu-04"]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_closed_full_is_busy_not_broken(self, lsf_backend):
        # A host closed because it is full is still perfectly schedulable; it
        # simply has nothing free. Treating it as down would hide real capacity
        # that appears the moment a job ends.
        n = _nodes(lsf_backend)["cpu-01"]
        assert n.schedulable is True
        assert n.cpus_free == 0


class TestQueueStatus:
    """LSF states the accept/start distinction more plainly than anyone."""

    def test_open_active(self, lsf_backend):
        q = _queues(lsf_backend)["gpu"]
        assert (q.enabled, q.started, q.usable) == (True, True, True)

    def test_open_inact_accepts_everything_and_runs_nothing(self, lsf_backend):
        q = _queues(lsf_backend)["frozen"]
        assert (q.enabled, q.started) == (True, False)
        assert q.usable is False
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}

    def test_closed_active(self, lsf_backend):
        q = _queues(lsf_backend)["shut"]
        assert (q.enabled, q.started) == (False, True)
        assert q.usable is False
        assert "QUEUE_DISABLED" in {b.code for b in q.structural_blockers()}

    def test_status_is_read_from_the_positional_table(self, lsf_backend):
        # The value is not labelled "STATUS:" -- it sits under a column
        # header. Searching for a label finds nothing and silently yields a
        # healthy-looking queue.
        assert _queues(lsf_backend)["frozen"].state_raw == "Open:Inact"


class TestQueueEntitlement:
    def test_named_users(self, lsf_backend):
        assert set(_queues(lsf_backend)["gpu"].allow_users) == {"alice", "bob", "carol"}

    def test_all_users_is_not_a_restriction(self, lsf_backend):
        assert _queues(lsf_backend)["normal"].allow_users == ()

    def test_host_list_limits_membership(self, lsf_backend):
        assert set(_queues(lsf_backend)["gpu"].node_names) == {
            "gpu-01", "gpu-02", "gpu-03", "gpu-04"
        }

    def test_hosts_all_means_every_host(self, lsf_backend):
        assert len(_queues(lsf_backend)["normal"].node_names) == 6


class TestLimits:
    def test_runlimit_is_minutes(self, lsf_backend):
        # LSF RUNLIMIT is expressed in minutes: 2880 -> 48 hours.
        assert _queues(lsf_backend)["gpu"].max_walltime_seconds == 2880 * 60

    def test_max_jobs_per_user(self, lsf_backend):
        assert lsf_backend.load_limits()["gpu"].max_jobs == 12


class TestNoProbe:
    def test_probe_returns_none(self, lsf_backend):
        assert lsf_backend.probe("gpu", JobShape()) is None

    def test_capabilities_explain_the_gap(self, lsf_backend):
        caps = lsf_backend.capabilities()
        assert caps.probe is False
        assert any("Open:Inact" in n for n in caps.notes)

    def test_no_free_time_estimate_is_offered(self, lsf_backend):
        # Getting per-job remaining time needs one call per job; returning
        # nothing is more honest than a bad estimate.
        assert lsf_backend.load_node_free_times() == {}


class TestSubmitFlags:
    def test_flags(self, lsf_backend):
        flags = " ".join(
            lsf_backend.submit_flags("gpu", JobShape(nodes=2, cpus_per_task=8,
                                                     gpus_per_node=4, memory_gb=64,
                                                     walltime="4h"))
        )
        assert "-q gpu" in flags
        assert "-n 16" in flags
        assert "ptile=8" in flags
        assert "num=4" in flags
        assert "-W 240" in flags


BMGROUP = """GROUP_NAME   HOSTS
gpu_hosts    gpu-01 gpu-02+1 gpu-03/ gpu-04
cpu_hosts    cpu-01 cpu-02
"""


def _lsf(bqueues: str, bmgroup: tuple[int, str, str]):
    from conftest import read

    from nodetop.backends.lsf import LsfBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    return Cluster.load(
        LsfBackend(RecordedRunner({
            "bhosts -gpu": (0, read("lsf", "bhosts_gpu.txt"), ""),
            "bhosts -w": (0, read("lsf", "bhosts.txt"), ""),
            "lshosts": (0, read("lsf", "lshosts.txt"), ""),
            "bqueues": (0, bqueues, ""),
            "bmgroup": bmgroup,
        })),
        with_free_times=False,
    )


def _grouped_bqueues() -> str:
    from conftest import read

    return read("lsf", "bqueues_l.txt").replace(
        "HOSTS:  gpu-01 gpu-02 gpu-03 gpu-04", "HOSTS:  gpu_hosts/"
    )


class TestHostGroups:
    def test_group_members_are_parsed(self, lsf_backend):
        got = lsf_backend.parse_host_groups(BMGROUP)
        assert got["gpu_hosts"] == ["gpu-01", "gpu-02", "gpu-03", "gpu-04"]

    def test_a_slice_suffix_is_stripped_but_the_hostname_survives(self, lsf_backend):
        # "gpu-02+1" -> "gpu-02". A blanket strip of digits would give "gpu-",
        # which matches no host and silently empties the queue.
        got = lsf_backend.parse_host_groups(BMGROUP)
        assert "gpu-02" in got["gpu_hosts"]
        assert "gpu-" not in got["gpu_hosts"]

    def test_the_header_row_is_skipped(self, lsf_backend):
        assert "GROUP_NAME" not in lsf_backend.parse_host_groups(BMGROUP)

    def test_a_group_scoped_queue_expands(self):
        q = _lsf(_grouped_bqueues(), (0, BMGROUP, "")).queues["gpu"]
        assert len(q.node_names) == 4
        assert q.unresolved_nodes == 0


class TestUnresolvableHosts:
    """An unexpandable group is reported, never guessed at."""

    def test_capacity_is_not_invented(self):
        # The previous behaviour fell back to every host in the cluster, handing
        # a queue restricted to four machines the free capacity of all six --
        # manufacturing exactly the phantom capacity this tool exists to catch.
        q = _lsf(_grouped_bqueues(), (1, "", "bmgroup: not found")).queues["gpu"]
        assert q.node_names == ()
        assert q.effective_free_nodes == 0
        assert q.effective_free_gpus == 0

    def test_the_gap_is_declared(self):
        q = _lsf(_grouped_bqueues(), (1, "", "bmgroup: not found")).queues["gpu"]
        # Surfaces in the report as "+N claimed but unresolved".
        assert q.declared_nodes == 1
        assert q.unresolved_nodes == 1

    def test_an_explicit_host_list_still_resolves_without_bmgroup(self):
        from conftest import read

        q = _lsf(read("lsf", "bqueues_l.txt"), (1, "", "nope")).queues["gpu"]
        assert len(q.node_names) == 4
        assert q.unresolved_nodes == 0

    def test_hosts_all_still_means_every_host(self):
        q = _lsf(_grouped_bqueues(), (1, "", "nope")).queues["normal"]
        assert len(q.node_names) == 6
        assert q.unresolved_nodes == 0


class TestResolveHosts:
    def test_a_mix_of_hostnames_and_groups(self, lsf_backend):
        groups = {"gpu_hosts": ["gpu-01", "gpu-02"]}
        members, unresolved = lsf_backend.resolve_hosts(
            "cpu-01 gpu_hosts/ mystery", ("gpu-01", "gpu-02", "cpu-01"), groups
        )
        assert set(members) == {"gpu-01", "gpu-02", "cpu-01"}
        assert unresolved == ("mystery",)

    def test_cluster_order_is_preserved_and_duplicates_dropped(self, lsf_backend):
        groups = {"a": ["h2", "h1"], "b": ["h1"]}
        members, _ = lsf_backend.resolve_hosts("a b", ("h1", "h2", "h3"), groups)
        assert members == ("h1", "h2")

    def test_a_group_member_not_in_the_cluster_is_ignored(self, lsf_backend):
        groups = {"a": ["h1", "retired-host"]}
        members, unresolved = lsf_backend.resolve_hosts("a", ("h1",), groups)
        assert members == ("h1",)
        assert unresolved == ()


class TestGroupLookupFailureIsReported:
    """Unix groups are LSF's only entitlement signal, so losing them matters.

    An empty group set reads as "cannot tell" downstream, so every
    `USER_ADVOCATES`/group restriction is silently ignored; a partial one reads
    as authoritative and produces a false denial. Neither is acceptable, so the
    failure is raised and `Cluster.load` records it -- leaving `identity` as
    None, which the entitlement filter treats as "cannot filter".
    """

    def test_a_broken_group_database_raises(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.lsf import LsfBackend

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: (_ for _ in ()).throw(OSError("nss unavailable")))
        with pytest.raises(OSError):
            LsfBackend(RecordedRunner({})).load_identity()

    def test_the_cluster_records_it_rather_than_crashing(self, monkeypatch):
        import grp as grp_mod

        from nodetop.backends.lsf import LsfBackend
        from nodetop.core.cluster import Cluster

        monkeypatch.setattr(grp_mod, "getgrall",
                            lambda: (_ for _ in ()).throw(OSError("nss unavailable")))
        cluster = Cluster.load(LsfBackend(RecordedRunner({})))
        assert cluster.identity is None
        assert "identity" in cluster.errors


class TestWrappedQueueValuesAreNotTruncated:
    """`bqueues -l` wraps a long value onto indented following lines.

    `_after` stopped at the end of the label's own line, so a queue whose
    `USERS:` list ran past one line had the tail silently dropped. A truncated
    allowlist is read as authoritative and produces a FALSE DENIAL -- the tool
    reporting no access to a queue that would have taken the job. Same defect
    the PBS backend had in its own dialect.
    """

    BODY = ("QUEUE: gpuq\n"
            "PRIO: 30  STATUS: Open:Active\n"
            "USERS: alice bob carol dave erin frank grace heidi ivan judy\n"
            "    karl liam mona nina\n"
            "HOSTS:  gpu-01 gpu-02\n"
            "    gpu-03 gpu-04\n")

    def test_the_user_list_keeps_every_name(self):
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert len(queue.allow_users) == 14
        assert "nina" in queue.allow_users

    def test_the_host_list_keeps_every_host(self):
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert "gpu-04" in getattr(queue, "labels_hosts", "")

    def test_a_following_label_stops_the_value(self):
        # The continuation must not swallow the next section.
        queue = LsfBackend(RecordedRunner({})).parse_queues(self.BODY)[0]
        assert "gpu-01" not in queue.allow_users
        assert queue.priority == 30

    def test_the_real_fixture_parses_unchanged(self, lsf_backend):
        names = {q.name: q for q in lsf_backend.load_queues()}
        assert names["gpu"].allow_users == ("alice", "bob", "carol")
        assert names["normal"].allow_users == ()


BHOSTS_SUSPENDED = """\
HOST_NAME          STATUS       JL/U    MAX  NJOBS    RUN  SSUSP  USUSP    RSV
susp-01            ok              -     40     40      0     24     16      0
resv-01            ok              -     40     32      2      0      0     30
run-01             ok              -     40     16     16      0      0      0
idle-01            ok              -     40      0      0      0      0      0
"""

BHOSTS_GPU_SUSPENDED = """\
HOST_NAME     GPU_ID   MODEL          MUSED    MRSV  NJOBS    RUN   SUSP    RSV
susp-01       0        NVIDIA_A100     30G      0M      1      0      1      0
susp-01       1        NVIDIA_A100      0M      0M      0      0      0      0
"""


class TestSuspendedAndReservedSlotsAreNotFreeCapacity:
    """`bhosts` NJOBS is the dispatched slots; RUN is only the executing ones.

    LSF documents NJOBS as the slots used by every job dispatched to the host --
    running, suspended AND reserved -- and a suspended job gives nothing back:
    LSF keeps its slots, its memory and its GPUs and resumes it in place. Its
    preemption SUSPENDS rather than requeues by default, so on a preempting
    cluster this is the ordinary state of a busy host and not a corner case.

    Occupancy was read from RUN, so every SSUSP, USUSP and RSV slot on the
    cluster was reported as free capacity: a 40-slot host holding 40 preempted
    slots read as **empty**. Capacity on the screen that no job can have is the
    one failure this tool exists to catch.
    """

    def _nodes(self, bhosts: str = BHOSTS_SUSPENDED, gpu: str = ""):
        parsed = LsfBackend(RecordedRunner({})).parse_nodes(bhosts, "", gpu)
        return {n.name: n for n in parsed}

    def test_a_fully_suspended_host_has_nothing_free(self):
        n = self._nodes()["susp-01"]
        assert (n.cpus_alloc, n.cpus_free) == (40, 0)

    def test_reserved_slots_are_held_not_free(self):
        # RSV is held by the scheduler for a pending job it is backfilling
        # around; handing those slots out is how that job never starts.
        n = self._nodes()["resv-01"]
        assert (n.cpus_alloc, n.cpus_free) == (32, 8)

    def test_a_suspended_job_still_holds_its_card(self):
        # GPU 0 is SUSP with 30G still resident on it; GPU 1 is genuinely free.
        n = self._nodes(gpu=BHOSTS_GPU_SUSPENDED)["susp-01"]
        assert (n.gpus_total, n.gpus_alloc, n.gpus_free) == (2, 1, 1)

    # -- controls: these hold before and after -----------------------------
    def test_control_a_plainly_running_host_is_read_the_same(self):
        n = self._nodes()["run-01"]
        assert (n.cpus_alloc, n.cpus_free) == (16, 24)

    def test_control_an_idle_host_is_still_wholly_idle(self):
        n = self._nodes()["idle-01"]
        assert (n.cpus_alloc, n.cpus_free) == (0, 40)

    def test_control_the_recorded_fixture_is_unchanged(self, lsf_backend):
        # Every SSUSP/USUSP/RSV in it is zero, so NJOBS == RUN throughout and
        # the recorded cluster must read exactly as it did.
        got = {n.name: n.cpus_alloc for n in lsf_backend.load_nodes()}
        assert got == {"gpu-01": 0, "gpu-02": 40, "gpu-03": 0, "gpu-04": 0,
                       "cpu-01": 128, "cpu-02": 16}


BHOSTS_STATUSES = """\
HOST_NAME          STATUS       JL/U    MAX  NJOBS    RUN  SSUSP  USUSP    RSV
lim-01             closed_LIM      -     40      0      0      0      0      0
wind-01            closed_Wind     -     40      0      0      0      0      0
later-01           closed_Zorp     -     40      0      0      0      0      0
busy-01            closed_Busy     -     40      8      8      0      0      0
full-01            closed_Full     -     40     40     40      0      0      0
ok-01              ok              -     40      0      0      0      0      0
"""


class TestAnUnrecognisedStatusDoesNotReadAsHealthy:
    """A `bhosts` STATUS the table does not list must not read as `ok`.

    The lookup was `_STATUS_TO_CONDITION.get(status)`, and `None` is the
    table's word for *nothing wrong here* -- the answer it gives `ok`. So every
    status not in it produced a healthy host advertising its whole complement as
    free. `closed_LIM` (sbatchd unreachable) and `closed_Wind` (shut by its own
    run window) are documented `bhosts -w` values and were both missing, and so
    is anything a newer LSF or a site's wrapper prints: 40 phantom slots per
    host, in the direction that gets a job sent to a machine that cannot run it.
    """

    def _nodes(self):
        parsed = LsfBackend(RecordedRunner({})).parse_nodes(BHOSTS_STATUSES)
        return {n.name: n for n in parsed}

    def test_closed_lim_is_down(self):
        n = self._nodes()["lim-01"]
        assert n.schedulable is False
        assert "DOWN" in n.conditions

    def test_closed_wind_is_drained(self):
        n = self._nodes()["wind-01"]
        assert n.schedulable is False
        assert "DRAIN" in n.conditions

    def test_a_status_nobody_recognises_degrades_to_unknown(self):
        n = self._nodes()["later-01"]
        assert n.conditions == frozenset({"UNKNOWN"})
        assert n.schedulable is False

    def test_what_lsf_actually_said_is_still_visible(self):
        # Degrading must not hide the evidence: the raw status is carried
        # through so the reader can see the word the table did not know.
        n = self._nodes()["later-01"]
        assert n.state_raw == "closed_Zorp"
        assert "closed_Zorp" in n.reason

    # -- controls: these hold before and after -----------------------------
    def test_control_closed_full_and_closed_busy_stay_schedulable(self):
        # The two deliberate `None` entries: busy, not broken. A blanket
        # "anything closed is out" would hide capacity that reappears the
        # moment a job ends.
        nodes = self._nodes()
        assert nodes["full-01"].schedulable is True
        assert nodes["busy-01"].schedulable is True
        assert nodes["busy-01"].conditions == frozenset()

    def test_control_ok_carries_no_condition_and_no_reason(self):
        n = self._nodes()["ok-01"]
        assert n.conditions == frozenset()
        assert n.reason == ""

    def test_control_the_recorded_fixture_keeps_its_verdicts(self, lsf_backend):
        got = {n.name: sorted(n.conditions) for n in lsf_backend.load_nodes()}
        assert got == {"gpu-01": [], "gpu-02": [], "gpu-03": ["DRAIN"],
                       "gpu-04": ["DOWN"], "cpu-01": [], "cpu-02": []}


#: A minimal `lshosts -w` / `bhosts -w` pair, so a size can be driven through
#: the real parser rather than only through `_mem_to_mb`. `maxmem` is the
#: caller under test; everything else is filler that parses.
def _lsf_pair(maxmem: str) -> tuple[str, str]:
    lshosts = (
        "HOST_NAME      type    model            cpuf ncpus maxmem maxswp server RESOURCES\n"
        f"c-01           X86_64  AMD_EPYC        55.0   128 {maxmem}    16G      Yes (mg)\n"
    )
    bhosts = (
        "HOST_NAME            STATUS  JL/U  MAX  NJOBS  RUN SSUSP USUSP  RSV\n"
        "c-01                 ok         -  128      0    0     0     0    0\n"
    )
    return bhosts, lshosts


class TestLsfMemorySizes:
    """LSF's suffixes are binary and case-blind; its BARE number is a setting.

    Nothing here changes a number LSF actually prints -- every suffixed form
    keeps its exact value, and that is the control the rest of this class is
    measured against. LSF is the ordinary convention, unlike `sge._mem_to_mb`
    where case selects decimal vs binary.

    A bare size is the one that moved. It has no unit in the string at all:
    LSF takes it from ``LSF_UNIT_FOR_LIMITS`` in ``lsf.conf`` (MB by default
    in 10.1, ``KB`` in older releases, and ``GB`` at plenty of sites).
    `_mem_to_mb` used to assume MB, which on a ``KB`` site overstates a host's
    memory by **1024x** -- the largest wrong answer this backend can produce,
    and in the phantom-capacity direction. The assumption was kept last round
    as unexercised, because ``lshosts``'s ``maxmem`` "always" prints a suffix;
    that is an inference from IBM's example output rather than anything the
    ``lshosts`` reference guarantees, and it is not a claim worth 1024x. So
    the unit is now read from ``lsf.conf`` where that is possible, and
    otherwise the answer is 0 = "not read", which `capacity.hardware_ok`
    already handles by gating on ``memory_mb > 0``.

    Also fixed earlier and still pinned: ``[\\d.]+`` accepted a multi-dot
    string that `float` then refused, raising `ValueError` out of
    `LsfBackend.parse_nodes` and taking every node with it.
    """

    @pytest.fixture(autouse=True)
    def _no_site_unit_from_this_host(self, tmp_path, monkeypatch):
        """No test in this class reads the real ``lsf.conf``, wherever it runs.

        `_site_unit_mb` looks in ``$LSF_ENVDIR`` and then ``/etc/lsf.conf``,
        which is right for the product and would make every "unknown unit"
        assertion here depend on whether the host has LSF installed -- the
        non-hermetic suite `conftest._detect_finds_a_recorded_backend`
        describes. `raising=False` on both, so this is a no-op against a build
        that has neither name and the controls stay state-independent.
        """
        monkeypatch.delenv("LSF_ENVDIR", raising=False)
        monkeypatch.setattr(
            "nodetop.backends.lsf._LSF_CONF_FALLBACK",
            str(tmp_path / "no-such-dir" / "lsf.conf"),
            raising=False,
        )

    #: CONTROL: every SUFFIXED form, exact. None of these consults the site
    #: unit -- a suffix says what it means -- so every entry holds both before
    #: and after the bare-number change, which is what makes it the control.
    KNOWN_GOOD = [
        ("256G", 262144), ("512G", 524288), ("2016M", 2016),
        ("15.9G", 16281), ("1K", 0), ("1024K", 1), ("1000000 K", 976),
        ("1T", 1048576), ("1P", 1073741824),
        ("1M", 1), ("16G", 16384),
        ("512g", 524288), ("256g", 262144),   # LSF has no case distinction
    ]

    @pytest.mark.parametrize("text,mb", KNOWN_GOOD)
    def test_control_every_readable_size_keeps_its_exact_value(self, text, mb):
        from nodetop.backends.lsf import _mem_to_mb

        assert _mem_to_mb(text) == mb

    @pytest.mark.parametrize("text,mb", KNOWN_GOOD)
    def test_control_a_suffix_outranks_the_site_unit(self, text, mb, tmp_path, monkeypatch):
        """CONTROL, and the one that matters: ``256G`` is 256 GiB on every site.

        ``LSF_UNIT_FOR_LIMITS`` scales the sizes LSF prints *without* a letter.
        A suffixed size is already explicit, so the setting must not touch it.
        Deliberately state-independent: this holds under the old code (which
        never looked at ``lsf.conf``) and under the new (which does), so a
        bare-number fix that leaked into the suffixed branch fails here.
        """
        from nodetop.backends.lsf import LsfBackend, _mem_to_mb

        (tmp_path / "lsf.conf").write_text("LSF_UNIT_FOR_LIMITS=KB\n")
        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path))
        assert _mem_to_mb(text) == mb
        if " " not in text:
            # Space-free forms only, and the exclusion is load-bearing: a
            # positional `lshosts` column cannot hold a space, so `1000000 K`
            # is a `_mem_to_mb`-level form rather than a cell. Fed to
            # `parse_nodes` it splits in two, `maxmem` reads as bare `1000000`,
            # and on this KB site that lands on 976 by arithmetic accident --
            # a control that agrees with the fix for the wrong reason, and it
            # was the teeth that showed it up.
            node = LsfBackend().parse_nodes(*_lsf_pair(text))[0]
            assert node.memory_mb == mb

    def test_control_the_recorded_fixture_keeps_its_memory_figures(self, lsf_backend):
        """CONTROL: the recorded `lshosts` is all suffixed, so nothing moves.

        Resolving the site unit added file access to `parse_nodes`; this pins
        that the recorded path is unaffected by it either way.
        """
        got = {n.name: n.memory_mb for n in lsf_backend.load_nodes()}
        assert got == {"gpu-01": 524288, "gpu-02": 524288, "gpu-03": 524288,
                       "gpu-04": 524288, "cpu-01": 262144, "cpu-02": 262144}

    @pytest.mark.parametrize("bare", ["2048", "1", "1000000", "0"])
    def test_a_bare_size_with_no_readable_site_unit_is_not_read(self, bare, monkeypatch):
        """A bare number whose unit nobody can look up answers 0, not MB.

        This is the 1024x assumption. ``LSF_UNIT_FOR_LIMITS`` is cluster-wide
        and lives in ``lsf.conf``; with no ``LSF_ENVDIR`` and no
        ``/etc/lsf.conf`` there is nothing to read it from, and "MB" would be a
        guess that is 1024x wrong on a ``KB`` site. 0 is this package's "not
        read" -- `hardware_ok` gates on ``memory_mb > 0``.
        """
        from nodetop.backends.lsf import LsfBackend, _mem_to_mb

        monkeypatch.delenv("LSF_ENVDIR", raising=False)
        monkeypatch.setattr("nodetop.backends.lsf._site_unit_mb", lambda: None)
        assert _mem_to_mb(bare) == 0
        # And through the real caller: the host still appears, memory unread.
        node = LsfBackend().parse_nodes(*_lsf_pair(bare))[0]
        assert node.name == "c-01"
        assert node.cpus_total == 128
        assert node.memory_mb == 0

    @pytest.mark.parametrize("unit,mb", [
        ("KB", 2), ("MB", 2048), ("GB", 2048 * 1024), ("TB", 2048 * 1024 * 1024),
    ])
    def test_a_bare_size_reads_its_unit_from_lsf_conf(self, unit, mb, tmp_path, monkeypatch):
        """Where the site setting IS readable, the bare number is read right.

        LSF finds ``lsf.conf`` via ``$LSF_ENVDIR`` (else the ``/etc/lsf.conf``
        symlink), so any client that could run ``lshosts`` had one -- which is
        why looking it up beats answering 0 everywhere. The 1024x spread
        between these rows is the size of the assumption being removed.
        """
        from nodetop.backends.lsf import LsfBackend

        (tmp_path / "lsf.conf").write_text(
            f"LSB_SHAREDIR=/x/work\nLSF_UNIT_FOR_LIMITS={unit}\n"
        )
        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path))
        assert LsfBackend().parse_nodes(*_lsf_pair("2048"))[0].memory_mb == mb

    def test_a_commented_out_setting_is_not_a_setting(self, tmp_path, monkeypatch):
        """``lsf.conf`` ships parameters present-but-commented.

        Reading ``#LSF_UNIT_FOR_LIMITS=GB`` as GB would invent a site decision
        nobody made -- and be 1024x off in the other direction.
        """
        from nodetop.backends.lsf import _site_unit_mb

        (tmp_path / "lsf.conf").write_text(
            "# LSF_UNIT_FOR_LIMITS=GB\n#LSF_UNIT_FOR_LIMITS=TB\n"
        )
        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path))
        assert _site_unit_mb() is None

    @pytest.mark.parametrize("body,mb", [
        ("LSF_UNIT_FOR_LIMITS=GB\n", 1024),
        ("  LSF_UNIT_FOR_LIMITS = GB\n", 1024),
        ('LSF_UNIT_FOR_LIMITS="GB"\n', 1024),
        ("LSF_UNIT_FOR_LIMITS=G\n", 1024),
        ("#LSF_UNIT_FOR_LIMITS=GB\nLSF_UNIT_FOR_LIMITS=KB\n", 1 / 1024),
        ("LSF_UNIT_FOR_LIMITS=EB\n", None),      # documented value, no row
        ("LSF_UNIT_FOR_LIMITS=\n", None),
        ("LSB_SHAREDIR=/x\n", None),
        ("", None),
    ])
    def test_the_site_unit_is_read_or_declared_unknown(self, body, mb, tmp_path, monkeypatch):
        from nodetop.backends.lsf import _site_unit_mb

        (tmp_path / "lsf.conf").write_text(body)
        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path))
        assert _site_unit_mb() == mb

    def test_an_unreadable_lsf_conf_is_not_a_crash(self, tmp_path, monkeypatch):
        """A missing or unreadable ``lsf.conf`` must degrade, not raise.

        `load_nodes` would otherwise lose every host to an `OSError` from a
        directory named ``lsf.conf`` or an unset-but-wrong ``LSF_ENVDIR`` --
        an empty listing this tool reports as "wrong backend or dead control
        plane", which is the misdiagnosis the rest of this file guards against.
        """
        from nodetop.backends.lsf import LsfBackend, _site_unit_mb

        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path / "nope"))
        assert _site_unit_mb() is None
        (tmp_path / "lsf.conf").mkdir()          # a directory, not a file
        monkeypatch.setenv("LSF_ENVDIR", str(tmp_path))
        assert _site_unit_mb() is None
        assert LsfBackend().parse_nodes(*_lsf_pair("256G"))[0].memory_mb == 262144

    @pytest.mark.parametrize("text", [
        "1.2.3.4G", "256.1.2G", ".", "..", "1..2", "8.8.8.8",
    ])
    def test_a_multi_dot_size_reads_as_unknown_instead_of_raising(self, text):
        from nodetop.backends.lsf import _mem_to_mb

        assert _mem_to_mb(text) == 0

    @pytest.mark.parametrize("text,note", [
        ("1E", "EB is a documented LSF_UNIT_FOR_LIMITS value; the table stops at P"),
        ("1Gi", "`Gi` is Kubernetes' spelling, not LSF's"),
        ("1gw", "`w` (words) is PBS vocabulary"),
        ("512G (mg)", "a whole lshosts line is not a size"),
        ("infinity", ""),
        ("-", "lshosts prints this when it has no info for a host"),
    ])
    def test_a_size_lsf_cannot_read_is_unknown_not_one_megabyte(self, text, note):
        # Unanchored, these all fell through to the bare-number branch and
        # became their leading digits in MB -- `1E` (an exabyte, if LSF ever
        # prints one) read as 1 MB. 0 is this package's "not read", and
        # `hardware_ok` gates on `memory_mb > 0`.
        from nodetop.backends.lsf import _mem_to_mb

        assert _mem_to_mb(text) == 0, note

    def test_one_malformed_maxmem_does_not_empty_the_node_listing(self):
        from nodetop.backends.lsf import LsfBackend

        lshosts = (
            "HOST_NAME      type    model            cpuf ncpus maxmem maxswp server RESOURCES\n"
            "good-01        X86_64  AMD_EPYC        55.0   128   256G    16G      Yes (mg)\n"
            "bad-01         X86_64  AMD_EPYC        55.0   128 1.2.3.4G  16G      Yes (mg)\n"
        )
        bhosts = (
            "HOST_NAME            STATUS  JL/U  MAX  NJOBS  RUN SSUSP USUSP  RSV\n"
            "good-01              ok         -  128      0    0     0     0    0\n"
            "bad-01               ok         -  128      0    0     0     0    0\n"
        )
        nodes = {n.name: n for n in LsfBackend().parse_nodes(bhosts, lshosts)}
        assert sorted(nodes) == ["bad-01", "good-01"]
        assert nodes["good-01"].memory_mb == 262144
        assert nodes["bad-01"].memory_mb == 0

    def test_control_the_recorded_fixture_totals_are_unchanged(self, lsf_backend):
        got = {n.name: n.memory_mb for n in lsf_backend.load_nodes()}
        assert got == {"gpu-01": 524288, "gpu-02": 524288, "gpu-03": 524288,
                       "gpu-04": 524288, "cpu-01": 262144, "cpu-02": 262144}
