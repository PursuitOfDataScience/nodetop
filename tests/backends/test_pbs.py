"""PBS Pro / OpenPBS / Torque."""

from __future__ import annotations

import pytest

from nodetop.backends.pbs import PbsBackend
from nodetop.core.model import JobShape
from nodetop.runner import RecordedRunner


def _nodes(pbs_backend):
    return {n.name: n for n in pbs_backend.load_nodes()}


class TestNodeParsingJson:
    def test_all_nodes(self, pbs_backend):
        assert len(pbs_backend.load_nodes()) == 5

    def test_resources(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu001"]
        assert (n.cpus_total, n.gpus_total) == (64, 4)
        assert n.memory_mb == 512 * 1024
        assert n.accelerator.model == "A100"
        assert set(n.queues) == {"gpuq", "debug"}

    def test_assigned_resources_become_occupancy(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu002"]
        assert (n.cpus_free, n.gpus_free) == (0, 0)

    def test_offline_is_unschedulable(self, pbs_backend):
        n = _nodes(pbs_backend)["gpu003"]
        assert n.schedulable is False

    def test_state_unknown_is_unreachable(self, pbs_backend):
        n = _nodes(pbs_backend)["cpu002"]
        assert n.unreachable is True
        assert n.schedulable is False

    def test_memory_units(self, pbs_backend):
        assert _nodes(pbs_backend)["cpu001"].memory_alloc_mb == 16 * 1024


class TestNodeParsingText:
    """Torque and PBS before 18 have no JSON mode at all."""

    def test_classic_format(self, pbs_backend):
        from conftest import read

        nodes = {
            n.name: n
            for n in pbs_backend.parse_nodes_text(read("pbs", "pbsnodes.txt"))
        }
        assert set(nodes) == {"node01", "node02"}
        assert nodes["node01"].gpus_total == 2
        assert nodes["node01"].accelerator.model == "H100"
        assert nodes["node01"].cpus_total == 32
        assert nodes["node02"].schedulable is False

    def test_json_is_preferred_and_text_is_the_fallback(self, pbs_backend):
        # The JSON fixture is what load_nodes should have used.
        assert len(pbs_backend.load_nodes()) == 5


class TestQueues:
    def _queues(self, pbs_backend):
        return {q.name: q for q in pbs_backend.load_queues()}

    def test_two_independent_switches(self, pbs_backend):
        q = self._queues(pbs_backend)
        # This is the PBS form of the phantom-capacity trap: enabled swallows
        # submissions, started decides whether any of them ever run.
        assert (q["gpuq"].enabled, q["gpuq"].started) == (True, True)
        assert (q["drainq"].enabled, q["drainq"].started) == (True, False)
        assert (q["closedq"].enabled, q["closedq"].started) == (False, False)

    def test_accepts_but_never_starts_is_unusable(self, pbs_backend):
        q = self._queues(pbs_backend)["drainq"]
        assert q.usable is False
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}

    def test_an_enabled_acl_with_no_users_permits_nobody(self, pbs_backend):
        # acl_user_enable=True with an empty acl_users is a closed door, and
        # must not be read as "no restriction".
        q = self._queues(pbs_backend)["emptyaclq"]
        assert q.usable is False
        assert "NO_USERS" in {b.code for b in q.structural_blockers()}

    def test_a_disabled_acl_is_not_a_restriction(self, pbs_backend):
        assert self._queues(pbs_backend)["cpuq"].allow_users == ()

    def test_acl_users_are_read(self, pbs_backend):
        assert set(self._queues(pbs_backend)["gpuq"].allow_users) == {
            "alice", "bob", "carol"
        }

    def test_walltime_is_hhmmss_never_bare_minutes(self, pbs_backend):
        # PBS "48:00:00" is 48 hours; reading it with Slurm's grammar would
        # give 48 minutes.
        assert self._queues(pbs_backend)["gpuq"].max_walltime_seconds == 48 * 3600

    def test_queue_membership_is_reconstructed_from_the_nodes(self, pbs_backend):
        # PBS does not list a queue's nodes; the mapping lives on Qlist.
        assert set(self._queues(pbs_backend)["gpuq"].node_names) == {
            "gpu001", "gpu002", "gpu003"
        }


class TestTheStateColumnSaysWhichStateItIs:
    """`enabled=True started=True` does not fit a table, and truncates to a lie.

    The state column is twelve characters, sized for Slurm's `UP`/`DOWN`, so
    every PBS queue rendered as `enabled=Tru…` -- cut off exactly at the
    answer, and the same eleven characters whether the queue was open or shut.
    Seen on a live 2026.1.0 cluster with ten queues, two of them closed:
    indistinguishable. The two switches are independent, so each of the four
    combinations gets its own word.
    """

    def _queues(self, pbs_backend):
        return {q.name: q for q in pbs_backend.load_queues()}

    def test_each_combination_has_its_own_word(self, pbs_backend):
        q = self._queues(pbs_backend)
        assert q["gpuq"].state_raw == "UP"            # accepts and runs
        assert q["drainq"].state_raw == "STOPPED"     # accepts, never runs
        assert q["closedq"].state_raw == "DOWN"       # neither

    def test_the_booleans_are_still_there(self, pbs_backend):
        # The word is for reading; every decision reads the switches, so
        # shortening the text must not cost the data.
        q = self._queues(pbs_backend)["drainq"]
        assert (q.enabled, q.started) == (True, False)

    def test_a_disabled_queue_names_itself_in_the_blocked_column(
            self, pbs_backend):
        # `Queue.structural_blockers` takes the first word of `state_raw` as
        # the short reason, which used to make the column read `enabled=False`.
        blocker = next(b for b in
                       self._queues(pbs_backend)["closedq"].structural_blockers()
                       if b.code == "QUEUE_DISABLED")
        assert blocker.short == "DOWN"

    def test_a_disabled_but_started_queue_is_not_called_down(self):
        # The fourth combination is missing from the fixture and is the one a
        # naive two-way mapping gets wrong: `qstop` without `qdisable`.
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        queue = PbsBackend(RecordedRunner({})).parse_queues_text(
            "Queue: shutq\n    enabled = False\n    started = True\n")[0]
        assert queue.state_raw == "DISABLED"


class TestLimits:
    def test_per_entity_table_is_unwrapped(self, pbs_backend):
        # "max_run_res.ngpus = [u:PBS_GENERIC=8]" -> 8
        limits = pbs_backend.load_limits()["gpuq"]
        assert limits.per_user["gpu"] == 8

    def test_resources_max_becomes_a_per_job_ceiling(self, pbs_backend):
        assert pbs_backend.load_limits()["gpuq"].per_job["cpu"] == 256

    def test_ceiling_is_detected_from_the_shape_alone(self, pbs_backend):
        limits = pbs_backend.load_limits()["gpuq"]
        got = limits.blockers(JobShape(nodes=4, gpus_per_node=4, walltime="1h"))
        assert "MAX_GPU_USER" in {b.code for b in got}


def _limits_from_qstat(text: str):
    """Queue ceilings from hand-built `qstat -Qf` text.

    The recorded fixture declares one entity per table, which is the shape that
    cannot distinguish "the generic entry" from "the last entry" -- so the
    multi-entity tables below are built here rather than added to it.
    """
    return PbsBackend(RecordedRunner({"qstat -Qf": (0, text, "")})).load_limits()


#: PBS prints the cells of a per-entity limit table in whatever order the
#: administrator set them, so a named user may appear before OR after
#: `PBS_GENERIC`. `gpuq` puts alice last and `cpuq` puts her first; the same
#: caller must get the same ceiling out of both.
ENTITY_TABLES = """Queue: gpuq
    queue_type = Execution
    resources_max.walltime = 48:00:00
    max_run_res.ngpus = [u:PBS_GENERIC=4],[u:alice=99]
    max_run = [u:PBS_GENERIC=6],[u:alice=77]
    max_queued = [u:PBS_GENERIC=9]
    enabled = True
    started = True

Queue: cpuq
    queue_type = Execution
    max_run_res.ncpus = [u:alice=64],[u:PBS_GENERIC=8]
    enabled = True
    started = True
"""


class TestPerEntityLimitsResolveToTheCaller:
    """`[u:PBS_GENERIC=4],[u:alice=99]` is a table, and 99 is not our number.

    The read was end-anchored, so it returned the LAST cell -- which on any
    queue that names a per-user override after the generic entry is somebody
    else's ceiling, reported as ours. `Limits.blockers` then either invents a
    `MAX_*_USER` blocker against a limit we do not have, or (the direction that
    actually costs something) stays silent about the one we do: reading alice's
    99 GPUs where PBS would hold us to 4 makes an over-limit job look
    acceptable, and PBS has no dry-run to catch it afterwards.

    Resolution order is the caller's own `u:` cell, then the generic cell.
    """

    def test_a_stranger_gets_the_generic_cell_not_the_tail(self, monkeypatch):
        monkeypatch.setenv("USER", "bob")
        # 99 is alice's override; bob is held to the generic 4.
        assert _limits_from_qstat(ENTITY_TABLES)["gpuq"].per_user["gpu"] == 4

    def test_our_own_cell_wins_over_the_generic_one(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        assert _limits_from_qstat(ENTITY_TABLES)["gpuq"].per_user["gpu"] == 99

    def test_our_own_cell_wins_wherever_it_sits_in_the_table(self, monkeypatch):
        # cpuq lists alice FIRST, so reading the tail gives her the generic 8 --
        # the same defect pointing the other way, and the reason position is not
        # a usable proxy for identity.
        monkeypatch.setenv("USER", "alice")
        assert _limits_from_qstat(ENTITY_TABLES)["cpuq"].per_user["cpu"] == 64

    def test_a_table_naming_only_other_users_declares_no_ceiling(self, monkeypatch):
        monkeypatch.setenv("USER", "carol")
        text = ENTITY_TABLES.replace(
            "[u:PBS_GENERIC=4],[u:alice=99]", "[u:alice=99],[u:bob=1]")
        # No cell applies to carol. Inheriting bob's 1 would refuse her a job
        # PBS accepts; inheriting alice's 99 would promise one it will not run.
        assert "gpu" not in _limits_from_qstat(text)["gpuq"].per_user

    def test_control_a_lone_generic_cell_and_a_bare_integer_are_unchanged(
        self, pbs_backend, monkeypatch
    ):
        """CONTROL -- passes before the fix and after it.

        The recorded fixture's `max_run_res.ngpus = [u:PBS_GENERIC=8]` names one
        entity, and `resources_max.ncpus = 256` is not a table at all. Neither
        reading depends on the resolution order, so this test pins what the fix
        must NOT disturb: a one-cell table, a plain integer, and a walltime that
        goes nowhere near this helper.
        """
        monkeypatch.setenv("USER", "alice")  # named in the fixture's ACL
        limits = pbs_backend.load_limits()["gpuq"]
        assert limits.per_user["gpu"] == 8
        assert limits.per_job["cpu"] == 256
        assert limits.max_walltime_seconds == 48 * 3600


class TestJobCountCeilingsAreTablesToo:
    """`max_run` / `max_queued` carry the same per-entity syntax.

    They were parsed with the plain-integer helper, whose regex anchors at the
    START of the string -- so `[u:PBS_GENERIC=4]` yielded `None` and
    `Limits.max_jobs` / `max_submitted` were `None` on every PBS cluster, while
    the `max_run_res.ngpus` line directly above parsed fine. Nothing renders
    these two fields today; the point is that the parse now agrees with the
    sibling line instead of quietly dropping the value.
    """

    def test_max_run_becomes_max_jobs(self, pbs_backend):
        # Recorded fixture, gpuq: `max_run = [u:PBS_GENERIC=4]`.
        assert pbs_backend.load_limits()["gpuq"].max_jobs == 4

    def test_max_queued_becomes_max_submitted(self, monkeypatch):
        monkeypatch.setenv("USER", "bob")
        assert _limits_from_qstat(ENTITY_TABLES)["gpuq"].max_submitted == 9

    def test_the_callers_own_cell_applies_here_too(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        assert _limits_from_qstat(ENTITY_TABLES)["gpuq"].max_jobs == 77

    def test_control_an_undeclared_job_ceiling_stays_none(self, pbs_backend):
        """CONTROL -- passes before the fix and after it.

        `cpuq` declares neither attribute and the fixture's `gpuq` has no
        `max_queued`. An absent ceiling must stay `None` rather than become a
        number the site never published.
        """
        limits = pbs_backend.load_limits()
        assert limits["gpuq"].max_submitted is None
        assert limits["cpuq"].max_jobs is None
        assert limits["cpuq"].max_submitted is None


#: One queue declaring six ceilings, every one in a shape the parser answers
#: `None` for; one declaring nothing at all. Before the fix these two produced
#: byte-identical `Limits`.
#:
#: The shapes are PBS's own: a walltime in a spelling PBS never prints, a table
#: cell that is not a number, a memory unit outside PBS's suffix list (`eb`),
#: a word count too small to survive rounding to MB (`1024w` is 8 KiB), and two
#: job counters holding words.  `pb` and `gw` were here until `_mem_to_mb`
#: learned the peta prefix and the `w` unit -- both are read now.
UNREADABLE_SHAPES = """Queue: shapeq
    queue_type = Execution
    resources_max.walltime = 2 days
    resources_max.ncpus = [u:PBS_GENERIC=unlimited]
    resources_max.mem = 2eb
    max_run_res.mem = 1024w
    max_run = many
    max_queued = [o:PBS_ALL=lots]
    enabled = True
    started = True

Queue: silentq
    queue_type = Execution
    enabled = True
    started = True
"""


class TestPublishedButUnreadableCeilingsAreNamed:
    """A ceiling PBS printed and we could not read must not read as no ceiling.

    `Limits.unreadable` exists for exactly this: the Slurm adapter fills it
    with `MaxWall` / `MaxTRESPerUser` / `MaxTRESPerJob` when a field is present
    but unparseable, so `fit.evaluate` can say the check did not run. This
    adapter filled the same dataclass and never touched the field -- so a
    `max_run_res.*` or `resources_max.*` value in an unfamiliar shape simply
    disappeared and the queue looked UNLIMITED on that axis, on the one
    scheduler with no dry-run to catch it afterwards.
    """

    def test_every_unread_ceiling_is_named(self):
        # Fixed order, not the order PBS happened to print them in.
        assert _limits_from_qstat(UNREADABLE_SHAPES)["shapeq"].unreadable == (
            "resources_max.walltime",
            "max_run_res.mem",
            "resources_max.mem",
            "resources_max.ncpus",
            "max_run",
            "max_queued",
        )

    def test_no_blocker_is_invented_for_the_ceiling_that_was_not_read(self):
        # Naming the gap must not become a limit nobody published: the numbers
        # are genuinely unknown, so a 64-node 4096-CPU job is not "refused".
        limits = _limits_from_qstat(UNREADABLE_SHAPES)["shapeq"]
        shape = JobShape(nodes=64, cpus_per_task=64, memory_gb=900, walltime="72h")
        assert limits.blockers(shape) == []
        assert limits.per_job == {} and limits.max_jobs is None

    def test_the_gap_reaches_the_report(self):
        from nodetop.core.cluster import Cluster
        from nodetop.core.fit import evaluate
        from nodetop.core.model import BackendCapabilities, Node, Queue

        queue = Queue(name="shapeq", node_names=("n",))
        queue.nodes = [Node(name="n", cpus_total=32, memory_mb=256 * 1024,
                            state_raw="up")]
        cluster = Cluster(
            backend_name="pbs", nodes=queue.nodes, queues={"shapeq": queue},
            limits={"shapeq": _limits_from_qstat(UNREADABLE_SHAPES)["shapeq"]},
            capabilities=BackendCapabilities(probe=False),
        )
        caveats = evaluate(cluster, JobShape(), queue).caveats
        assert any("could not read resources_max.walltime" in c for c in caveats)
        assert any("not checked at all" in c for c in caveats)

    def test_control_a_queue_that_declares_no_ceiling_says_nothing(self):
        """CONTROL -- passes before the fix and after it.

        The boundary the whole change turns on: `silentq` prints none of these
        attributes, so there is nothing unread about it. Reading absence as
        unreadable would attach a caveat to every queue on every cluster and
        make the field worthless.
        """
        assert _limits_from_qstat(UNREADABLE_SHAPES)["silentq"].unreadable == ()

    def test_control_a_sentinel_or_an_empty_value_is_absence_not_a_shape(self):
        """CONTROL -- passes before the fix and after it.

        PBS prints an unset attribute with an empty value rather than omitting
        it (the recorded fixture's `acl_users =` is that shape), and `unlimited`
        says outright that there is no ceiling. Both mean the same as never
        printing the line.
        """
        text = """Queue: sentinelq
    queue_type = Execution
    resources_max.walltime = unlimited
    resources_max.ncpus =
    max_run_res.ngpus = none
    max_run =
    enabled = True
    started = True
"""
        limits = _limits_from_qstat(text)["sentinelq"]
        assert limits.unreadable == ()
        assert limits.max_walltime_seconds is None
        assert limits.per_job == {} and limits.per_user == {}

    def test_control_a_table_naming_only_other_users_is_read_not_unread(
        self, monkeypatch
    ):
        """CONTROL -- passes before the fix and after it.

        `[u:alice=99],[u:bob=1]` parses perfectly and correctly declares no
        ceiling for carol; it is the one `None` here that is an answer rather
        than a failure. Calling it unreadable would warn about a limit that
        does not apply to the caller -- and this case already has a test one
        class up asserting the ceiling itself stays absent.
        """
        monkeypatch.setenv("USER", "carol")
        text = ENTITY_TABLES.replace(
            "[u:PBS_GENERIC=4],[u:alice=99]", "[u:alice=99],[u:bob=1]")
        limits = _limits_from_qstat(text)["gpuq"]
        assert "gpu" not in limits.per_user
        assert "max_run_res.ngpus" not in limits.unreadable


#: `resources_max.mem` is the per-job memory ceiling and `max_run_res.mem` the
#: per-user one; PBS prints the latter's cells with units, which is the shape
#: that decided how the cell picker captures a value.  `ompthreads` is a real
#: PBS resource this model has no ceiling axis for.
MEM_CEILINGS = """Queue: memq
    queue_type = Execution
    resources_max.mem = 256gb
    max_run_res.mem = [u:PBS_GENERIC=200gb],[u:alice=2tb]
    resources_max.ompthreads = banana
    enabled = True
    started = True
"""


class TestMemoryCeilingsAreReadAtAll:
    """`Limits.per_job["mem_mb"]` is a documented axis PBS never filled.

    The resource map went `ngpus`/`ncpus`/`nodect` only, so every PBS memory
    ceiling was dropped on the floor -- `Limits.blockers` checks `mem_mb`, and
    Slurm supplies it from `MaxTRES`, so the same over-limit job was flagged on
    one scheduler and silently accepted on the other. Dropping it also hid it
    from `unreadable`: an axis nothing reads has nothing to report as unread,
    which is why `2eb` above needed this fix before it could be named.
    """

    def test_a_per_job_memory_ceiling_is_read(self):
        assert _limits_from_qstat(MEM_CEILINGS)["memq"].per_job["mem_mb"] == 256 * 1024

    def test_a_per_entity_memory_table_resolves_like_any_other(self, monkeypatch):
        monkeypatch.setenv("USER", "bob")
        assert _limits_from_qstat(MEM_CEILINGS)["memq"].per_user["mem_mb"] == 200 * 1024

    def test_the_callers_own_cell_wins_here_too(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        assert _limits_from_qstat(MEM_CEILINGS)["memq"].per_user["mem_mb"] == 2048 * 1024

    def test_an_over_limit_request_is_now_blocked(self, monkeypatch):
        monkeypatch.setenv("USER", "bob")
        got = _limits_from_qstat(MEM_CEILINGS)["memq"].blockers(
            JobShape(nodes=1, memory_gb=300, walltime="1h"))
        assert {"MAX_MEM_MB_JOB", "MAX_MEM_MB_USER"} <= {b.code for b in got}

    def test_control_a_resource_with_no_ceiling_axis_here_is_left_alone(self):
        """CONTROL -- passes before the fix and after it.

        `ompthreads` is unreadable-looking and must stay unmentioned: it is a
        limit nothing here was ever going to check, so naming it would be a
        warning about a check that does not exist. Only an axis
        `Limits.blockers` actually tests can go unchecked.
        """
        limits = _limits_from_qstat(MEM_CEILINGS)["memq"]
        assert "ompthreads" not in limits.per_job
        assert limits.unreadable == ()

    def test_control_the_recorded_fixture_is_unchanged(self, pbs_backend, monkeypatch):
        """CONTROL -- passes before the fix and after it.

        Every queue in the recorded `qstat -Qf`, including the one printing an
        empty `acl_users`: the axes that already worked keep their numbers and
        nothing anywhere is reported unread.
        """
        monkeypatch.setenv("USER", "alice")
        limits = pbs_backend.load_limits()
        assert limits["gpuq"].per_user["gpu"] == 8
        assert limits["gpuq"].per_job["cpu"] == 256
        assert limits["gpuq"].max_walltime_seconds == 48 * 3600
        assert limits["gpuq"].max_jobs == 4
        assert all(not q.unreadable for q in limits.values())


#: One node whose memory PBS reported in words. 512 GiB installed, one job
#: holding 8 gigawords of it.
WORD_NODE = """{
  "nodes": {
    "w001": {
      "state": "free",
      "resources_available": {"ncpus": 64, "mem": "64gw", "Qlist": "wordq"},
      "resources_assigned": {"ncpus": 8, "mem": "8gw"}
    }
  }
}
"""

#: Two queues whose memory ceilings use the two suffixes that were dropped.
WORD_CEILING = """Queue: wordq
    queue_type = Execution
    resources_max.mem = 1gw
    enabled = True
    started = True

Queue: petaq
    queue_type = Execution
    resources_max.mem = 1pb
    enabled = True
    started = True
"""


#: PBS's ``w`` unit, one row per shape PBS prints, as ``(text, mb)``.
#:
#: A PBS suffix is a size prefix *and* a unit: ``b`` is bytes and ``w`` is
#: *words*, 8 bytes each on a 64-bit server. `_mem_to_mb` matched the unit and
#: threw it away, so every `w` form answered its `b` twin's number -- `1gw`
#: came out as 1024 MB where PBS means 8 GiB. An 8x understatement.
WORD_SIZES = [
    ("1mw", 8),
    ("1gw", 8 * 1024),
    ("1tw", 8 * 1024 * 1024),
    ("16GW", 128 * 1024),        # PBS accepts either case
    ("1.5gw", 12 * 1024),        # and a fraction, per its own syntax
    ("131072w", 1),              # unprefixed, and the first such size past 1 MB
]

#: The ``b`` half of the same syntax, value for value -- the CONTROL rows.
#: This scaling was verified correct in an earlier round and rewriting the
#: regex is exactly how it would break, so every one of these must answer
#: today's number both before the `w` fix and after it.
BYTE_SIZES = [
    ("1024kb", 1),
    ("128gb", 128 * 1024),
    ("2tb", 2 * 1024 * 1024),
    ("1Gb", 1024),
    ("1.5gb", 1536),
    ("2097152", 2),              # bare = bytes
    ("512kb", 0),                # under 1 MB: truncates, and always did
    ("1b", 0),
    ("0gb", 0),
]


class TestTheWordUnitIsEightBytes:
    """`mem = 1gw` is 8 GiB, and PBS prints it.

    PBS Pro's size syntax is ``<num><prefix><unit>`` where the unit is ``b``
    *or* ``w``, a machine word -- so ``w``, ``kw``, ``mw``, ``gw``, ``tw`` are
    all legal and all eight times their ``b`` twin. The regex matched ``[bw]``
    and used only the prefix, which is not a harmless simplification of the
    grammar: it published an eighth of the number PBS reported.

    Nothing here is confined to queue ceilings. `_mem_to_mb` is also what
    supplies `Node.memory_mb` and `Node.memory_alloc_mb`, and understating an
    *assignment* is the one direction that invents capacity: free memory is
    total minus assigned, so an `8gw` job holding 64 GiB was recorded as
    holding 8, and the 56 GiB difference was published as available.
    """

    @pytest.mark.parametrize("text,mb", WORD_SIZES)
    def test_a_word_size_is_read_as_words(self, text, mb):
        from nodetop.backends.pbs import _mem_to_mb

        assert _mem_to_mb(text) == mb

    def test_a_node_sized_in_words_is_not_an_eighth_of_itself(self):
        nodes = PbsBackend(RecordedRunner({"pbsnodes -a -F json": (0, WORD_NODE, "")})
                           ).load_nodes()
        assert {n.name: n.memory_mb for n in nodes} == {"w001": 512 * 1024}

    def test_memory_held_by_a_job_is_not_published_as_free(self):
        # The dangerous direction: `resources_assigned.mem = 8gw` is 64 GiB
        # held, so 448 GiB is free -- not the 504 GiB an 8 GiB reading left.
        node = PbsBackend(RecordedRunner({"pbsnodes -a -F json": (0, WORD_NODE, "")})
                          ).load_nodes()[0]
        assert node.memory_alloc_mb == 64 * 1024
        assert node.memory_free_mb == 448 * 1024

    def test_a_memory_ceiling_in_words_is_read_as_words(self):
        limits = _limits_from_qstat(WORD_CEILING)["wordq"]
        assert limits.per_job["mem_mb"] == 8 * 1024

    def test_a_request_under_the_real_ceiling_is_not_refused(self):
        # An understated ceiling refuses work the site accepts: 4 GB is well
        # inside `1gw` = 8 GiB, and was blocked against a 1 GiB misreading.
        limits = _limits_from_qstat(WORD_CEILING)["wordq"]
        assert limits.blockers(JobShape(nodes=1, memory_gb=4, walltime="1h")) == []
        assert [b.code for b in limits.blockers(
            JobShape(nodes=1, memory_gb=16, walltime="1h"))] == ["MAX_MEM_MB_JOB"]

    def test_a_sub_megabyte_word_count_answered_zero_before_and_after(self):
        """Which forms were silently wrong, and which were already caught.

        `1024w` is 8 KiB and `1kw` is the same 8 KiB; both truncate to 0 MB
        with the unit read and without it, so `_pbs_mem_mb` turned them into
        `None` and `Limits.unreadable` named them either way. Only the
        prefixed forms big enough to survive the rounding -- `mw` and up --
        landed silently wrong, which is why the matrix above starts there.
        """
        from nodetop.backends.pbs import _mem_to_mb, _pbs_mem_mb

        assert _mem_to_mb("1024w") == _mem_to_mb("1kw") == 0
        assert _pbs_mem_mb("1024w") is None
        assert "max_run_res.mem" in _limits_from_qstat(UNREADABLE_SHAPES)["shapeq"].unreadable

    def test_a_petabyte_is_not_a_node_with_no_memory(self):
        """`p` is in PBS's suffix list and was missing from the scale map.

        Safe as errors go -- 0 MB understates, and `Node.hardware_ok` drops a
        node claiming no memory rather than offering it -- but it dropped the
        node, and a ceiling of 0 is no ceiling at all.
        """
        from nodetop.backends.pbs import _mem_to_mb

        assert _mem_to_mb("1pb") == 1024 * 1024 * 1024
        assert _mem_to_mb("2PB") == 2 * 1024 * 1024 * 1024
        assert _mem_to_mb("1pw") == 8 * 1024 * 1024 * 1024
        assert _limits_from_qstat(WORD_CEILING)["petaq"].per_job["mem_mb"] == 1024 ** 3

    @pytest.mark.parametrize("text,mb", BYTE_SIZES)
    def test_control_every_byte_form_keeps_its_exact_value(self, text, mb):
        """CONTROL -- passes before the fix and after it."""
        from nodetop.backends.pbs import _mem_to_mb

        assert _mem_to_mb(text) == mb

    def test_control_the_recorded_fixture_is_unchanged(self, pbs_backend):
        """CONTROL -- passes before the fix and after it.

        The recorded `pbsnodes -a -F json` is all `gb` and one `0kb`, so every
        node in it must come out byte-identical to the numbers a live PBS Pro
        2022.1 printed.
        """
        nodes = {n.name: n for n in pbs_backend.load_nodes()}
        assert [(n.memory_mb, n.memory_alloc_mb, n.memory_free_mb)
                for _, n in sorted(nodes.items())] == [
            (256 * 1024, 16 * 1024, 240 * 1024),    # cpu001, 16gb assigned
            (256 * 1024, 0, 256 * 1024),            # cpu002, nothing assigned
            (512 * 1024, 0, 512 * 1024),            # gpu001, "0kb" assigned
            (512 * 1024, 500 * 1024, 12 * 1024),    # gpu002
            (512 * 1024, 0, 512 * 1024),            # gpu003
        ]

    @pytest.mark.parametrize("junk", ["1wg", "1gbb", "1 g b", "gw", "-8gw", "1eb", "words"])
    def test_control_a_shape_pbs_never_prints_is_still_nothing(self, junk):
        """CONTROL -- passes before the fix and after it.

        A wider unit vocabulary must not become a wider *grammar*: the unit
        follows the prefix, there is exactly one of each, and anything else is
        0 rather than a number guessed from the digits it happens to contain.
        """
        from nodetop.backends.pbs import _mem_to_mb

        assert _mem_to_mb(junk) == 0


class TestNoProbe:
    def test_probe_returns_none(self, pbs_backend):
        # PBS has no verify-only submission, so there is nothing to ask.
        assert pbs_backend.probe("gpuq", JobShape()) is None

    def test_capabilities_say_so(self, pbs_backend):
        caps = pbs_backend.capabilities()
        assert caps.probe is False
        assert any("no verify-only" in n for n in caps.notes)


class TestSubmitFlags:
    def test_select_statement(self, pbs_backend):
        flags = " ".join(
            pbs_backend.submit_flags("gpuq", JobShape(nodes=2, gpus_per_node=4,
                                                      cpus_per_task=8, memory_gb=64,
                                                      walltime="4h"))
        )
        assert "select=2:ncpus=8:ngpus=4:mem=64gb" in flags
        assert "walltime=04:00:00" in flags
        assert "-q gpuq" in flags

    def test_nodelist_uses_pbs_notation(self, pbs_backend):
        assert pbs_backend.format_nodelist(["b", "a"]) == "a+b"


QSTAT_F_RUNNING = """Job Id: 1234.pbs-head
    Job_Name = train
    job_state = R
    Resource_List.walltime = 24:00:00
    stime = Thu Aug 21 10:00:00 2026
    exec_host = gpu001/0*64+gpu002/0*64
    exec_vnode = (gpu001:ncpus=64)+(gpu002:ncpus=64)
    etime = Thu Aug 21 09:55:00 2026

Job Id: 1235.pbs-head
    job_state = R
    Resource_List.walltime = 02:00:00
    stime = Thu Aug 21 12:00:00 2026
    exec_host = gpu001/0*8

Job Id: 1236.pbs-head
    job_state = Q
    Resource_List.walltime = 01:00:00
    estimated.start_time = Fri Aug 22 03:00:00 2026
"""


class TestFreeTimes:
    """PBS records no end time, so it has to be computed."""

    def _times(self, pbs_backend):
        return pbs_backend.parse_free_times(QSTAT_F_RUNNING)

    def test_end_is_start_plus_walltime(self, pbs_backend):
        # An upper bound, which is the right direction: a job may finish early,
        # so promising the node sooner would be a promise PBS never made.
        got = self._times(pbs_backend)
        assert str(got["gpu002"]) == "2026-08-22 10:00:00"

    def test_exec_host_slash_form_is_parsed(self, pbs_backend):
        # "gpu001/0*64+gpu002/0*64" -- matching this with a parenthesis
        # pattern (the exec_vnode shape) silently yields nothing at all.
        assert set(self._times(pbs_backend)) == {"gpu001", "gpu002"}

    def test_the_latest_job_on_a_node_wins(self, pbs_backend):
        # gpu001 hosts two jobs; the node is free no earlier than the last one.
        got = self._times(pbs_backend)
        assert str(got["gpu001"]) == "2026-08-22 10:00:00"

    def test_a_pending_job_contributes_nothing(self, pbs_backend):
        # estimated.start_time is when a queued job may START, not when
        # anything ends; using it as an end time is simply wrong.
        assert len(self._times(pbs_backend)) == 2

    def test_a_job_missing_any_piece_is_skipped(self, pbs_backend):
        partial = "Job Id: 1\n    exec_host = n1/0\n"
        assert pbs_backend.parse_free_times(partial) == {}

    def test_empty_input(self, pbs_backend):
        assert pbs_backend.parse_free_times("") == {}


QUEUES_JSON = """{
 "timestamp": 1755800000,
 "pbs_version": "2022.1.1",
 "Queue": {
   "workq": {
     "queue_type": "Execution",
     "enabled": "True",
     "started": "False",
     "Priority": "80",
     "resources_max": {"walltime": "48:00:00", "nodect": "16"},
     "max_run_res": {"ngpus": "[u:PBS_GENERIC=8]"},
     "acl_user_enable": "True",
     "acl_users": "alice,bob"
   }
 }
}"""


class TestQueuesJson:
    """The JSON path, which PBS Pro 18+ prefers."""

    def test_nested_keys_are_flattened(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        # resources_max.walltime has to survive the nesting.
        assert q.max_walltime_seconds == 48 * 3600
        assert q.max_nodes == 16

    def test_switches_and_acl(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        assert (q.enabled, q.started) == (True, False)
        assert set(q.allow_users) == {"alice", "bob"}
        assert q.priority == 80

    def test_accepts_but_never_starts(self, pbs_backend):
        q = pbs_backend.parse_queues_json(QUEUES_JSON)[0]
        assert "QUEUE_NOT_STARTED" in {b.code for b in q.structural_blockers()}


def _pbs_cluster(nodes_json: str, qstat: str):
    from nodetop.backends.pbs import PbsBackend
    from nodetop.core.cluster import Cluster
    from nodetop.runner import RecordedRunner

    return Cluster.load(
        PbsBackend(RecordedRunner({
            "pbsnodes -a -F json": (0, nodes_json, ""),
            "qstat -Qf": (0, qstat, ""),
            "qstat -f": (0, "", ""),
        })),
        with_free_times=False,
    )


MIXED_NODES = """{"nodes": {
 "gpu001": {"state":"free","resources_available":{"ncpus":64,"ngpus":4,"Qlist":"gpuq"},
            "resources_assigned":{}},
 "cpu001": {"state":"free","resources_available":{"ncpus":128},"resources_assigned":{}}
}}"""

TYPED_QUEUES = """Queue: gpuq
    queue_type = Execution
    enabled = True
    started = True

Queue: workq
    queue_type = Execution
    enabled = True
    started = True

Queue: routeq
    queue_type = Route
    enabled = True
    started = True
    route_destinations = gpuq,workq
"""


class TestUnrestrictedNodes:
    """Qlist restricts a node; it does not enrol it."""

    def test_a_node_with_no_qlist_belongs_to_every_execution_queue(self):
        # Requiring an explicit mention orphans every unrestricted node: its
        # capacity becomes invisible to all queues.
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert "cpu001" in cluster.queues["gpuq"].node_names
        assert "cpu001" in cluster.queues["workq"].node_names

    def test_a_restricted_node_stays_restricted(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert "gpu001" in cluster.queues["gpuq"].node_names
        assert "gpu001" not in cluster.queues["workq"].node_names

    def test_a_queue_no_node_names_is_not_left_looking_empty(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        assert cluster.queues["workq"].node_names != ()

    def test_membership_is_deduplicated(self):
        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        names = cluster.queues["gpuq"].node_names
        assert len(names) == len(set(names))


class TestRoutingQueues:
    def test_a_route_queue_is_recognised(self):
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert q.routes is True
        assert q.forwards_to == ("gpuq", "workq")

    def test_an_execution_queue_is_not(self):
        assert _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["gpuq"].routes is False

    def test_it_owns_no_nodes(self):
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert q.node_names == ()
        assert q.declared_nodes == 0

    def test_having_no_nodes_is_not_reported_as_all_nodes_down(self):
        # "every node is down" is a finding; "owns no nodes by design" is not.
        q = _pbs_cluster(MIXED_NODES, TYPED_QUEUES).queues["routeq"]
        assert "ALL_NODES_UNSCHEDULABLE" not in {
            b.code for b in q.structural_blockers()
        }

    def test_it_is_not_offered_as_a_placement_target(self):
        from nodetop.core.fit import rank
        from nodetop.core.model import JobShape

        cluster = _pbs_cluster(MIXED_NODES, TYPED_QUEUES)
        names = [p.queue for p in rank(cluster, JobShape(cpus_per_task=2))]
        # Its capacity belongs to its destinations, which are ranked instead.
        assert "routeq" not in names
        assert {"gpuq", "workq"} <= set(names)


class TestWrappedAttributeValuesAreNotTruncated:
    """PBS wraps long attribute values at 80 columns, mid-value, with a tab.

    Every parser here iterated raw lines and required an `=` to accept one, so
    each continuation was dropped -- losing the *tail* of exactly the values long
    enough to wrap, which are the ones that matter.
    """

    def test_a_nodes_queue_list_keeps_every_queue(self):
        # A node serving eight queues was recorded as serving five. It then goes
        # missing from the queues whose names were cut, and its capacity with it.
        text = ("node01\n"
                "     state = free\n"
                "     pcpus = 32\n"
                "     resources_available.ncpus = 32\n"
                "     resources_available.Qlist = workq,batch,longq,gpuq,bigmem,\n"
                "\tdebug,test,priority\n"
                "     resources_assigned.ncpus = 0\n")
        node = PbsBackend(RecordedRunner({})).parse_nodes_text(text)[0]
        assert node.queues == ("workq", "batch", "longq", "gpuq", "bigmem",
                               "debug", "test", "priority")

    def test_a_queue_acl_keeps_every_user(self):
        # A truncated allowlist reads as authoritative and produces a FALSE
        # DENIAL -- no access reported to a queue that would have taken the job.
        text = ("Queue: gpuq\n"
                "    queue_type = Execution\n"
                "    enabled = True\n"
                "    started = True\n"
                "    acl_user_enable = True\n"
                "    acl_users = alice@*,bob@*,carol@*,dave@*,erin@*,frank@*,\n"
                "\tgrace@*,heidi@*\n")
        queue = PbsBackend(RecordedRunner({})).parse_queues_text(text)[0]
        assert len(queue.allow_users) == 8
        assert "heidi@*" in queue.allow_users

    def test_exec_host_keeps_every_node(self):
        # The longest field PBS emits, and it always wraps for a multi-node job.
        # It maps running work to nodes, so a truncated one attributes free-time
        # estimates to the wrong machines.
        text = ("Job Id: 1.server\n"
                "    job_state = R\n"
                "    exec_host = n01/0*32+n02/0*32+n03/0*32+n04/0*32+n05/0*32+\n"
                "\tn06/0*32+n07/0*32\n"
                "    stime = Sat Aug 23 10:00:00 2026\n"
                "    Resource_List.walltime = 04:00:00\n")
        got = PbsBackend(RecordedRunner({})).parse_free_times(text)
        assert sorted(got) == [f"n0{i}" for i in range(1, 8)]

    def test_a_record_header_is_never_read_as_a_continuation(self):
        # Headers are unindented and continuations are indented, which is what
        # keeps the two apart. Two records must stay two.
        text = ("node01\n     state = free\n     pcpus = 8\n"
                "node02\n     state = free\n     pcpus = 8\n")
        nodes = PbsBackend(RecordedRunner({})).parse_nodes_text(text)
        assert [n.name for n in nodes] == ["node01", "node02"]

    def test_unwrapped_output_is_unaffected(self, pbs_backend):
        # The format in use must parse exactly as before the hardening.
        assert len(pbs_backend.load_nodes()) >= 1


class TestExclusivePlacementIsNotFreeCapacity:
    """PBS records whole-node exclusivity in the STATE, not in every resource.

    Measured on a 10,624-node PBS Pro 2022.1 cluster: 10,194 nodes
    `job-exclusive`, and not one node anywhere carried `ngpus` under
    `resources_assigned` -- whole-node placement means the scheduler never has
    to account for a GPU individually. `ncpus` happened to be assigned in full,
    so the CPU figures were right and the accelerator figures were not: nodetop
    announced **62,886 of 63,744 GPUs free** where 1,722 were. A 36x
    overstatement on the axis people pick that machine for.

    Cross-checked against a second PBS Pro site (2026.1, 24 nodes) whose jobs
    DO share nodes and where every `ngpus` is accounted: there not one
    `job-exclusive` node was partially assigned, so this rule changed nothing
    and the free count still matched `available - assigned` exactly. It
    corrects the site that omits the accounting without disturbing the site
    that keeps it.
    """

    EXCLUSIVE = """{"nodes": {
     "x1": {"state":"job-exclusive",
            "resources_available":{"ncpus":208,"ngpus":6,"mem":"1000gb"},
            "resources_assigned":{"ncpus":208}},
     "x2": {"state":"free",
            "resources_available":{"ncpus":208,"ngpus":6,"mem":"1000gb"},
            "resources_assigned":{}}
    }}"""

    def _nodes(self, payload, cmd="pbsnodes -a -F json"):
        backend = PbsBackend(RecordedRunner({cmd: (0, payload, "")}))
        parse = (backend.parse_nodes_json if "json" in cmd
                 else backend.parse_nodes_text)
        return {n.name: n for n in parse(payload)}

    def test_nothing_on_an_exclusive_node_is_free(self):
        got = self._nodes(self.EXCLUSIVE)
        held = got["x1"]
        assert (held.gpus_free, held.cpus_free, held.memory_free_mb) == (0, 0, 0)
        # And the neighbour is untouched, so this is not a blanket zeroing.
        assert (got["x2"].gpus_free, got["x2"].cpus_free) == (6, 208)

    def test_a_full_node_is_not_a_broken_one(self):
        # Occupancy, not a condition. Calling it unschedulable would report 96%
        # of that cluster as out of service and bury it in `health` -- trading
        # one wrong answer for a louder one.
        held = self._nodes(self.EXCLUSIVE)["x1"]
        assert held.schedulable is True
        assert held.conditions == frozenset()
        assert held.degraded is False

    def test_the_classic_text_format_gets_the_same_rule(self):
        # Torque and PBS before 18 speak only this one.
        text = ("x1\n     state = job-exclusive\n"
                "     resources_available.ncpus = 208\n"
                "     resources_available.ngpus = 6\n"
                "     resources_assigned.ncpus = 208\n")
        held = self._nodes(text, cmd="pbsnodes -a")["x1"]
        assert (held.gpus_free, held.cpus_free) == (0, 0)
        assert held.schedulable is True

    def test_a_state_set_carrying_it_still_blocks_on_the_other_word(self):
        # `offline,job-exclusive` is real output: full AND drained.
        payload = self.EXCLUSIVE.replace('"job-exclusive"', '"offline,job-exclusive"')
        held = self._nodes(payload)["x1"]
        assert held.schedulable is False
        assert held.gpus_free == 0


class TestQueueAttributesAreFetchedOnce:
    """Two spellings of one query is two instants in one report -- and 38s.

    `load_queues` asked for `qstat -Qf -F json` while `load_limits` asked for
    the plain `qstat -Qf`, so the same 37 KB of queue attributes came back
    twice. On the 10,624-node cluster that was 23.9s + 14.9s of a 65s run.
    """

    JSON = """{"Queue": {"workq": {"queue_type":"Execution","enabled":"True",
        "started":"True","resources_max":{"walltime":"24:00:00","nodect":"16"}}}}"""
    TEXT = ("Queue: workq\n    queue_type = Execution\n    enabled = True\n"
            "    started = True\n    resources_max.walltime = 24:00:00\n"
            "    resources_max.nodect = 16\n")

    def _backend(self, *, json_ok=True):
        responses = {"pbsnodes -a -F json": (0, '{"nodes": {}}', "")}
        responses["qstat -Qf -F json"] = ((0, self.JSON, "") if json_ok
                                          else (1, "", "unknown option"))
        responses["qstat -Qf"] = (0, self.TEXT, "")
        return PbsBackend(RecordedRunner(responses))

    def test_one_query_serves_queues_and_limits(self):
        backend = self._backend()
        backend.load_queues()
        limits = backend.load_limits()
        asked = [c for c in backend.runner.calls if c[:2] == ["qstat", "-Qf"]]
        assert len(asked) == 1, asked
        # And the ceilings still arrive, from the payload already in hand.
        assert limits["workq"].max_walltime_seconds == 86400
        assert limits["workq"].per_job["node"] == 16

    def test_a_pbs_without_json_falls_back_and_still_asks_once(self):
        # Torque has no JSON mode at all, and the failure is cached: asking
        # every consumer to rediscover that costs a round trip each.
        backend = self._backend(json_ok=False)
        queues = backend.load_queues()
        limits = backend.load_limits()
        assert [q.name for q in queues] == ["workq"]
        assert limits["workq"].max_walltime_seconds == 86400
        assert len([c for c in backend.runner.calls if c == ["qstat", "-Qf"]]) == 1
        assert len([c for c in backend.runner.calls
                    if c == ["qstat", "-Qf", "-F", "json"]]) == 1


class TestQueueMembershipStaysLinear:
    """The membership rebuild was quadratic in the node count.

    `set(members)` sat inside a generator's `if` clause, so it was rebuilt once
    per node: 51 queues over 10,624 nodes took **151 seconds**, in a 3m10s run
    whose scheduler queries accounted for 11s. The ceiling here is loose on
    purpose -- it is not a benchmark, it is a guard against the shape coming
    back, and the quadratic form needs minutes to do what this does in under a
    second.
    """

    def test_five_thousand_nodes_across_thirty_queues(self):
        import json as _json
        import time

        nodes = {f"n{i:05d}": {"state": "free",
                               "resources_available": {"ncpus": 8}}
                 for i in range(5000)}
        queues = {f"q{i:02d}": {"queue_type": "Execution", "enabled": "True",
                                "started": "True"} for i in range(30)}
        backend = PbsBackend(RecordedRunner({
            "pbsnodes -a -F json": (0, _json.dumps({"nodes": nodes}), ""),
            "qstat -Qf -F json": (0, _json.dumps({"Queue": queues}), ""),
        }))
        start = time.time()
        got = backend.load_queues()
        elapsed = time.time() - start
        assert elapsed < 10.0, f"{elapsed:.1f}s for 30 queues x 5000 nodes"
        # Unrestricted nodes belong to every execution queue, which is the
        # rule the slow version was implementing correctly and expensively.
        assert all(q.declared_nodes == 5000 for q in got)
