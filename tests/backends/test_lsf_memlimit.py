"""The LSF queue memory ceiling: filled, or named, but never guessed.

``Limits.blockers`` checks a ``mem_mb`` per-job ceiling, and until now nothing in
this backend ever filled one: ``load_limits`` read ``MAX_JOBS_PER_USER``,
``PJOB_LIMIT``, ``UJOB_LIMIT`` and ``RUNLIMIT``, so a queue publishing
``MEMLIMIT`` produced **zero** blockers for a job asking many times that memory.
The PBS backend had the same gap and closed it; this is the same axis on the
other backend.

What makes LSF's version awkward is scope, not parsing. ``MEMLIMIT`` is enforced
**per process** by default and only bounds the whole job under
``LSB_JOB_MEMLIMIT=Y``, so filling the job-wide axis from it unconditionally
understates the ceiling by the process count -- and `_memlimit_is_per_job`
already spells out why that is the worst available outcome: it "invents a blocker
for a job that would have run". So the ceiling is filled only when it is
comparable, and otherwise named in ``Limits.unreadable``, which exists to keep
exactly this gap visible "without inventing a limit nobody published".
"""

from nodetop.backends import lsf as lsfmod
from nodetop.backends.lsf import LsfBackend
from nodetop.runner import RecordedRunner

# MEMLIMIT alone on its line: the unambiguous layout, and the one the recorded
# `bqueues -l` fixture uses for RUNLIMIT.
_ALONE = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

RUNLIMIT
 1440.0 min

MEMLIMIT
 4194304 K
"""

# IBM's own documented combined table. The next line's first size is
# FILELIMIT's, not MEMLIMIT's.
_COMBINED = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

 FILELIMIT   DATALIMIT   STACKLIMIT  CORELIMIT   MEMLIMIT
    8000 K    10000 K      2000 K     20000 K     5000 K
"""

_NO_MEMLIMIT = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

RUNLIMIT
 1440.0 min
"""


def _limits(text, per_job_scope, site_unit=None, monkeypatch=None):
    monkeypatch.setattr(lsfmod, "_memlimit_is_per_job", lambda: per_job_scope)
    monkeypatch.setattr(lsfmod, "_site_unit_mb", lambda: site_unit)
    backend = LsfBackend(RecordedRunner({"bqueues": (0, text, "")}))
    return backend.load_limits()["normal"]


def test_a_job_scoped_memlimit_fills_the_per_job_axis(monkeypatch):
    """``LSB_JOB_MEMLIMIT=Y``: the ceiling bounds the job, so it is comparable.

    4194304 K is 4096 MB -- the K scale is binary in `_LSF_SCALE`.
    """
    lim = _limits(_ALONE, True, monkeypatch=monkeypatch)
    assert lim.per_job["mem_mb"] == 4096
    assert lim.unreadable == ()
    # and the axis now actually blocks something, which was the whole point
    assert "mem_mb" in lim.per_job


def test_a_per_process_memlimit_is_named_not_converted(monkeypatch):
    """The default scope. Naming it keeps the gap visible; converting it lies."""
    lim = _limits(_ALONE, False, monkeypatch=monkeypatch)
    assert "mem_mb" not in lim.per_job
    assert lim.unreadable == ("MEMLIMIT as a job-wide ceiling (LSB_JOB_MEMLIMIT is not set)",)


def test_the_two_not_comparable_causes_read_differently(monkeypatch):
    """``fit.py`` spells this field into "could not read {entry} on ...".

    A per-process ceiling WAS read; what could not be read is a job-wide one. A
    bare "MEMLIMIT" there makes that sentence untrue, and untrue in the direction
    that sends an admin hunting a parse bug instead of setting
    ``LSB_JOB_MEMLIMIT``. The genuinely unreadable causes keep the bare name --
    which is what every other producer of this field means by it.
    """
    per_process = _limits(_ALONE, False, monkeypatch=monkeypatch).unreadable[0]
    assert per_process.startswith("MEMLIMIT as a job-wide ceiling")
    assert "LSB_JOB_MEMLIMIT" in per_process, "name the setting to change"

    # a size no unit can resolve is a real read failure, and says so
    bare = _ALONE.replace(" 4194304 K", " 4194304")
    assert _limits(bare, True, site_unit=None, monkeypatch=monkeypatch).unreadable == (
        "MEMLIMIT",
    )
    # and so is the layout that will not be guessed at
    assert _limits(_COMBINED, True, monkeypatch=monkeypatch).unreadable == ("MEMLIMIT",)

    # the sentence a user actually sees, built the way `fit.py` builds it
    sentence = f"could not read {per_process} on lsf queue limits normal"
    assert "could not read MEMLIMIT as a job-wide ceiling" in sentence


def test_the_combined_table_is_not_guessed_at(monkeypatch):
    """The layout with no recorded sample: reported unreadable, not misread.

    The specific wrong answer this rules out is FILELIMIT's 8000 K arriving as
    the memory ceiling, which is what reading the next line's first size gives.
    """
    lim = _limits(_COMBINED, True, monkeypatch=monkeypatch)
    assert "mem_mb" not in lim.per_job, "no cell may be guessed from a column layout"
    assert lim.unreadable == ("MEMLIMIT",)


def test_control_a_queue_declaring_no_memlimit_stays_silent(monkeypatch):
    """CONTROL, passing with the change present or absent.

    An absent ceiling must produce neither a limit nor a warning, or every queue
    on a cluster that sets no MEMLIMIT gains an ``unreadable`` entry -- noise on
    the common case, which is worse than the gap being closed.
    """
    lim = _limits(_NO_MEMLIMIT, True, monkeypatch=monkeypatch)
    assert "mem_mb" not in lim.per_job
    assert lim.unreadable == ()
    assert lim.max_walltime_seconds == 1440 * 60, "the neighbouring limit still reads"


def test_control_b_the_recorded_fixture_is_unchanged(lsf_backend):
    """CONTROL, passing in both states. The fixture declares no MEMLIMIT.

    Its existing figures are what a regression here would disturb, so they are
    asserted rather than assumed: RUNLIMIT in minutes and MAX_JOBS_PER_USER.
    """
    limits = lsf_backend.load_limits()
    assert limits["gpu"].max_walltime_seconds == 2880 * 60
    assert limits["gpu"].max_jobs == 12
    for lim in limits.values():
        assert lim.unreadable == (), lim.name
        assert "mem_mb" not in lim.per_job, lim.name


def test_a_bare_memlimit_needs_the_site_unit(monkeypatch):
    """A bare size with no readable ``LSF_UNIT_FOR_LIMITS`` is not assumed to be MB.

    Not a control -- both halves exercise the new branch, which is why it is not
    named as one.

    `_mem_to_mb` answers 0 = "not read" there, and 0 is not a ceiling, so it
    lands in ``unreadable`` alongside the per-process case. The 1024x guess this
    avoids is the largest wrong answer available in this backend.
    """
    bare = _ALONE.replace(" 4194304 K", " 4194304")
    lim = _limits(bare, True, site_unit=None, monkeypatch=monkeypatch)
    assert "mem_mb" not in lim.per_job
    assert lim.unreadable == ("MEMLIMIT",), "a size nothing can resolve is a real read failure"

    # with the site unit readable as KB, the same string resolves
    lim = _limits(bare, True, site_unit=1 / 1024, monkeypatch=monkeypatch)
    assert lim.per_job["mem_mb"] == 4096
    assert lim.unreadable == ()
