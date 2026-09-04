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

The other awkward part *was* parsing, and is now handled. ``bqueues -l`` prints
limits two ways: one label alone on its line with its value below, and a
*combined* table where several limits share one header row. The combined form
used to answer ``unreadable`` outright, because reading the next line's first
size there yields ``FILELIMIT``'s value and nothing here would guess at a
column. `_memlimit_cell` reads the character columns instead, and still answers
``unreadable`` for any row whose cell order and column positions disagree -- so
the layout is either read correctly or named, never misread. No cluster is
reachable to record a sample of it, so the shapes below are built from IBM's
documented output and from the layout rule it shows.
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

# IBM's own documented combined table, verbatim. The next line's FIRST size is
# FILELIMIT's, not MEMLIMIT's -- that is the wrong answer this layout invites.
#
# The five columns are the reason this row makes a good anchor: read as MB they
# are 7, 9, 1, 19 and 4, all distinct, so asserting 4 rules out every other
# column rather than merely agreeing with the right one.
_COMBINED = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

 FILELIMIT   DATALIMIT   STACKLIMIT  CORELIMIT   MEMLIMIT
    8000 K    10000 K      2000 K     20000 K     5000 K
"""

# The same row with MEMLIMIT's label glued to CORELIMIT's by a single space, so
# the two rows no longer split into the same number of cells. Nothing can place
# the columns, so nothing may be read from them.
_GLUED = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

 FILELIMIT   DATALIMIT   STACKLIMIT  CORELIMIT MEMLIMIT
    8000 K    10000 K      2000 K     20000 K   5000 K
"""

_NO_MEMLIMIT = """QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

RUNLIMIT
 1440.0 min
"""


def _combined(pairs, gutter="   "):
    """A combined limit row laid out the way IBM's documented one reads.

    Each column is as wide as the wider of its label and its value, both
    right-aligned in it, with a gutter between. Built here rather than recorded
    because ``tests/fixtures/lsf/bqueues_l.txt`` has no combined table at all --
    it declares only ``RUNLIMIT``, alone on its line -- and no LSF cluster is
    reachable to record one. So the shapes come from the layout rule, and
    `_COMBINED` above keeps the documented row itself as the anchor.

    Note this builds *exact* right edges while IBM's published row is off by up
    to two characters. Both must read, which is why the reader tests nearest
    column rather than equal edges.
    """
    cells = [(k, v, max(len(k), len(v))) for k, v in pairs]
    header = gutter.join(k.rjust(w) for k, _v, w in cells)
    values = gutter.join(v.rjust(w) for _k, v, w in cells)
    return f"""QUEUE: normal
PARAM: PRIO NICE STATUS          MAX JL/U
       50    20  Open:Active       -    8

 {header}
 {values}
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
    # and so is a combined row whose columns cannot be placed
    assert _limits(_GLUED, True, monkeypatch=monkeypatch).unreadable == ("MEMLIMIT",)

    # the sentence a user actually sees, built the way `fit.py` builds it
    sentence = f"could not read {per_process} on lsf queue limits normal"
    assert "could not read MEMLIMIT as a job-wide ceiling" in sentence


def test_the_combined_table_reads_its_own_column(monkeypatch):
    """IBM's documented combined row now resolves to MEMLIMIT's own cell.

    The specific wrong answer this rules out is FILELIMIT's 8000 K arriving as
    the memory ceiling, which is what reading the next line's first size gives.
    5000 K is 4 MB on `_LSF_SCALE`'s binary K -- IBM's example values are toys,
    but they are the published ones, and the five columns land on five distinct
    MB figures so the assertion identifies the column rather than just agreeing
    with it.
    """
    lim = _limits(_COMBINED, True, monkeypatch=monkeypatch)
    assert lim.per_job["mem_mb"] == 4, "MEMLIMIT's 5000 K, not another column's size"
    assert lim.unreadable == (), "read, so nothing to name"
    # every other column, spelled out, so a future column-off-by-one names itself
    assert lim.per_job["mem_mb"] not in (7, 9, 1, 19), "FILE/DATA/STACK/CORELIMIT"


def test_the_column_holds_wherever_memlimit_sits(monkeypatch):
    """First column, last column, and a value narrower or wider than its label.

    Token index cannot do this: the header row has five tokens and the value row
    ten, because every size is ``number`` + ``unit``. These are the boundary
    cases the layout itself produces, and each is read from the column rule, not
    from a position that happens to work on one sample.
    """
    kb = ("FILELIMIT", "8000 K"), ("DATALIMIT", "10000 K")

    # last column, and first, of the same limits
    assert _limits(
        _combined([*kb, ("MEMLIMIT", "2097152 K")]), True, monkeypatch=monkeypatch
    ).per_job["mem_mb"] == 2048
    assert _limits(
        _combined([("MEMLIMIT", "2097152 K"), *kb]), True, monkeypatch=monkeypatch
    ).per_job["mem_mb"] == 2048

    # value NARROWER than its label: "1 G" under an eight-character MEMLIMIT
    assert _limits(
        _combined([*kb, ("MEMLIMIT", "1 G"), ("CORELIMIT", "20000 K")]),
        True, monkeypatch=monkeypatch,
    ).per_job["mem_mb"] == 1024

    # value WIDER than its label, which pushes the column out rather than
    # sliding the row: a ceiling this size is what a real fat node publishes
    assert _limits(
        _combined([*kb, ("MEMLIMIT", "1073741824 K"), ("CORELIMIT", "20000 K")]),
        True, monkeypatch=monkeypatch,
    ).per_job["mem_mb"] == 1024 * 1024

    # an unset NEIGHBOUR must not shift the count: "-" is its own cell
    assert _limits(
        _combined(
            [("FILELIMIT", "-"), ("DATALIMIT", "10000 K"), ("STACKLIMIT", "-"),
             ("MEMLIMIT", "2097152 K")]
        ),
        True, monkeypatch=monkeypatch,
    ).per_job["mem_mb"] == 2048


def test_a_combined_row_that_cannot_be_placed_is_still_named(monkeypatch):
    """The fallback stays. A wrong ceiling is worse than a named gap.

    `_memlimit_is_per_job` says why: filling the job-wide axis from the wrong
    figure "invents a blocker for a job that would have run". So every shape
    where the two readings -- cell order and character columns -- disagree keeps
    answering ``unreadable``, which is what this whole layout used to answer.
    That is the property that makes the reader safe: it can turn "unreadable"
    into a right number or leave it alone, never into a wrong one.
    """
    header = " FILELIMIT   DATALIMIT   STACKLIMIT  CORELIMIT   MEMLIMIT"
    for name, row in [
        # a label glued to its neighbour, so the rows do not split alike
        ("glued label", _GLUED),
        # cells merged by a single space: nine cells of header, four of value
        ("merged values", f"QUEUE: normal\n\n{header}\n    8000 K 10000 K"
                          "      2000 K     20000 K     5000 K\n"),
        # the row ran out and the next section arrived
        ("no value row", f"QUEUE: normal\n\n{header}\nUSERS: all users\n"),
        # the next header arrived instead of values
        ("next header", f"QUEUE: normal\n\n{header}\n SWAPLIMIT   THREADLIMIT\n"),
        # right count, wrong columns: the whole row slid left
        ("row slid", f"QUEUE: normal\n\n{header}\n"
                     " 8000 K    10000 K      2000 K     20000 K  5000 K\n"),
    ]:
        lim = _limits(row, True, monkeypatch=monkeypatch)
        assert "mem_mb" not in lim.per_job, f"{name}: no cell may be guessed"
        assert lim.unreadable == ("MEMLIMIT",), name


def test_an_unset_memlimit_is_no_ceiling_not_an_unreadable_one(monkeypatch):
    """``-`` is an answer -- "no limit" -- not a field that could not be read.

    It used to reach `_mem_to_mb`, which reads it as 0 = "not read", so a queue
    whose memory limit is *unset* reported its ceiling as unreadable. That is a
    warning about a field published perfectly clearly, and it is the same false
    warning `test_control_a...` guards against for an absent MEMLIMIT -- LSF
    prints ``-`` for an unset limit (the recorded fixture's ``PARAM`` row uses it
    four times) and IBM documents the resource-limit default as infinity.

    The other two backends already rule this way: `pbs._looks_present`'s
    docstring says an attribute never printed and one printed with a no-limit
    sentinel "are the same thing -- no ceiling, nothing to warn about".

    The combined table is where it stops being hypothetical, since a row of five
    limits usually has some of them unset.
    """
    for name, row in [
        ("alone on its line", _ALONE.replace(" 4194304 K", " -")),
        ("its own column", _combined(
            [("FILELIMIT", "8000 K"), ("DATALIMIT", "10000 K"), ("MEMLIMIT", "-")]
        )),
    ]:
        lim = _limits(row, True, monkeypatch=monkeypatch)
        assert "mem_mb" not in lim.per_job, f"{name}: unset is not a ceiling"
        assert lim.unreadable == (), f"{name}: nothing failed to read"


def test_control_the_label_on_its_own_line_still_resolves(monkeypatch):
    """CONTROL, passing with the column reader present or absent.

    The layout the recorded fixture actually uses, and the one that already
    worked. A column reader that took this path over is the regression that
    would matter most, since it is the shape seen in the wild here.
    """
    lim = _limits(_ALONE, True, monkeypatch=monkeypatch)
    assert lim.per_job["mem_mb"] == 4096, "4194304 K, read from the next line"
    assert lim.unreadable == ()
    assert lim.max_walltime_seconds == 1440 * 60, "the neighbouring limit still reads"


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
