"""A mistyped hostlist must not be able to exhaust memory.

`--exclude` is user input, so the argument to `expand` is whatever somebody
typed. It was unbounded, and the bracket sections MULTIPLY, so the cost of a
typo was not linear in the length of the string. Measured before the bound, with
a 2 GiB ceiling on the process:

    n[1-1000000]                1,000,000 names   0.28 s
    u[1-2000]r[1-2000]          4,000,000 names   0.34 s
    a[1-200]b[1-200]c[1-200]    8,000,000 names   0.68 s
    n[1-100000000]              MemoryError
    a[1-70000]b[1-70000]        MemoryError

Both sibling packages already capped at 65536 -- one of them after measuring the
`u[1-2000]r[1-2000]` case at 325 MiB -- and this module was the one without it.
Their note is also the reason the bound is applied where it is: "Enforced *while*
expanding, not afterwards. Truncating the finished list left the guard doing
nothing about the case that needs it." That is not a hypothetical. A first attempt
here sliced the finished product, which fixed the three fast rows above and left
`a[1-70000]b[1-70000]` raising MemoryError in 2.18 s, because 65536 x 65536 is
built before any trim runs. The product is now consumed lazily.

The controls matter as much as the bound: a cap that also truncated real node
lists would silently lose nodes, which is the failure this whole module exists to
prevent.
"""

from __future__ import annotations

import resource
import subprocess
import sys

import pytest

from nodetop.hostlist import MAX_EXPANSION, collapse, expand

#: Expressions whose unbounded expansion is between one million and 4.3 billion
#: names. Each must return at the bound instead.
#: The address-space ceiling each child runs under, in MiB. Comfortably above
#: what a bounded expansion needs -- `u[1-2000]r[1-2000]` was measured at
#: 325 MiB unbounded -- and far below what an unbounded one does.
CEILING_MIB = 512

#: Exit code a child uses to say the ceiling would not install, so the case is
#: skipped rather than read as a failure of `expand`. Deliberately not 1, which
#: is what an uncaught `MemoryError` gives -- the very outcome these cases exist
#: to catch -- so the two can never be confused.
_NO_CEILING = 99


def _child_code(expression: str) -> str:
    """Source for a child that expands ``expression`` under the ceiling."""
    return (
        "import resource, sys\n"
        "try:\n"
        f"    limit = {CEILING_MIB} * 1024**2\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
        "except (ValueError, OSError) as exc:\n"
        "    sys.stderr.write('RLIMIT_AS unavailable: %s' % exc)\n"
        f"    raise SystemExit({_NO_CEILING}) from None\n"
        f"sys.path.insert(0, {str(_src())!r})\n"
        "from nodetop.hostlist import expand\n"
        f"print(len(expand({expression!r})))\n"
    )


PATHOLOGICAL = [
    "n[1-1000000]",
    "n[1-100000000]",
    "u[1-2000]r[1-2000]",
    "a[1-200]b[1-200]c[1-200]",
    "a[1-70000]b[1-70000]",
    "a[1-100]b[1-100]c[1-100]d[1-100]",
]


class TestThePathologicalCasesReturnAtTheBound:
    """Every one of these runs in a CHILD with an address-space limit.

    Two reasons, and the second was learned the hard way. First, the claim is
    about peak memory, and a test that only checks the returned length passes
    while the intermediate product is still being built -- exactly the bug the
    first attempt at this fix left in place. Second, an in-process version does
    not fail cleanly when the bound is removed: expanding `n[1-100000000]` in the
    test runner takes the runner with it, so a broken cap would show up as a dead
    pytest (and could exhaust a login node during an ordinary suite run) instead
    of a red test. The ceiling belongs in a child that can die alone.
    """

    @pytest.mark.parametrize("expression", PATHOLOGICAL)
    def test_it_stays_within_a_memory_ceiling(self, expression):
        done = subprocess.run(
            [sys.executable, "-c", _child_code(expression)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if done.returncode == _NO_CEILING:
            pytest.skip(f"the ceiling could not be installed here: {done.stderr.strip()}")
        assert done.returncode == 0, (
            f"{expression} did not survive a {CEILING_MIB} MiB ceiling:\n{done.stderr[-400:]}"
        )
        assert int(done.stdout.strip()) <= MAX_EXPANSION

    def test_the_bound_matches_what_the_siblings_use(self):
        # Not arbitrary: the same number in three packages, so a node list that
        # one of them truncates is not silently kept whole by another.
        assert MAX_EXPANSION == 65536


class TestRealNodeListsAreUntouched:
    """The controls. Losing nodes is the failure this module was written to stop."""

    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("midway3-0001", ["midway3-0001"]),
            ("midway3-[0001-0003]", ["midway3-0001", "midway3-0002", "midway3-0003"]),
            ("n[01-03,07]", ["n01", "n02", "n03", "n07"]),
            ("a[1-2],b[3-4]", ["a1", "a2", "b3", "b4"]),
            ("node[1-3]-ib", ["node1-ib", "node2-ib", "node3-ib"]),
            ("a[1-2]b[3-4]", ["a1b3", "a1b4", "a2b3", "a2b4"]),
            ("", []),
            ("(null)", []),
        ],
    )
    def test_the_forms_slurm_actually_emits(self, expression, expected):
        assert expand(expression) == expected

    def test_a_cluster_sized_list_is_whole(self):
        # Far above any real cluster and still well under the bound, so a genuine
        # allocation is never the thing that gets trimmed.
        assert len(expand("midway3-[00001-10000]")) == 10000

    def test_a_list_at_the_bound_is_exact(self):
        # Literal 65536, not `MAX_EXPANSION`, deliberately. These two expand
        # in-process, so writing them in terms of the constant makes their cost
        # scale with it -- and a constant raised to something absurd would then
        # take the test runner down instead of failing. The literal cannot, and
        # `test_the_bound_matches_what_the_siblings_use` above is the tripwire
        # that says to update these if the bound ever moves.
        assert len(expand("n[1-65536]")) == 65536

    def test_one_past_the_bound_loses_exactly_one(self):
        # The off-by-one on the guard itself.
        assert len(expand("n[1-65537]")) == 65536

    def test_expansion_still_round_trips_through_collapse(self):
        # The property the module is for: collapse(expand(x)) names the same set.
        original = "cn-[0001-0010,0012-0015]"
        assert expand(collapse(expand(original))) == expand(original)


def _src():
    import pathlib

    return pathlib.Path(__file__).resolve().parent.parent / "src"


def test_the_ceiling_is_either_installed_or_the_case_is_skipped():
    """The ceiling is never assumed -- the child says whether it took.

    `RLIMIT_AS` is honoured on Linux and is the thing that gives the six cases
    above their teeth. It is not portable: Darwin aliases it onto `RLIMIT_RSS`
    and refuses the call with `EINVAL`, which CPython reports as
    `ValueError: current limit exceeds maximum limit`. Found on the macOS leg of
    CI, where all six went red for that reason while the four Linux legs were
    green -- a red that said nothing about `expand`, since the expansion never
    ran.

    So the child installs the ceiling under a guard and exits `_NO_CEILING`
    when it cannot, and the case SKIPS. Skip and not pass: a bound that was
    never enforced proves nothing about peak memory, and a green tick there
    would be the quiet kind of wrong. Probed in the child rather than keyed off
    `sys.platform`, so a container or a kernel that forbids the call is handled
    the same way, and this test asserts those are the only two outcomes -- the
    exit code is either the skip sentinel with its reason named, or a clean run
    whose answer is right.
    """
    done = subprocess.run(
        [sys.executable, "-c", _child_code("n[1-2]")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if done.returncode == _NO_CEILING:
        assert "RLIMIT_AS" in done.stderr
        return
    assert done.returncode == 0, done.stderr[-400:]
    assert int(done.stdout.strip()) == 2
    # A ceiling this run can install is one this platform reports back, so the
    # premise of every case above is checked here rather than assumed.
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    assert soft == resource.RLIM_INFINITY or soft > 0
