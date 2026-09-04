"""`_OPERATIONAL_BLOCKERS` says a mistake either way is silent, and nothing checked it.

Its own comment states the invariant and the cost of breaking it::

    Every code here is one a `Blocker` in this codebase actually carries, and
    each was checked against where it is raised -- a set with a code nobody
    emits silently stops saving anything, and a missing one silently keeps
    paying.

"Checked against where it is raised" was a manual pass at some past time. The
consumer is `fit.evaluate`::

    dead = [b for b in blockers if b.fatal and b.code in _OPERATIONAL_BLOCKERS]

so a code in the set that is never emitted **fatal** can never fire, and an
operational code left out of the set costs a dry-run round trip against a queue
that accepts nothing from anyone. Measured on this tree: 14 blocker codes are
emitted, 6 are operational, and every one of the 8 exclusions matches a reason
the source states -- but a fifteenth code added tomorrow is triaged by nobody.

Both directions are swept from the AST rather than from a list, so the sweep
cannot fall behind the code it guards.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from nodetop.core.fit import _OPERATIONAL_BLOCKERS
from nodetop.core.model import Blocker

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "nodetop"

#: Emitted codes that are deliberately NOT operational, each with the reason the
#: source gives. A new code must be added here or to `_OPERATIONAL_BLOCKERS`, and
#: until it is, `test_every_emitted_code_is_triaged` fails -- which is the whole
#: point: the comment says a missing one "silently keeps paying".
_NOT_OPERATIONAL = {
    # "Distinguished from an ACCESS blocker (which the control plane may
    # contradict, and that contradiction is the point of probing at all)."
    "ACCOUNT_NOT_ALLOWED": "access",
    "GROUP_NOT_ALLOWED": "access",
    "QOS_NOT_ALLOWED": "access",
    "USER_NOT_ALLOWED": "access",
    # "...and from a soft one (which is about the size of the request, not the
    # queue)."
    "MAX_WALLTIME": "soft",
    "QUEUE_MAX_NODES": "soft",
    "QUEUE_MAX_WALLTIME": "soft",
    # "`REQUIRES_RESERVATION` is deliberately absent: that queue does accept
    # work, from a job that names a reservation, so the dry-run's answer is real."
    "REQUIRES_RESERVATION": "soft",
}


def _emitted() -> dict[str, set[str]]:
    """``{code: {fatal values seen at construction}}`` across ``src/nodetop``.

    Read off `Blocker(...)` calls: the codes are string literals at the call
    site, so the AST is the authority and a list in this file would be the thing
    that rots. ``"<default>"`` stands for an omitted ``fatal=``, which is
    ``True`` on the dataclass.
    """
    rows: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            if name != "Blocker":
                continue
            code = None
            fatal = "<default>"
            for kw in node.keywords:
                if kw.arg == "code" and isinstance(kw.value, ast.Constant):
                    code = kw.value.value
                if kw.arg == "fatal" and isinstance(kw.value, ast.Constant):
                    fatal = str(kw.value.value)
            if code is None and node.args and isinstance(node.args[0], ast.Constant):
                code = node.args[0].value
            if isinstance(code, str):
                rows.setdefault(code, set()).add(fatal)
    return rows


class TestTheSweepCanProveSomething:
    def test_it_finds_the_codes(self) -> None:
        """Vacuity guard: a sweep that found nothing would pass every assertion
        below. 14 codes on the tree this was written against."""
        emitted = _emitted()
        assert len(emitted) >= 10, sorted(emitted)
        assert "QUEUE_DISABLED" in emitted
        assert _OPERATIONAL_BLOCKERS, "the set under test is empty"

    def test_fatal_defaults_to_true(self) -> None:
        """The sweep records an omitted `fatal=` as `<default>`, and the checks
        below read that as fatal. If the dataclass default flipped, they would be
        silently inverted."""
        assert Blocker("X", "y").fatal is True


class TestEveryOperationalCodeCanActuallyFire:
    """The "silently stops saving anything" direction."""

    @pytest.mark.parametrize("code", sorted(_OPERATIONAL_BLOCKERS))
    def test_it_is_emitted_somewhere(self, code: str) -> None:
        assert code in _emitted(), code

    @pytest.mark.parametrize("code", sorted(_OPERATIONAL_BLOCKERS))
    def test_it_is_emitted_fatal(self, code: str) -> None:
        """`evaluate` requires `b.fatal AND b.code in _OPERATIONAL_BLOCKERS`, so a
        code only ever raised non-fatal is in the set and unreachable."""
        seen = _emitted()[code]
        assert seen <= {"<default>", "True"}, (code, seen)


class TestEveryEmittedCodeIsTriaged:
    """The "silently keeps paying" direction -- a new code must be classified."""

    def test_no_code_is_unclassified(self) -> None:
        emitted = set(_emitted())
        classified = set(_OPERATIONAL_BLOCKERS) | set(_NOT_OPERATIONAL)
        assert emitted - classified == set(), sorted(emitted - classified)

    def test_the_two_groups_do_not_overlap(self) -> None:
        assert set(_OPERATIONAL_BLOCKERS) & set(_NOT_OPERATIONAL) == set()

    def test_the_exclusion_list_names_only_real_codes(self) -> None:
        """An exclusion for a code nobody emits would let a real one hide behind
        a stale name."""
        emitted = set(_emitted())
        assert set(_NOT_OPERATIONAL) <= emitted, sorted(set(_NOT_OPERATIONAL) - emitted)

    def test_the_soft_exclusions_really_are_non_fatal(self) -> None:
        """The source's own reason for excluding them: "about the size of the
        request, not the queue". A soft code raised fatal would be a queue-level
        refusal wearing a request-level label, and `evaluate` would still skip it
        only if it were in the set -- so the label has to match the flag."""
        emitted = _emitted()
        for code, why in _NOT_OPERATIONAL.items():
            if why != "soft":
                continue
            assert emitted[code] == {"False"}, (code, emitted[code])

    def test_the_access_exclusions_are_fatal_and_still_excluded(self) -> None:
        """The other half, and the more surprising one: an ACCESS blocker IS
        fatal, and is excluded anyway, because "the control plane may contradict
        it, and that contradiction is the point of probing at all"."""
        emitted = _emitted()
        access = [c for c, why in _NOT_OPERATIONAL.items() if why == "access"]
        assert access, _NOT_OPERATIONAL
        for code in access:
            assert emitted[code] <= {"<default>", "True"}, (code, emitted[code])
            assert code not in _OPERATIONAL_BLOCKERS


class TestControls:
    """The consumer's behaviour, which must hold whatever this file asserts."""

    def test_control_a_fatal_operational_blocker_is_dead(self) -> None:
        blockers = [Blocker("QUEUE_DISABLED", "state=DOWN")]
        dead = [b for b in blockers if b.fatal and b.code in _OPERATIONAL_BLOCKERS]
        assert dead, blockers

    def test_control_a_fatal_access_blocker_is_not_dead(self) -> None:
        """The distinction `evaluate` rests on: a refusal about YOU is still worth
        probing, because the control plane may disagree."""
        blockers = [Blocker("ACCOUNT_NOT_ALLOWED", "not on the allowlist")]
        dead = [b for b in blockers if b.fatal and b.code in _OPERATIONAL_BLOCKERS]
        assert dead == [], blockers

    def test_control_a_soft_blocker_is_not_dead(self) -> None:
        blockers = [Blocker("QUEUE_MAX_NODES", "asks for more than the queue has", fatal=False)]
        dead = [b for b in blockers if b.fatal and b.code in _OPERATIONAL_BLOCKERS]
        assert dead == [], blockers
