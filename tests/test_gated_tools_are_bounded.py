"""A tool that decides whether CI passes needs an upper bound.

`ruff check src tests` and `mypy src` are gates in both workflows, so those two
tools decide whether a push is green. `pyproject.toml` pinned `ruff>=0.3` and
`mypy>=1.8` with no ceiling, which makes CI a moving target: the next release
can fail a tree nobody touched, on somebody else's push, with no change to
blame.

The mechanism is specific here. `[tool.ruff.lint] select` names whole rule
FAMILIES -- E, F, W, I, N, UP, B, SIM, ARG, C4, T20 -- rather than individual
codes, so every rule ruff adds to any of those families is opted into
automatically by a version bump. Two siblings already pin for this reason:
slurmpast's comment records "a minor ruff release changing its rule defaults is
exactly what broke the first CI run here", and rapidu bounds its formatter on
the same argument.

Shaped as an implication that only fires while the hazard exists, with a control
asserting the hazard is really there -- so the pin cannot be satisfied by
quietly deleting the gate. `pyproject.toml` is read as text: `tomllib` is 3.11+
and this package supports 3.10.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ["ci.yml", "release.yml"]


def _requirement(name: str) -> str:
    """The version spec for `name` in the `dev` extra, e.g. `>=0.15,<0.17`."""
    text = (ROOT / "pyproject.toml").read_text()
    found = re.findall(rf'^\s*"{name}([^"]*)",\s*$', text, re.MULTILINE)
    assert found, f"{name} is not pinned in pyproject.toml at all"
    assert len(found) == 1, f"{name} is pinned {len(found)} times: {found}"
    return str(found[0])


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text()


class TestTheGatedToolsAreBounded:
    @pytest.mark.parametrize("tool", ["ruff", "mypy"])
    def test_a_tool_a_gate_runs_has_an_upper_bound(self, tool: str) -> None:
        gated = [w for w in WORKFLOWS if re.search(rf"run:\s*{tool}\b", _workflow(w))]
        assert gated, f"no workflow runs {tool}; this test's premise is gone"
        spec = _requirement(tool)
        assert "<" in spec, (
            f"{tool} decides CI in {gated} but is pinned {tool}{spec} with no ceiling: "
            f"a minor release can fail a tree nobody touched"
        )


class TestControls:
    """These read no ceiling, so they hold with the pins bounded or not."""

    def test_the_lint_gate_really_runs_so_the_pin_is_not_vacuous(self) -> None:
        # If this fails, the implication above went quiet for the wrong reason:
        # the gate was removed rather than the pin fixed.
        running = [w for w in WORKFLOWS if re.search(r"run:\s*ruff check", _workflow(w))]
        assert running == WORKFLOWS, f"only {running} run `ruff check`"

    @pytest.mark.parametrize("tool", ["ruff", "mypy"])
    def test_a_floor_is_still_declared(self, tool: str) -> None:
        # A ceiling is not a substitute for a floor: a reproducible rule set
        # needs bounding at both ends.
        assert ">=" in _requirement(tool)

    def test_the_selection_still_names_families_rather_than_codes(self) -> None:
        """The premise of the ceiling, asserted rather than assumed.

        A family (`SIM`) opts into every rule ruff ever adds to it; a code
        (`SIM117`) does not. If this ever fails, the argument in the pin's
        comment has changed and wants re-reading -- not silently keeping a
        ceiling justified by something no longer true.
        """
        text = (ROOT / "pyproject.toml").read_text()
        select = re.search(r"^select\s*=\s*\[([^\]]*)\]", text, re.MULTILINE)
        assert select, "no [tool.ruff.lint] select found"
        chosen = re.findall(r'"([A-Z0-9]+)"', select.group(1))
        assert chosen, select.group(1)
        families = [c for c in chosen if not re.search(r"\d", c)]
        assert families, f"select names no rule families, only codes: {chosen}"
