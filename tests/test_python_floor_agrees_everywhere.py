"""Five places state the supported Python range, and nothing checked they agree.

`README.md` carries a `python-3.10+` badge, `pyproject.toml` declares
`requires-python = ">=3.10"` and four `Programming Language :: Python :: 3.x`
classifiers, ruff is pinned to `target-version = "py310"`, and `ci.yml` runs the
matrix. All five say the same thing today.

Four of them move on their own when the floor is raised — `requires-python` because
installs break, the matrix because CI is what runs, `target-version` because ruff
starts allowing newer syntax, the classifiers because PyPI shows them. **The badge is
a shields.io URL nobody greps**, so it is the one that silently keeps saying 3.10.
That is the same failure slurmpast's `tests-1029` badge had, and this repo already
guards its other badge (`test_the_readme_badge_still_says_none`) — the Python one was
simply not covered.

Getting it wrong is not cosmetic in either direction: a badge promising a version the
package refuses turns into a failed install, and a badge lagging the real floor sends
someone to build a 3.10 environment for a package that needs 3.11.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _version_key(text):
    return tuple(int(part) for part in text.split("."))


def badge_floor():
    """The `python-3.10+` badge's floor. `%2B` is the URL-encoded `+`."""
    found = re.search(r"badge/python-(\d+\.\d+)%2B", README.read_text())
    assert found, "the python badge should still be in the README"
    return found.group(1)


def requires_python_floor():
    found = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', PYPROJECT.read_text())
    assert found, "requires-python should still be declared"
    return found.group(1)


def ruff_target():
    found = re.search(r'target-version\s*=\s*"py(\d)(\d+)"', PYPROJECT.read_text())
    assert found, "ruff target-version should still be pinned"
    return f"{found.group(1)}.{found.group(2)}"


def classifier_versions():
    return re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', PYPROJECT.read_text())


def matrix_versions():
    found = re.search(r"python-version:\s*\[([^\]]+)\]", CI.read_text())
    assert found, "ci.yml should still declare a python matrix"
    return [v.strip().strip('"').strip("'") for v in found.group(1).split(",")]


class TestEverySurfaceStatesTheSameFloor:
    def test_the_badge_matches_requires_python(self):
        assert badge_floor() == requires_python_floor(), (
            f"README badge says {badge_floor()}+, pyproject requires "
            f">={requires_python_floor()} — the badge is the copy nobody greps"
        )

    def test_ruff_targets_the_declared_floor(self):
        assert ruff_target() == requires_python_floor(), (
            f"ruff target-version is py{ruff_target().replace('.', '')} but the package "
            f"claims >={requires_python_floor()}; ruff would allow syntax the floor forbids"
        )

    def test_ci_runs_the_declared_floor(self):
        lowest = min(matrix_versions(), key=_version_key)
        assert lowest == requires_python_floor(), (
            f"ci.yml's lowest job is {lowest}, the package claims "
            f">={requires_python_floor()} — the floor would be untested"
        )

    def test_the_classifiers_are_exactly_the_tested_versions(self):
        assert sorted(classifier_versions(), key=_version_key) == sorted(
            matrix_versions(), key=_version_key
        ), {"classifiers": classifier_versions(), "matrix": matrix_versions()}


class TestControls:
    """These read no floor value, so each holds whatever the surfaces say."""

    def test_every_surface_was_actually_found(self):
        # A guard that cannot fail is not a guard: if any regex stopped matching,
        # the assertions above would compare nothing.
        assert len(matrix_versions()) >= 2
        assert len(classifier_versions()) >= 2
        assert re.fullmatch(r"\d+\.\d+", badge_floor())
        assert re.fullmatch(r"\d+\.\d+", requires_python_floor())
        assert re.fullmatch(r"\d+\.\d+", ruff_target())

    def test_the_version_key_orders_numerically_not_lexically(self):
        # "3.9" > "3.10" as strings, which would pick the wrong floor.
        versions = ["3.9", "3.10", "3.13"]
        assert min(versions, key=_version_key) == "3.9"
        assert min(versions) == "3.10", "string ordering really does differ here"

    def test_the_other_readme_badge_is_untouched(self):
        # The badge this repo already guards, asserted independently.
        assert "dependencies-none" in README.read_text()
