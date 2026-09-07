"""Deprecated stdlib APIs, scanned across the WHOLE package including subpackages.

`requires-python = ">=3.10"` has no upper bound, so pip installs this on whatever
comes next. Nothing held the source away from the APIs that a later interpreter
removes: `src/nodetop` uses none of them, but only because nobody had reached for one.

The three sibling packages carry the same scan. Copying theirs verbatim would have
been wrong here, and measurably so: they are flat single-directory packages, so their
`SRC.glob("*.py")` sees everything. **nodetop is the only one with subpackages** —
`core/` and `backends/` hold 16 of its 25 modules, so a flat glob would have covered
9 and reported clean. `test_the_scan_reaches_the_subpackages` is the guard against
that specific mistake.

Why a test at all, when CI runs 3.10-3.13: because this list is the part CI *cannot*
catch. A module a newer Python **removed** is an `ImportError` in the matrix and needs
no test. These are **deprecated but still importable** — verified on the interpreter
this suite runs on (3.11): `distutils`, `imp` and `pkg_resources` all import, and
`datetime.utcnow`, `locale.getdefaultlocale`, `typing.ByteString` and
`importlib.find_loader` all still exist. They pass every job today and break later.

The list is kept identical to the siblings' rather than trimmed to the ones that
outlive 3.13, because which entry is removed at which version is not something this
machine can verify — only 3.11 is installed here.
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "nodetop"

#: Deprecated or removed, but importable on the versions this package supports.
#: Identical to slurmate's, slurmpast's, slurmwatch's and rapidu's, so a change in
#: one transfers to all five.
BANNED = re.compile(
    r"\b(distutils|import imp\b|utcnow|getdefaultlocale|find_loader"
    r"|pkg_resources|typing\.ByteString)\b"
)


def _sources():
    """Every shipped module, subpackages included.

    `rglob`, not `glob`: see the module docstring. This is the one package in the
    family where the difference is 9 files against 25.
    """
    return sorted(SRC.rglob("*.py"))


class TestNoDeprecatedStdlibApis:
    def test_the_source_is_clean(self):
        offenders = [
            f"{path.relative_to(SRC)}:{n}"
            for path in _sources()
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if BANNED.search(line) and not line.lstrip().startswith("#")
        ]
        assert offenders == [], offenders

    def test_the_scan_reaches_the_subpackages(self):
        """The mistake a copied flat glob would make, pinned directly.

        `core/` and `backends/` are most of this package. A scan that missed them
        would still report clean, which is the worst kind of green.
        """
        reached = {str(p.relative_to(SRC).parent) for p in _sources()}
        assert {"core", "backends"} <= reached, sorted(reached)

    def test_there_are_as_many_sources_as_the_tree_holds(self):
        # A silent glob miss would make the scan vacuously clean; 25 today, and a
        # flat glob would see 9.
        assert len(_sources()) >= 20, len(_sources())

    def test_the_scanner_would_notice_a_real_offender(self):
        """A guard that cannot fail is not a guard."""
        for planted in (
            "from distutils.util import strtobool",
            "import imp",
            "datetime.datetime.utcnow()",
            "locale.getdefaultlocale()",
            "importlib.find_loader('x')",
            "import pkg_resources",
            "typing.ByteString",
        ):
            assert BANNED.search(planted), planted


class TestControls:
    """None of these reads the source tree, so each holds whatever it contains."""

    def test_word_boundaries_keep_innocent_names_safe(self):
        assert not BANNED.search("find_loaders(x)")
        assert not BANNED.search("my_utcnowish_helper")
        assert not BANNED.search("implementation")

    def test_the_banned_names_are_matched_bare(self):
        assert BANNED.search("distutils") and BANNED.search("import imp")

    def test_the_source_tree_is_where_this_expects(self):
        assert (SRC / "cli.py").is_file()
        assert (SRC / "core" / "cluster.py").is_file()
        assert (SRC / "backends" / "pbs.py").is_file()

    def test_the_dependency_claim_is_still_guarded_elsewhere(self):
        # This file scans stdlib usage; the "no third-party imports" claim is
        # test_readme.py's. Asserted so the split cannot go unnoticed.
        text = (pathlib.Path(__file__).parent / "test_readme.py").read_text()
        assert "dependencies-none" in text
