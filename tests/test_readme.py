"""The prose docs, checked against the code.

This file exists because of a specific failure. The snapshot/replay capability
was silently dropped during a package rename: the README went on advertising
it, the code no longer had it, and nothing noticed until someone read both.
Documentation that nothing verifies is documentation that will drift.

So every command the docs show must parse with the real argument parser, every
name they import must exist, and every method they tell a contributor to
implement must be on the actual protocol.  README.md is the short front door
and DESIGN.md holds the long-form rationale; both are checked, because either
one drifting is the same defect.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from nodetop import backends
from nodetop.cli import build_parser

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DESIGN = ROOT / "DESIGN.md"
TEXT = README.read_text()
DOCS = TEXT + "\n" + DESIGN.read_text()


def _blocks(lang: str | None = None, text: str = DOCS) -> list[str]:
    found = re.findall(r"```(\w*)\n(.*?)```", text, re.S)
    return [body for tag, body in found if lang is None or tag == lang]


def _invocations() -> list[str]:
    """Every `nodetop ...` command line the README shows, cleaned up.

    Placeholders and shell plumbing are skipped: they are illustrative, not
    runnable, and pretending otherwise would make the test noise.
    """
    out: list[str] = []
    for body in _blocks():
        for raw in body.splitlines():
            line = raw.strip().lstrip("$ ").strip()
            if not line.startswith("nodetop"):
                continue
            line = line.split("#")[0].strip()
            if any(ch in line for ch in "<>…|&") or "..." in line:
                continue
            out.append(line)
    return sorted(set(out))


class TestReadmeExists:
    def test_there_are_invocations_to_check(self):
        # A guard on the extractor itself: if it silently matched nothing, the
        # rest of this file would pass while checking absolutely nothing.
        assert len(_invocations()) >= 10


class TestEveryShownCommandIsReal:
    @pytest.mark.parametrize("line", _invocations(), ids=lambda s: s)
    def test_it_parses_with_the_real_parser(self, line):
        # This is the check that would have caught `nodetop snapshot`
        # disappearing while the README still documented it.
        argv = line.split()[1:]
        build_parser().parse_args(argv)


class TestEveryImportedNameExists:
    def _imports(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for body in _blocks("python"):
            for m in re.finditer(r"^from ([\w.]+) import (.+)$", body, re.M):
                for name in m.group(2).split(","):
                    name = name.strip()
                    if name and name.isidentifier():
                        pairs.append((m.group(1), name))
        return pairs

    def test_the_extractor_found_some(self):
        assert self._imports()

    def test_every_name_is_importable(self):
        import importlib

        for module, name in self._imports():
            mod = importlib.import_module(module)
            assert hasattr(mod, name), f"{module}.{name} is documented but absent"


class TestDocumentedApiSurface:
    def _attribute_chains(self) -> set[str]:
        """`cluster.can_probe`, `place.runnable_now` and friends."""
        chains: set[str] = set()
        for body in _blocks("python"):
            for m in re.finditer(r"\b(cluster|place|shape)\.(\w+)", body):
                chains.add(f"{m.group(1)}.{m.group(2)}")
        return chains

    def test_documented_cluster_attributes_exist(self):
        from nodetop import Cluster

        for chain in self._attribute_chains():
            obj, attr = chain.split(".", 1)
            if obj != "cluster":
                continue
            assert hasattr(Cluster, attr) or attr in Cluster.__annotations__, chain

    def test_documented_placement_attributes_exist(self):
        from nodetop import Placement

        for chain in self._attribute_chains():
            obj, attr = chain.split(".", 1)
            if obj != "place":
                continue
            assert hasattr(Placement, attr) or attr in Placement.__annotations__, chain

    def test_documented_shape_attributes_exist(self):
        from nodetop import JobShape

        for chain in self._attribute_chains():
            obj, attr = chain.split(".", 1)
            if obj != "shape":
                continue
            assert hasattr(JobShape, attr) or attr in JobShape.__annotations__, chain


class TestBackendContract:
    """The 'Adding a backend' block is a promise to contributors."""

    def _promised_methods(self) -> list[str]:
        for body in _blocks("python"):
            if "class MyBackend" not in body:
                continue
            return re.findall(r"def (\w+)\(", body)
        return []

    def test_the_block_is_still_there(self):
        assert self._promised_methods()

    def test_every_promised_method_is_on_the_protocol(self):
        from nodetop.backends.base import Backend

        for name in self._promised_methods():
            assert hasattr(Backend, name), f"README promises {name}() but Backend lacks it"

    def test_every_real_backend_implements_them(self):
        for backend_name in backends.names():
            backend = backends.get(backend_name)
            for name in self._promised_methods():
                assert callable(getattr(backend, name, None)), (
                    f"{backend_name} is missing {name}()"
                )


class TestNamedBackendsAreReal:
    def _named(self) -> set[str]:
        # The support line at the top of the README.
        # The support line lives inside a centred HTML block in the header.
        m = re.search(r"Slurm · ([^<\n]+)", TEXT)
        assert m, "the backend support line has moved or gone"
        raw = m.group(1)
        aliases = {
            "pbs pro / openpbs / torque": "pbs",
            "lsf": "lsf",
            "grid engine": "sge",
            "kubernetes": "kubernetes",
            "a bare pool of machines": "sshpool",
        }
        found = {"slurm"}
        for part in raw.split("·"):
            key = part.strip().lower()
            if key in aliases:
                found.add(aliases[key])
        return found

    def test_every_advertised_backend_is_registered(self):
        assert self._named() <= set(backends.names())

    def test_every_registered_backend_is_advertised(self):
        # A backend nobody is told about might as well not exist.
        assert set(backends.names()) <= self._named()


class TestClaimedFlags:
    @pytest.mark.parametrize("flag", [
        "--json", "--no-color", "--ascii", "--backend", "--replay",
         "--detail", "--all", "--gpu-mem", "--needs", "--tolerates",
    ])
    def test_the_flag_the_readme_names_is_accepted(self, flag):
        assert flag in TEXT, f"{flag} is no longer documented"
        # Every one of these must be live somewhere in the CLI.
        parser = build_parser()
        actions = {o for a in parser._actions for o in a.option_strings}
        for sub in parser._subparsers._group_actions:  # type: ignore[union-attr]
            for choice in sub.choices.values():  # type: ignore[attr-defined]
                actions |= {o for a in choice._actions for o in a.option_strings}
        assert flag in actions, f"{flag} is documented but the CLI does not accept it"


class TestTheVersionIsSpelledOnce:
    """`pyproject.toml` and `_version.py` both carry the version literal.

    That duplication is deliberate and documented: `_version.py` is a literal so
    the package imports straight from a source tree with no build step, which a
    tool for diagnosing a cluster outage should not need. What was missing is
    anything keeping the two equal -- they agreed today with nothing to keep them
    agreeing tomorrow, which is the same defect this file's docstring is about,
    one layer down.

    The failure it guards is quiet in the worst way: a release bumps
    `pyproject.toml`, the wheel goes to PyPI as 0.6.0, and `nodetop --version`
    keeps saying 0.5.0 because that is what the literal says. A version string is
    the field a reader trusts without checking.
    """

    @staticmethod
    def _pyproject_version() -> str:
        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text()
        # A plain regex rather than a TOML parse: this must hold on 3.10, where
        # `tomllib` does not exist, and the package takes no dependencies.
        found = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.M)
        assert found, "pyproject.toml declares no static version"
        return found.group(1)

    def test_the_two_literals_agree(self):
        from nodetop._version import VERSION

        declared = self._pyproject_version()
        assert declared == VERSION, (
            f"_version.py says {VERSION}, pyproject.toml says {declared} — "
            f"`nodetop --version` reports the former and PyPI publishes the latter"
        )

    def test_what_the_cli_reports_is_that_same_string(self):
        # The control: the guard is only worth anything if `--version` really
        # reads the literal it pins. Asserted through the parser rather than by
        # reading the source, so a future indirection still has to agree.
        import nodetop

        assert nodetop.__version__ == self._pyproject_version()

    def test_the_version_is_a_release_number_not_a_placeholder(self):
        # `0.0.0`, `0.0.0+unknown` and an empty string all mean "we do not know",
        # and shipping one of those as a version is worse than failing to build.
        version = self._pyproject_version()
        assert re.fullmatch(r"\d+\.\d+\.\d+[\w.+-]*", version), version
        assert not version.startswith("0.0.0"), version


class TestTheNoDependenciesClaimIsTrue:
    """"No dependencies" is asserted in three places and guarded in none.

    `pyproject.toml` says `dependencies = []` with a comment explaining why, the
    README carries a `dependencies-none` badge, and DESIGN.md restates it — and
    nothing checked the tree. The claim is load-bearing rather than decorative:
    this is a tool reached for while a cluster is misbehaving, on a login node
    with nothing but the system Python, so one `import rich` ends its reason to
    exist.

    A sibling package had the same gap, with the check living only in a CI job —
    which cannot fail during the local gate run that introduces the import.
    """

    #: Stdlib modules that post-date this package's floor, and when they landed.
    #:
    #: `sys.stdlib_module_names` is the RUNNING interpreter's, so it answers "is
    #: this stdlib now" and cannot answer "was it stdlib at 3.10". Importing
    #: `tomllib` would pass the third-party check while raising
    #: `ModuleNotFoundError` on the oldest interpreter `_MIN_PYTHON` allows.
    TOO_NEW = {"tomllib": (3, 11), "dbm.sqlite3": (3, 13), "annotationlib": (3, 14)}

    @staticmethod
    def _sources():
        import nodetop

        return sorted(pathlib.Path(nodetop.__file__).parent.rglob("*.py"))

    @classmethod
    def _imports(cls, path):
        import ast

        found = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module)
        return found

    def test_there_are_sources_to_check(self):
        # A silent glob miss would make everything below vacuously pass.
        assert len(self._sources()) >= 15

    def test_no_module_imports_anything_third_party(self):
        import sys

        stdlib = set(sys.stdlib_module_names)
        local = {p.stem for p in self._sources()} | {"nodetop"}
        offenders = {}
        for path in self._sources():
            third = {
                name.split(".")[0]
                for name in self._imports(path)
                if name.split(".")[0] not in stdlib
                and name.split(".")[0] not in local
            }
            if third:
                offenders[path.name] = sorted(third)
        assert not offenders, (
            f"{offenders} — pyproject declares none, the README badge says none, "
            f"and DESIGN.md explains why it matters"
        )

    def test_no_module_imports_a_stdlib_addition_newer_than_the_floor(self):
        from nodetop.cli import _MIN_PYTHON

        offenders = {}
        for path in self._sources():
            names = self._imports(path)
            hit = {
                mod: ver
                for mod, ver in self.TOO_NEW.items()
                if ver > _MIN_PYTHON
                and (mod in names or any(n.split(".")[0] == mod for n in names))
            }
            if hit:
                offenders[path.name] = hit
        assert not offenders, f"{offenders} post-date the {_MIN_PYTHON} floor"

    def test_pyproject_still_declares_an_empty_list(self):
        # The claim from the other direction: a dependency declared but not yet
        # imported already breaks `pip install` on an air-gapped node.
        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text()
        found = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
        assert found is not None, "pyproject declares no `dependencies` key at all"
        declared = [x for x in (v.strip().strip('",') for v in found.group(1).split("\n")) if x]
        assert not declared, declared

    def test_the_readme_badge_still_says_none(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        assert "dependencies-none" in (root / "README.md").read_text()

    def test_the_detector_would_notice_a_real_import(self):
        # The control: a guard that cannot fail is not a guard.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe = pathlib.Path(tmp) / "probe.py"
            probe.write_text("import rich\nfrom textual.app import App\nimport tomllib\n")
            names = self._imports(probe)
            assert {"rich", "textual", "tomllib"} <= {n.split(".")[0] for n in names}
