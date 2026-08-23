"""The README, checked against the code.

This file exists because of a specific failure. The snapshot/replay capability
was silently dropped during a package rename: the README went on advertising
it, the code no longer had it, and nothing noticed until someone read both.
Documentation that nothing verifies is documentation that will drift.

So every command the README shows must parse with the real argument parser,
every name it imports must exist, and every method it tells a contributor to
implement must be on the actual protocol.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from nodetop import backends
from nodetop.cli import build_parser

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
TEXT = README.read_text()


def _blocks(lang: str | None = None) -> list[str]:
    found = re.findall(r"```(\w*)\n(.*?)```", TEXT, re.S)
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
