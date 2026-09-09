"""`nodetop.core` imports no backend -- the claim the design rests on.

DESIGN.md opens with eight rows of the same lie told by five different
schedulers and draws an architectural conclusion from them: "Not one of those
rows is about Slurm. That's the point -- the *reasoning* is
scheduler-independent, so it lives in `nodetop.core`, which imports no backend
and knows what no scheduler is." `core/cluster.py` says it again at the one
place it could go wrong, as a comment on its annotation-only import: "only for
annotations: core never imports a backend".

Both are prose, and nothing read either of them, so a `from ..backends.slurm
import SlurmBackend` added to a core module would have passed every gate. That
is the hole `test_startup.py` exists to close for deferred imports -- "nothing
stops a later edit from hoisting it back to the top, and nothing would fail".

Checked two ways, because each alone has a gap. The source scan sees an import
that never executes -- inside a function, or under `TYPE_CHECKING` -- which no
runtime probe can observe. The runtime probe sees one that arrives through a
module core imports rather than from core's own source.

And the deferred case is the one that matters, which measuring the neuters
showed. A concrete import at a core module's TOP LEVEL is already caught by
Python: `from ..backends.slurm import SlurmBackend` in `core/capacity.py` makes
`nodetop` uninstallable-in-practice, `ImportError: cannot import name
'BackendCapabilities' from partially initialized module
'nodetop.backends.base' (most likely due to a circular import)`, and collection
stops. Put the same import INSIDE a function and nothing breaks at all: the
runtime probe stays green because the line never executes, and only the two
source scans fail. That silent case is what this file is for.

Measured when written: importing all seven core modules loads
`nodetop.backends` and `nodetop.backends.base`, and no concrete backend. Those
two arrive from `nodetop/__init__.py`, which re-exports the abstract `Backend`,
and `backends/__init__.py` is a name table on purpose so that "asking *which*
backends exist costs no imports at all and detecting one costs only the one".
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

import nodetop.core

#: The scheduler-specific modules. `base` is deliberately absent: it is the
#: abstract interface, and the annotation-only import of it is the exception
#: `cluster.py` documents at the site.
CONCRETE = ("slurm", "pbs", "lsf", "sge", "kubernetes", "sshpool")

CORE = pathlib.Path(nodetop.core.__file__).parent


def _core_sources() -> list[pathlib.Path]:
    return sorted(CORE.glob("*.py"))


def _type_checking_lines(tree: ast.Module) -> set[int]:
    """Line numbers sitting inside an ``if TYPE_CHECKING:`` block."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            for child in ast.walk(node):
                line = getattr(child, "lineno", None)
                if line is not None:
                    lines.add(line)
    return lines


def _backend_imports(path: pathlib.Path) -> list[tuple[str, int, bool]]:
    """Imports naming ``backends``, as ``(target, line, under_type_checking)``."""
    tree = ast.parse(path.read_text())
    guarded = _type_checking_lines(tree)
    found: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = "." * (node.level or 0) + (node.module or "")
            joiner = "." if node.module else ""
            targets = [f"{base}{joiner}{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            continue
        found.extend(
            (target, node.lineno, node.lineno in guarded)
            for target in targets
            if "backends" in target
        )
    return found


def _dynamic_backend_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """``(line, module)`` for every dynamic import naming a concrete backend.

    The scan above reads `ast.Import`/`ast.ImportFrom`, so three forms walk
    straight past it — tabulated against crafted snippets rather than read off
    the code:

    ===================================================  =======
    form                                                 static scan
    ===================================================  =======
    ``from ..backends.slurm import SlurmBackend``         catches
    ``import nodetop.backends.slurm``                     catches
    ``importlib.import_module("nodetop.backends.slurm")`` MISSES
    ``import_module(".backends.slurm", __package__)``      MISSES
    ``__import__("nodetop.backends.slurm")``              MISSES
    ===================================================  =======

    Not a theoretical gap: `core/access.py` already writes
    ``__import__("contextlib")`` twice, so the form is idiomatic in the very
    directory this guards, and a developer following that local style would not
    be flagged. Worse, both of those sit inside functions — and a dynamic import
    inside a function executes at call time, so it escapes the runtime
    `sys.modules` probe as well. That is the same "the deferred case is the one
    that matters" shape the module docstring already records for a plain import
    hoisted into a function body.

    `backends/__init__.py` uses `import_module` legitimately, by design ("asking
    *which* backends exist costs no imports at all") — but that is not `core/`,
    and this scan is only ever pointed at `core/`.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if not (callee == "__import__" or callee.split(".")[-1] == "import_module"):
            continue
        for arg in node.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            if any(f"backends.{name}" in arg.value for name in CONCRETE):
                found.append((node.lineno, f"{callee}({arg.value!r})"))
    return found


class TestCoreNeverImportsAConcreteBackend:
    def test_no_core_source_reaches_a_backend_dynamically(self) -> None:
        offenders = [
            f"{path.name}:{line} {what}"
            for path in _core_sources()
            for line, what in _dynamic_backend_imports(path)
        ]
        assert offenders == [], f"core must not import a backend dynamically; found {offenders}"

    def test_no_core_source_names_a_scheduler_specific_module(self) -> None:
        offenders = [
            (path.name, target, line)
            for path in _core_sources()
            for target, line, _ in _backend_imports(path)
            if any(f"backends.{name}" in target for name in CONCRETE)
        ]
        assert offenders == [], f"core must import no backend (DESIGN.md); found {offenders}"

    def test_the_only_backend_references_are_the_two_deliberate_ones(self) -> None:
        """A whitelist, so a NEW kind of backend reference has to be argued for.

        ``..backends.base`` is annotations-only and guarded; ``..backends`` is
        the lazy import behind ``snapshot(backend=None)``, which calls
        ``detect()`` -- generic, and deferred so that importing core does not
        pay for it.
        """
        refs = {
            (path.name, target, guarded)
            for path in _core_sources()
            for target, _, guarded in _backend_imports(path)
        }
        assert refs == {
            ("cluster.py", "..backends.base.Backend", True),
            ("cluster.py", "..backends", False),
        }, refs

    def test_importing_every_core_module_loads_no_concrete_backend(self) -> None:
        modules = [p.stem for p in _core_sources() if p.stem != "__init__"]
        probe = (
            "import sys\n"
            + "".join(f"import nodetop.core.{name}\n" for name in modules)
            + "print('|'.join(sorted(m for m in sys.modules if 'nodetop.backends' in m)))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
        loaded = [m for m in done.stdout.strip().split("|") if m]
        assert loaded == ["nodetop.backends", "nodetop.backends.base"], loaded


class TestControls:
    """These hold whether or not the rule above does."""

    def test_the_scan_actually_found_the_core_modules(self) -> None:
        names = {p.name for p in _core_sources()}
        assert {"cluster.py", "capacity.py", "fit.py", "model.py"} <= names, names

    def test_the_scan_reads_imports_at_all(self) -> None:
        # Were `_backend_imports` silently returning [] for everything, the
        # rule above would pass vacuously.
        tree = ast.parse((CORE / "cluster.py").read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
        assert len(imports) >= 5, len(imports)

    def test_the_dependency_runs_the_other_way(self) -> None:
        # One-way, not "nobody imports anybody": a concrete backend is entitled
        # to import core, and does.
        text = (CORE.parent / "backends" / "slurm.py").read_text()
        assert "core" in text, "a backend should still be free to import core"


class TestTheScanSeesDynamicImportsToo:
    """Three import forms walked past the static scan.

    Found by tabulating, not by reading: `ast.Import`/`ast.ImportFrom` cover the
    two spellings a linter thinks about, and `importlib.import_module(...)` /
    `__import__(...)` are neither. The gap is real rather than theoretical —
    `core/access.py` already uses `__import__("contextlib")` twice, inside
    functions, so the form is idiomatic here AND invisible to the runtime probe
    (a call-time import never appears in `sys.modules` at module-import time).
    """

    CAUGHT = {
        "importlib absolute": (
            'import importlib\nimportlib.import_module("nodetop.backends.slurm")\n'
        ),
        "import_module relative": (
            'from importlib import import_module\nimport_module(".backends.pbs", __package__)\n'
        ),
        "dunder import": '__import__("nodetop.backends.kubernetes")\n',
        "inside a function": (
            "def f():\n    import importlib\n"
            '    return importlib.import_module("nodetop.backends.lsf")\n'
        ),
    }

    def _scan(self, source: str, tmp_path: pathlib.Path) -> list[tuple[int, str]]:
        path = tmp_path / "probe.py"
        path.write_text(source)
        return _dynamic_backend_imports(path)

    @pytest.mark.parametrize("form", sorted(CAUGHT))
    def test_a_dynamic_backend_import_is_caught(self, form: str, tmp_path: pathlib.Path) -> None:
        assert self._scan(self.CAUGHT[form], tmp_path), form

    @pytest.mark.parametrize(
        "source",
        [
            '__import__("contextlib")\n',  # the idiom core/access.py already uses
            # core, not a backend
            'import importlib\nimportlib.import_module("nodetop.core.model")\n',
            # the package, not a scheduler
            'import importlib\nimportlib.import_module("nodetop.backends")\n',
            # not a constant, so unknowable
            "import importlib\nimportlib.import_module(name)\n",
            "importlib.reload(mod)\n",  # a different importlib call
        ],
    )
    def test_it_does_not_cry_wolf(self, source: str, tmp_path: pathlib.Path) -> None:
        assert self._scan(source, tmp_path) == [], source

    def test_the_real_core_has_no_dynamic_backend_import(self) -> None:
        # The same claim the static scan makes, for the forms it cannot see.
        found = [
            (path.name, hit) for path in _core_sources() for hit in _dynamic_backend_imports(path)
        ]
        assert found == []

    def test_the_static_scan_is_untouched(self) -> None:
        # The control on the widening: the original two forms must still be the
        # static scan's business, not silently handed to the new one.
        assert _dynamic_backend_imports(CORE / "cluster.py") == []
        assert {t for t, _, _ in _backend_imports(CORE / "cluster.py")} == {
            "..backends.base.Backend",
            "..backends",
        }
