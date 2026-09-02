"""The single-file build, which exists because module count is a cost.

On a cluster whose home is NFS, starting the tool is dominated by filesystem
round trips rather than by anything the code does: an interpreter floor of 64 ms
against 443 ms to `import nodetop`, at roughly 19 ms per module file. A zip on
`sys.path` is one open instead of two dozen. Measured on that home, thirteen
runs each, medians:

===================  ========  ==============  =============
command              files     pyz (sources)   pyz --fast
===================  ========  ==============  =============
`--version`          406.5 ms  351.2 ms        **222.5 ms**
`nodes --all`        603.7 ms  402.1 ms        **240.6 ms**
`queues`             474.9 ms  337.6 ms        **239.6 ms**
===================  ========  ==============  =============

Tested rather than left to rot: an artifact nobody exercises is the one that
turns out to be broken during an outage, which is exactly when somebody reaches
for the fast-starting copy.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile

import pytest

pytest.importorskip("zipapp")

#: Resolved from this file, not from the working directory: `pytest` is run from
#: wherever the reader happens to be standing, and a relative path made the
#: builder unfindable from anywhere but the repo root.
TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "build_pyz.py"

# Skipped rather than failed when the builder is not in the tree. An sdist that
# omits `tools/` still ships this file, and seven red tests for a missing build
# script look precisely like a portability failure in the code -- the same shape
# as a sibling package's `conftest.py` absence, which made a whole suite collect
# zero tests and report success.
pytestmark = pytest.mark.skipif(
    not TOOL.is_file(), reason=f"{TOOL.name} is not in this tree (sdist without tools/)"
)


def _build(tmp_path, *flags):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "nodetop.pyz"
    done = subprocess.run(
        [sys.executable, str(TOOL), "-o", str(out), *flags],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    return out, done.stdout


class TestItBuildsAndRuns:
    def test_the_portable_build_runs(self, tmp_path):
        out, said = _build(tmp_path)
        assert out.is_file()
        # The builder checks this itself and refuses to claim success otherwise,
        # which is the property being pinned here.
        assert "nodetop" in said
        got = subprocess.run([sys.executable, str(out), "--version"],
                             capture_output=True, text=True)
        assert got.returncode == 0
        assert "nodetop" in got.stdout

    def test_it_carries_sources_by_default(self, tmp_path):
        # Portable means any interpreter that satisfies `requires-python`, so
        # the archive must not be bytecode locked to whatever built it.
        out, _ = _build(tmp_path)
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        assert any(n.endswith("nodetop/cli.py") for n in names), names[:5]
        assert not any(n.endswith(".pyc") for n in names)

    def test_the_fast_build_carries_stripped_bytecode(self, tmp_path):
        # `-OO` is the point: this codebase's docstrings are 61% of its
        # compiled size (1,303,123 bytes against 501,758), and every start
        # reads them off the network to throw them away.
        out, said = _build(tmp_path, "--fast")
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        assert any(n.endswith("nodetop/cli.pyc") for n in names), names[:5]
        # No package source at all. `__main__.py` is exempt: `zipapp` writes
        # that shim itself, and it is two lines calling `nodetop.cli:main`.
        assert not [n for n in names
                    if n.endswith(".py") and n != "__main__.py"]
        assert "locked to Python" in said

    def test_a_real_command_works_from_the_archive(self, tmp_path):
        # `--version` only proves the entry point. `backends` builds the parser,
        # walks the registry and renders a table, which is the shape of a real
        # run without needing a scheduler.
        out, _ = _build(tmp_path)
        got = subprocess.run([sys.executable, str(out), "backends", "--no-color"],
                             capture_output=True, text=True)
        assert got.returncode == 0
        assert "batch systems" in got.stdout

    def test_the_exit_status_survives(self, tmp_path):
        """`nodetop check && sbatch ...` has to mean something from the archive.

        zipapp's own generated shim is `import nodetop.cli; nodetop.cli.main()`
        -- the return value dropped, so every command exits 0. Measured against
        the installed package before the fix: `queues -q nope` exits 2 there and
        **0** from the archive, which would wave a caller through a refusal. The
        archive carries its own `raise SystemExit(main())` instead.
        """
        out, _ = _build(tmp_path)
        for argv, expected in ((["queues", "-q", "definitely-not-a-partition"], 2),
                               (["--version"], 0)):
            installed = subprocess.run([sys.executable, "-m", "nodetop", *argv],
                                       capture_output=True, text=True)
            archived = subprocess.run([sys.executable, str(out), *argv],
                                      capture_output=True, text=True)
            assert installed.returncode == expected, installed.stderr[-200:]
            assert archived.returncode == installed.returncode, argv

    def test_the_fast_build_names_its_own_interpreter(self, tmp_path):
        """A version-locked archive cannot be launched by `python3`, whatever that is.

        Found on a cluster whose system `python3` is 3.9: a `--fast` archive
        built under 3.12 with `#!/usr/bin/env python3` failed *every* command
        with a `runpy` traceback -- unloadable bytecode fails before `main` can
        say "nodetop needs Python 3.10". The bytecode build therefore names the
        interpreter that made it; the portable build keeps `python3`, because
        its sources reach `main` and get told the floor in one line.
        """
        fast, _ = _build(tmp_path / "fast", "--fast")
        portable, _ = _build(tmp_path / "portable")
        assert fast.read_bytes().split(b"\n", 1)[0] == f"#!{sys.executable}".encode()
        assert portable.read_bytes().split(b"\n", 1)[0] == b"#!/usr/bin/env python3"

    def test_it_is_executable(self, tmp_path):
        # It carries a shebang, so `./nodetop.pyz` has to work on a login node
        # where nobody wants to type the interpreter.
        out, _ = _build(tmp_path)
        assert out.stat().st_mode & 0o111
        with out.open("rb") as fh:
            assert fh.read(2) == b"#!"
