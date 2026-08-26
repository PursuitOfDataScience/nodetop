#!/usr/bin/env python3
"""Build ``nodetop.pyz``: the whole tool as one file, for a fast start.

Why this exists, in numbers. On a cluster whose home directory is NFS -- which
is most of them -- starting `nodetop` costs more in filesystem round trips than
in anything the code does:

    python -c pass                  64 ms      <- the interpreter alone
    import nodetop                 443 ms
    nodetop --version              419 ms

That is not CPU. It is two dozen module files, each an open and a read across
the network, at roughly 19 ms apiece. A zip on ``sys.path`` is **one** open, and
``zipimport`` reads the members out of memory. Measured on that same NFS home,
thirteen runs each, medians:

    ==========================  ========
    two dozen files (today)     419.0 ms
    one zip of sources          318.7 ms
    one zip of stripped bytecode 249.6 ms
    ==========================  ========

The last row is worth explaining, because it says something about this codebase
specifically: its docstrings are **61% of its compiled size** (1,303,123 bytes
of ``.pyc`` against 501,758 with docstrings stripped). Every run reads them off
the network to throw them away. That is a packaging problem and not a reason to
write fewer of them -- the comments and docstrings here *are* the design
record. This script leaves them in the repository and out of the artifact.

Two builds, and the difference matters:

``--portable`` (default)
    Ships ``.py`` sources. Runs on any interpreter that satisfies
    ``requires-python``, and ``zipimport`` compiles them in memory on each run
    -- which still beats reading two dozen files off NFS.

``--fast``
    Ships ``-OO`` bytecode built by *this* interpreter. Faster again, and
    **locked to this Python version**: bytecode carries a magic number, and
    zipimport refuses a mismatch. Build it on the machine that will run it --
    the archive's shebang names the building interpreter by absolute path,
    because a bare ``python3`` on a cluster whose system Python is 3.9 produces
    a ``runpy`` traceback rather than a usable error.

Usage::

    python tools/build_pyz.py                 # dist/nodetop.pyz, portable
    python tools/build_pyz.py --fast          # smaller and faster, version-locked
    python tools/build_pyz.py -o /tmp/nt.pyz  # somewhere else

Then run it directly -- it carries a shebang and the executable bit::

    ./dist/nodetop.pyz status
    python dist/nodetop.pyz status            # or explicitly

The installed package and its `nodetop`/`nt` console scripts are untouched by
any of this; the pyz is an extra way to run the same code, for people who pay
NFS latency on every invocation.
"""

from __future__ import annotations

import argparse
import compileall
import py_compile
import shutil
import subprocess
import sys
import tempfile
import zipapp
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "nodetop"


def _staged(destination: Path, *, fast: bool) -> None:
    """Copy the package into ``destination``, as sources or as bytecode."""
    shutil.copytree(PACKAGE, destination / "nodetop",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if not fast:
        return
    # `legacy=True` puts each `.pyc` where its `.py` was, which is the layout
    # zipimport expects when the source is absent. optimize=2 is what strips
    # the docstrings.
    compileall.compile_dir(
        destination / "nodetop", quiet=2, optimize=2, legacy=True,
        force=True, ddir="nodetop",
    )
    for source in (destination / "nodetop").rglob("*.py"):
        source.unlink()
    for cache in (destination / "nodetop").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def build(output: Path, *, fast: bool = False) -> Path:
    """Write the archive and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        _staged(staging, fast=fast)
        # The shebang has to match what the archive can actually run on, and
        # for `--fast` that is exactly one interpreter.
        #
        # Found the hard way: a `--fast` archive built under 3.12 and given
        # `/usr/bin/env python3` failed every command on a cluster whose system
        # `python3` is 3.9 -- with a `runpy` traceback, because unloadable
        # bytecode fails before `main` can say "nodetop needs Python 3.10". The
        # version-locked build therefore names the interpreter that built it,
        # which is the same machine it is meant to run on.
        #
        # The portable build keeps `python3`: it carries sources, so an old
        # interpreter reaches `main` and gets told the floor in one line.
        interpreter = sys.executable if fast else "/usr/bin/env python3"
        # Our own entry shim, NOT zipapp's `main=`. The one zipapp generates is
        #
        #     import nodetop.cli
        #     nodetop.cli.main()
        #
        # which discards the return value, so every command exits 0. That
        # silently breaks the contract `cmd_check` is built around --
        # `nodetop check && sbatch ...` would wave the caller through a refusal
        # -- and it breaks it in the direction where nobody notices: measured
        # against the installed package, `queues -q nope` exits 2 there and 0
        # from the archive. `raise SystemExit(main())` is what
        # `nodetop/__main__.py` already does for `python -m nodetop`.
        (staging / "__main__.py").write_text(
            "from nodetop.cli import main\n\nraise SystemExit(main())\n")
        zipapp.create_archive(
            staging, target=output, interpreter=interpreter, compressed=True,
        )
    output.chmod(0o755)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output", type=Path,
                        default=ROOT / "dist" / "nodetop.pyz",
                        help="where to write the archive")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--portable", action="store_true", default=True,
                      help="ship sources: runs on any supported Python (default)")
    mode.add_argument("--fast", action="store_true",
                      help="ship this interpreter's stripped bytecode: faster, "
                           "and only runs on this Python version")
    args = parser.parse_args(argv)

    built = build(args.output, fast=args.fast)
    size = built.stat().st_size
    with zipfile.ZipFile(built) as archive:
        members = len(archive.namelist())

    # Prove it runs before saying it is built. A pyz that imports nothing is a
    # 500 KB file that looks fine until somebody needs it during an outage.
    check = subprocess.run([sys.executable, str(built), "--version"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        print(f"built {built} but it does not run:\n{check.stderr}", file=sys.stderr)
        return 1

    # And again through the SHEBANG, which is a different question and the one
    # that actually went wrong: the check above names the interpreter, so it
    # passed happily for an archive whose `#!/usr/bin/env python3` resolved to a
    # 3.9 that could not load it. Every command failed with a `runpy` traceback
    # and the build had called itself good.
    # Resolved: `exec` searches PATH for a bare name, so a relative output path
    # would make this check fail as "not found" and say nothing about the pyz.
    direct = subprocess.run([str(built.resolve()), "--version"],
                            capture_output=True, text=True)
    if direct.returncode != 0:
        # A portable archive on a host whose `python3` predates the floor is not
        # a broken build -- the archive says so itself, in one line, which is
        # the whole reason the portable build ships sources. Say which it is.
        old_python = "needs Python" in direct.stderr
        where = "  (built fine; this host's `python3` is too old to run it)"
        print(f"{'note' if old_python else 'ERROR'}: {built} does not run via its "
              f"own shebang:\n  {direct.stderr.strip().splitlines()[-1][:120]}"
              + (where if old_python else ""), file=sys.stderr)
        if not old_python:
            return 1

    kind = "this interpreter's stripped bytecode" if args.fast else "sources"
    print(f"{built}  ({size // 1024} KiB, {members} members, {kind})")
    print(f"  {check.stdout.strip()}")
    if args.fast:
        print(f"  locked to Python {sys.version_info.major}."
              f"{sys.version_info.minor} (magic {py_compile.importlib.util.MAGIC_NUMBER.hex()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
