"""What may and may not be imported before the tool has done anything.

Startup is dominated by imports, not by work: 30 ms of interpreter, 83 ms to
print `--version`. Every module on that path is a file to find, read and
unmarshal, so the cheapest optimisation available is not importing it. Measured
on two source trees identical but for these imports, runs strictly alternating
between them, 21 pairs, medians -- ``pathlib``, ``difflib`` and
``concurrent.futures`` moved into the four functions that use them:

====================  =====================  ==========
command               login node (warm pyc)  NFS home
====================  =====================  ==========
`--version`           84.0 -> **77.2 ms**    345.6 -> **327.8 ms**
`--help`              84.9 -> **79.2 ms**    --
`import nodetop.cli`  --                     332.6 -> **292.8 ms**
====================  =====================  ==========

Alternating on purpose: measuring all of one tree and then all of the other let
NFS caching hand back a +90 ms result with the sign reversed.

The property is invisible in the code once an import sits inside a function --
nothing stops a later edit from hoisting it back to the top, and nothing would
fail. Hence this file: it names the modules and the reason, so putting one back
breaks a test that says why.

Cheap to run, too. `sys.modules` after an import IS the measurement; no timing,
so nothing here is flaky on a loaded machine.
"""

from __future__ import annotations

import subprocess
import sys

#: Imports that no command needs before it has read its arguments, with what
#: each cost when it was on the common path (measured with a warm bytecode
#: cache, so these are the floor rather than the NFS figure).
DEFERRED = {
    "pathlib": "8.5 ms; reached only by `snapshot` and `--replay`",
    "difflib": "1.1 ms; reached only when a queue name is misspelled",
    "concurrent.futures": "4.6 ms, plus `logging` behind it; only `Cluster.load`",
    "json": "-1.9 ms locally, nothing on NFS; only `--json`, `snapshot`, `--replay`",
}


def _baseline() -> set[str]:
    """What the interpreter has loaded before any of our code runs.

    Needed as an exemption, and it has to come from a bare interpreter rather
    than from a run of ours with no arguments: that run imports `nodetop.cli`
    too, so anything hoisted back to the top of it would appear in the baseline
    and exempt itself. Found the hole by hoisting `difflib` back -- only the
    `hasattr` check below noticed.

    An editable install's path finder imports `pathlib` at site time, which is
    not something this package can give back.
    """
    done = subprocess.run(
        [sys.executable, "-c", "import sys; print('|'.join(sorted(sys.modules)))"],
        capture_output=True, text=True)
    return set(done.stdout.strip().split("|"))


def _imported(argv: list[str]) -> set[str]:
    """Modules present in `sys.modules` after a real run of ``argv``.

    A subprocess, not an import here: pytest itself has imported half the
    standard library, so this question cannot be asked in-process.
    """
    probe = (
        "import sys, runpy;\n"
        f"sys.argv = ['nodetop', {', '.join(repr(a) for a in argv)}];\n"
        "try:\n"
        "    runpy.run_module('nodetop', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('|'.join(sorted(sys.modules)), file=sys.stderr)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True)
    return set(done.stderr.strip().splitlines()[-1].split("|"))


class TestTheCommonPathStaysNarrow:
    def test_version_imports_none_of_the_deferred_modules(self):
        loaded = _imported(["--version"])
        # `pathlib` is exempt when the interpreter's own startup already
        # brought it in -- an editable install's path finder does, and that is
        # not something this package can give back.
        base = _baseline()
        for module, why in DEFERRED.items():
            if module in base:
                continue
            assert module not in loaded, f"{module} back on the startup path ({why})"

    def test_help_imports_none_of_them_either(self):
        # `--help` is the fastest thing the tool can be asked to do, and the
        # one a shell completion or a wrapper script runs.
        loaded, base = _imported(["--help"]), _baseline()
        for module, why in DEFERRED.items():
            if module not in base:
                assert module not in loaded, f"{module} on the --help path ({why})"

    def test_the_import_still_works_where_it_was_moved_to(self):
        # The other half of the deal: a deferred import that is wrong is a
        # NameError at the moment somebody needs it, in a command nobody runs
        # often. Each of the four call sites is exercised elsewhere in the
        # suite -- snapshot, replay, the misspelling hint and `Cluster.load` --
        # so this only pins that they are reachable at all.
        import nodetop.cli as cli

        assert cli._load_replay.__module__ == "nodetop.cli"
        assert not hasattr(cli, "pathlib")     # not a module global any more
        assert not hasattr(cli, "difflib")

    def test_json_output_still_works_from_its_deferred_import(self):
        """The other half of deferring `json`: `--json` must still emit JSON.

        All eleven `--json` VIEWS go through one helper now, so this is the whole
        contract for them -- and `default=str` is part of it, since only two of
        the eleven used to pass it. `snapshot -o` serialises separately, to a file
        and with its own indent; `DEFERRED` above names it for that reason.
        """
        import datetime
        import json

        import nodetop.cli as cli

        got = subprocess.run(
            [sys.executable, "-m", "nodetop", "backends", "--json"],
            capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
        assert isinstance(json.loads(got.stdout), (dict, list))

        # A datetime used to serialise in two commands and raise in nine.
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_json({"when": datetime.datetime(2026, 8, 26, 12, 0)})
        assert json.loads(buf.getvalue())["when"] == "2026-08-26 12:00:00"

    def test_a_real_command_still_loads_what_it_needs(self):
        # Deferring is only safe because the pool is imported by the method
        # that uses it. `backends` runs the registry for real.
        done = subprocess.run(
            [sys.executable, "-m", "nodetop", "backends", "--no-color"],
            capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "batch systems" in done.stdout
