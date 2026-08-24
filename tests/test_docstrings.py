"""Docstring examples, executed.

A docstring that has quietly become false is worse than no docstring: it is a
confident wrong answer at the exact moment someone is trying to understand the
code. `gauge` had drifted this way — its example showed eight filled cells and
a shaded trough for a value that actually renders as one partial cell over
dotted background.

Every worked example below is copied from a docstring in `src/`. If one of them
fails, the docstring is lying.
"""

from __future__ import annotations

import pytest


class TestHostlistExamples:
    def test_split_groups_example(self):
        # split_groups: '"a-[1-2,4],b-[1-3]"' -> '["a-[1-2,4]", "b-[1-3]"]'
        from nodetop.hostlist import split_groups

        assert split_groups("a-[1-2,4],b-[1-3]") == ["a-[1-2,4]", "b-[1-3]"]

    def test_the_module_example_really_is_two_groups(self):
        # module: 'cn-[0001-0010,0012-0015],gn-bigmem[1-4]' is two
        # groups, not five.
        from nodetop.hostlist import split_groups

        text = "cn-[0001-0010,0012-0015],gn-bigmem[1-4]"
        assert len(split_groups(text)) == 2

    def test_range_body_example(self):
        # _expand_range_body: '"1-3,7"' -> 1,2,3,7
        from nodetop.hostlist import _expand_range_body

        assert _expand_range_body("1-3,7") == ["1", "2", "3", "7"]

    def test_zero_padding_example(self):
        # _expand_range_body: '0001-0003' yields '0001 0002 0003', not '1 2 3'
        from nodetop.hostlist import _expand_range_body

        assert _expand_range_body("0001-0003") == ["0001", "0002", "0003"]

    def test_collapse_width_example(self):
        # collapse: 'node-0001' and 'node-1' stay in separate groups
        from nodetop.hostlist import collapse, expand

        out = collapse(["node-0001", "node-1"])
        assert set(expand(out)) == {"node-0001", "node-1"}

    def test_multi_dimensional_example(self):
        # _expand_group: 'rack[1-2]node[1-4]' is eight nodes
        from nodetop.hostlist import expand

        assert len(expand("rack[1-2]node[1-4]")) == 8


class TestDurationTable:
    """The table in `parse_duration`, row by row."""

    @pytest.mark.parametrize("text,seconds,note", [
        (3600, 3600, "<int> seconds -- already seconds"),
        ("2-00:00:00", 2 * 86400, "D-HH[:MM[:SS]] -- days first"),
        ("2:00:00", 2 * 3600, "HH:MM:SS -- three fields"),
        ("2:00", 120, "MM:SS -- two fields, '2:00' = 2 min"),
        ("60", 3600, "MM -- bare number is minutes, '60' = 1 hour"),
    ])
    def test_row(self, text, seconds, note):
        from nodetop.core.duration import parse_duration

        assert parse_duration(text) == seconds, note

    def test_the_sixty_times_error_the_module_warns_about(self):
        # module: "Reading '2:00' as two hours is a 60x error."
        from nodetop.core.duration import parse_duration

        assert parse_duration("2:00") * 60 == parse_duration("2:00:00")


class TestSlurmDurationExamples:
    def test_the_two_field_ambiguity(self):
        # parse_slurm_duration: '2:00' is MM:SS (two minutes) but '1-2:00' is
        # D-HH:MM.
        from nodetop.backends.slurm import parse_slurm_duration

        assert parse_slurm_duration("2:00") == 120
        assert parse_slurm_duration("1-2:00") == 86400 + 2 * 3600

    def test_the_tres_example(self):
        # _parse_tres_map: 'cpu=256,gres/gpu=32,node=8,mem=100G' -> neutral keys
        from nodetop.backends.slurm import _parse_tres_map

        got = _parse_tres_map("cpu=256,gres/gpu=32,node=8,mem=100G")
        assert got == {"cpu": 256, "gpu": 32, "node": 8, "mem_mb": 100 * 1024}


class TestBackendExamples:
    def test_sge_complex_values_example(self):
        # parse_complex_values: 'complex_values  gpu=4,slots=32'
        from nodetop.backends.sge import SgeBackend
        from nodetop.runner import RecordedRunner

        backend = SgeBackend(RecordedRunner({}))
        got = backend.parse_complex_values("complex_values  gpu=4,slots=32\n")
        assert got == {"gpu": 4, "slots": 32}

    def test_pbs_exec_host_and_exec_vnode_shapes(self):
        # parse_free_times documents both:
        #   exec_host  = gpu001/0*64+gpu002/0*64      (slash-separated)
        #   exec_vnode = (gpu001:ncpus=64)+(...)      (parenthesised)
        # Only exec_host is guaranteed present, and it is the one parsed.
        from nodetop.backends.pbs import PbsBackend
        from nodetop.runner import RecordedRunner

        backend = PbsBackend(RecordedRunner({}))
        text = (
            "Job Id: 1\n"
            "    Resource_List.walltime = 01:00:00\n"
            "    stime = Thu Aug 21 10:00:00 2026\n"
            "    exec_host = gpu001/0*64+gpu002/0*64\n"
            "    exec_vnode = (gpu001:ncpus=64)+(gpu002:ncpus=64)\n"
        )
        assert set(backend.parse_free_times(text)) == {"gpu001", "gpu002"}

    def test_lsf_param_table_example(self):
        # _param_status documents the positional PARAM table it reads.
        from nodetop.backends.lsf import _param_status

        body = (
            "PARAM: PRIO NICE STATUS          MAX JL/U\n"
            "       50    20  Open:Active       -    8\n"
        )
        assert _param_status(body) == "Open:Active"


class TestRenderExamples:
    def test_the_bar_resolution_claim(self):
        # bar: "Eighth-blocks let a 16-cell bar resolve ~1/128, so a
        # nearly-empty queue still shows *something*."
        from nodetop.render import Style, bar

        st = Style(depth=0)
        assert bar(1 / 128, 16, st)[0] != st.g.empty
        assert bar(0.0, 16, st)[0] == st.g.empty

    def test_the_gauge_example(self):
        # gauge's docstring shows these two lines verbatim.
        from nodetop.render import Style, gauge

        st = Style(depth=0)
        assert gauge(88, 176, 14, st, "gpu") == "███████░░░░░░░ 88/176 gpu"
        assert gauge(12, 176, 14, st, "gpu") == "█░░░░░░░░░░░░░ 12/176 gpu"

    def test_the_trough_is_a_shaded_block(self):
        # It was a dot leader, on the argument that dots survive a font without
        # the block glyphs. They do not help: the trough and the fill are in the
        # same Unicode block, so a font missing one is missing both, and
        # Glyphs.ascii() is the real fallback. The shaded trough makes the
        # filled/empty boundary unmistakable.
        from nodetop.render import Glyphs

        assert Glyphs().empty == "\u2591"
        assert Glyphs.ascii().empty == "."

    @pytest.mark.parametrize("depth,label", [
        (0, "no colour"), (4, "16-colour"), (8, "256-colour"), (24, "truecolor"),
    ])
    def test_the_depth_ladder(self, depth, label):
        # _depth: "0 = no colour, 4 = 16-colour, 8 = 256-colour, 24 = truecolor"
        from nodetop.render import Style

        st = Style(depth=depth)
        assert st.enabled is (depth > 0), label

    def test_the_width_examples(self):
        # width(): the three cases its docstring names.
        from nodetop.render import width

        assert width("\033[31mabc\033[0m") == 3
        assert width("日本語") == 6
        assert width("é") == 1


class TestHardwareExamples:
    def test_the_sm_example(self):
        # sm: "(``sm_80`` -> 80), else None"
        from nodetop.core.hardware import ACCELERATORS

        assert ACCELERATORS["A100"].sm == 80
        assert ACCELERATORS["MI300X"].sm is None

    def test_the_memory_ambiguity_example(self):
        # module: "``A100`` alone does not say 40 GB or 80 GB"
        from nodetop.core.hardware import ACCELERATORS

        spec = ACCELERATORS["A100"]
        assert set(spec.memory_variants) == {40, 80}
        assert spec.memory_certain is False


class TestVocabularyTable:
    """The mapping table at the top of `core/model.py`."""

    @pytest.mark.parametrize("backend,term", [
        ("slurm", "partition"),
        ("pbs", "queue"),
        ("lsf", "queue"),
        ("sge", "queue"),
        ("kubernetes", "namespace"),
    ])
    def test_each_row_matches_the_backend(self, backend, term):
        from nodetop import backends

        assert backends.get(backend).queue_term == term

    def test_the_table_names_every_backend_that_exists(self):
        # A backend absent from the table is one a reader will not know about.
        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src" / "nodetop" / "core" / "model.py"
        ).read_text()
        header = text[: text.index('"""', 10)].lower()
        spellings = {
            "slurm": "slurm", "pbs": "pbs", "lsf": "lsf", "sge": "sge",
            "kubernetes": "kubernetes", "sshpool": "ssh pool",
        }
        from nodetop import backends

        for name in backends.names():
            assert spellings[name] in header, f"{name} is missing from the table"

    def test_the_table_names_every_dry_run_and_every_absence(self):
        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src" / "nodetop" / "core" / "model.py"
        ).read_text()
        header = text[: text.index('"""', 10)]
        from nodetop import backends

        for name in backends.names():
            caps = backends.get(name).capabilities()
            if caps.probe and caps.probe_command:
                # The command that confirms entitlement should be named.
                token = caps.probe_command.split()[0]
                assert token in header, f"{name}: {token} missing from the table"


class TestPublicSurfaceIsDeclared:
    """``__all__`` must list every name another module imports from here.

    Not cosmetic: a name that other modules use but ``__all__`` omits reads as
    private, so the next person to tidy up deletes or renames it freely.
    ``capability_gap`` was in that position.
    """

    def test_model_exports_everything_it_is_imported_for(self):
        import pathlib
        import re

        import nodetop.core.model as model

        root = pathlib.Path(model.__file__).parent.parent.parent.parent
        wanted: set[str] = set()
        pattern = re.compile(
            r"from (?:\.|\.\.core\.|\.core\.|nodetop\.core\.)model import \(?([^)\n]*)\)?"
        )
        for path in list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py")):
            if path.name == "model.py":
                continue
            for match in pattern.finditer(path.read_text()):
                for name in match.group(1).split(","):
                    name = name.strip()
                    if name and not name.startswith("#"):
                        wanted.add(name)
        missing = sorted(wanted - set(model.__all__))
        assert not missing, f"imported elsewhere but not in __all__: {missing}"

    def test_all_is_sorted(self):
        import nodetop.core.model as model

        assert list(model.__all__) == sorted(model.__all__)

    def test_every_declared_name_exists(self):
        import nodetop.core.model as model

        assert not [n for n in model.__all__ if not hasattr(model, n)]
