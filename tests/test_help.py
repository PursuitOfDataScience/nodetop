"""The --help text, checked against what the flags actually do.

Every help string is a claim, and an untested claim drifts. One of these had
gone wrong: `check --gpu-mem` said the value was "checked against the hardware
model" while the command's own output said it "took no part", and a request for
999 GiB was accepted. A user reading the help would have believed the
requirement was validated.
"""

from __future__ import annotations

import json

import pytest

from nodetop.cli import (
    build_parser,
    cmd_exclude,
    cmd_nodes,
    cmd_queues,
    cmd_status,
    cmd_where,
)
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())
ASCII = Style(depth=0, glyphs=Glyphs.ascii())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _all_parsers():
    root = build_parser()
    yield "(global)", root
    for sub in root._subparsers._group_actions:  # type: ignore[union-attr]
        seen = set()
        for name, choice in sub.choices.items():  # type: ignore[attr-defined]
            if id(choice) in seen:
                continue
            seen.add(id(choice))
            yield name, choice


def _options():
    for command, parser in _all_parsers():
        for action in parser._actions:
            if action.option_strings:
                yield command, action


class TestNoInvisibleFlags:
    """A flag with no description cannot be discovered."""

    def test_every_option_has_a_help_string(self):
        missing = [
            f"{command} {'/'.join(a.option_strings)}"
            for command, a in _options()
            if not a.help
        ]
        assert not missing, f"undocumented flags: {missing}"

    def test_every_value_taking_option_names_its_metavar(self):
        vague = [
            f"{command} {'/'.join(a.option_strings)}"
            for command, a in _options()
            if a.nargs != 0 and a.metavar is None and a.dest not in {"help", "version"}
            and not isinstance(a.const, bool)
            and a.__class__.__name__ == "_StoreAction"
        ]
        # A bare NODES/GIB/NAME tells the reader what shape the value takes.
        assert not vague, f"options without a metavar: {vague}"


class TestFilterFlagsFilter:
    """`--gpu` says "accelerator nodes only". It has to be only those."""

    def _rows(self, cluster, capsys, argv):
        cmd_nodes(cluster, _args(argv), PLAIN)
        return json.loads(capsys.readouterr().out)

    def test_gpu_returns_only_accelerator_nodes(self, cluster, capsys):
        rows = self._rows(cluster, capsys, ["--json", "nodes", "--gpu"])
        assert rows
        assert all(r["accelerators"][1] > 0 for r in rows)

    def test_cpu_returns_only_nodes_without_accelerators(self, cluster, capsys):
        # --all: the entitlement filter is on by default and this asserts the
        # --cpu flag's behaviour, not the caller's access.
        rows = self._rows(cluster, capsys, ["--json", "nodes", "--cpu", "--all"])
        assert rows
        assert all(r["accelerators"][1] == 0 for r in rows)

    def test_free_returns_only_nodes_with_something_free(self, cluster, capsys):
        rows = self._rows(cluster, capsys, ["--json", "nodes", "--free"])
        for r in rows:
            assert r["schedulable"] is True
            assert r["cpus"][0] > 0 or r["accelerators"][0] > 0

    def test_the_three_filters_partition_the_nodes(self, cluster, capsys):
        every = {r["name"] for r in self._rows(cluster, capsys, ["--json", "nodes"])}
        gpu = {r["name"] for r in self._rows(cluster, capsys, ["--json", "nodes", "--gpu"])}
        cpu = {r["name"] for r in self._rows(cluster, capsys, ["--json", "nodes", "--cpu"])}
        assert gpu | cpu == every
        assert not (gpu & cpu)


class TestExcludeSelectors:
    """Each selector claims a specific set; it must return exactly that set."""

    def _nodes(self, cluster, capsys, selector):
        cmd_exclude(cluster, _args(["--json", "exclude", selector]), PLAIN)
        return set(json.loads(capsys.readouterr().out)["nodes"])

    def test_gpu_nodes_is_every_accelerator_node(self, cluster, capsys):
        got = self._nodes(cluster, capsys, "--gpu-nodes")
        assert got == {n.name for n in cluster.nodes if n.is_gpu_node}

    def test_unschedulable_is_every_down_or_drained_node(self, cluster, capsys):
        got = self._nodes(cluster, capsys, "--unschedulable")
        assert got == {n.name for n in cluster.nodes if not n.schedulable}

    def test_degraded_is_every_impaired_but_schedulable_node(self, cluster, capsys):
        got = self._nodes(cluster, capsys, "--degraded")
        assert got == {n.name for n in cluster.nodes if n.degraded}

    def test_selectors_combine_as_a_union(self, cluster, capsys):
        cmd_exclude(cluster,
                    _args(["--json", "exclude", "--gpu-nodes", "--unschedulable"]),
                    PLAIN)
        both = set(json.loads(capsys.readouterr().out)["nodes"])
        gpu = self._nodes(cluster, capsys, "--gpu-nodes")
        down = self._nodes(cluster, capsys, "--unschedulable")
        assert both == gpu | down


class TestAllFlags:
    def test_status_all_includes_queues_with_no_free_capacity(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), PLAIN)
        narrow = capsys.readouterr().out
        cmd_status(cluster, _args(["status", "--all"]), PLAIN)
        wide = capsys.readouterr().out
        assert len(wide) >= len(narrow)

    def test_where_all_includes_ruled_out_queues(self, cluster, capsys):
        cmd_where(cluster, _args(["--json", "where", "-g", "1"]), PLAIN)
        narrow = {r["queue"] for r in json.loads(capsys.readouterr().out)}
        cmd_where(cluster, _args(["--json", "where", "-g", "1", "--all"]), PLAIN)
        wide = {r["queue"] for r in json.loads(capsys.readouterr().out)}
        assert narrow < wide

    def test_queues_detail_is_implied_by_naming_a_queue(self, cluster, capsys):
        # The help says so explicitly.
        cmd_queues(cluster, _args(["queues", "-q", "test"]), PLAIN)
        named = capsys.readouterr().out
        cmd_queues(cluster, _args(["queues", "--detail", "-q", "test"]), PLAIN)
        assert named == capsys.readouterr().out


class TestPresentationFlags:
    def test_ascii_emits_no_non_ascii_bytes(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), ASCII)
        capsys.readouterr().out.encode("ascii")

    def test_no_color_emits_no_escape_sequences(self, cluster, capsys):
        cmd_status(cluster, _args(["status"]), PLAIN)
        assert "\033" not in capsys.readouterr().out


class TestWalltimeHelpIsAccurate:
    """The help spells out four forms and one convention."""

    @pytest.mark.parametrize("text,seconds", [
        ("4:00:00", 4 * 3600),
        ("2-00:00:00", 2 * 86400),
        ("90m", 5400),
        ("36h", 36 * 3600),
        ("60", 3600),          # "a bare number is minutes"
    ])
    def test_every_documented_form_parses_as_documented(self, text, seconds):
        from nodetop.cli import _shape_from_args

        shape = _shape_from_args(_args(["where", "-t", text]))
        assert shape.walltime_seconds == seconds

    def test_the_help_still_documents_the_bare_number_convention(self):
        parser = build_parser()
        for sub in parser._subparsers._group_actions:  # type: ignore[union-attr]
            where = sub.choices["where"]  # type: ignore[attr-defined]
            action = next(a for a in where._actions if "--time" in a.option_strings)
            assert "minutes" in action.help


class TestCheckHelpMatchesCheckBehaviour:
    """The claim that had actually gone wrong."""

    def _check_action(self, name):
        parser = build_parser()
        for sub in parser._subparsers._group_actions:  # type: ignore[union-attr]
            check = sub.choices["check"]  # type: ignore[attr-defined]
            return next(a for a in check._actions if name in a.option_strings)
        raise AssertionError("check subparser not found")

    def _where_action(self, name):
        parser = build_parser()
        for sub in parser._subparsers._group_actions:  # type: ignore[union-attr]
            where = sub.choices["where"]  # type: ignore[attr-defined]
            return next(a for a in where._actions if name in a.option_strings)
        raise AssertionError("where subparser not found")

    @pytest.mark.parametrize("flag", ["--gpu-mem", "--needs"])
    def test_check_says_it_does_not_evaluate_them(self, flag):
        help_text = self._check_action(flag).help
        assert "NOT CHECKED" in help_text

    @pytest.mark.parametrize("flag", ["--gpu-mem", "--needs"])
    def test_check_does_not_claim_a_hardware_check(self, flag):
        # This is the exact wording that was wrong: check evaluates neither.
        assert "checked against the hardware model" not in self._check_action(flag).help

    def test_where_still_claims_the_check_it_does_perform(self):
        assert "checked against the hardware model" in self._where_action("--gpu-mem").help

    @pytest.mark.parametrize("flag", ["--gpu-mem", "--needs"])
    def test_the_two_commands_describe_them_differently(self, flag):
        assert self._check_action(flag).help != self._where_action(flag).help


class TestMemorySizesAcceptTheSchedulersOwnSpelling:
    """`--mem 64G` is what a Slurm user types, because `sbatch` takes it.

    The argument was a bare float in GiB, so the natural thing to type was an
    argparse error -- a tool whose premise is scheduler fluency refusing the
    scheduler's own notation. `Gi` is accepted too, because this tool speaks
    Kubernetes and that is how Kubernetes writes it.
    """

    @pytest.mark.parametrize("text,expected", [
        ("64", 64.0),          # bare numbers still mean GiB: nothing changes
        ("64G", 64.0),
        ("64GB", 64.0),
        ("64Gi", 64.0),
        ("64GiB", 64.0),
        ("40g", 40.0),         # case-insensitive
        ("65536M", 64.0),
        ("65536Mi", 64.0),
        ("2T", 2048.0),
        ("2Ti", 2048.0),
        ("0.5T", 512.0),
        (" 64G ", 64.0),       # whitespace from a shell
    ])
    def test_it_parses(self, text, expected):
        from nodetop.cli import memory_gb

        assert memory_gb(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "bogus", "64X", "G", "--", "1.2.3",
                                      "64 GB extra", "-8G"])
    def test_garbage_is_refused_with_a_useful_message(self, text):
        import argparse

        from nodetop.cli import memory_gb

        with pytest.raises(argparse.ArgumentTypeError) as exc:
            memory_gb(text)
        # The message has to say what good input looks like; "invalid float
        # value" did not.
        assert "64G" in str(exc.value)

    def test_the_suffixes_are_binary_like_sbatch(self):
        # There is no second convention to guess between: sbatch's K/M/G/T are
        # binary multiples, so these are too.
        from nodetop.cli import memory_gb

        assert memory_gb("1024M") == pytest.approx(1.0)
        assert memory_gb("1T") == pytest.approx(1024.0)

    @pytest.mark.parametrize("flag", ["--mem", "--gpu-mem"])
    def test_both_memory_flags_accept_it(self, flag):
        args = build_parser().parse_args(["where", flag, "64G"])
        assert getattr(args, flag.lstrip("-").replace("-", "_")) == 64.0

    def test_a_bad_size_is_a_usage_error_not_a_traceback(self, capsys):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["where", "--mem", "banana"])
        assert exc.value.code == 2
        assert "64G" in capsys.readouterr().err
