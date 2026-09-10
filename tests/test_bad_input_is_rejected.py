"""Bad input fails, and `exclude` puts nothing unusable on stdout.

Five findings from the v0.5.2 audit, each verified by running the code before
being fixed:

* `exclude` printed `(no matching nodes)` — ANSI-wrapped on a tty — to
  **stdout** when nothing matched, while `--json` answered `{"count": 0,
  "nodelist": ""}`. The command exists to be substituted into
  `sbatch --exclude=$(...)`, so that sentence became a node name.
* `--needs bf16typo` matched EVERY node: `supports` answers `None` for an
  unknown capability and `capability_gap` records only an explicit `False`.
  Measured on an A100 spec, `hardware_ok(node, JobShape(requires=("bf16typo",)))`
  returned `(True, ())`.
* `--time garbage` (and `1w`, `1h30`, `1.5h`) silently meant "unlimited",
  because `parse_duration` answers `None` for both the sentinels and anything
  unreadable, and every ceiling check skips a `None`. `--mem` has always
  rejected bad input; `--time` had no `type=` at all.
* PBS `_mem_to_mb("10GiB")` was `0` — "not read" — because the regex had no
  slot for the `i` of the IEC spelling.
* `expand("n[10-1]")` returned `n01 … n10`, padding to the width of the
  endpoint as *written before* the swap. `n01` is a different node from `n1`,
  which is the whole reason `n[1-10]` was fixed to answer `n1 … n10`.
"""

from __future__ import annotations

import pytest

from nodetop.backends.pbs import _mem_to_mb
from nodetop.cli import build_parser, cmd_exclude
from nodetop.core.cluster import Cluster
from nodetop.core.duration import parse_duration, understood
from nodetop.core.hardware import CAPABILITIES, supports
from nodetop.core.model import Node
from nodetop.hostlist import expand
from nodetop.render import Style

PLAIN = Style(enabled=False)


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _plain_cluster() -> Cluster:
    """One healthy node with no accelerator, so `--gpu-nodes` picks nothing."""
    node = Node(name="n1", state_raw="idle", cpus_total=8, cpus_alloc=0,
                memory_mb=1000, memory_alloc_mb=0)
    return Cluster(backend_name="slurm", queue_term="partition",
                   nodes=[node], queues={}, errors={})


class TestExcludeKeepsStdoutSubstitutable:
    def test_nothing_matching_writes_nothing_to_stdout(self, capsys) -> None:
        assert cmd_exclude(_plain_cluster(), _args(["exclude", "--gpu-nodes"]), PLAIN) == 0
        captured = capsys.readouterr()
        assert captured.out == "", captured.out
        # The human still gets told -- on the stream that is not being
        # substituted into a scheduler argument.
        assert "no matching nodes" in captured.err, captured.err

    def test_the_json_surface_is_unchanged(self, capsys) -> None:
        # The control: `--json` already answered the empty case honestly, and
        # this fix is about making text agree with it, not about changing it.
        import json

        assert cmd_exclude(
            _plain_cluster(), _args(["--json", "exclude", "--gpu-nodes"]), PLAIN
        ) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 0
        assert data["nodelist"] == ""


class TestAWalltimeTypoIsAnError:
    @pytest.mark.parametrize("bad", ["garbage", "1w", "1h30", "1.5h", "-5", "tomorrow"])
    def test_unreadable_walltime_exits_two(self, bad: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _args(["where", "-c", "1", "-t", bad])
        assert exit_info.value.code == 2

    @pytest.mark.parametrize("good", ["4:00:00", "2-00:00:00", "90m", "36h", "120"])
    def test_a_real_walltime_is_still_accepted(self, good: str) -> None:
        assert _args(["where", "-c", "1", "-t", good]).time == good

    @pytest.mark.parametrize("sentinel", ["unlimited", "infinite", "none", "n/a", "0"])
    def test_asking_for_no_limit_is_still_allowed(self, sentinel: str) -> None:
        # "No limit" is a real thing to ask for, and it is what the sentinels
        # mean. Rejecting these would have been the opposite over-correction.
        assert _args(["where", "-c", "1", "-t", sentinel]).time == sentinel

    def test_understood_separates_the_two_meanings_of_none(self) -> None:
        # The root cause: `parse_duration` cannot tell them apart, and this is
        # the predicate that can.
        assert parse_duration("garbage") is None
        assert parse_duration("unlimited") is None
        assert understood("unlimited") is True
        assert understood("garbage") is False


class TestAnUnknownCapabilityIsAnError:
    @pytest.mark.parametrize("bad", ["bf16typo", "cuda2", "fp16", "bfloat"])
    def test_unknown_capability_exits_two(self, bad: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _args(["where", "-c", "1", "--needs", bad])
        assert exit_info.value.code == 2

    @pytest.mark.parametrize("good", ["bf16", "fp8", "tf32", "flash", "cuda", "rocm"])
    def test_a_known_capability_is_still_accepted(self, good: str) -> None:
        assert _args(["where", "-c", "1", "--needs", good]).needs == good

    def test_a_list_is_checked_token_by_token(self) -> None:
        assert _args(["where", "-c", "1", "--needs", "bf16,flash"]).needs == "bf16,flash"
        with pytest.raises(SystemExit):
            _args(["where", "-c", "1", "--needs", "bf16,nonsense"])

    def test_empty_and_blank_still_mean_no_filter(self) -> None:
        # `--needs ""` and stray commas are "no requirement", not a typo.
        assert _args(["where", "-c", "1", "--needs", ""]).needs == ""
        assert _args(["where", "-c", "1", "--needs", " , "]).needs == " , "

    def test_the_advertised_set_is_what_supports_understands(self) -> None:
        # The control that keeps the validator honest: every name it accepts
        # must actually be answerable, and `supports` must know no others.
        from nodetop.core.hardware import ACCELERATORS

        spec = ACCELERATORS["A100"]
        for name in CAPABILITIES:
            assert supports(spec, name) is not None, name
        assert supports(spec, "bf16typo") is None


class TestParsingGapsThatReadAsUnknown:
    @pytest.mark.parametrize(
        ("text", "want"),
        [("10GiB", 10240), ("10Gib", 10240), ("10gb", 10240), ("10g", 10240),
         ("1.5gb", 1536), ("10 gb", 10240), ("1pb", 1024 * 1024 * 1024)],
    )
    def test_pbs_memory_suffixes(self, text: str, want: int) -> None:
        assert _mem_to_mb(text) == want

    def test_pbs_still_reads_words_and_rejects_nonsense(self) -> None:
        # Controls: the `w` suffix still scales (and still truncates below a
        # megabyte, which `_pbs_mem_mb` turns back into "unread"), and a
        # genuinely unreadable size still answers 0.
        assert _mem_to_mb("1024w") == 0
        assert _mem_to_mb("banana") == 0
        assert _mem_to_mb("") == 0

    def test_a_reversed_range_names_the_same_nodes_as_the_forward_one(self) -> None:
        assert expand("n[10-1]") == expand("n[1-10]")
        assert expand("n[10-1]")[0] == "n1"

    def test_a_padded_range_is_still_padded(self) -> None:
        # The control: the width rule itself is unchanged -- it is taken from
        # the low endpoint as written, which is what keeps `n[0001-0003]` at
        # four digits and `n[1-10]` at one.
        assert expand("n[0001-0003]") == ["n0001", "n0002", "n0003"]
        assert expand("n[0003-0001]") == ["n0001", "n0002", "n0003"]
        assert expand("n[1-10]")[0] == "n1"
