"""The `nodes` headline adds up under every filter, not just unfiltered.

`cmd_status`' funnel states the rule this line borrows: "every partition on the
cluster is in exactly one of these terms, so the line answers 'why five rows'
by arithmetic rather than by asking the reader to trust it." The `nodes`
headline has the same shape -- `330 of 608 ... 278 not on your allowlist` --
and only the entitlement filter had a term. The four flag filters (`--queue`,
`--gpu`, `--cpu`, `--free`) removed nodes silently, so measured on one snapshot:

    default        330 of 608 ... 278 not on your allowlist    330 + 278 = 608
    --gpu           58 of 608 ... 278 not on your allowlist     58 + 278 = 336
    --free         177 of 608 ... 278 not on your allowlist    177 + 278 = 455
    --gpu --free    41 of 608 ... 278 not on your allowlist     41 + 278 = 319

Under a filter the line left up to 289 of 608 nodes in no term at all -- or,
read the other way round, invited the reader to blame the allowlist for nodes
their own flags had removed. `status`' funnel is pinned this way already
(`test_unconfirmed_names_published.py` asserts its terms sum to the total);
this is the same pin on the sibling line.

`--json` returns before the headline is built, so the payload is untouched.
"""

from __future__ import annotations

import re

import pytest

from nodetop.cli import build_parser, cmd_nodes
from nodetop.core.cluster import Cluster
from nodetop.core.model import Identity, Node, Queue
from nodetop.render import Style

PLAIN = Style(enabled=False)

_SHOWN = re.compile(r"(\d+) of (\d+)")
_ALL = re.compile(r"all (\d+)")
_HIDDEN = re.compile(r"(\d+) not on your allowlist")
_FILTERED = re.compile(r"(\d+) filtered out")


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _node(name: str, queue: str, *, gpus: int = 0, full: bool = False) -> Node:
    """One node. `full` leaves it schedulable but with nothing free."""
    return Node(
        name=name, state_raw="IDLE", cpus_total=8,
        cpus_alloc=8 if full else 0, memory_mb=64000,
        memory_alloc_mb=64000 if full else 0,
        gpus_total=gpus, gpus_alloc=gpus if full else 0, queues=(queue,),
    )


#: 7 nodes I may use (2 GPU idle, 3 CPU idle, 2 CPU full) and 4 I may not.
#: Every flag filter therefore removes a different, non-zero number of them.
def _cluster() -> Cluster:
    mine = (
        [_node(f"gpu{i}", "open", gpus=4) for i in range(2)]
        + [_node(f"cpu{i}", "open") for i in range(3)]
        + [_node(f"busy{i}", "open", full=True) for i in range(2)]
    )
    theirs = [_node(f"other{i}", "theirs") for i in range(4)]
    queues = {
        "open": Queue(name="open", node_names=tuple(n.name for n in mine),
                      declared_nodes=len(mine), nodes=mine, allow_accounts=()),
        # Not an empty allowlist: it has to name an account that is not ours.
        "theirs": Queue(name="theirs", node_names=tuple(n.name for n in theirs),
                        declared_nodes=len(theirs), nodes=theirs,
                        allow_accounts=("someone-else",)),
    }
    return Cluster(
        backend_name="synthetic", queue_term="partition",
        nodes=mine + theirs, queues=queues,
        identity=Identity(user="me", accounts=("mine",), qos=("q",)),
    )


TOTAL = 11
FILTERS = [[], ["--gpu"], ["--cpu"], ["--free"], ["--gpu", "--free"],
           ["--all", "--gpu"], ["--queue", "open"]]


def _headline(argv: list[str], capsys) -> str:
    cmd_nodes(_cluster(), _args(["nodes", *argv]), PLAIN)
    out = capsys.readouterr().out
    return next(ln for ln in out.splitlines() if "nodes" in ln)


def _terms(line: str) -> tuple[int, int, int, int]:
    """`(shown, total, not_on_allowlist, filtered_out)` off the rendered line."""
    shown_match = _SHOWN.search(line)
    if shown_match:
        shown, total = int(shown_match.group(1)), int(shown_match.group(2))
    else:
        total = int(_ALL.search(line).group(1))
        shown = total
    hidden = _HIDDEN.search(line)
    filtered = _FILTERED.search(line)
    return (shown, total,
            int(hidden.group(1)) if hidden else 0,
            int(filtered.group(1)) if filtered else 0)


class TestTheFunnelAddsUp:
    @pytest.mark.parametrize("argv", FILTERS, ids=lambda a: " ".join(a) or "default")
    def test_the_terms_sum_to_the_total(self, argv, capsys) -> None:
        line = _headline(argv, capsys)
        shown, total, hidden, filtered = _terms(line)
        assert total == TOTAL, line
        assert shown + hidden + filtered == total, line

    @pytest.mark.parametrize(
        "argv,expected",
        [(["--gpu"], 5), (["--cpu"], 2), (["--free"], 2), (["--gpu", "--free"], 5)],
        ids=["gpu", "cpu", "free", "gpu+free"],
    )
    def test_each_filter_reports_what_it_removed(self, argv, expected, capsys) -> None:
        """Not just that it sums -- that the term is the right size."""
        _, _, _, filtered = _terms(_headline(argv, capsys))
        assert filtered == expected

    def test_the_two_halves_of_one_filter_partition_the_allowlist(
        self, capsys
    ) -> None:
        """`--gpu` and `--cpu` are complements, so their filtered-out counts
        must add to the allowlist population -- 5 + 2 = 7."""
        _, _, _, gpu_filtered = _terms(_headline(["--gpu"], capsys))
        _, _, _, cpu_filtered = _terms(_headline(["--cpu"], capsys))
        assert gpu_filtered + cpu_filtered == 7


class TestControls:
    """Each passes with the `filtered out` term removed as well as with it.

    They cover what the line already got right, so a neuter that reddens one of
    them means the change went further than the missing term -- verified by
    running it.
    """

    def test_the_unfiltered_line_already_summed(self, capsys) -> None:
        """The case that was correct: no flag filter, so no term is needed."""
        line = _headline([], capsys)
        shown, total, hidden, _ = _terms(line)
        assert (shown, hidden, total) == (7, 4, TOTAL), line
        assert "filtered out" not in line, line

    def test_all_says_all_and_claims_no_funnel(self, capsys) -> None:
        """`--all` lifts the entitlement filter, so there is nothing to explain."""
        line = _headline(["--all"], capsys)
        assert "all 11" in line, line
        assert "not on your allowlist" not in line, line
        assert "filtered out" not in line, line

    @pytest.mark.parametrize("argv", FILTERS, ids=lambda a: " ".join(a) or "default")
    def test_the_shown_and_total_pair_is_still_printed(self, argv, capsys) -> None:
        line = _headline(argv, capsys)
        assert _SHOWN.search(line) or _ALL.search(line), line

    def test_the_gpu_and_out_counts_are_subsets_not_terms(self, capsys) -> None:
        """They describe the shown rows, so they must never enter the sum --
        which is why `2 with GPUs` sits beside a `shown` of 7, not inside it."""
        line = _headline([], capsys)
        assert "2 with GPUs" in line, line
        assert "0 out" in line, line
        shown, _, _, _ = _terms(line)
        assert shown == 7
