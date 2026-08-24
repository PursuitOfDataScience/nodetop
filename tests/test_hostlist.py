"""Hostlist expansion and collapse."""

from __future__ import annotations

import pytest

from nodetop.hostlist import collapse, expand, split_groups


class TestSplitGroups:
    def test_commas_inside_brackets_do_not_split(self):
        # The whole reason this function exists: a naive split(",") turns two
        # groups into five and silently loses nodes.
        assert split_groups("a-[1-2,4],b-[1-3]") == ["a-[1-2,4]", "b-[1-3]"]

    def test_plain_list(self):
        assert split_groups("n1,n2,n3") == ["n1", "n2", "n3"]

    def test_unbalanced_bracket_degrades_without_raising(self):
        assert split_groups("a-[1-2,b-3") == ["a-[1-2,b-3"]


class TestExpand:
    def test_simple_range(self):
        assert expand("node-[1-3]") == ["node-1", "node-2", "node-3"]

    def test_zero_padding_is_preserved(self):
        # node-[7-8] and node-[0007-0008] are different node names.
        assert expand("node-[0007-0009]") == ["node-0007", "node-0008", "node-0009"]

    def test_mixed_ranges_and_singletons(self):
        assert expand("n-[1-2,5,7-8]") == ["n-1", "n-2", "n-5", "n-7", "n-8"]

    def test_multiple_groups(self):
        assert expand("a-[1-2],b-[1-2]") == ["a-1", "a-2", "b-1", "b-2"]

    def test_no_brackets(self):
        assert expand("single-node") == ["single-node"]

    def test_suffix_after_bracket(self):
        assert expand("gpu[1-2]-ib") == ["gpu1-ib", "gpu2-ib"]

    def test_multi_dimensional_names_take_the_product(self):
        # Multi-dimensional expressions are legal and real: this is eight
        # nodes, not two names with a literal bracket inside them.
        assert expand("rack[1-2]node[1-4]") == [
            "rack1node1", "rack1node2", "rack1node3", "rack1node4",
            "rack2node1", "rack2node2", "rack2node3", "rack2node4",
        ]

    def test_three_sections_with_a_trailing_suffix(self):
        assert expand("x[1-2]y[1-2]z") == ["x1y1z", "x1y2z", "x2y1z", "x2y2z"]

    def test_unbalanced_bracket_is_kept_literal(self):
        # A truncated field is far more likely than a syntax error, so it
        # degrades instead of raising.
        assert expand("n[1-2") == ["n[1-2"]

    @pytest.mark.parametrize("empty", ["", None, "(null)", "None", "n/a"])
    def test_slurms_several_spellings_of_nothing(self, empty):
        assert expand(empty) == []

    def test_reversed_range_is_tolerated(self):
        assert expand("n[3-1]") == ["n1", "n2", "n3"]

    def test_real_610_node_partition(self):
        # Verbatim from `scontrol show partition test`, which reported
        # TotalNodes=610 for exactly this string.
        real = (
            "gn-[0001-0044],gn-bigmem[1-4],climate-[001-048],"
            "cn-[0001-0010,0012-0015,0017-0216,0219-0308,0310-0324,"
            "0329-0427,0429-0442,0444-0456,0501-0562,0600-0606]"
        )
        assert len(expand(real)) == 610


class TestCollapse:
    def test_contiguous_run(self):
        assert collapse(["n-1", "n-2", "n-3"]) == "n-[1-3]"

    def test_single_node_gets_no_brackets(self):
        assert collapse(["n-1"]) == "n-1"

    def test_gaps_become_separate_runs(self):
        assert collapse(["n-1", "n-2", "n-4"]) == "n-[1-2,4]"

    def test_padding_width_is_kept(self):
        assert collapse(["n-0001", "n-0002"]) == "n-[0001-0002]"

    def test_differing_widths_do_not_merge(self):
        # n-1 and n-0001 are different names; merging them would emit a set
        # that does not round-trip.
        out = collapse(["n-1", "n-0001"])
        assert set(expand(out)) == {"n-1", "n-0001"}

    def test_unsorted_input(self):
        assert collapse(["n-3", "n-1", "n-2"]) == "n-[1-3]"

    def test_names_without_numbers(self):
        assert collapse(["login", "n-1", "n-2"]) == "login,n-[1-2]"

    def test_duplicates_are_dropped(self):
        assert collapse(["n-1", "n-1", "n-2"]) == "n-[1-2]"

    def test_real_exclude_list(self):
        names = ["cn-0423", "cn-0298", "cn-0377", "cn-0378",
                 "cn-0603", "cn-0604", "cn-0605", "cn-0606"]
        assert collapse(names) == "cn-[0298,0377-0378,0423,0603-0606]"


class TestRoundTrip:
    @pytest.mark.parametrize("nodelist", [
        "n-[1-5]",
        "n-[0001-0044]",
        "a-[1-2],b-[10-12]",
        "bigmem[1-4]",
        "n-[1,3,5,7]",
        "rack[1-2]node[1-4]",
    ])
    def test_expand_collapse_expand_is_stable(self, nodelist):
        first = expand(nodelist)
        assert expand(collapse(first)) == first


class TestCollapseExpandRoundTrip:
    """``expand(collapse(names)) == names`` is the contract that matters.

    The collapsed form is what goes into ``sbatch --exclude=`` and what the
    health view prints, so a name lost or invented in the round trip changes
    which machines a job avoids. Probed over 300 generated shapes; the cases
    below are the ones that pin the edges.
    """

    @pytest.mark.parametrize("names", [
        [f"gn-{i:04d}" for i in range(1, 45)],
        [f"climate-{i:03d}" for i in
         (1, 2, 3, 7, 10, 11, 13, 15, 17, 22, 24, 25, 27, 28, 29, 30, 33, 37,
          39, 40, 43, 45)],
        ["cn-0385"],
        ["a", "b", "c"],
        ["node1", "node10", "node2"],          # numeric vs lexical order
        ["n-1-1", "n-1-2", "n-2-1"],           # more than one number per name
        ["gpu01", "gpu02", "gpu10"],
        ["host-001", "host-1"],                # same value, different padding
        ["pool", "pool-1"],                    # bare stem beside a numbered one
        [f"r{i}c{j}" for i in range(3) for j in range(3)],
        [],
    ])
    def test_it_round_trips(self, names):
        assert sorted(set(expand(collapse(sorted(names))))) == sorted(set(names))

    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", range(12))
    def test_generated_sets_round_trip(self, seed, width):
        import random

        rng = random.Random(seed * 31 + width)
        prefix = rng.choice(["nd", "gn-", "c", "mx-"])
        names = [f"{prefix}{i:0{width}d}"
                 for i in rng.sample(range(200), rng.randint(1, 25))]
        assert sorted(set(expand(collapse(sorted(names))))) == sorted(set(names))

    def test_a_name_containing_the_delimiters_cannot_round_trip(self):
        # Pinned rather than fixed: '[' and ']' ARE the collection syntax, so a
        # name using them is not expressible in the notation every scheduler
        # reads, and no escaping exists that sbatch/qsub would accept. Real
        # hostnames cannot contain them. Documented here so the limit is known
        # rather than discovered.
        assert expand(collapse(["x[1]", "y"])) != ["x[1]", "y"]
        assert sorted(expand(collapse(["x[1]", "y"]))) == ["x1", "y"]
