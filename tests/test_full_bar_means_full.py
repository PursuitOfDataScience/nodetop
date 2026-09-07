"""A meter drawn completely full must mean the whole of what it measures.

Every caller of :func:`nodetop.render.bar` prints an EXACT ratio beside it --
``5115/5120`` under ``cores free``, ``231/232`` in the feature table,
``88/176 gpu`` from :func:`nodetop.render.gauge`. There is no rounded label for
the bar to agree with, so the only reading a solid bar has is
numerator == denominator. Both draw paths rounded there and both claimed it
early: ASCII went solid at 96.6% of 8 cells, Unicode at 99.4%, and a partition
with 5115 of 5120 cores free rendered as every core free.

Same invariant the two siblings settled, keyed the way each one's own label
allows: ``slurmwatch.tui`` labels with no decimals and reserves through
99.5-99.99, ``slurmpast.render`` labels with one decimal and reserves through
99.95-99.99. Here the label is an integer ratio, so the reserve runs all the
way to 1.0.
"""

import pytest

from nodetop.render import Glyphs, Style, bar, gauge, width

PLAIN = Style(depth=0, glyphs=Glyphs())
ASCII = Style(depth=0, glyphs=Glyphs.ascii())

# The boundary, swept either side of both rounding thresholds: ASCII turns over
# at 1 - 1/(2 * size), Unicode at 1 - 1/(16 * size).
BOUNDARY = [90.0, 96.6, 98.0, 99.0, 99.4, 99.6, 99.9, 99.94, 99.96, 100.0]
SIZES = [8, 18]


def solid(drawn, style):
    """Is every cell of `drawn` the FULL block -- no partial tip, no trough?"""
    return set(drawn) == {style.g.blocks[-1]}


class TestTheBoundary:
    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("percent", BOUNDARY)
    def test_a_solid_meter_means_every_last_unit(self, size, percent):
        for style in (PLAIN, ASCII):
            # A fresh style per value: the memo is keyed on the rounded eighths,
            # which cannot tell 99.9% from 100% at eight cells.
            st = Style(depth=0, glyphs=style.g)
            drawn = bar(percent / 100.0, size, st)
            assert solid(drawn, st) is (percent >= 100.0), (percent, size, drawn)

    @pytest.mark.parametrize("size", SIZES)
    def test_the_ratios_the_tables_actually_print(self, size):
        # `cores free` on this cluster's biggest partition, and the feature
        # table's "231 of 232 identified". Neither is the whole of anything.
        for free, total in ((5115, 5120), (231, 232), (39, 40)):
            for glyphs in (Glyphs(), Glyphs.ascii()):
                st = Style(depth=0, glyphs=glyphs)
                drawn = bar(free / total, size, st)
                assert not solid(drawn, st), (free, total, size, drawn)
                st = Style(depth=0, glyphs=glyphs)
                assert solid(bar(total / total, size, st), st)

    def test_the_gauge_carries_the_reserve_to_its_own_numbers(self):
        # `gauge` draws the accelerator inventory at nine cells, where Unicode
        # turned solid from 99.3% up: 175 of 176 free read as all of them.
        assert gauge(175, 176, 9, Style(depth=0)) == "████████▉ 175/176"
        assert gauge(176, 176, 9, Style(depth=0)) == "█████████ 176/176"


class TestTheMemoDoesNotLendOutAFullBar:
    """The memo key has to carry the predicate the body now branches on.

    ``int(round(fraction * size * 8))`` is 64 for both 0.999 and 1.0 at eight
    cells, and ``int(round(fraction * size))`` is 8 for both, so before the
    reserve those two values genuinely were one picture and one key. They are
    two pictures now, and a key that cannot separate them hands the solid bar
    drawn for a 5120/5120 row to the 5115/5120 row underneath it.
    """

    @pytest.mark.parametrize("size", SIZES)
    def test_the_whole_and_a_hair_under_it_are_two_pictures(self, size):
        for glyphs in (Glyphs(), Glyphs.ascii()):
            warm = Style(depth=0, glyphs=glyphs)
            first = bar(1.0, size, warm)          # fills the memo
            second = bar(0.9999, size, warm)      # must not be handed `first`
            assert first != second, (size, glyphs.unicode, first)
            assert solid(first, warm) and not solid(second, warm)

    @pytest.mark.parametrize("size", SIZES)
    def test_a_warm_memo_still_answers_what_a_cold_one_would(self, size):
        for glyphs in (Glyphs(), Glyphs.ascii()):
            warm = Style(depth=0, glyphs=glyphs)
            for percent in BOUNDARY:
                f = percent / 100.0
                cold = Style(depth=0, glyphs=glyphs)
                assert bar(f, size, warm) == bar(f, size, cold), (percent, size)


class TestControls:
    """These hold in both states -- before the reserve and after it."""

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("percent", BOUNDARY)
    def test_control_the_meter_is_still_exactly_size_cells_wide(self, size, percent):
        # The reserve moves a cell from fill to trough; it must never move one
        # out of the column, which is what breaks every table the bar sits in.
        for glyphs in (Glyphs(), Glyphs.ascii()):
            st = Style(depth=0, glyphs=glyphs)
            assert width(bar(percent / 100.0, size, st)) == size

    def test_control_the_whole_is_still_drawn_solid(self):
        # The reserve withholds the last unit BELOW the boundary only. At the
        # boundary a solid bar is the correct and only reading.
        for glyphs in (Glyphs(), Glyphs.ascii()):
            for size in SIZES:
                st = Style(depth=0, glyphs=glyphs)
                assert solid(bar(1.0, size, st), st)
                st = Style(depth=0, glyphs=glyphs)
                assert solid(bar(4.0, size, st), st), "clamped from above"

    def test_control_the_low_end_still_resolves_one_eighth(self):
        # The other half of the honesty rule, and the one this change must not
        # touch: 1/128 of a 16-cell bar is a visible tip, and a true zero is an
        # empty track. (`test_docstrings` pins the same claim.)
        st = Style(depth=0)
        assert bar(1 / 128, 16, st) == "▏░░░░░░░░░░░░░░░"
        assert bar(0.0, 16, st) == "░" * 16

    @pytest.mark.parametrize("fraction,plain,ascii_", [
        (0.0, "░░░░░░░░", "........"),
        (0.25, "██░░░░░░", "##......"),
        (0.5, "████░░░░", "####...."),
        (0.75, "██████░░", "######.."),
        (0.875, "███████░", "#######."),
        (0.9, "███████▎", "#######."),
    ])
    def test_control_nothing_below_the_boundary_moved(self, fraction, plain, ascii_):
        # The reserve can only bite where the uncapped fill would have reached
        # `size` whole cells -- above 98.4% at eight cells. Everything a reader
        # normally looks at draws exactly what it drew before.
        assert bar(fraction, 8, Style(depth=0, glyphs=Glyphs())) == plain
        assert bar(fraction, 8, Style(depth=0, glyphs=Glyphs.ascii())) == ascii_

    def test_control_a_zero_size_meter_is_still_the_empty_string(self):
        # `size - 1` is -1 there, and a reserve that forgot to floor at zero
        # would print a stray trough cell into a column of no width.
        for glyphs in (Glyphs(), Glyphs.ascii()):
            for fraction in (0.0, 0.5, 0.99, 1.0):
                st = Style(depth=0, glyphs=glyphs)
                assert bar(fraction, 0, st) == ""
