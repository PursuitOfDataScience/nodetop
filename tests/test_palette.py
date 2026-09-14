"""The palette, measured rather than admired.

Every assertion here exists because the previous palette failed it, and the
complaint that started the rewrite was not subtle: "the colors are very
confusing and not pleasant at all, several same colors go together." Three
separate defects produced that, and each has a test below.

1. **Hue was carrying an ordered quantity.** The heat ramp ran blue -> cyan ->
   green with its brightest step in the *middle*: L* climbed to 91 at cyan and
   fell to 76 at the green end, six reversals in twelve steps. That is the
   defect that got the jet colormap retired from scientific plotting -- a
   mid-range value reads as the extreme because it is the brightest thing on
   the screen. Lightness is the only channel the eye reads as "more", so the
   ramp is now monotonic in it.

2. **The steps were not evenly spaced.** They ran from dE2000 2.8 to 15.3, a
   5.5x spread, and four of the top five were one green: steps 7, 8, 9 and 10
   sat 2.8, 6.4, 4.8 and 4.7 apart. Two rows an order of magnitude apart came
   out the same colour, which is the literal reading of "several same colors
   go together".

3. **Verdict colours and scale colours shared hues.** `ok` green was dE2000 8
   from the ramp's top step, so an idle node's green dot and its green core
   count were one colour by accident, and a reader could not tell whether a
   colour meant "this is fine" or "this is a large number". The ramp now stops
   at turquoise and green belongs to `ok` alone.

The thresholds are not round numbers picked to pass. dE2000 of about 2 is a
just-noticeable difference between two large patches side by side; small
coloured text in a table, separated by other columns, needs several times that
before two cells read as different, which is where 18 and 25 come from.
"""

from __future__ import annotations

import itertools

from colour import (
    BACKGROUND,
    XTERM256,
    contrast_ratio,
    delta_e,
    lightness,
)

from nodetop.render import (
    _FRAME_ANCHORS,
    _PALETTE,
    _RAMP,
    _WASH,
    RAMP_STEPS,
)

#: Roles that carry a judgement. These are the ones a reader must never
#: mistake for a measurement.
VERDICTS = ("ok", "warn", "bad")

#: Every role that is painted onto text a reader has to read.
READABLE = tuple(name for name in _PALETTE if name != "track")


def _truecolor(tones):
    return [tone[0] for tone in tones]


def _at_depth(tones, depth):
    """The colours this terminal actually draws for the ramp."""
    if depth >= 24:
        return [tone[0] for tone in tones]
    if depth >= 8:
        return [XTERM256[tone[1]] for tone in tones]
    # SGR 30-37 and 90-97, as an xterm draws them.
    basic = {30: (0, 0, 0), 31: (205, 0, 0), 32: (0, 205, 0), 33: (205, 205, 0),
             34: (0, 0, 238), 35: (205, 0, 205), 36: (0, 205, 205),
             37: (229, 229, 229), 90: (127, 127, 127), 91: (255, 0, 0),
             92: (0, 255, 0), 93: (255, 255, 0), 94: (92, 92, 255),
             95: (255, 0, 255), 96: (0, 255, 255), 97: (255, 255, 255)}
    return [basic[tone[2]] for tone in tones]


class TestTheRampIsOrderedByLightness:
    """A scale has to look like a scale, and only one channel does that.

    Hue is a categorical channel: the eye tells red from green instantly and
    cannot tell you which is *larger*. Lightness is the ordered one. A ramp
    whose lightness goes up and then down is a ramp that claims its middle is
    its extreme -- which is exactly what the old one claimed, and why a
    half-empty partition drew the loudest row on the screen.
    """

    def test_every_step_is_lighter_than_the_one_below_it(self):
        levels = [lightness(rgb) for rgb in _truecolor(_RAMP)]
        reversals = [i for i in range(1, len(levels)) if levels[i] <= levels[i - 1]]
        assert not reversals, (
            f"lightness falls back at steps {reversals}: "
            f"{[round(x, 1) for x in levels]}"
        )

    def test_the_fill_tones_are_ordered_too(self):
        # The wash is what a filled meter is drawn in, so a reversal there is
        # a bar that gets darker as it gets fuller.
        levels = [lightness(rgb) for rgb in _truecolor(_WASH)]
        assert levels == sorted(levels), [round(x, 1) for x in levels]

    def test_the_ordering_survives_a_256_colour_terminal(self):
        # Nearest-match to the xterm cube reintroduces reversals, because the
        # cube is sparse through blue-cyan -- the 256 column is chosen to be
        # monotonic, not merely close.
        for name, tones in (("ramp", _RAMP), ("fill", _WASH)):
            levels = [lightness(rgb) for rgb in _at_depth(tones, 8)]
            assert levels == sorted(levels), f"{name}: {[round(x, 1) for x in levels]}"

    def test_the_ordering_survives_a_16_colour_terminal(self):
        for name, tones in (("ramp", _RAMP), ("fill", _WASH)):
            levels = [lightness(rgb) for rgb in _at_depth(tones, 4)]
            assert levels == sorted(levels), f"{name}: {[round(x, 1) for x in levels]}"

    def test_the_span_is_wide_enough_to_read_as_a_range(self):
        levels = [lightness(rgb) for rgb in _truecolor(_RAMP)]
        assert levels[-1] - levels[0] > 25, (
            "the emptiest and the fullest should not be the same brightness"
        )


class TestTheRampStepsAreEvenlySpaced:
    """"Several same colors go together" was a measurement, not an impression.

    A ramp with one step at dE2000 2.8 and another at 15.3 is really a ramp
    with fewer steps than it advertises: the 2.8 is spent, and the rows that
    land either side of it are told they are equal when they are not.
    """

    def _gaps(self):
        tones = _truecolor(_RAMP)
        return [delta_e(tones[i], tones[i + 1]) for i in range(len(tones) - 1)]

    def test_no_two_neighbouring_steps_are_the_same_colour(self):
        gaps = self._gaps()
        assert min(gaps) > 3.5, (
            f"steps {gaps.index(min(gaps))} and {gaps.index(min(gaps)) + 1} are "
            f"dE {min(gaps):.1f} apart -- indistinguishable in table text"
        )

    def test_the_steps_are_the_same_size_as_each_other(self):
        # The old ramp's widest gap was 5.5x its narrowest, so the scale was
        # coarse at one end and invisible at the other.
        gaps = self._gaps()
        assert max(gaps) / min(gaps) < 2.0, (
            f"uneven: {[round(g, 1) for g in gaps]}"
        )

    def test_distant_steps_cannot_be_confused(self):
        # Rows three apart is the realistic worst case: a reader comparing two
        # partitions is rarely comparing neighbours.
        tones = _truecolor(_RAMP)
        for i in range(len(tones) - 3):
            assert delta_e(tones[i], tones[i + 3]) > 12, f"steps {i} and {i + 3}"


class TestAColourIsEitherAVerdictOrAQuantity:
    """The single worst thing the old palette did.

    `ok` was rgb(110, 205, 130) and the ramp's top step was rgb(0, 215, 95):
    dE2000 8, which in a table is the same green. So a row could show a green
    "idle" dot, a green core count and a green memory figure, two of them
    meaning "large" and one meaning "fine", and nothing on screen said which
    was which. The ramp now stops at turquoise.
    """

    def test_no_verdict_colour_is_near_any_step_of_the_ramp(self):
        for role in VERDICTS:
            colour = _PALETTE[role][0]
            gap, step = min((delta_e(colour, tone), i)
                            for i, tone in enumerate(_truecolor(_RAMP)))
            assert gap > 18, (
                f"`{role}` is dE {gap:.1f} from ramp step {step} -- a reader "
                f"cannot tell the verdict from the measurement"
            )

    def test_the_accent_is_not_near_the_ramp_either(self):
        # `accent` marks identity: a backend name, an accelerator model. If it
        # landed in the ramp it would read as a number that had lost its digits.
        gap = min(delta_e(_PALETTE["accent"][0], tone)
                  for tone in _truecolor(_RAMP))
        assert gap > 18, f"dE {gap:.1f}"

    def test_the_ramp_never_reaches_green(self):
        # Green is spoken for. Turquoise is the top of the scale.
        for i, (red, green, blue) in enumerate(_truecolor(_RAMP)):
            assert blue > green * 0.78, (
                f"step {i} rgb({red}, {green}, {blue}) has drifted into green"
            )

    def test_the_ramp_never_reaches_red_or_amber(self):
        # Amber and red are warning colours. The top of this ramp is the most
        # AVAILABLE resource, which is the opposite of a warning -- and the
        # bottom is merely the least, which is not a warning either.
        for tone in _truecolor(_RAMP) + _truecolor(_WASH):
            red, green, blue = tone
            assert red < max(green, blue), f"rgb{tone} reads as warm"


class TestTheVerdictColoursSurviveColourBlindness:
    """Roughly one man in twelve cannot separate red from green by hue.

    Which is why every verdict in this tool is also a glyph and a word -- the
    colour is never the only carrier. But the colours should still do as much
    as they can, and what they can do is differ in LIGHTNESS, which no form of
    colour vision deficiency takes away. The old `ok` and `warn` were L* 75.2
    and 74.3: the same brightness, so once hue was gone there was nothing left.
    """

    def test_the_verdicts_are_separated_in_lightness_not_only_in_hue(self):
        levels = sorted(lightness(_PALETTE[role][0]) for role in VERDICTS)
        for lower, higher in itertools.pairwise(levels):
            assert higher - lower > 7, (
                f"verdicts sit at L* {[round(x, 1) for x in levels]} -- too close "
                f"to tell apart without hue"
            )

    def test_the_verdicts_are_far_apart_for_a_reader_who_does_see_hue(self):
        for one, two in itertools.combinations(VERDICTS, 2):
            gap = delta_e(_PALETTE[one][0], _PALETTE[two][0])
            assert gap > 25, f"`{one}` vs `{two}`: dE {gap:.1f}"

    def test_identity_is_not_confusable_with_failure(self):
        # `accent` was terracotta rgb(217, 119, 87) and `bad` salmon
        # rgb(235, 110, 105) -- dE2000 9.1. The colour for "this is the thing
        # you asked about" was a near-match for the colour for "this is
        # broken", on a tool whose whole job is telling you which is which.
        gap = delta_e(_PALETTE["accent"][0], _PALETTE["bad"][0])
        assert gap > 25, f"dE {gap:.1f}"


class TestEverythingPaintedOntoTextCanBeRead:
    """A palette that fails contrast is not a style choice, it is a bug."""

    def test_every_role_clears_wcag_aa_against_a_dark_terminal(self):
        for role in READABLE:
            ratio = contrast_ratio(_PALETTE[role][0], BACKGROUND)
            assert ratio >= 4.5, f"`{role}`: {ratio:.2f}:1"

    def test_every_step_of_the_ramp_clears_it_too(self):
        # Including the coldest, which is what an empty queue's `0` is drawn
        # in: the least interesting number still has to be legible.
        for i, tone in enumerate(_truecolor(_RAMP)):
            ratio = contrast_ratio(tone, BACKGROUND)
            assert ratio >= 4.5, f"step {i}: {ratio:.2f}:1"

    def test_the_track_recedes_without_disappearing(self):
        # The one role deliberately below AA: it is a meter's empty channel,
        # a reference mark for "all of it" rather than content.
        ratio = contrast_ratio(_PALETTE["track"][0], BACKGROUND)
        assert 1.3 < ratio < 3.0, ratio

    def test_the_greys_are_an_evenly_spaced_ladder(self):
        # text > muted > dim > track, and far enough apart that the ladder is
        # visible. These four are told apart by lightness alone -- they have no
        # hue to spare -- so an uneven ladder is a ladder with a missing rung.
        levels = [lightness(_PALETTE[role][0])
                  for role in ("text", "muted", "dim", "track")]
        gaps = [a - b for a, b in itertools.pairwise(levels)]
        assert all(gap > 14 for gap in gaps), [round(x, 1) for x in levels]
        assert max(gaps) / min(gaps) < 1.6, [round(g, 1) for g in gaps]


class TestTheDepthsDoNotCollideWhereItMatters:
    """Sixteen colours is a real terminal, not a theoretical one.

    `TERM=screen` and most tmux defaults advertise it. At that depth the old
    palette collapsed `accent` onto `warn` (both SGR 33) -- brand
    indistinguishable from warning -- and `text` onto `muted` (both 37).
    """

    def test_no_two_meaningful_roles_share_a_16_colour_code(self):
        codes: dict[int, list[str]] = {}
        for role, (_rgb, _c256, c16) in _PALETTE.items():
            codes.setdefault(c16, []).append(role)
        collisions = {code: roles for code, roles in codes.items() if len(roles) > 1}
        # `dim` and `track` are allowed to share: both mean "recede", and at
        # sixteen colours there is exactly one grey to recede into.
        assert collisions in ({}, {90: ["dim", "track"]}), collisions

    def test_no_two_roles_share_a_256_colour_index(self):
        indexes = [tone[1] for tone in _PALETTE.values()]
        assert len(set(indexes)) == len(indexes), indexes

    def test_the_ramp_does_not_lose_more_than_half_its_steps_at_256(self):
        # It used to hand back the same index for steps 1 and 2 -- one of
        # twelve gone outright -- while collapsing to six distinct colours.
        assert len({tone[1] for tone in _RAMP}) >= RAMP_STEPS // 2

    def test_the_verdicts_stay_apart_at_every_depth(self):
        # Looser at sixteen, because at sixteen the colours are the terminal's
        # and not ours: `ok` and `warn` become its own green and its own
        # yellow, dE2000 23 apart, and no choice of SGR code moves them
        # further. What the palette owes at that depth is not to make it
        # worse -- which the old one did, by spending SGR 33 on `accent` as
        # well as `warn`.
        for depth, floor in ((24, 25), (8, 25), (4, 20)):
            tones = {role: _at_depth([_PALETTE[role]], depth)[0] for role in VERDICTS}
            for one, two in itertools.combinations(VERDICTS, 2):
                gap = delta_e(tones[one], tones[two])
                assert gap > floor, f"depth {depth}: `{one}` vs `{two}`: dE {gap:.1f}"


class TestTheFrameIsChromeAndNotContent:
    """A border drawn in the colours of the numbers inside it.

    The gradient used to sweep light cyan through aqua, which is the ramp's own
    territory: a cyan frame around a cyan column, chrome competing with the
    content it exists to contain. It now runs periwinkle to light orchid --
    pale enough that nothing in a table is ever mistaken for it, and in a hue
    family no measurement uses.
    """

    def test_the_gradient_avoids_every_colour_the_data_uses(self):
        for anchor in _FRAME_ANCHORS:
            gap, step = min((delta_e(anchor, tone), i)
                            for i, tone in enumerate(_truecolor(_RAMP)))
            assert gap > 18, f"rgb{anchor} is dE {gap:.1f} from ramp step {step}"
            for role in VERDICTS:
                assert delta_e(anchor, _PALETTE[role][0]) > 18, f"rgb{anchor} vs {role}"

    def test_the_gradient_stays_in_the_light_band(self):
        # A frame that sweeps into the dark puts its bottom-right corner below
        # the visible range of a dark terminal, and the box loses a side.
        for anchor in _FRAME_ANCHORS:
            assert lightness(anchor) > 70, f"rgb{anchor} will vanish"

    def test_the_gradient_is_paler_than_anything_in_a_table(self):
        palest = max(lightness(tone) for tone in
                     _truecolor(_RAMP) + [_PALETTE[r][0] for r in READABLE])
        for anchor in _FRAME_ANCHORS:
            # Not brighter than *everything* -- `text` is a light grey -- but
            # never darker than the brightest thing it frames by more than a
            # little, or it starts reading as a row.
            assert lightness(anchor) > palest - 12, f"rgb{anchor}"


class TestTheFillIsDarkerThanTheTextAndBrighterThanTheTrack:
    """Three properties a meter needs, none of which the ramp had by accident.

    A bar is a slab and text is a line: the tone that reads as bright in a
    number reads as shouting across sixteen filled cells. But the coldest fill
    once came out *darker than the track it sat in*, so a barely-full bar read
    inverted -- the emptiness looking more present than the fill.
    """

    def test_every_fill_is_brighter_than_the_empty_track(self):
        track = lightness(_PALETTE["track"][0])
        for step, tone in enumerate(_truecolor(_WASH)):
            assert lightness(tone) > track, f"step {step} fill is darker"

    def test_every_fill_is_darker_than_its_own_text_tone(self):
        for step, (text, fill) in enumerate(zip(_truecolor(_RAMP),
                                                _truecolor(_WASH), strict=True)):
            assert lightness(fill) < lightness(text), f"step {step}"

    def test_the_fill_keeps_its_step_s_hue(self):
        # A darker twin, not a different colour: the bar and the number beside
        # it are one reading, and they have to look related.
        for step, (text, fill) in enumerate(zip(_truecolor(_RAMP),
                                                _truecolor(_WASH), strict=True)):
            lighter = tuple(min(255, round(c * lightness(text) / lightness(fill)))
                            for c in fill)
            assert delta_e(lighter, text) < 22, f"step {step}: fill hue has drifted"
