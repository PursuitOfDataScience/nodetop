"""Terminal rendering, with no third-party dependencies.

A tool whose job is to work on a login node during an outage should not need
anything installed to run, so the colour, box drawing, gauges and width-correct
tables are all built here against the standard library.

Three things are handled that a naive implementation gets wrong, and all three
are correctness rather than decoration:

* **Display width is not string length.**  Padding a cell with ``len()`` breaks
  alignment for any wide character and for the ANSI escapes we emit ourselves.
  :func:`width` measures what the terminal will actually show.
* **Not every terminal speaks UTF-8.**  A ``LANG=C`` session or a bare console
  renders box-drawing characters as mojibake, so every glyph has an ASCII
  twin and the set is chosen from the real stdout encoding.
* **Colour support is a spectrum, not a boolean.**  Truecolor, 256-colour and
  16-colour terminals all exist, and a palette picked for one looks wrong or
  fails outright on another.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "Glyphs",
    "sanitize",
    "colorize_help",
    "FALLBACK_WIDTH",
    "MIN_WIDTH",
    "PROSE_WIDTH",
    "RAMP_STEPS",
    "Style",
    "badge",
    "bar",
    "columns",
    "flow",
    "gauge",
    "heat_step",
    "heat_steps",
    "kv",
    "panel",
    "plural",
    "rule",
    "section",
    "sparkline",
    "table",
    "tree",
    "truncate",
    "width",
    "wrap_indent",
]

#: Columns the layout will use. **The terminal's, not a fixed measure.**
#:
#: It was 100, so a frame stopped at column 100 however wide the window was
#: and the tool sat in a box off to the left of a wide terminal: "i think the
#: app should take the entire hortizontal space. the current one looks so
#: squeezed and unnatural." The cap existed to stop a box being ruled out well
#: past its content -- which is a real failure, and the answer to it is for the
#: CONTENT to use the room (see the meter in `status`), not for the window to
#: be pretended smaller than it is.
#:
#: The ceiling is a sanity bound rather than a design one: a terminal claiming
#: tens of thousands of columns is a terminal lying, and every cell of every
#: table is padded to whatever this returns.
MAX_WIDTH = 400

#: Columns *prose* wraps to, whatever the window is doing.
#:
#: Tables and frames take the terminal; sentences do not. A paragraph set 300
#: columns wide is measurably harder to read than the same paragraph at 90 --
#: the eye loses the line it was on when it sweeps back -- and typography has
#: settled this at roughly 45-90 characters for long enough that a terminal is
#: not going to overturn it. So the two are separate numbers, and this one is
#: also a cap and not a width: a narrow window still wins.
PROSE_WIDTH = 96


# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------
def _strip_ansi(text: str) -> str:
    # Nothing to strip, which is the common case: a node name, a count, a
    # state. `in` on a string is a C-level scan, where the loop below is a
    # Python one per character -- and this function was the single largest
    # entry in a profile of rendering a 10,624-row table (53,537 calls,
    # 200 ms of 819 ms under the profiler).
    if "\033" not in text:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\033":
            # CSI ... final-byte, or a short two-character escape.
            j = i + 1
            if j < len(text) and text[j] == "[":
                j += 1
                while j < len(text) and not (0x40 <= ord(text[j]) <= 0x7E):
                    j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


#: Every C0 control character, DEL, and the C1 block. Tab is in here too: it
#: measures as one column and the terminal expands it to the next tab stop,
#: which is the same alignment break by a quieter route.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Make scheduler-supplied text safe to print. Control characters out.

    A node's ``Reason``, a Kubernetes condition message and a dry-run's stderr
    are all operator- or controller-authored free text, and they go straight
    into a table cell. Left alone they do damage that :func:`width` cannot see,
    because it measures what the text *occupies* and these characters act
    instead:

    * ``ESC [ 2 J`` clears the caller's terminal in the middle of the report;
    * ``\r`` returns to column zero, so the rest of the row overwrites what was
      already drawn and the alignment this module exists to protect is gone --
      silently, with content hidden rather than mangled;
    * ``\n`` splits one row across two lines and the table stops being a table;
    * ``\t`` measures as one column and expands to a tab stop.

    Applied to *incoming* data only, never to the tool's own output: styling is
    added after this runs, so the escapes nodetop emits deliberately are
    untouched. Each offender becomes a space rather than being deleted, so the
    text keeps its shape and a mangled field still reads as a mangled field
    instead of quietly closing up.

    This is also the ``--replay`` boundary. A snapshot is a JSON file that may
    have been handed over by someone else, and replaying one should not be able
    to repaint the terminal of whoever reads it.
    """
    return _CONTROL.sub(" ", text) if text else text


#: Measured cell widths, kept because the same string is measured many times.
#:
#: A table sizes each column by the widest cell and then pads every cell to it,
#: so one render measures each cell at least twice -- and a listing repeats
#: itself heavily: `MIXED`, `64/64`, a meter, a dash. Profiling `nodes --all` on
#: a 607-node cluster: **`width` called 19,600 times per render**, and it plus
#: `_strip_ansi` were the top two entries by cumulative time.
#:
#: It matters most where it is least visible: the interactive browse re-renders
#: the whole frame on every keypress, so this is the cost of moving the cursor.
#:
#: Bounded, and small: cells are short strings, 8192 of them is a few hundred
#: kilobytes, and a listing of 10,000 nodes reuses far more than it adds.
@functools.lru_cache(maxsize=8192)
def width(text: str) -> int:
    """Columns this string occupies in a terminal.

    Ignores ANSI escapes, counts East-Asian wide and fullwidth characters as
    two columns, and counts zero-width combining marks as none.  Using
    ``len()`` instead is what makes a coloured or non-Latin table drift.
    """
    stripped = _strip_ansi(text)
    # ASCII cannot be wide and cannot combine, so its display width IS its
    # length -- and `str.isascii()` is a flag check on the string object rather
    # than a scan. This is the path essentially every cell takes: names,
    # counts, states, meters drawn from box-drawing... no, those are not ASCII,
    # so they fall through to the loop, which is exactly right.
    #
    # It matters because the memo above cannot help at scale: 10,624 nodes
    # produce far more than 8,192 distinct strings, so a large listing is
    # mostly cache misses, and a miss used to mean two `unicodedata` lookups
    # per character -- 385,333 of each in one render of that table.
    if stripped.isascii():
        return len(stripped)
    total = 0
    for ch in stripped:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def truncate(text: str, limit: int, ellipsis: str = "\u2026") -> str:
    """Cut to ``limit`` display columns, keeping any ANSI styling intact.

    Escape sequences are copied through without being counted, and a reset is
    appended when the cut lands inside a styled run -- otherwise the colour
    would bleed into the rest of the line and, worse, the visible text would be
    silently shortened by however many bytes the escapes occupied.
    """
    if limit <= 0:
        return ""
    if width(text) <= limit:
        return text

    keep = max(0, limit - width(ellipsis))
    out: list[str] = []
    used = 0
    styled = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\033":
            j = i + 1
            if j < len(text) and text[j] == "[":
                j += 1
                while j < len(text) and not (0x40 <= ord(text[j]) <= 0x7E):
                    j += 1
            out.append(text[i : j + 1])
            styled = True
            i = j + 1
            continue
        w = 0 if unicodedata.combining(ch) else (
            2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        )
        if used + w > keep:
            break
        out.append(ch)
        used += w
        i += 1
    return "".join(out) + ellipsis + ("\033[0m" if styled else "")


def pad(text: str, size: int, align: str = "left", have: int | None = None) -> str:
    """Pad to ``size`` display columns. ``have`` is the width, if already known.

    A table measures every cell to size its columns and then pads every cell to
    that size, so the width was already in hand and this used to ask for it
    again: 85,000 redundant measurements on a 10,624-row listing.
    """
    gap = max(0, size - (width(text) if have is None else have))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


# ---------------------------------------------------------------------------
# glyphs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Glyphs:
    """The character set to draw with.

    Two complete sets exist so the output degrades to something readable
    instead of to mojibake.  ``unicode`` is chosen only when stdout can
    actually encode it.
    """

    # box drawing
    h: str = "─"
    v: str = "│"
    tl: str = "╭"
    tr: str = "╮"
    bl: str = "╰"
    br: str = "╯"
    # tree
    branch: str = "├─"
    last: str = "╰─"
    pipe: str = "│ "
    # status
    #: The selection cursor. A glyph rather than colour alone, because inverse
    #: video is invisible under NO_COLOR and unobvious even with it -- the
    #: complaint was "the users don't know if they can move the cursor up or
    #: down". A pointer at the left edge of the row says which row and implies
    #: the axis.
    #: Marks the column a table is ordered by, so the reader is not left to
    #: infer it from the row order.
    sort_down: str = "↓"
    cursor: str = "❯"
    bullet: str = "⏺"
    ok: str = "●"
    partial: str = "◐"
    off: str = "○"
    bad: str = "✗"
    warn: str = "▲"
    arrow: str = "→"
    #: For the key hints in a browse footer: the pair a reader presses to move
    #: between rows, and the one that steps back out. Glyphs rather than the
    #: words "up/down arrow" because the footer has to fit on one line beside
    #: four other hints, and because the shape IS the key.
    arrow_pair: str = "↑↓"
    arrow_back: str = "←"
    #: A SEPARATOR, and only ever that.
    #:
    #: It kept being reached for as an empty table cell, where it means nothing
    #: a reader can act on and twice stood in for an actual number -- a job
    #: holding none of a node's accelerators, and a job spanning exactly one
    #: node. "putting a dot there means nothing"; "what does . mean in the node
    #: column?" A count goes in a count column, and :attr:`dash` is what a
    #: question that does not arise looks like.
    sep: str = "·"
    ellipsis: str = "…"
    #: "This question does not arise here", in a column that asks one: a node
    #: with no accelerator under `gpu free`, a partition that can start nothing
    #: under `start`, a field the control plane did not report. Distinct from a
    #: zero, which is a measurement.
    dash: str = "—"
    # meters: eighth-blocks give sub-cell resolution
    blocks: str = "▏▎▍▌▋▊▉█"
    empty: str = "░"
    spark: str = "▁▂▃▄▅▆▇█"
    unicode: bool = True

    @classmethod
    def ascii(cls) -> Glyphs:
        return cls(
            h="-", v="|", tl="+", tr="+", bl="+", br="+",
            branch="|-", last="`-", pipe="| ",
            sort_down="v", cursor=">", bullet="*", ok="o", partial="%", off=".", bad="x",
            warn="!",
            arrow="->", arrow_pair="up/dn", arrow_back="<-",
            sep="-", ellipsis="...", dash="--",
            blocks="#", empty=".", spark="_.-=+*#%",
            unicode=False,
        )

    @classmethod
    def detect(cls, stream: object | None = None) -> Glyphs:
        """Unicode when stdout can encode it, ASCII otherwise."""
        if os.environ.get("NODETOP_ASCII"):
            return cls.ascii()
        stream = stream or sys.stdout
        encoding = (getattr(stream, "encoding", None) or "").lower()
        if "utf" in encoding:
            return cls()
        # A terminal that cannot encode the glyph would raise or print
        # replacement characters; ASCII is strictly better than either.
        try:
            "─●".encode(encoding or "ascii")
        except (LookupError, UnicodeEncodeError):
            return cls.ascii()
        return cls()


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
#: (truecolor rgb, 256-colour index, 16-colour SGR) for each semantic role.
#:
#: **A role is a category, so it is told apart by HUE; a magnitude is an order,
#: so the ramp below is told apart by LIGHTNESS.** Mixing the two is what made
#: this palette unreadable: the old ramp ended in the same green ``ok`` is
#: painted in, so a row could show an ``ok`` dot, a ramp-green core count and a
#: ramp-green memory figure side by side in three greens nobody could separate
#: -- "several same colors go together". The ramp now stops at turquoise and
#: never enters the green, amber, red or magenta that carry meaning, so a
#: colour on this screen is either a verdict or a quantity and never ambiguous.
#:
#: Measured, not eyeballed. Against a ``#0d1016`` terminal every role clears
#: WCAG AA 4.5:1 (``track`` excepted -- it is a meter's empty channel, which
#: must recede), and no two verdict colours are closer than dE2000 28. The
#: verdict trio is additionally separated in LIGHTNESS (``ok`` L* 79,
#: ``warn`` L* 69, ``bad`` L* 56), because deuteranopia and protanopia collapse
#: green/amber/red toward one hue and lightness is what survives -- which is
#: also why each one is spoken with a glyph and a word, never colour alone.
#:
#: Picked so the same intent survives all three depths rather than only the
#: richest one. At sixteen colours the only remaining collision is
#: ``dim``/``track``, two roles that both mean "recede"; ``accent`` used to
#: share SGR 33 with ``warn`` -- brand indistinguishable from warning -- and
#: ``text`` used to share 37 with ``muted``.
_PALETTE: dict[str, tuple[tuple[int, int, int], int, int]] = {
    # Identity: the backend's name, an accelerator model, the submit line.
    # Magenta, because it was terracotta -- dE2000 9.1 from `bad`, i.e. the
    # colour for "this is what you are looking at" was a near-match for the
    # colour for "this is broken". Nothing else on screen is in this family.
    "accent": ((242, 132, 200), 212, 95),
    "ok": ((114, 218, 104), 77, 92),
    "warn": ((236, 151, 0), 172, 33),
    "bad": ((233, 82, 60), 196, 91),
    # A neutral notice -- a routing arrow, a node with no room left -- and
    # the colour of everything you can type in `--help`: flags, sub-commands,
    # example commands. It is a periwinkle rather than a ramp blue, and it
    # sits a full 17 points of L* above the ramp's coldest step, which is what
    # keeps a flag from reading as a small number. At sixteen colours it is
    # bright blue and not plain: navy on a dark terminal is a rumour.
    "info": ((144, 166, 247), 111, 94),
    "dim": ((119, 124, 131), 244, 90),
    "text": ((217, 219, 221), 253, 97),
    # Secondary *content* -- a real measurement that is not the one the view
    # was ranked by. Distinct from "dim", which means context rather than
    # content: a snapshot age, a caveat, a hint. A free-core count is content
    # even when the bar beside it is what the eye goes to first, and painting
    # the two the same grey is what makes a numeric column read as furniture.
    # The four greys are an evenly spaced L* ladder -- 87, 69, 52, 26 -- so
    # they rank by brightness alone, with colour vision or without it.
    "muted": ((163, 168, 174), 248, 37),
    # The unfilled part of a meter. A reference mark for "all of it", not
    # content at all, so it drops below "dim" to a near-background grey: it
    # should frame the bar without competing with it.
    "track": ((55, 62, 69), 237, 90),
}


# ---------------------------------------------------------------------------
# heat: magnitude as colour
# ---------------------------------------------------------------------------
#
# A meter's fill is coloured by the size of what it measures, on one ordered
# ramp -- and the ramp is ordered by LIGHTNESS, which is the only visual
# channel the eye reads as a quantity without being taught to.
#
# This replaces a flat single-colour fill, which was chosen after an earlier
# version painted bars green above half and amber below.  That version deserved
# to go: two colours either side of a threshold is a *verdict*, and the same
# amber then meant "40% of GPUs are free" -- a warning about nothing.  Going
# flat fixed the false alarm by giving up on colour carrying any quantity at
# all, which is the other extreme.
#
# A ramp is neither.  Twelve ordered steps read as a scale rather than a
# judgement, the way a heatmap legend does, and **no step is red at either
# end** -- so a full bar and an empty one are both unremarkable, and red is
# left to mean what it means everywhere else in this tool: something is
# actually wrong, said with a glyph and a word.
#
# **The previous ramp was a rainbow, and rainbows lie about order.**  It ran
# blue -> cyan -> green with its brightest step in the MIDDLE: L* climbed to
# 91 at cyan and fell back to 76 at the green end, six reversals in twelve
# steps.  That is the defect that got jet retired from scientific plotting --
# a mid-range value reads as the extreme because it is the brightest thing on
# screen, and the true extreme reads as mid-range.  Worse, its steps were
# spaced between dE2000 2.8 and 15.3, a 5.5x spread, so four of the top five
# were one green: steps 7, 8, 9 and 10 sat 2.8, 6.4, 4.8 and 4.7 apart and
# a reader comparing two rows could not tell which was larger.  On a
# 256-colour terminal steps 1 and 2 were literally the same index, and at
# sixteen colours the twelve collapsed into six.
#
# This ramp is built the way a perceptually uniform colormap is built: a path
# through OKLCH from blue (L 0.60) through cyan to turquoise (L 0.87),
# resampled at twelve points of EQUAL dE2000 arc length.  The result is
# monotonic in lightness with steps 4.3-5.1 apart -- a 1.18x spread instead of
# 5.5x -- so adjacent rows differ by a visible, constant amount and distant
# rows cannot be confused.  Every step clears 4.5:1 against a dark terminal,
# so the coldest count is still readable rather than merely present.
#
# **It stops at turquoise and does not reach green.** Green belongs to `ok`,
# and the old ramp's top step was dE2000 8 from it -- an idle node's dot and
# its core count were the same colour by accident. The gap is now 23, and the
# ramp's whole span sits at least 36 from `warn`, `bad` and `accent`. So the
# question "is this colour a verdict or a quantity?" has an answer.
#
# Each step has a darker twin, used for bar *fill*.  A bar is a slab and text
# is a line: the colour that reads as bright in a number reads as shouting
# across eighteen filled cells, and ten shouting bars are a wall.  Terminals
# have no alpha channel, so the wash is a genuinely darker colour of the same
# hue -- which is what compositing that hue at ~60% over a dark background
# would have produced anyway.
_Tone = tuple[tuple[int, int, int], int, int]

#: Text tones, least first. Monotonic in lightness, evenly spaced in dE2000.
#:
#: **The warm end was removed first, and the green end after it.** The ramp
#: once ran blue -> cyan -> green -> yellow -> amber, on the reasoning that any
#: ordered sweep reads as a scale. It does not, because these numbers are
#: *availability*: the top of the ramp is the emptiest node, and amber is read
#: as heat -- "why use the orange colour to denote an unoccupied cpu?" A fully
#: idle machine drew the most alarming colour on the screen.
#:
#: Ending in green fixed that and introduced a quieter version of the same
#: problem: green is this tool's word for "fine", so the scale and the verdict
#: spoke with one voice. The sweep now ends in turquoise -- still the brightest
#: step, still unmistakably the top of the scale, and no longer a claim.
#:
#: The 256-colour column is not a nearest-match: nearest-match reintroduces
#: lightness reversals, because the xterm cube is sparse through blue-cyan.
#: It is the closest sequence that is still monotonic in L*.
_RAMP: tuple[_Tone, ...] = (
    ((55, 126, 227), 33, 34),    # blue        -- least free
    ((39, 136, 230), 33, 34),
    ((13, 146, 231), 33, 94),
    ((0, 156, 228), 39, 94),
    ((0, 166, 225), 39, 94),
    ((0, 177, 223), 39, 36),
    ((0, 187, 222), 45, 36),
    ((0, 198, 223), 45, 36),
    ((0, 210, 224), 80, 96),
    ((0, 222, 225), 80, 96),
    ((0, 233, 224), 51, 96),
    ((0, 245, 222), 86, 96),     # turquoise   -- most free
)

#: Fill tones: the same hue, darker, one per step of :data:`_RAMP`.  At 16
#: colours there is no room for a second copy of every step -- SGR faint
#: collapses bright blue onto blue on the terminals that implement it as "not
#: bold" -- so the fill simply keeps the text tone there.
#: Every fill must be BRIGHTER than the track it sits in.
#:
#: The coldest fill used to be rgb(0,0,135), luminance 10, against a track of
#: luminance 65 -- so a bar with a little free capacity drew its filled part
#: *darker* than its empty part and read inverted, the emptiness looking more
#: present than the fill. A meter whose two halves swap roles at the bottom of
#: its range is worse than no meter.
_WASH: tuple[_Tone, ...] = (
    ((26, 82, 161), 25, 34),     # blue
    ((4, 90, 163), 25, 34),
    ((0, 99, 159), 25, 94),
    ((0, 107, 155), 25, 94),
    ((0, 115, 154), 31, 94),
    ((0, 122, 153), 31, 36),
    ((0, 131, 153), 31, 36),
    ((0, 138, 154), 31, 36),
    ((0, 145, 155), 37, 96),
    ((0, 153, 156), 37, 96),
    ((0, 162, 156), 37, 96),
    ((0, 171, 155), 37, 96),     # turquoise
)

RAMP_STEPS = len(_RAMP)

# Counts across partitions are heavily skewed -- one partition holds a third of
# the free cores on this cluster and the whole tail is a few percent each -- so
# indexing the ramp linearly by value/peak spends most of it on the top two
# rows and crushes the rest into one colour.  The exponent spreads the tail
# back out.  0.45 is the sRGB encoding gamma, and it is the right neighbourhood
# for the same reason it is there: perceived brightness follows roughly the
# same curve.
#
# It applies to :func:`heat_steps` (ranking a set of raw counts) and NOT to
# :func:`heat_step` (a share that is already a fraction of its own total).
# Gamma on a share would be a lie about the number: 26% of cores free would
# come out mid-ramp, which is where 50% belongs.
_RANK_GAMMA = 0.45

# Two values count as "the same size" for colouring when they are within this
# of each other.  Below it a visible colour step would claim a difference the
# reader cannot check; above it, :func:`heat_steps` insists on one.
_SAME = 0.01


def heat_step(fraction: float | None, gamma: float = 1.0) -> int:
    """Ramp index for a 0..1 position, clamped."""
    if not fraction or fraction <= 0:
        return 0
    if fraction >= 1:
        return RAMP_STEPS - 1
    return min(RAMP_STEPS - 1, int((fraction**gamma) * RAMP_STEPS))


def heat_steps(values: Sequence[float]) -> list[int]:
    """Ramp indices for a whole column at once, in the order given.

    Colouring each row independently is what produces "these three look the
    same": a fixed number of bands cannot know that 1336, 128 and 128 happen to
    fall inside one of them, so a partition with ten times another's room comes
    out the same colour as it.

    Colouring the set fixes it.  Every row starts at the step its own magnitude
    earns against the largest value; the rows are then walked largest-first,
    and one that is *measurably* smaller than the row above -- more than
    :data:`_SAME` apart -- is forced at least one step cooler.  Rows that
    really are equal stay equal, which is the property that makes the ramp mean
    anything.  Ties in the tail bottom out at the coldest step rather than
    wrapping round, so a long listing ends in flat blue instead of starting
    over in amber.
    """
    if not values:
        return []
    peak = max(values)
    if peak <= 0:
        return [0] * len(values)
    out = [0] * len(values)
    previous: tuple[int, float] | None = None
    for at in sorted(range(len(values)), key=lambda i: values[i], reverse=True):
        value = values[at]
        step = heat_step(value / float(peak), _RANK_GAMMA)
        if previous is not None:
            prev_step, prev_value = previous
            if value >= prev_value * (1.0 - _SAME):
                step = prev_step
            elif step >= prev_step:
                step = max(0, prev_step - 1)
        out[at] = step
        previous = (step, value)
    return out


def _depth() -> int:
    """0 = no colour, 4 = 16-colour, 8 = 256-colour, 24 = truecolor."""
    if os.environ.get("NO_COLOR"):
        return 0
    forced = os.environ.get("FORCE_COLOR")
    # `FORCE_COLOR=0` asks for NO colour, and used to get the most there is: any
    # value other than `1`/`true` was read as "some other level" and mapped to 24
    # below, so `FORCE_COLOR=0` -- and `false` -- came back TRUECOLOR. A value
    # naming zero cannot select the top of a ladder whose zero this docstring
    # defines as "no colour". Treated like `NO_COLOR`, and spelled with the same
    # `false`/`true` vocabulary the level mapping already uses.
    if forced in ("0", "false"):
        return 0
    if not forced and not sys.stdout.isatty():
        return 0
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return 0
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return 24
    if "256color" in term or "direct" in term:
        return 8
    if forced:
        return 24 if forced not in ("1", "true") else 8
    return 4 if term else 0


class Style:
    """Semantic colour and glyph access, degrading cleanly.

    Call sites ask for meaning (``st.ok``, ``st.bad``) rather than for a
    colour, so a change of palette or a drop to a 16-colour terminal does not
    ripple outward.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        depth: int | None = None,
        glyphs: Glyphs | None = None,
    ) -> None:
        self.depth = 0 if enabled is False else (_depth() if depth is None else depth)
        if enabled is True and self.depth == 0:
            self.depth = 8
        self.g = glyphs or Glyphs.detect()
        #: Read on every painted string -- 74,526 times in one 10,624-row frame
        #: -- and derived from a `depth` that cannot change after construction,
        #: so it is computed here instead of on each read.
        self.enabled = self.depth > 0
        #: Tone -> escape, filled on demand. See :meth:`_sgr`.
        self._escapes: dict[_Tone, str] = {}
        #: (role, bold) and (step, fill, bold) -> the escape prefix that opens
        #: a painted run. Same reasoning as `_escapes`, one level up: `paint`
        #: and `tint` were called 74,526 times between them for one frame and
        #: rebuilt the same few prefixes every time.
        self._prefixes: dict[tuple[object, ...], str] = {}
        #: Meter shape -> rendered bar, filled on demand. See :func:`bar`.
        self._bars: dict[tuple[object, ...], str] = {}

    # -- primitives ---------------------------------------------------------
    def _sgr(self, tone: _Tone) -> str:
        """One ``(rgb, 256, 16)`` tone as the escape this terminal can show.

        Memoised per style, because `depth` is fixed at construction and the
        answer is otherwise rebuilt for every cell of every row: **94,294 calls
        in one render** of a 10,624-row table, each formatting the same handful
        of strings. The domain is the palette plus the ramp -- a few dozen tones
        -- so the dictionary stays tiny and every entry is hit thousands of
        times. Per instance rather than an `lru_cache`, since the escape depends
        on this style's colour depth.
        """
        hit = self._escapes.get(tone)
        if hit is None:
            rgb, c256, c16 = tone
            if self.depth >= 24:
                hit = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
            elif self.depth >= 8:
                hit = f"\033[38;5;{c256}m"
            else:
                hit = f"\033[{c16}m"
            self._escapes[tone] = hit
        return hit

    def _fg(self, role: str) -> str:
        return self._sgr(_PALETTE[role])

    def paint(self, role: str, text: str, bold: bool = False) -> str:
        if not self.enabled or not text:
            return text
        key = (role, bold)
        prefix = self._prefixes.get(key)
        if prefix is None:
            prefix = ("\033[1m" if bold else "") + self._fg(role)
            self._prefixes[key] = prefix
        return f"{prefix}{text}\033[0m"

    def tint(self, text: str, step: int, fill: bool = False, bold: bool = False) -> str:
        """Paint with ramp step ``step``; ``fill`` picks the darker twin.

        The one way call sites reach the heat ramp, so a row's number and the
        bar beside it are guaranteed to be the same hue -- which is the whole
        point of colouring either of them.
        """
        if not self.enabled or not text:
            return text
        # Clamped before it is used as a key, so an out-of-range step shares
        # the entry it shares the colour with and the dictionary stays bounded
        # at RAMP_STEPS x fill x bold.
        step = max(0, min(RAMP_STEPS - 1, step))
        key = (step, fill, bold)
        prefix = self._prefixes.get(key)
        if prefix is None:
            prefix = (("\033[1m" if bold else "")
                      + self._sgr((_WASH if fill else _RAMP)[step]))
            self._prefixes[key] = prefix
        return f"{prefix}{text}\033[0m"

    def heat(self, text: str, fraction: float | None, bold: bool = False) -> str:
        """Paint by a share of a known total: 0 is coolest, 1 is warmest."""
        return self.tint(text, heat_step(fraction), bold=bold)

    def bold(self, text: str) -> str:
        return f"\033[1m{text}\033[0m" if self.enabled else text

    def inverse(self, text: str) -> str:
        """Reverse video across the whole string, embedded resets and all.

        **Re-armed after every reset the text already contains, which is the
        only reason this works.** A rendered table row is full of coloured
        cells, and each one ends in ``ESC [ 0 m`` -- which clears reverse video
        along with the colour. Wrapping such a row in ``ESC [ 7 m`` therefore
        highlighted it as far as the first coloured cell and no further, so the
        selected row read as a smudge on the left rather than as a selected row.
        That was the "the highlight should be obvious" complaint, and it was not
        a question of taste.

        Re-opening after each reset costs one escape per cell and leaves the
        row's own colours intact underneath.
        """
        if not self.enabled:
            return text
        armed = text.replace("\033[0m", "\033[0m\033[7m")
        return f"\033[7m{armed}\033[0m"

    # -- semantic roles -----------------------------------------------------
    def accent(self, t: str, bold: bool = False) -> str:
        return self.paint("accent", t, bold)

    def ok(self, t: str, bold: bool = False) -> str:
        return self.paint("ok", t, bold)

    def warn(self, t: str, bold: bool = False) -> str:
        return self.paint("warn", t, bold)

    def bad(self, t: str, bold: bool = False) -> str:
        return self.paint("bad", t, bold)

    def info(self, t: str, bold: bool = False) -> str:
        return self.paint("info", t, bold)

    def dim(self, t: str) -> str:
        return self.paint("dim", t)

    def muted(self, t: str) -> str:
        """Content that is not what this view was ranked by. See "muted"."""
        return self.paint("muted", t)

    def track(self, t: str) -> str:
        return self.paint("track", t)

    def head(self, t: str) -> str:
        return self.paint("text", t, bold=True)


MIN_WIDTH = 40


def plural(count: int, word: str, suffix: str = "s") -> str:
    """``1 queue`` / ``2 queues``.

    Worth a helper rather than an inline ``f"{n} {term}s"``: the noun comes
    from the backend, so it is never a literal the author can eyeball, and
    "1 queues considered" is the kind of thing that reads as carelessness.
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}{suffix}"


#: Columns to assume when there is no terminal to ask -- a pipe, a file, a CI
#: log. Not :data:`MAX_WIDTH`: the cap is what a *window* may be, and a pipe
#: has no window, so inheriting the cap would set every redirected table to
#: four hundred columns.
FALLBACK_WIDTH = 100


def term_width(cap: int = MAX_WIDTH) -> int:
    """Usable width, clamped to a range the layout can actually work in."""
    columns = shutil.get_terminal_size((FALLBACK_WIDTH, 24)).columns
    return max(MIN_WIDTH, min(columns, cap))


#: Rows a full-screen frame may occupy, before the window is consulted.
#:
#: The counterpart of :data:`MAX_WIDTH` and for the same reason: a box ruled out
#: to sixty rows around eight rows of content reads as an empty room.
MAX_HEIGHT = 30

#: Below this a full-screen frame cannot hold its own chrome, and the repaint
#: starts overwriting the screen instead of itself.
MIN_HEIGHT = 8


def term_height(cap: int = MAX_HEIGHT) -> int:
    """Rows one full-screen frame may occupy, borders included.

    **One less than the window, and that spare line is load-bearing.** A frame
    exactly as tall as the terminal scrolls it by one on its final newline, so
    the repaint's cursor-up lands a line low and every keypress orphans the top
    border -- a growing stack of ``╭────╮``.

    Every view in an interactive session sizes itself from this one number, so
    the box stays put as the reader moves between levels instead of shrinking
    to fit whatever is inside it.
    """
    lines = shutil.get_terminal_size((FALLBACK_WIDTH, 24)).lines
    return max(MIN_HEIGHT, min(lines - 1, cap))


# ---------------------------------------------------------------------------
# meters
# ---------------------------------------------------------------------------
def bar(
    fraction: float,
    size: int = 16,
    style: Style | None = None,
    role: str | None = None,
    step: int | None = None,
) -> str:
    """A horizontal meter with sub-cell resolution.

    Eighth-blocks let a 16-cell bar resolve ~1/128, so a nearly-empty queue
    still shows *something* rather than rounding to nothing -- which matters
    when the difference between 0 and 2 free GPUs is the whole question.

    The fill is coloured by the heat ramp (see :func:`heat_step`) and the
    unfilled remainder by the near-background ``track`` grey, so the bar reads
    as a box with a level in it rather than as a stripe trailing off into
    nothing.  ``step`` overrides which ramp step to use, for a caller that has
    already ranked a whole column with :func:`heat_steps` and wants the bar to
    match the number beside it.  ``role`` overrides the colour entirely, for
    the rare bar that genuinely is a verdict.

    **A meter drawn completely full means the whole of it, and nothing less.**
    Every caller prints an EXACT ratio beside the bar -- ``5115/5120``,
    ``231/232``, ``88/176 gpu`` -- so unlike the family's percentage gauges
    there is no rounded label to agree with: the only fullness the number
    admits is numerator == denominator.  Both draw paths reached full early
    anyway, because both round.  Measured before this, at ``NODETOP_ASCII`` and
    without: ASCII drew ``########`` for 96.6% of 8 cells and
    ``##################`` for 98% of 18; Unicode drew a solid ``████████``
    from 99.4% of 8 cells and a solid 18 from 99.9%.  A partition with 5115 of
    5120 cores free therefore rendered as *every core free*, which is the one
    thing the ``cores free`` column exists to distinguish.  So the last eighth
    is withheld until ``fraction`` reaches 1.0 -- the eighth, not the whole
    cell, because sub-cell resolution is this meter's stated point and one
    eighth of one cell is all it takes for the tip to stop being ``█``.  The
    siblings settled the same invariant the same way: ``slurmwatch.tui``
    reserves the final eighth in ``_color_bar`` and the final cell in
    ``_bar_cells``, and ``slurmpast.render.bar_cells`` reserves the final whole
    cell on the two paths that have no eighths to spare.
    """
    style = style or Style()
    g = style.g
    fraction = max(0.0, min(1.0, fraction))
    tone = heat_step(fraction) if step is None else step
    # A meter is one of very few pictures: `size` cells at eighth resolution,
    # in one of nine ramp steps. A 10,624-row listing drew 10,624 of them and
    # the drawing is not free -- 105 ms of that render, in `heat_step`, the
    # block arithmetic and two `paint` calls per bar. Keyed on the *rounded*
    # numbers the body actually uses, both of them, so a hit is the same string
    # the miss would have built rather than one that merely looks like it.
    #
    # `whole` is in the key for exactly that reason. The rounded eighths cannot
    # tell 0.999 from 1.0 at size 8 -- both round to 64 -- and since the reserve
    # below keys on the UNROUNDED fraction, the two now draw different pictures.
    # Without this term the first of the pair to be drawn would be handed back
    # for the other, which is how a 5115/5120 row would inherit a solid bar from
    # the 5120/5120 row above it.
    whole = fraction >= 1.0
    key = (int(round(fraction * size * 8)), int(round(fraction * size)),
           size, role, tone, whole)
    hit = style._bars.get(key)
    if hit is not None:
        return hit

    def painted(fill: str, trough: str) -> str:
        if role is not None:
            return style.paint(role, fill + trough)
        return style.tint(fill, tone, fill=True) + style.track(trough)

    if not g.unicode:
        filled = int(round(fraction * size))
        # Short of the whole, keep one cell of trough: ASCII has no partial
        # block to spend, so the cell is the smallest reserve there is.
        if not whole:
            filled = min(filled, max(0, size - 1))
        drawn = painted(g.blocks * filled, g.empty * (size - filled))
    else:
        total_eighths = int(round(fraction * size * 8))
        # Short of the whole, keep one eighth back, so the tip is `▉` and not
        # `█`. See the reserve paragraph in the docstring.
        if not whole:
            total_eighths = min(total_eighths, max(0, size * 8 - 1))
        full, remainder = divmod(total_eighths, 8)
        fill = g.blocks[-1] * full
        if remainder:
            fill += g.blocks[remainder - 1]
        drawn = painted(fill, g.empty * max(0, size - width(fill)))
    # Bounded: the domain is (eighths x size x role x step), and a listing
    # reuses far more than it adds. The cap is a backstop against a caller that
    # varies `size` per row rather than per column.
    if len(style._bars) >= 4096:
        style._bars.clear()
    style._bars[key] = drawn
    return drawn


def gauge(
    free: int,
    total: int,
    size: int = 14,
    style: Style | None = None,
    unit: str = "",
) -> str:
    """A meter with its own numbers::

        ███████░░░░░░░ 88/176 gpu
        █░░░░░░░░░░░░░ 12/176 gpu

    The trough is ``░`` (``.`` in ASCII mode).  It was a dot leader, on the
    argument that dots stay legible where the block glyphs are missing -- but
    that argument does not hold: ``░`` is U+2591 and ``█`` is U+2588, the same
    Unicode block, so a font lacking one lacks the other and the bar is
    unreadable either way.  :meth:`Glyphs.ascii` is the real fallback for that
    case.  A shaded trough also makes the boundary between filled and empty
    unmistakable at a glance, which a dot leader does not.
    """
    style = style or Style()
    if total <= 0:
        # From the glyph set, not a literal: a hardcoded em dash is exactly how
        # a non-ASCII character sneaks past the ASCII fallback.
        return style.dim(style.g.dash)
    share = free / total
    meter = bar(share, size, style)
    count = f"{free}/{total}"
    # The count is content and the unit is a label, so they are not the same
    # grey. The count also takes the bar's own ramp tone: the number and the
    # meter are one reading, and colouring only one of them splits it.
    label = style.heat(count, share) + (style.dim(f" {unit}") if unit else "")
    return f"{meter} {label}"


def sparkline(values: Sequence[float], style: Style | None = None) -> str:
    """A one-line distribution, for showing shape without a whole chart."""
    style = style or Style()
    if not values:
        return ""
    ramp = style.g.spark
    hi = max(values)
    if hi <= 0:
        return style.track(ramp[0] * len(values))
    # Height and colour both, from the same value. A spark column is one cell
    # wide, so height alone gives it eight distinguishable levels; the ramp
    # gives the short ones somewhere to be visible.
    steps = heat_steps(list(values))
    return "".join(
        style.tint(ramp[min(len(ramp) - 1, int(v / hi * (len(ramp) - 1)))], steps[i])
        for i, v in enumerate(values)
    )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def rule(title: str = "", style: Style | None = None, size: int | None = None) -> str:
    style = style or Style()
    g = style.g
    size = size or term_width()
    if not title:
        return style.dim(g.h * size)
    left = f"{g.h}{g.h} "
    # Clip the title: a rule that overflows is worse than a rule with a short
    # label, and the fill going to zero does not stop the line growing.
    room = size - width(left) - 1
    if room < 4:
        return style.dim(g.h * size)
    title = truncate(title, room, g.ellipsis)
    label = style.head(title)
    used = width(left) + width(title) + 1
    return style.dim(left) + label + " " + style.dim(g.h * max(0, size - used))


# The frame's gradient, as anchor colours it is interpolated between. Every one
# of them is a *light* colour, and that is the whole point.
#
# A frame that sweeps light-to-deep puts the highlight at the top-left like
# gloss on a card -- and puts the darkest end of the ramp at the bottom-right,
# where on a dark terminal it simply disappears. A gradient whose range leaves
# the visible band is not a gradient with a subtle end, it is one that is broken
# for half its length.
#
# So the sweep moves in *hue* and stays put in brightness. It used to move
# through light cyan and aqua, **which is the data ramp's own territory**: the
# border of a table was drawn in the colours of the numbers inside it, a cyan
# frame around a cyan column, and the chrome competed with the content it was
# supposed to contain. The sweep now runs periwinkle -> lilac -> light orchid,
# at least dE2000 21 from every step of the ramp and from `ok`, `warn` and
# `bad`. Nothing in any table is ever this pale, and now nothing in any table
# is this hue either, so a frame reads as a frame at a glance.
_FRAME_ANCHORS = ((195, 209, 246), (200, 194, 246), (215, 183, 234), (230, 190, 222))

#: The same sweep on the xterm-256 cube, held to the same rule: nothing below
#: the bright band, or the bottom border vanishes. The cube is thin on pale
#: violets, so this is three tones rather than ten -- which is what a frame
#: needs, the gradient being a texture and not a scale.
_FRAME_256 = (189, 189, 189, 189, 183, 183, 182, 182, 182, 182)

#: Sixteen colours, which is what ``TERM=screen`` and most tmux defaults
#: advertise, and the depth with no room to be clever. Bright variants only:
#: plain blue at this depth is a murky navy that disappears against a dark
#: background, and because the sweep runs diagonally that is exactly where the
#: bottom border lands. Two bright tones read as a deliberate two-tone frame.
#: Bright cyan is gone from the pair for the reason above -- at sixteen colours
#: it is also ramp step eight.
_FRAME_16 = (94, 95)

#: Steps to quantise the truecolor sweep into: fine enough that the bands are
#: invisible, coarse enough that runs of equal colour still group into one
#: escape sequence instead of one per column.
_FRAME_STEPS = 24


def _frame_ramp(style: Style) -> list[_Tone]:
    """Tones for the frame gradient, lightest first; empty when colour is off."""
    if not style.enabled:
        return []
    if style.depth >= 24:
        span = len(_FRAME_ANCHORS) - 1
        out: list[_Tone] = []
        for i in range(_FRAME_STEPS):
            scaled = (i / float(_FRAME_STEPS - 1)) * span
            low = min(span, int(scaled))
            high = min(span, low + 1)
            f = scaled - low
            rgb = tuple(
                int(round(_FRAME_ANCHORS[low][c]
                          + (_FRAME_ANCHORS[high][c] - _FRAME_ANCHORS[low][c]) * f))
                for c in range(3)
            )
            out.append((rgb, 0, 0))  # type: ignore[arg-type]
        return out
    if style.depth >= 8:
        return [((0, 0, 0), code, 0) for code in _FRAME_256]
    return [((0, 0, 0), 0, code) for code in _FRAME_16]


def panel(
    lines: Sequence[str],
    title: str = "",
    style: Style | None = None,
    size: int | None = None,
    role: str | None = None,
    shrink: bool = True,
) -> str:
    """A framed block, for content that should read as one unit.

    ``shrink`` sizes the frame to its widest line instead of stretching it to
    the window. A box ruled out to 100 columns around 60 columns of content
    reads as an empty room; sized to the content it reads as one object, which
    is the only reason to draw a frame at all.

    The border carries a **diagonal colour sweep**: hue advances with ``x + y``,
    so the lightest point is the top-left corner and the sweep travels round to
    the bottom-right the way a highlight falls across a glossy surface. It is
    drawn in runs of equal tone rather than per character, which costs about ten
    escape sequences per border instead of one per column.

    ``role`` forces a single flat palette colour instead, for a frame that has
    to mean something -- and it is deliberately not the default: the frame is
    chrome, and chrome that shouts a semantic colour competes with the numbers
    inside it.
    """
    style = style or Style()
    g = style.g
    size = size or term_width()
    if shrink and lines:
        needed = max(width(line) for line in lines) + 4
        if title:
            needed = max(needed, width(title) + 6)
        size = min(size, max(needed, MIN_WIDTH // 2))
    inner = size - 4
    # +1 for the title, which is now a content line rather than part of the edge.
    height = len(lines) + 2 + (1 if title else 0)
    ramp = [] if role is not None else _frame_ramp(style)

    def tone_at(x: int, y: int) -> str | None:
        """Hue for one frame cell, sweeping diagonally from the top-left."""
        if not ramp:
            return None
        across = x / float(size - 1) if size > 1 else 0.0
        down = y / float(height - 1) if height > 1 else 0.0
        at = 0.5 * across + 0.5 * down
        return style._sgr(ramp[min(len(ramp) - 1, max(0, int(at * len(ramp))))])

    def cell(text: str, x: int, y: int) -> str:
        """One border character, or a short run of them at a fixed x."""
        prefix = tone_at(x, y)
        if prefix is None:
            return style.paint(role or "dim", text)
        return f"{prefix}{text}\033[0m"

    def sweep(text: str, y: int, x0: int = 0) -> str:
        """A horizontal border run, grouping equal tones into one escape."""
        if not ramp:
            return style.paint(role or "dim", text)
        out: list[str] = []
        buffered: list[str] = []
        current: str | None = None
        for i, ch in enumerate(text):
            tone = tone_at(x0 + i, y)
            if current is not None and tone != current:
                out.append(f"{current}{''.join(buffered)}\033[0m")
                buffered = []
            current = tone
            buffered.append(ch)
        if buffered and current is not None:
            out.append(f"{current}{''.join(buffered)}\033[0m")
        return "".join(out)

    # The frame is unbroken, and the title lives inside it.
    #
    # A title used to be inlaid into the top edge -- `╭─ nodetop · slurm ────╮`
    # -- which cuts the border where the eye expects it to continue: "i hope the
    # ui is completely sealed, not broke by ' nodetop · slurm ───'". A box that
    # is a box everywhere is worth more than a label saving one line, and the
    # title reads better as content than as a gap in the chrome.
    def stretched(line: str) -> str:
        """A divider inside a frame spans the frame.

        :func:`_grid` rules to its own widest column, which is narrower than
        the frame whenever some other line -- the facts header -- is longer.
        A separator that stops four columns short of the border reads as a
        rendering fault rather than as a separator, and the frame only knows
        its final inner width here.
        """
        bare = _strip_ansi(line)
        body = bare.lstrip()
        if len(body) < 3 or set(body) != {g.h}:
            return line
        lead = len(bare) - len(body)
        prefix = line[: line.index(g.h)]
        tail = "\033[0m" if "\033" in prefix else ""
        return prefix + g.h * max(1, inner - lead) + tail

    out = [sweep(g.tl + g.h * (size - 2) + g.tr, 0)]
    if title:
        lines = [title, *lines]
    for i, line in enumerate(lines):
        # Truncate to the inner width: a content line longer than the frame
        # pushes the right border off the screen and breaks the box.
        fitted = pad(truncate(stretched(line), inner, g.ellipsis), inner)
        y = i + 1
        out.append(cell(g.v, 0, y) + " " + fitted + " " + cell(g.v, size - 1, y))
    out.append(sweep(g.bl + g.h * (size - 2) + g.br, height - 1))
    return "\n".join(out)


def section(
    title: str, style: Style | None = None, note: str = "", size: int | None = None
) -> str:
    """A Claude-Code-style section marker: an accent bullet and a bold title.

    The trailing note is clipped to whatever room is left, since it is a hint
    rather than content and must not be the thing that overflows the window.

    The *title* is clipped too, if it has to be.  It only ever needs to be on a
    narrow window with a long heading -- "the submit filter and the scheduler
    disagree" is 46 columns -- and that combination went unseen because the
    heading renders only when a submit filter contradicts its scheduler, which
    no width sweep had a fixture for.  Same omission as :func:`rule` had.
    """
    style = style or Style()
    size = size or term_width()
    bullet = f"{style.accent(style.g.bullet)} "
    room = size - width(bullet)
    if width(title) > room:
        title = truncate(title, max(1, room), style.g.ellipsis)
    line = bullet + style.head(title)
    if not note:
        return line
    room = size - width(line) - 2
    if room < 8:
        return line
    return line + "  " + style.dim(truncate(note, room, style.g.ellipsis))


def tree(
    items: Sequence[tuple[str, str]],
    style: Style | None = None,
    indent: str = "  ",
    size: int | None = None,
) -> str:
    """``├─ tag  detail`` lines, with the last branch closed.

    Reads as one grouped finding rather than a list of unrelated lines, which
    matters when four blockers apply to the same queue.
    """
    style = style or Style()
    g = style.g
    size = size or term_width()
    room = max(12, size - len(indent) - width(g.branch) - 1)
    out: list[str] = []
    for i, (tag, detail) in enumerate(items):
        stem = g.last if i == len(items) - 1 else g.branch
        # The tag is a label, not content: clip it rather than let it push the
        # line past the window edge.
        out.append(f"{indent}{style.dim(stem)} {truncate(tag, room, g.ellipsis)}")
        if detail:
            cont = "  " if i == len(items) - 1 else g.pipe
            out.append(
                wrap_indent(detail, indent=indent + style.dim(cont) + "  ", raw_prefix=True)
            )
    return "\n".join(out)


def badge(text: str, role: str, style: Style | None = None) -> str:
    """A short inverse-video label, for a verdict that must not be missed."""
    style = style or Style()
    if not style.enabled:
        return f"[{text}]"
    rgb, c256, c16 = _PALETTE[role]
    if style.depth >= 24:
        bg = f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    elif style.depth >= 8:
        bg = f"\033[48;5;{c256}m"
    else:
        bg = f"\033[{c16 + 10}m"
    return f"{bg}\033[30;1m {text} \033[0m"


def kv(
    pairs: Sequence[tuple[str, str]],
    style: Style | None = None,
    indent: str = "  ",
    size: int | None = None,
) -> str:
    """Aligned ``key  value`` lines, wrapping a value that does not fit.

    The continuation is indented to the value column so a wrapped value still
    reads as belonging to its key rather than as a new entry.
    """
    style = style or Style()
    if not pairs:
        return ""
    keysize = max(width(k) for k, _ in pairs)
    size = size or term_width()
    room = max(20, size - len(indent) - keysize - 2)
    out: list[str] = []
    for key, value in pairs:
        lead = f"{indent}{style.dim(pad(key, keysize))}  "
        if width(value) <= room or "\033" in value:
            # A value carrying its own styling (a gauge, a coloured verdict) is
            # laid out by whoever built it; re-wrapping would split an escape
            # sequence across lines.
            out.append(lead + value)
            continue
        import textwrap

        wrapped = textwrap.wrap(value, width=room,
                                break_on_hyphens=False) or [value]
        out.append(lead + wrapped[0])
        pad_to = " " * (len(indent) + keysize + 2)
        out.extend(pad_to + line for line in wrapped[1:])
    return "\n".join(out)


def _cell(text: str, i: int, sizes: Sequence[int], aligns: Sequence[str],
          have: int | None = None) -> str:
    """One padded cell -- except the last, when nothing follows it.

    **The trailing padding on a final left-aligned cell is the widest thing
    this module produces, and nobody can see it.** A column sized to
    `V100, RTX6000` pads the heading `gpu model` out by four spaces; the reader
    sees a heading.

    It stops being invisible the moment anything *measures* the table. `status`
    draws its rule at ``max(width(x) for x in out)``, so the rule came out 97
    columns for content that ended at 79, and the interactive frame was then
    held open to the rule -- eighteen columns of empty box at every level,
    traceable to spaces after a heading.

    Not fixed with ``rstrip``, which was tried and is wrong: a padded cell
    keeps its spaces INSIDE the styled run, so ``ESC[38;5;248mgpu
    moduleESC[0m`` loses them and the coloured twin does not. That makes the
    line a different width with colour on than off, which is the one property
    :class:`TestLayoutStability` exists to protect -- argparse and every table
    here lay out with `len()`. So the padding is never added instead.

    Only the last column, and only left-aligned: a right-aligned cell pads on
    the LEFT, which is what puts its digits under the ones above them.
    """
    if i == len(sizes) - 1 and aligns[i] == "left":
        return text
    return pad(text, sizes[i], aligns[i],
               **({} if have is None else {"have": have}))


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    aligns: Sequence[str] | None = None,
    style: Style | None = None,
    indent: str = "",
    limits: Sequence[int] | None = None,
    fit: bool = True,
    atomic: Sequence[bool] | None = None,
    size: int | None = None,
    keep: int = 2,
    drop_empty: bool = True,
    header_role: str = "head",
    header_paint: Sequence[Callable[[str], str] | None] | None = None,
    show_header: bool = True,
    underline: bool = True,
) -> str:
    """An aligned table, measured in display columns and fitted to the window.

    ``limits`` caps individual columns up front; ``fit`` then shrinks whatever
    still does not fit the terminal.  Cells are truncated with an ellipsis
    rather than wrapped, because a wrapped cell destroys the alignment that
    makes a table readable in the first place.

    Shrinking has a floor, so a wide table in a narrow window eventually
    cannot fit at all.  Past that point trailing columns are *dropped* -- the
    leading ``keep`` are treated as the identity of the row and never go -- and
    a closing line says how many went, because a table that silently loses
    columns reads as a table that had none.

    ``drop_empty`` removes a column that is blank in *every* row before any of
    that happens.  Kept distinct from the width-driven drop, and deliberately
    silent, because the two lose different things: a blank column carries no
    information, so removing it costs nothing and hands its width to the
    columns that do carry some.  ``nodes --gpu`` was spending 29 of 100
    columns on a REASON that was empty on all 91 rows.
    """
    style = style or Style()
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    if not body:
        return indent + style.dim("(nothing to show)")
    # Every cell's display width, measured ONCE and carried through.
    #
    # The three passes below -- cap to `limits`, size the columns, shrink to the
    # window -- each used to ask `width` for every cell again, and so did `pad`
    # at the end: four measurements per cell. On a 10,624-row listing that was
    # ~255,000 calls costing 86 ms of a 280 ms frame, for 85,000 distinct
    # answers. The array is updated wherever a cell is rewritten, which is the
    # only way it can go stale.
    wide = [[width(c) for c in r] for r in body]

    headers = list(headers)
    ncol = len(headers)
    aligns = list(aligns or ["left"] * ncol)
    aligns += ["left"] * (ncol - len(aligns))

    if limits:
        for r, w in zip(body, wide, strict=True):
            for i, cap in enumerate(limits[:ncol]):
                if cap and i < len(r) and w[i] > cap:
                    r[i] = truncate(r[i], cap, style.g.ellipsis)
                    w[i] = width(r[i])

    if drop_empty and body:
        # A column blank on every row. Never drop the leading `keep`: those
        # identify the row, and a blank identity column is a data problem the
        # reader should see rather than a layout one to tidy away.
        blank = [
            i for i in range(keep, ncol)
            if not any(_strip_ansi(r[i]).strip() for r in body if i < len(r))
        ]
        for i in reversed(blank):
            del headers[i]
            del aligns[i]
            for r, w in zip(body, wide, strict=True):
                if i < len(r):
                    del r[i]
                    del w[i]
        ncol -= len(blank)

    sizes = [width(h) for h in headers]
    for w in wide:
        for i in range(min(ncol, len(w))):
            if w[i] > sizes[i]:
                sizes[i] = w[i]

    # Shrink to the terminal rather than overflowing it. A table wider than the
    # window wraps, and a wrapped row destroys the column alignment that makes
    # a table worth using -- so the widest columns give ground first, down to a
    # floor that keeps each header legible.
    dropped: list[str] = []
    if fit:
        window = size or term_width()
        floors = [min(width(h), 6) if width(h) else 3 for h in headers]
        # **A column the caller calls atomic does not shrink at all.**
        #
        # The floor above is derived from the HEADER, so a column headed
        # `cores free` may be squeezed to six columns -- and six columns of
        # `1744/2000G` is `1744/…`, a fraction with its denominator eaten.
        # That is not a smaller version of the fact, it is a different and
        # false one, and the reader cannot tell which digits went.
        #
        # Only the caller knows which cells are indivisible. A list of
        # accelerator models shortens usefully; a ratio, a duration and a node
        # shape do not. Marked ones floor at their own content width, so the
        # loop below spends the squeeze on the columns that can take it and
        # the drop-from-the-right path handles a window that genuinely cannot
        # hold them all -- which at least says `N more columns (widen, or
        # --json)` instead of mangling four numbers in silence.
        if atomic:
            for i, fixed in enumerate(atomic[:ncol]):
                if fixed:
                    floors[i] = max(floors[i], sizes[i])
        # Drop from the right while even the floors cannot fit, never going
        # below `keep` columns -- the row has to stay identifiable.
        while ncol > max(1, keep):
            if sum(floors[:ncol]) + 2 * (ncol - 1) + len(indent) <= window:
                break
            ncol -= 1
            dropped.append(headers[ncol])
        headers = headers[:ncol]
        aligns = aligns[:ncol]
        sizes = sizes[:ncol]
        floors = floors[:ncol]
        body = [r[:ncol] for r in body]
        wide = [w[:ncol] for w in wide]
        available = window - len(indent) - 2 * (ncol - 1)
        guard = 0
        while sum(sizes) > available and guard < 4096:
            slack = [sizes[i] - floors[i] for i in range(ncol)]
            if max(slack) <= 0:
                break
            sizes[slack.index(max(slack))] -= 1
            guard += 1
        for r, w in zip(body, wide, strict=True):
            for i in range(min(ncol, len(r))):
                if w[i] > sizes[i]:
                    r[i] = truncate(r[i], sizes[i], style.g.ellipsis)
                    w[i] = width(r[i])
        # Headers shrink with their columns too. Truncating only the data
        # leaves the header row wider than every row beneath it, which is the
        # one line guaranteed to overflow.
        headers = [
            truncate(h, sizes[i], style.g.ellipsis) if width(h) > sizes[i] else h
            for i, h in enumerate(headers)
        ]

    lines = []
    if show_header:
        # Dispatched through the named Style method, not paint(): "head" is a
        # bold-plus-colour combination rather than a palette entry, so
        # paint("head", ...) raises KeyError.
        wear = getattr(style, header_role, None) or (
            lambda t: style.paint(header_role, t))

        def dress(i: int, text: str) -> str:
            """Paint one header cell.

            ``header_paint`` gives a column its own hue, which is how a table
            with two `free` columns tells the reader which resource each one
            belongs to before they read a word. Padding happens first, so the
            colour covers the cell and not just the label.
            """
            if header_paint is not None and i < len(header_paint):
                own = header_paint[i]
                if own is not None:
                    return own(text)
            return wear(text)

        lines.append(indent + "  ".join(
            dress(i, _cell(h, i, sizes, aligns)) for i, h in enumerate(headers)
        ))
        if underline:
            lines.append(
                indent + "  ".join(style.dim(style.g.h * sizes[i])
                                   for i in range(ncol)))
    for r, w in zip(body, wide, strict=True):
        cells = [
            _cell(r[i] if i < len(r) else "", i, sizes, aligns,
                  have=w[i] if i < len(w) else 0)
            for i in range(ncol)
        ]
        lines.append((indent + "  ".join(cells)).rstrip())
    if dropped:
        lines.append(indent + style.dim(truncate(
            f"{style.g.ellipsis} {plural(len(dropped), 'more column')} "
            f"(widen, or --json)",
            max(4, (size or term_width()) - len(indent)), style.g.ellipsis,
        )))
    return "\n".join(lines)


def flow(
    items: Sequence[str],
    style: Style | None = None,
    size: int | None = None,
    indent: str = "  ",
    sep: str | None = None,
) -> str:
    """Join items with a separator, wrapping when the next one will not fit.

    For legends and hint lines: unlike :func:`columns` it does not pad to a
    common width, so short items stay short.  Unlike ``sep.join`` it respects
    the window -- which is the whole reason it exists, a hand-joined legend
    having overflowed at 104 columns.
    """
    style = style or Style()
    if not items:
        return ""
    size = size or term_width()
    joiner = sep if sep is not None else f"  {style.dim(style.g.sep)}  "
    room = max(4, size - width(indent))
    lines: list[str] = []
    current = ""
    for item in items:
        # An item too wide to sit on a line of its own has to be clipped;
        # starting a fresh line for it would still overflow.
        if width(item) > room:
            item = truncate(item, room, style.g.ellipsis)
        candidate = item if not current else current + joiner + item
        if current and width(indent) + width(candidate) > size:
            lines.append(indent + current)
            current = item
        else:
            current = candidate
    lines.append(indent + current)
    return "\n".join(lines)


def columns(items: Sequence[str], style: Style | None = None, size: int | None = None,
            indent: str = "  ") -> str:
    """Lay short strings out in as many columns as fit."""
    style = style or Style()
    if not items:
        return ""
    size = size or term_width()
    cell = max(width(i) for i in items) + 2
    per_row = max(1, (size - len(indent)) // cell)
    lines = []
    for i in range(0, len(items), per_row):
        lines.append(indent + "".join(pad(x, cell) for x in items[i : i + per_row]).rstrip())
    return "\n".join(lines)


def wrap_indent(
    text: str,
    indent: str = "      ",
    size: int | None = None,
    raw_prefix: bool = False,
    first: str | None = None,
) -> str:
    """Wrap prose to the terminal, keeping a hanging indent.

    ``raw_prefix`` allows a coloured indent: textwrap measures the escape
    sequence as visible text, so the indent is applied afterwards instead.

    ``first`` sets a different prefix for the first line, which is what a
    bullet needs: without it the continuation lines align under the bullet
    glyph rather than under the sentence, and the list stops looking like a
    list.  The hanging indent is padded to match its width automatically --
    requiring the caller to keep the two in step is a trap, because the ASCII
    glyph twin is a different width from the Unicode one (``--`` against
    ``—``) and only one of the two spellings looks right.
    """
    import textwrap

    # Prose, so `PROSE_WIDTH` and not the window. A sentence set to the full
    # width of a wide terminal is harder to read than the same sentence at
    # ninety columns, and tables taking the whole window is no reason for
    # paragraphs to. See `PROSE_WIDTH`.
    size = size or min(term_width(), PROSE_WIDTH)
    if first is not None:
        # Pad whichever is narrower so the block lines up in both glyph sets.
        pad_to = max(width(first), width(indent))
        first = first + " " * (pad_to - width(first))
        indent = indent + " " * (pad_to - width(indent))
    # break_on_hyphens=False is load-bearing: the default splits "--test-only"
    # into "--test-" / "only", which reads as a different flag entirely, and
    # does the same to hyphenated node names like "gn-0001".
    if raw_prefix:
        visible = width(indent)
        wrapped = textwrap.fill(
            text, width=max(20, size - visible), break_on_hyphens=False
        )
        lines = wrapped.splitlines()
        head = first if first is not None else indent
        return "\n".join(
            (head if i == 0 else indent) + line for i, line in enumerate(lines)
        )
    return textwrap.fill(
        text, width=size,
        initial_indent=first if first is not None else indent,
        subsequent_indent=indent, break_on_hyphens=False,
    )


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------
#
# argparse lays its help out in columns using ``len()``, so colouring the
# strings it is handed throws every column off by the width of the escape
# sequences.  Colour is applied to the finished block instead: escapes occupy no
# columns, so the layout argparse already computed survives untouched.
#
# Four roles, and deliberately no more -- a help screen wearing a dozen colours
# is harder to read than one wearing none:
#
#     info    what you type          flags, and the flags inside prose
#     warn    what you substitute    NAME, N, FILE
#     head    structure              section headings, the program name
#     dim     context                defaults, the example explanations
#
# ``warn``'s amber is borrowed rather than meaning "warning" here.  A help
# screen contains no warnings, so there is nothing for it to be confused with,
# and adding two palette roles used by one function would be worse.
_HELP_HEADING = re.compile(r"^([a-zA-Z][a-zA-Z ]*:)\s*$")
_HELP_SPLIT = re.compile(r"^(\s+)(\S.*?)(\s\s+|$)(.*)$")

# One alternation rather than a pass per role, because ``re.sub`` never overlaps
# its own matches: painting flags and then backticked spans separately nests a
# reset inside the span and drops the tail of it back to plain.  Leftmost-first
# also gives the precedence that is wanted -- a flag inside a quoted command
# belongs to the command.
_HELP_SPANS = re.compile(
    r"(?P<default>\(default:[^)]*\))"
    r"|(?P<code>`[^`]+`)"
    r"|(?P<flag>(?<![\w-])--?[a-zA-Z][\w-]*)"
    r"|(?P<caps>\b[A-Z][A-Z_]{2,}\b)"
)
_HELP_TONES = {"default": "dim", "code": "info", "flag": "info",
               "caps": "accent"}
_HELP_METAVAR = re.compile(r"^[A-Z][A-Z_]*$")


def _is_invocation(line: str) -> bool:
    """argparse indents an invocation and wraps its help text much further in.

    Two columns of indent for a flag, four for a sub-command inside the
    positional list.  The indent alone tells an invocation from continued help
    prose, which matters for a wrapped help line that happens to begin with a
    flag.
    """
    return bool(line.strip()) and (len(line) - len(line.lstrip())) in (2, 4)


def _metavars(text: str) -> frozenset[str]:
    """The placeholders this parser actually declares.

    The placeholder tone has to mean "you substitute this", so it is spent only
    on words the invocation column proves are placeholders.  A blanket all-caps rule paints
    ``GPU`` and ``QOS`` -- prose nouns, and this tool's prose is full of them --
    the same colour as ``FILE``, which drains the colour of its meaning.
    """
    found = set()
    for line in text.split("\n"):
        m = _HELP_SPLIT.match(line) if _is_invocation(line) else None
        if not m:
            continue
        for token in re.split(r"[\s,=]+", m.group(2)):
            if token and not token.startswith("-") and _HELP_METAVAR.match(token):
                found.add(token)
    return frozenset(found)


def colorize_help(text: str, style: Style | None = None) -> str:
    """Paint an already-formatted argparse help block."""
    style = style or Style()
    if not style.enabled:
        return text
    names = _metavars(text)
    out: list[str] = []
    section_name = ""
    prog = ""
    for line in text.split("\n"):
        head = _HELP_HEADING.match(line)
        if head:
            section_name = head.group(1)[:-1]
            out.append(style.head(head.group(1)))
        elif line.startswith("usage:"):
            label, _, rest = line.partition(":")
            prog = rest.split()[0] if rest.split() else ""
            out.append(style.dim(label + ":") + _paint_words(rest, style, first=True))
        elif not line.strip():
            out.append(line)
        elif section_name == "examples":
            # A command line, or the note that closes the section? Both are
            # indented, so the indent cannot decide it -- the note came out
            # painted as a shell command, every word of it a substitutable
            # value. What separates them is that an example invokes the program:
            # its first word is the program's own name.
            if prog and line.lstrip().startswith(prog):
                out.append(_paint_example(line, style, names))
            else:
                out.append(_paint_prose(line, style, names))
        elif section_name and _is_invocation(line):
            out.append(_paint_invocation(line, style, names))
        else:
            out.append(_paint_prose(line, style, names))
    return "\n".join(out)


def _paint_words(text: str, style: Style, first: bool = False) -> str:
    """A shell command line: literal words in one tone, substitutables in the other.

    ``first`` marks the leading word as the program name -- structure, not
    something you would change -- and it is the first *visible* token, not token
    index 1: a leading space makes those different, and that is how the
    program's own name came out painted as a placeholder.

    A bare word takes the placeholder tone only when it follows a flag, because
    that is what makes it an argument to one.  A sub-command follows the program name instead, so it
    stays the literal-text tone and matches the same word where it appears in the
    invocation list -- ``queues`` painted blue in one place and amber in the
    other is worse than either choice on its own.  (A flag that takes no value
    would fool this; none of the examples has a bare literal after one.)

    The placeholder tone is `accent`, and it used to be `warn`.  Amber is this
    tool's warning colour -- a degraded node, a queue that will not take your
    job -- and spending it on ``NAME`` in ``--backend NAME`` put the loudest
    thing in the palette on a piece of syntax.  A help page full of amber is a
    help page that has taught the reader amber means nothing.
    """
    out = []
    seen = False
    after_flag = False
    for tok in re.split(r"(\s+)", text):
        if not tok or not tok.strip():
            out.append(tok)
            continue
        if not seen and first:
            out.append(style.paint("text", tok, bold=True))
        elif tok.lstrip("[").startswith("-"):
            out.append(style.paint("info", tok))
            after_flag = True
            seen = True
            continue
        elif after_flag:
            out.append(style.paint("accent", tok))
        else:
            out.append(style.paint("info", tok))
        seen = True
        after_flag = False
    return "".join(out)


def _paint_example(line: str, style: Style, names: frozenset[str] = frozenset()) -> str:
    m = _HELP_SPLIT.match(line)
    if not m:
        return _paint_prose(line, style, names)
    lead, command, gap, note = m.groups()
    return lead + _paint_words(command, style, first=True) + gap + style.dim(note)


def _paint_invocation(line: str, style: Style,
                      names: frozenset[str] = frozenset()) -> str:
    m = _HELP_SPLIT.match(line)
    if not m:
        return line
    lead, invocation, gap, rest = m.groups()
    parts = []
    for tok in re.split(r"([\s,]+)", invocation):
        if not tok:
            continue
        if set(tok) <= {",", " "}:
            # The separator between two spellings of the same option. Dim, not
            # the placeholder tone: it is punctuation, and painting it as a
            # value is what made "-h, --help" look like a flag taking an
            # argument.
            parts.append(style.dim(tok))
        elif tok.startswith("-"):
            parts.append(style.paint("info", tok))
        elif _HELP_METAVAR.match(tok):
            parts.append(style.paint("accent", tok))  # you substitute this
        else:
            # A sub-command, or an alias list beside one. Literal text you
            # type, so it wears the same tone as a flag.
            parts.append(style.paint("info", tok))
    return lead + "".join(parts) + gap + _paint_prose(rest, style, names)


def _paint_prose(text: str, style: Style,
                 names: frozenset[str] = frozenset()) -> str:
    if not text:
        return text

    def paint(match: re.Match) -> str:
        word = match.group(0)
        # An all-caps word earns the placeholder tone by being a declared
        # placeholder, or by having the shape of an environment variable
        # (NO_COLOR).
        if match.lastgroup == "caps" and word not in names and "_" not in word:
            return word
        return style.paint(_HELP_TONES[match.lastgroup or "flag"], word)

    return _HELP_SPANS.sub(paint, text)
