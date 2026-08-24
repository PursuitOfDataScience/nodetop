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

import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "Glyphs",
    "sanitize",
    "colorize_help",
    "MIN_WIDTH",
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

MAX_WIDTH = 100


# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------
def _strip_ansi(text: str) -> str:
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


def width(text: str) -> int:
    """Columns this string occupies in a terminal.

    Ignores ANSI escapes, counts East-Asian wide and fullwidth characters as
    two columns, and counts zero-width combining marks as none.  Using
    ``len()`` instead is what makes a coloured or non-Latin table drift.
    """
    total = 0
    for ch in _strip_ansi(text):
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


def pad(text: str, size: int, align: str = "left") -> str:
    gap = max(0, size - width(text))
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
            arrow="->", sep="-", ellipsis="...", dash="--",
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
#: Picked so the same intent survives all three depths rather than only the
#: richest one.
_PALETTE: dict[str, tuple[tuple[int, int, int], int, int]] = {
    "accent": ((217, 119, 87), 173, 33),
    "ok": ((110, 205, 130), 114, 32),
    "warn": ((225, 175, 70), 214, 33),
    "bad": ((235, 110, 105), 203, 31),
    "info": ((130, 170, 225), 111, 34),
    "dim": ((128, 132, 140), 244, 90),
    "text": ((215, 215, 215), 252, 37),
    # Secondary *content* -- a real measurement that is not the one the view
    # was ranked by. Distinct from "dim", which means context rather than
    # content: a snapshot age, a caveat, a hint. A free-core count is content
    # even when the bar beside it is what the eye goes to first, and painting
    # the two the same grey is what makes a numeric column read as furniture.
    "muted": ((170, 173, 180), 247, 37),
    # The unfilled part of a meter. A reference mark for "all of it", not
    # content at all, so it drops below "dim" to a near-background grey: it
    # should frame the bar without competing with it.
    "track": ((62, 65, 70), 238, 90),
}


# ---------------------------------------------------------------------------
# heat: magnitude as colour
# ---------------------------------------------------------------------------
#
# A meter's fill is coloured by the size of what it measures, on one ordered
# ramp from deep blue through cyan and green to amber.
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
# Twelve steps, not more.  The xterm cube has little perceptual room between
# 5fff5f and 87ff5f, so a finer ramp spends several of its steps inside one
# green and produces exactly the "these rows look the same" complaint it was
# meant to fix.
#
# Each step has a darker twin, used for bar *fill*.  A bar is a slab and text
# is a line: the colour that reads as bright in a number reads as shouting
# across eighteen filled cells, and ten shouting bars are a wall.  Terminals
# have no alpha channel, so the wash is a genuinely darker colour of the same
# hue -- which is what compositing that hue at ~60% over a dark background
# would have produced anyway.
_Tone = tuple[tuple[int, int, int], int, int]

#: Text tones, least first. **The warm end was wrong and has been removed.**
#:
#: The ramp used to run blue -> cyan -> green -> yellow -> amber, on the
#: reasoning that any ordered sweep reads as a scale. It does not, because these
#: numbers are *availability*: the top of the ramp is the emptiest node, and
#: amber is read as heat -- "why use the orange colour to denote an unoccupied
#: cpu?" A fully idle machine drew the most alarming colour on the screen.
#:
#: So the sweep ends in green, which everything from a traffic light to a disk
#: gauge already agrees means "go", and starts in the deep blue that reads as
#: "nearly nothing". Amber and gold are gone entirely -- they are back to being
#: warning colours, which is what a reader assumes they are.
_RAMP: tuple[_Tone, ...] = (
    ((0, 130, 205), 32, 34),     # blue        -- least free
    ((0, 155, 225), 39, 94),     # bright blue
    ((0, 175, 255), 39, 94),     # azure
    ((0, 215, 255), 45, 94),     # sky
    ((0, 255, 255), 51, 36),     # cyan
    ((0, 255, 215), 50, 96),     # turquoise
    ((0, 255, 175), 49, 96),     # spring
    ((95, 255, 175), 85, 96),    # aquamarine
    ((95, 255, 135), 84, 92),    # mint
    ((95, 255, 95), 83, 92),     # light green
    ((0, 255, 0), 46, 92),       # green
    ((0, 215, 95), 41, 32),      # deep green  -- most free
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
    ((0, 90, 145), 24, 34),      # blue
    ((0, 110, 165), 25, 94),
    ((0, 135, 175), 31, 94),
    ((0, 135, 215), 32, 94),
    ((0, 175, 175), 37, 36),
    ((0, 175, 155), 36, 96),
    ((0, 175, 135), 36, 96),
    ((0, 175, 115), 35, 96),
    ((0, 175, 95), 35, 92),
    ((0, 155, 75), 29, 92),
    ((0, 135, 0), 28, 92),
    ((0, 115, 55), 22, 32),      # deep green
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

    @property
    def enabled(self) -> bool:
        return self.depth > 0

    # -- primitives ---------------------------------------------------------
    def _sgr(self, tone: _Tone) -> str:
        """One ``(rgb, 256, 16)`` tone as the escape this terminal can show."""
        rgb, c256, c16 = tone
        if self.depth >= 24:
            return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        if self.depth >= 8:
            return f"\033[38;5;{c256}m"
        return f"\033[{c16}m"

    def _fg(self, role: str) -> str:
        return self._sgr(_PALETTE[role])

    def paint(self, role: str, text: str, bold: bool = False) -> str:
        if not self.enabled or not text:
            return text
        prefix = ("\033[1m" if bold else "") + self._fg(role)
        return f"{prefix}{text}\033[0m"

    def tint(self, text: str, step: int, fill: bool = False, bold: bool = False) -> str:
        """Paint with ramp step ``step``; ``fill`` picks the darker twin.

        The one way call sites reach the heat ramp, so a row's number and the
        bar beside it are guaranteed to be the same hue -- which is the whole
        point of colouring either of them.
        """
        if not self.enabled or not text:
            return text
        ramp = _WASH if fill else _RAMP
        tone = ramp[max(0, min(RAMP_STEPS - 1, step))]
        prefix = ("\033[1m" if bold else "") + self._sgr(tone)
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


def term_width(cap: int = MAX_WIDTH) -> int:
    """Usable width, clamped to a range the layout can actually work in."""
    return max(MIN_WIDTH, min(shutil.get_terminal_size((cap, 24)).columns, cap))


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
    lines = shutil.get_terminal_size((MAX_WIDTH, 24)).lines
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
    """
    style = style or Style()
    g = style.g
    fraction = max(0.0, min(1.0, fraction))
    tone = heat_step(fraction) if step is None else step

    def painted(fill: str, trough: str) -> str:
        if role is not None:
            return style.paint(role, fill + trough)
        return style.tint(fill, tone, fill=True) + style.track(trough)

    if not g.unicode:
        filled = int(round(fraction * size))
        return painted(g.blocks * filled, g.empty * (size - filled))

    total_eighths = int(round(fraction * size * 8))
    full, remainder = divmod(total_eighths, 8)
    fill = g.blocks[-1] * full
    if remainder:
        fill += g.blocks[remainder - 1]
    return painted(fill, g.empty * max(0, size - width(fill)))


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
# So the sweep moves in *hue* and stays put in brightness: light cyan through
# aqua and periwinkle to light violet. Every step is legible against black,
# none is legible as data -- nothing in any table is ever this pale -- and the
# bottom border has the same weight as the top.
_FRAME_ANCHORS = ((140, 233, 255), (94, 234, 212), (129, 199, 255), (167, 160, 255))

#: The same sweep on the xterm-256 cube, held to the same rule: nothing below
#: the bright band, or the bottom border vanishes.
_FRAME_256 = (123, 87, 80, 74, 75, 111, 147, 141, 177, 183)

#: Sixteen colours, which is what ``TERM=screen`` and most tmux defaults
#: advertise, and the depth with no room to be clever. Bright variants only:
#: plain blue at this depth is a murky navy that disappears against a dark
#: background, and because the sweep runs diagonally that is exactly where the
#: bottom border lands. Two bright tones read as a deliberate two-tone frame.
_FRAME_16 = (96, 94)

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


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    aligns: Sequence[str] | None = None,
    style: Style | None = None,
    indent: str = "",
    limits: Sequence[int] | None = None,
    fit: bool = True,
    size: int | None = None,
    keep: int = 2,
    drop_empty: bool = True,
    header_role: str = "head",
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

    headers = list(headers)
    ncol = len(headers)
    aligns = list(aligns or ["left"] * ncol)
    aligns += ["left"] * (ncol - len(aligns))

    if limits:
        for r in body:
            for i, cap in enumerate(limits[:ncol]):
                if cap and i < len(r):
                    r[i] = truncate(r[i], cap, style.g.ellipsis)

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
            for r in body:
                if i < len(r):
                    del r[i]
        ncol -= len(blank)

    sizes = [width(h) for h in headers]
    for r in body:
        for i in range(min(ncol, len(r))):
            sizes[i] = max(sizes[i], width(r[i]))

    # Shrink to the terminal rather than overflowing it. A table wider than the
    # window wraps, and a wrapped row destroys the column alignment that makes
    # a table worth using -- so the widest columns give ground first, down to a
    # floor that keeps each header legible.
    dropped: list[str] = []
    if fit:
        window = size or term_width()
        floors = [min(width(h), 6) if width(h) else 3 for h in headers]
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
        available = window - len(indent) - 2 * (ncol - 1)
        guard = 0
        while sum(sizes) > available and guard < 4096:
            slack = [sizes[i] - floors[i] for i in range(ncol)]
            if max(slack) <= 0:
                break
            sizes[slack.index(max(slack))] -= 1
            guard += 1
        for r in body:
            for i in range(min(ncol, len(r))):
                if width(r[i]) > sizes[i]:
                    r[i] = truncate(r[i], sizes[i], style.g.ellipsis)
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
        lines.append(indent + "  ".join(
            wear(pad(h, sizes[i], aligns[i])) for i, h in enumerate(headers)
        ))
        if underline:
            lines.append(
                indent + "  ".join(style.dim(style.g.h * sizes[i])
                                   for i in range(ncol)))
    for r in body:
        cells = [
            pad(r[i] if i < len(r) else "", sizes[i], aligns[i]) for i in range(ncol)
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

    size = size or term_width()
    if first is not None:
        # Pad whichever is narrower so the block lines up in both glyph sets.
        pad_to = max(width(first), width(indent))
        first = first + " " * (pad_to - width(first))
        indent = indent + " " * (pad_to - width(indent))
    # break_on_hyphens=False is load-bearing: the default splits "--test-only"
    # into "--test-" / "only", which reads as a different flag entirely, and
    # does the same to hyphenated node names like "beagle3-0001".
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
_HELP_TONES = {"default": "dim", "code": "info", "flag": "info", "caps": "warn"}
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

    Amber has to mean "you substitute this", so it is spent only on words the
    invocation column proves are placeholders.  A blanket all-caps rule paints
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

    A bare word is amber only when it follows a flag, because that is what makes
    it an argument to one.  A sub-command follows the program name instead, so it
    stays the literal-text tone and matches the same word where it appears in the
    invocation list -- ``queues`` painted blue in one place and amber in the
    other is worse than either choice on its own.  (A flag that takes no value
    would fool this; none of the examples has a bare literal after one.)
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
            out.append(style.paint("warn", tok))
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
            # amber: it is punctuation, and painting it as a value is what made
            # "-h, --help" look like a flag taking an argument.
            parts.append(style.dim(tok))
        elif tok.startswith("-"):
            parts.append(style.paint("info", tok))
        elif _HELP_METAVAR.match(tok):
            parts.append(style.paint("warn", tok))  # you substitute this
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
        # An all-caps word earns amber by being a declared placeholder, or by
        # having the shape of an environment variable (NO_COLOR).
        if match.lastgroup == "caps" and word not in names and "_" not in word:
            return word
        return style.paint(_HELP_TONES[match.lastgroup or "flag"], word)

    return _HELP_SPANS.sub(paint, text)
