"""CIELAB, WCAG contrast and CIEDE2000, for tests that judge a palette.

The tool ships no third-party dependencies on purpose -- it has to run on a
login node during an outage -- so the colour science a palette test needs lives
here rather than coming from ``colormath``.  It is test-only: nothing in
``nodetop`` imports it.

Why a real colour difference and not ``abs(r1 - r2) + ...``: the sRGB cube is
nothing like perceptually uniform.  ``(0, 255, 175)`` and ``(95, 255, 175)`` are
95 apart in the red channel and *the same green* to look at, while ``(0, 130,
205)`` and ``(0, 155, 225)`` are a quarter of that distance and visibly
different.  A test that measured channels would have passed the ramp this one
was written to reject.
"""

from __future__ import annotations

import math

#: A dark terminal background.  Not ``#000000``: a pure-black surface makes
#: every hue vibrate against it, and the terminals people actually use --
#: iTerm2, Windows Terminal, GNOME, VS Code -- all default to a near-black in
#: this neighbourhood.  Contrast figures are quoted against it.
BACKGROUND = (13, 16, 22)

Rgb = tuple[int, int, int]


def _linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Rgb) -> float:
    """WCAG relative luminance, 0..1."""
    r, g, b = (_linear(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: Rgb, b: Rgb) -> float:
    """WCAG contrast ratio, 1..21.  AA wants 4.5 for text."""
    high, low = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def lab(rgb: Rgb) -> tuple[float, float, float]:
    """CIELAB under D65.  ``L*`` is perceived lightness, 0..100."""
    r, g, b = (_linear(v) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lightness(rgb: Rgb) -> float:
    """``L*`` alone -- the channel the eye reads as "more" or "less"."""
    return lab(rgb)[0]


def delta_e(one: Rgb, two: Rgb) -> float:
    """CIEDE2000.  Roughly: 1 is a just-noticeable difference between two large
    patches side by side, and small text in a table needs several times that
    before two cells read as different colours."""
    l1, a1, b1 = lab(one)
    l2, a2, b2 = lab(two)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar**7 / (cbar**7 + 25**7))) if cbar else 0.0
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0

    dlp, dcp = l2 - l1, c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dhp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2)

    lbar, cbarp = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbar = (h1p + h2p + 360) / 2
    else:
        hbar = (h1p + h2p - 360) / 2

    t = (1
         - 0.17 * math.cos(math.radians(hbar - 30))
         + 0.24 * math.cos(math.radians(2 * hbar))
         + 0.32 * math.cos(math.radians(3 * hbar + 6))
         - 0.20 * math.cos(math.radians(4 * hbar - 63)))
    sl = 1 + (0.015 * (lbar - 50) ** 2) / math.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cbarp
    sh = 1 + 0.015 * cbarp * t
    rt = (-math.sin(math.radians(2 * 30 * math.exp(-(((hbar - 275) / 25) ** 2))))
          * (2 * math.sqrt(cbarp**7 / (cbarp**7 + 25**7)) if cbarp else 0.0))
    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2
                     + rt * (dcp / sc) * (dhp / sh))


def _xterm256() -> list[Rgb]:
    palette: list[Rgb] = [
        (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
        (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
        (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ]
    levels = (0, 95, 135, 175, 215, 255)
    palette += [(r, g, b) for r in levels for g in levels for b in levels]
    palette += [(8 + i * 10,) * 3 for i in range(24)]
    return palette


#: What an xterm-compatible terminal actually draws for ``ESC [ 38;5;N m``.
#: The 256-colour column of the palette is checked against this rather than
#: against the truecolor tone it approximates, because the index is what the
#: reader sees on that terminal.
XTERM256 = _xterm256()
