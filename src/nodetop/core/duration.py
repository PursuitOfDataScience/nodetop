"""Walltime parsing that is explicit about an ambiguity every scheduler has.

The same string means different things to different batch systems, and the
disagreement is not obscure -- it is the most common unit in the interface:

* a bare number is **minutes** to Slurm (``--time=60``) and LSF (``-W 60``),
* PBS insists on ``HH:MM:SS``, so a bare number is not accepted at all,
* Kubernetes has no walltime concept; a deadline is seconds.

And within one system, two colon-separated fields flip meaning depending on
whether a day part is present: Slurm reads ``2:00`` as ``MM:SS`` (two minutes)
but ``1-2:00`` as ``D-HH:MM``.  Reading ``2:00`` as two hours is a 60x error.

:func:`parse_duration` therefore accepts the unambiguous forms plus explicit
suffixes (``90m``, ``2h``, ``1d12h``), and documents the one convention it has
to pick: a bare number is minutes, matching Slurm and LSF.  Backends that must
parse a *scheduler's own output* use their own exact parser instead, because
there the grammar is fixed by the system rather than chosen by us.
"""

from __future__ import annotations

import re
from datetime import datetime

__all__ = ["format_duration", "format_wait", "parse_duration", "parse_timestamp"]

_SENTINELS = frozenset(
    {"unlimited", "infinite", "infinity", "none", "n/a", "na", "(null)", "", "-", "0"}
)

#: 90m / 2h / 1d12h / 36h30m -- unambiguous because the unit is written down.
_SUFFIXED = re.compile(
    r"^(?:(?P<d>\d+)\s*d)?\s*(?:(?P<h>\d+)\s*h)?\s*"
    r"(?:(?P<m>\d+)\s*m(?:in)?)?\s*(?:(?P<s>\d+)\s*s)?$",
    re.IGNORECASE,
)
_COLON = re.compile(r"^(?:(?P<days>\d+)-)?(?P<rest>\d+(?::\d+)*)$")


def parse_duration(text: str | int | None) -> int | None:
    """Parse a walltime to seconds; ``None`` means no limit.

    Accepted, in order of precedence:

    ==========================  ==========================  =============
    form                        reading                     example
    ==========================  ==========================  =============
    ``<int>`` seconds           already seconds             ``3600``
    ``NdNhNmNs``                explicit units              ``1d12h``
    ``D-HH[:MM[:SS]]``          days first                  ``2-00:00:00``
    ``HH:MM:SS``                three fields                ``2:00:00``
    ``MM:SS``                   two fields                  ``2:00`` = 2 min
    ``MM``                      bare number is **minutes**  ``60`` = 1 hour
    ==========================  ==========================  =============
    """
    if text is None:
        return None
    if isinstance(text, int):
        # An int is taken as seconds: it came from code, not a command line,
        # so there is no unit convention to guess at.
        return text if text > 0 else None
    t = text.strip()
    if t.lower() in _SENTINELS:
        return None

    if not t[0].isdigit():
        return None

    # Suffixed form first: it is the only one that cannot be misread.
    if re.search(r"[dhms]", t, re.IGNORECASE):
        m = _SUFFIXED.match(t)
        if m and any(m.group(g) for g in ("d", "h", "m", "s")):
            return (
                int(m.group("d") or 0) * 86400
                + int(m.group("h") or 0) * 3600
                + int(m.group("m") or 0) * 60
                + int(m.group("s") or 0)
            )
        return None

    m = _COLON.match(t)
    if not m:
        return None
    days = int(m.group("days") or 0)
    fields = [int(x) for x in m.group("rest").split(":")]
    has_days = m.group("days") is not None

    if has_days:
        units = [3600, 60, 1][: len(fields)]      # D-HH[:MM[:SS]]
    elif len(fields) == 1:
        units = [60]                              # bare number is minutes
    elif len(fields) == 2:
        units = [60, 1]                           # MM:SS
    elif len(fields) == 3:
        units = [3600, 60, 1]                     # HH:MM:SS
    else:
        return None
    if len(fields) > len(units):
        return None
    return days * 86400 + sum(v * u for v, u in zip(fields, units, strict=True))


def format_duration(seconds: int | None) -> str:
    """Render seconds as ``D-HH:MM:SS`` / ``H:MM:SS``."""
    if seconds is None:
        return "unlimited"
    if seconds < 0:
        return "0:00:00"
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours}:{mins:02d}:{secs:02d}"


def format_wait(seconds: float | None) -> str:
    """Human-scale wait for a report column: ``now``, ``3h 20m``, ``2d 4h``.

    A negative interval is not "now".  It means the moment the scheduler
    predicted has already passed and the resource is still held -- a job
    overrunning its walltime, or a stale reading.  Reporting that as available
    sends someone at a node that is not free, so it is named instead.  A minute
    either side is treated as now, since that is clock jitter.
    """
    if seconds is None:
        return "?"
    if seconds < -60:
        return "overdue"
    if seconds <= 60:
        return "now"
    mins = int(seconds // 60)
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def format_age(seconds: float | None, tolerance: int = 60) -> str | None:
    """Elapsed magnitude for a past instant: ``3h 20m``, ``36d``, ``<1m``.

    ``None`` when the instant is *ahead* of the clock by more than
    ``tolerance`` -- there is no honest age to print for that, and the caller
    should say so rather than render a duration.

    Separate from :func:`format_wait`, which is about the future.  Reusing that
    one for an age prints "now ago" for anything under a minute and "overdue
    ago" for a recording made on a host whose clock runs fast: a clock-skew
    report dressed up as a duration.  The magnitudes are deliberately formatted
    the same way, so the two read alike where they appear side by side.
    """
    if seconds is None:
        return None
    if seconds < -tolerance:
        return None
    if seconds < 60:
        return "<1m"
    mins = int(seconds // 60)
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def parse_timestamp(text: str | None) -> datetime | None:
    """Parse the timestamp formats schedulers actually emit."""
    if not text:
        return None
    t = text.strip()
    if t.lower() in {"unknown", "none", "n/a", "(null)", "", "-"}:
        return None
    # ISO-8601 with a trailing Z, as Kubernetes emits. An aware timestamp is
    # converted to LOCAL time before the zone is dropped: everything downstream
    # compares against a naive datetime.now(), so merely stripping the zone
    # leaves a UTC wall-clock reading masquerading as a local one -- an error of
    # however many hours the host is offset.
    iso = t[:-1] + "+00:00" if t.endswith("Z") else t
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",   # Slurm
        "%Y-%m-%dT%H:%M",
        "%a %b %d %H:%M:%S %Y",  # PBS / LSF long form
        "%b %d %H:%M",           # LSF short form (no year)
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            got = datetime.strptime(t, fmt)
        except ValueError:
            continue
        # A format with no year defaults to 1900; assume the current year,
        # which is the only sane reading for a scheduler's near-term estimate.
        return got.replace(year=datetime.now().year) if got.year == 1900 else got
    return None
