"""Bracket-notation node list expansion and collapse.

Slurm writes node sets in a notation that nests commas *inside* the
brackets -- ``cn-[0001-0010,0012-0015],gn-bigmem[1-4]`` is two
groups, not five.  A naive ``split(",")`` therefore shreds it, which is the
single most common way a home-grown Slurm script silently loses nodes.

Collapse is the inverse, and it is the reason this module exists as much as
expansion is: generating an exclusion argument by hand produces a string long
enough to hit shell and scheduler argument limits, while the bracket form stays
short.

The notation originates with Slurm but is not unique to it -- PBS host lists,
LSF host groups and a good deal of site tooling use the same shape -- so this
lives outside the backends and any of them may use it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["collapse", "expand", "split_groups"]

# A trailing numeric suffix, captured so we can group ``node-0007`` with
# ``node-0008``.  The width matters: Slurm pads to a fixed width per group and
# ``node-[7-8]`` is NOT the same node set as ``node-[0007-0008]``.
_TRAILING_NUM = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


def split_groups(nodelist: str) -> list[str]:
    """Split a nodelist on commas that are *outside* bracket groups.

    ``"a-[1-2,4],b-[1-3]"`` -> ``["a-[1-2,4]", "b-[1-3]"]``
    """
    groups: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in nodelist:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            # Tolerate an unbalanced ']' rather than going negative; a
            # malformed nodelist should degrade, not raise, because it is
            # usually a truncated field rather than a real syntax error.
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            groups.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        groups.append("".join(current))
    return [g.strip() for g in groups if g.strip()]


def _expand_range_body(body: str) -> list[str]:
    """Expand the inside of one bracket group: ``"1-3,7"`` -> 1,2,3,7.

    Zero padding is preserved from the widest endpoint, matching Slurm: a
    range written ``0001-0003`` yields ``0001 0002 0003``, not ``1 2 3``.
    """
    out: list[str] = []
    for piece in body.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, _, hi_s = piece.partition("-")
            if not (lo_s.isdigit() and hi_s.isdigit()):
                out.append(piece)
                continue
            width = max(len(lo_s), len(hi_s))
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(str(n).zfill(width) for n in range(lo, hi + 1))
        else:
            out.append(piece)
    return out


def expand(nodelist: str | None) -> list[str]:
    """Expand a Slurm nodelist into concrete node names, order preserved.

    Returns ``[]`` for the empty string and for Slurm's several ways of
    spelling "nothing" (``(null)``, ``None``, ``n/a``) so callers do not have
    to special-case them at every site.
    """
    if not nodelist:
        return []
    text = nodelist.strip()
    if text.lower() in {"(null)", "none", "n/a", "(none)"}:
        return []

    names: list[str] = []
    for group in split_groups(text):
        names.extend(_expand_group(group))
    return names


def _expand_group(group: str) -> list[str]:
    """Expand one group, which may contain several bracket sections.

    Multi-dimensional names are legal and real: ``rack[1-2]node[1-4]`` is
    eight nodes, not two names with a literal bracket in them.  Each section
    multiplies the result, so this walks left to right taking the product.
    """
    parts: list[list[str]] = []
    rest = group
    while "[" in rest:
        head, _, after = rest.partition("[")
        body, closed, rest = after.partition("]")
        if not closed:
            # Unbalanced: treat the remainder literally rather than raising,
            # since a truncated field is far more likely than a syntax error.
            parts.append([head + "[" + body])
            rest = ""
            break
        if head:
            parts.append([head])
        parts.append(_expand_range_body(body))
    if rest:
        parts.append([rest])

    names = [""]
    for section in parts:
        names = [prefix + piece for prefix in names for piece in section]
    return names


def collapse(names: Iterable[str]) -> str:
    """Collapse node names back into Slurm bracket notation.

    Nodes are grouped by (prefix, digit width), so ``node-0001`` and
    ``node-1`` stay in separate groups -- they are different names and
    merging them would emit a set that does not round-trip.
    """
    # Group by prefix and zero-pad width; keep first-seen prefix order so the
    # output is stable and diffable rather than alphabetised surprise.
    buckets: dict[tuple[str, int], list[int]] = {}
    plain: list[str] = []
    order: list[tuple[str, int]] = []

    for name in names:
        m = _TRAILING_NUM.match(name)
        if not m:
            if name not in plain:
                plain.append(name)
            continue
        key = (m.group("prefix"), len(m.group("num")))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(int(m.group("num")))

    parts: list[str] = list(plain)
    for key in order:
        prefix, width = key
        nums = sorted(set(buckets[key]))
        runs: list[str] = []
        start = prev = nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            runs.append(_run(start, prev, width))
            start = prev = n
        runs.append(_run(start, prev, width))
        if len(runs) == 1 and "-" not in runs[0]:
            parts.append(f"{prefix}{runs[0]}")
        else:
            parts.append(f"{prefix}[{','.join(runs)}]")
    return ",".join(parts)


def _run(start: int, end: int, width: int) -> str:
    if start == end:
        return str(start).zfill(width)
    return f"{str(start).zfill(width)}-{str(end).zfill(width)}"
