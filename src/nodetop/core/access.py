"""Remembering what the control plane said about access, and re-asking anyway.

**Why this exists.** Starting the tool on a 607-node cluster costs 1.93s, and
**1.60s of it is dry-runs** -- nineteen `sbatch --test-only` calls, one per
partition the declared allowlists admit. Measured where that time goes:
``sbatch --help`` is 14.8 ms (the client, no RPC), ``scontrol ping`` 15.8 ms (a
round trip), and ``sbatch --test-only`` **98 ms** -- so ~82 ms of each probe is
the controller running the site's submit plugin, serialised against everyone
else's submissions. More concurrency does not help; batching partitions into one
request is refused by the plugin. That time is not ours to make smaller.

**So it is spent differently, not saved.** An interactive browse opens on the
answer from the last run and re-asks in the background; when the fresh answer
differs, the view reloads itself, which is the same thing `r` does and lands the
cursor back where it was. A reader sees a usable screen in ~0.35s instead of
1.9s, and within about two seconds the screen is exactly as authoritative as it
would have been if they had waited.

**What is deliberately NOT cached.**

* A printout. ``nodetop status | grep`` gets one shot at being right, so it
  probes and waits, exactly as before. Only a session -- which can correct
  itself on screen and says how old its answer is -- reads this file.
* Anything but a settled verdict. "The control plane did not answer" is not a
  finding and never gets written down; the next run asks again.
* Any answer to a *different* question. The key covers the backend, the
  cluster, the user, the accounts and the job shape. It deliberately does NOT
  cover which partitions were asked about: that set is "the partitions with room
  right now", which changes as jobs start and finish, and keying on it made
  every second run a miss -- measured, 2.3s instead of 0.33s, because one
  partition had filled up in between. The verdicts are stored per partition
  instead, and a run whose candidates are all present uses them.

The file is a convenience and never a dependency: every error here -- no HOME, a
read-only cache directory, truncated JSON, a schema from a future version -- ends
with "ask the cluster", which is what the tool did before this module existed.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Sequence

#: How old a remembered answer may be before it is ignored.
#:
#: A day, and the length is the point. Every session re-asks in the background
#: within seconds of opening, so this bounds nothing except how old the *first
#: frame* may be -- the window in which a reader could act on a stale answer is
#: the couple of seconds before the recheck lands, whatever this number says.
#: What it does control is how often the wait comes back: at fifteen minutes,
#: coming back to the terminal after lunch cost the full 1.9s again, which is
#: most of what this whole mechanism exists to avoid. An entitlement is
#: configuration an administrator edits, usually because the reader asked them
#: to; a day-old answer to "may I submit here" is very nearly always still the
#: answer, and the frame says how old it is.
DEFAULT_TTL = 86_400.0

#: At most this many remembered answers, newest kept. One per (cluster, user,
#: accounts, shape), so one or two per cluster in practice.
KEEP = 32

VERSION = 1


#: Values already complained about, so a run says it once rather than per call.
_TTL_COMPLAINED: set[str] = set()


def _ttl_ignored(raw: str, why: str) -> None:
    """Say that the variable is not in effect, once per distinct value.

    On stderr, not raised. This is a cache knob: erroring out because one is
    mistyped would take the tool down over something it can simply do without,
    and a sibling package makes that trade the other way only for a knob that
    bounds a query (`SLURMPAST_TIMEOUT`, which really can hang a run).

    But silence was wrong too. `ttl()` is consulted up to three times per run, so
    the note is deduplicated rather than printed each time.
    """
    if raw in _TTL_COMPLAINED:
        return
    _TTL_COMPLAINED.add(raw)
    # `sys.stderr.write`, not `print(file=...)`: `T20` is selected precisely to
    # keep `print` out of the library modules, and it caught this line. The
    # sibling package that writes everything this way is the reason the rule
    # costs nothing there.
    sys.stderr.write(
        f"nodetop: ignoring NODETOP_ACCESS_TTL={raw!r} ({why}); using the default "
        f"of {DEFAULT_TTL:.0f}s. Set it to a number of seconds, or 0 to disable "
        f"the remembered answers.\n"
    )


def ttl() -> float:
    """The freshness bound, honouring ``NODETOP_ACCESS_TTL`` (0 disables).

    One rule for every unusable value: the variable is ignored and the default
    stands. It used to be three different rules, all silent --

        NODETOP_ACCESS_TTL=abc    -> the default (86400s)
        NODETOP_ACCESS_TTL=-5     -> 0, i.e. caching DISABLED
        NODETOP_ACCESS_TTL=1e999  -> inf, i.e. answers never expire
        NODETOP_ACCESS_TTL=nan    -> 0, i.e. disabled (from `max`'s NaN ordering)

    -- so the same class of typo produced "default", "off" and "forever"
    depending on which typo it was, and nothing said which. `inf` is the one that
    could actually mislead: a remembered refusal would be reused for the life of
    the cache file, long after the account was fixed.
    """
    raw = os.environ.get("NODETOP_ACCESS_TTL")
    if raw is None:
        return DEFAULT_TTL
    try:
        value = float(raw)
    except ValueError:
        _ttl_ignored(raw, "not a number")
        return DEFAULT_TTL
    if not math.isfinite(value):
        _ttl_ignored(raw, "not a finite number of seconds")
        return DEFAULT_TTL
    if value < 0:
        _ttl_ignored(raw, "negative")
        return DEFAULT_TTL
    return value


def directory() -> str | None:
    """Where the answers live, or ``None`` if this machine has nowhere for them."""
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        home = os.environ.get("HOME")
        if not home:
            return None
        base = os.path.join(home, ".cache")
    return os.path.join(base, "nodetop", "access")


def path(name: str) -> str | None:
    """The file holding one question's answer.

    **One file per question, not one document holding all of them.** A login
    node runs several of these at once -- a browse, a printout, a background
    recheck -- and a shared document is read, merged and rewritten by each, so
    the last rename wins and the others' entries vanish. Measured with twelve
    concurrent writers on twelve different keys: **five of the twelve were
    lost**, which costs a full dry-run pass rather than the one probe a lost
    *verdict* costs. Separate files cannot collide, need no locking -- which
    would be its own adventure on an NFS home -- and prune by age.
    """
    where = directory()
    return None if where is None else os.path.join(where, f"{name}.json")


def key(
    *,
    backend: str,
    cluster: str,
    user: str,
    accounts: Sequence[str],
    shape: str,
) -> str:
    """A stable name for one question, so a different question cannot hit it.

    Hashed rather than spelled out because `cluster` is every partition name on
    the cluster, and the file is meant to stay small.

    Note what is *absent*: which partitions were asked about. That set is "the
    ones with room right now" and it moves as jobs start and finish -- keying on
    it meant a second run 40 seconds later was a miss, measured at 2.3s instead
    of 0.33s. Verdicts are per partition; see :func:`load`.
    """
    import hashlib

    parts = "\x00".join((
        f"v{VERSION}", backend, cluster, user,
        ",".join(sorted(accounts)), shape,
    ))
    return hashlib.sha256(parts.encode("utf-8", "replace")).hexdigest()[:32]


def _read(name: str) -> dict:
    """One answer, or an empty document if it cannot be trusted."""
    where = path(name)
    if where is None:
        return {}
    try:
        import json

        with open(where, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        # A file from another version is not corrupt, but it is not ours to
        # interpret either. Ignored, and overwritten on the next write.
        return {}
    return data


#: What a remembered verdict can say. "unsettled" is a real answer -- the
#: control plane was asked and could not settle it -- and is kept apart from a
#: refusal for the same reason the funnel keeps them apart.
YES, NO, MAYBE = "yes", "no", "unsettled"


def load(name: str, *, now: float | None = None
         ) -> tuple[dict[str, str], float] | None:
    """Whatever is remembered for question ``name``, and how old it is.

    Returns what it knows even when that does not cover everything the caller
    is about to ask about: the verdicts are independent questions, so the gap
    can be probed on its own -- one dry-run instead of nineteen. Measured before
    that split existed: a partition gaining room between two runs cost a full
    pass, 2.2s where the run before and after took 0.33s.

    ``None`` means the file has nothing fresh to say, which covers a stale
    entry and every failure mode of the file itself.
    """
    window = ttl()
    if window <= 0:
        return None
    entry = _read(name)
    at = entry.get("at")
    verdicts = entry.get("verdicts")
    if not isinstance(at, (int, float)) or not isinstance(verdicts, dict):
        return None
    clock = time.time() if now is None else now
    age = clock - at
    if age > window:
        return None
    # Each verdict expires on its own clock, not the file's. See :func:`save`:
    # the merge carries forward answers this run never re-asked, so ageing the
    # whole document together let one refusal be renewed indefinitely by writes
    # about other partitions. A file written before ``seen`` existed has no
    # per-verdict time and ages on the document's, exactly as it used to.
    stamps = entry.get("seen")
    stamps = stamps if isinstance(stamps, dict) else {}
    known: dict[str, str] = {}
    for k, v in verdicts.items():
        if not isinstance(k, str) or v not in (YES, NO, MAYBE):
            continue
        when = stamps.get(k)
        if isinstance(when, (int, float)) and clock - when > window:
            continue
        known[k] = v
    return known, max(0.0, age)


def save(
    name: str,
    verdicts: dict[str, str],
    *,
    now: float | None = None,
) -> bool:
    """Write one answer down. Returns whether it landed; failure is not an error.

    Written to a temporary file in the same directory and renamed over the
    target, so a reader never sees half a document and two tools running at once
    cannot interleave -- the loser's whole answer is simply lost, which costs one
    slow start.
    """
    where = path(name)
    if where is None or ttl() <= 0:
        return False
    # Merged, not replaced: a run only asks about the partitions with room right
    # now, and forgetting the others would make the next run -- with a slightly
    # different set -- a miss again. That is the bug this shape fixes.
    before = _read(name)
    previous = before.get("verdicts")
    merged = dict(previous) if isinstance(previous, dict) else {}
    merged.update(verdicts)
    stamp = time.time() if now is None else now
    # When each verdict was last *asked*, which is not when the file was last
    # written -- and the difference is the bug. The merge above carries forward
    # answers this run never re-asked, so one document timestamp restamped them
    # all as new: a partition that stops having room stops being a candidate, is
    # never re-probed, and its remembered refusal is renewed by every unrelated
    # write. Measured with a fake clock: a `no` recorded twelve days earlier was
    # still served, and reported as "checked 1s ago". That is the "reused long
    # after the account was fixed" failure `ttl()` refuses `inf` to prevent,
    # arriving through the merge instead of through the knob.
    #
    # Additive, and read defensively at both ends, so no version bump: a file
    # without it ages on `at` as before, and an older nodetop ignores it.
    seen: dict[str, float] = {}
    stamps = before.get("seen")
    if isinstance(stamps, dict):
        seen.update({k: float(v) for k, v in stamps.items()
                     if k in merged and isinstance(v, (int, float))})
    was = before.get("at")
    if isinstance(was, (int, float)):
        for k in merged:
            seen.setdefault(k, float(was))
    seen.update(dict.fromkeys(verdicts, stamp))
    document = {
        "version": VERSION,
        "at": stamp,
        "verdicts": {k: merged[k] for k in sorted(merged)},
        "seen": {k: seen[k] for k in sorted(seen)},
    }
    try:
        import json
        import tempfile

        os.makedirs(os.path.dirname(where), exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=os.path.dirname(where), prefix=".access-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(document, fh)
            os.chmod(temporary, 0o600)
            os.replace(temporary, where)
        except BaseException:
            with __import__("contextlib").suppress(OSError):
                os.unlink(temporary)
            raise
    except (OSError, ValueError, UnicodeError):
        return False
    _prune()
    return True


def _prune() -> None:
    """Keep the newest :data:`KEEP` answers. Never an error if it cannot."""
    where = directory()
    if where is None:
        return
    try:
        files = [os.path.join(where, f) for f in os.listdir(where)
                 if f.endswith(".json")]
        if len(files) <= KEEP:
            return
        for old in sorted(files, key=os.path.getmtime)[:-KEEP]:
            with __import__("contextlib").suppress(OSError):
                os.unlink(old)
    except OSError:
        return
