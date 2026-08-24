"""Keyboard selection over an already-rendered listing.

The reports this tool prints are read and then acted on by retyping a name into
another command, which is the friction this module removes: move a highlight
down the rows you are already looking at and press Enter.

Three constraints shaped it, and all three ruled out the obvious approach of
reaching for curses or a TUI library:

* **No dependencies.**  This is a tool you run on a login node while the cluster
  is misbehaving, so it must work with nothing but the system Python.
  ``termios`` and ``tty`` are standard library on every platform the package
  claims; a full-screen library is not available and would not be installable at
  the moment it is needed.
* **The same output.**  A second renderer for the interactive view would drift
  from the printed one -- this codebase has the scars, which is why ``_grid``
  and ``_node_rows`` exist.  So nothing here renders anything: it takes the
  finished lines and paints one of them in inverse video.
* **It must degrade to the printout.**  Redirected, piped, in a dumb terminal,
  or on a platform without ``termios``: the answer is the static report, not an
  error.  :func:`supported` is the single gate.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["Key", "raw_session", "read_key", "select", "supported"]


class Key:
    """Decoded keypresses.  Names rather than bytes, so callers read cleanly."""

    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    RIGHT = "right"
    BACK = "back"
    #: Escape, told apart from Left. Both step out of a nested view, but at the
    #: root they differ: Left there is a movement with nowhere to go, while
    #: Escape is how a reader leaves.
    ESCAPE = "escape"
    QUIT = "quit"
    TOP = "top"
    BOTTOM = "bottom"
    OTHER = "other"


#: Single characters that mean each action.  ``j``/``k`` alongside the arrows
#: because anyone who lives in a terminal reaches for them.
_KEYS = {
    "\r": Key.ENTER, "\n": Key.ENTER, " ": Key.ENTER,
    "k": Key.UP, "j": Key.DOWN,
    # Escape and Backspace step OUT rather than quitting. Once the view has
    # depth, "out" is what a reader means by both, and a key that leaves the
    # program from three levels down is a key you press once and then distrust.
    #
    # At the ROOT there is nothing to step out of, and there the two part ways:
    # Escape leaves the program, Left does nothing. Left had to stop leaving
    # because a stray press took the whole thing down; Escape has no such
    # excuse -- it is the key a reader reaches for to get out, and doing
    # nothing is indistinguishable from a hang.
    "\x1b": Key.ESCAPE, "\x7f": Key.BACK, "h": Key.BACK,
    "q": Key.QUIT, "\x03": Key.QUIT, "\x04": Key.QUIT,
    "g": Key.TOP, "G": Key.BOTTOM,
}

#: How long to wait for the rest of an escape sequence. Long enough that a
#: three-byte arrow key arriving in separate reads is not split, short enough
#: that a deliberate Escape does not feel stuck.
_ESCAPE_GRACE = 0.05

#: The tails of the CSI sequences an arrow key sends.  Read after ``ESC [``.
_ARROWS = {"A": Key.UP, "B": Key.DOWN, "C": Key.RIGHT, "D": Key.BACK,
           "H": Key.TOP, "F": Key.BOTTOM}


#: Rows a full-screen view needs before it is worth entering.
#:
#: Not a taste judgement -- below this the redraw is destructive.  A frame is
#: repainted by moving the cursor up by its own height, and the chrome alone
#: (two borders, the facts line, a blank, the funnel, a rule, the column names,
#: one row and the position line) is nine rows that nothing is allowed to drop.
#: On a shorter terminal the cursor clamps at the top of the screen, the
#: clear-to-end lands in the wrong place, and every keypress leaves another copy
#: behind.  The static print scrolls, which is merely inconvenient.
MIN_LINES = 10


def supported(stream: object | None = None) -> bool:
    """Can this session take keystrokes at all?

    **Both streams matter, and for different reasons.**  Output must be a
    terminal or there is nothing to paint a highlight on; *input* must be a
    terminal or there is no one to read from -- and a run with stdin redirected
    from a file would otherwise consume that file as keystrokes.  ``NO_COLOR``
    is not consulted: a highlight is structure, not decoration, and inverse
    video is how it degrades.

    A terminal too short to hold one frame counts as unsupported, and falls
    back to the static print rather than to a screen that repaints on top of
    itself.  See :data:`MIN_LINES`.
    """
    out = stream if stream is not None else sys.stdout
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except Exception:  # pragma: no cover - non-POSIX
        return False
    if shutil.get_terminal_size().lines < MIN_LINES:
        return False
    return bool(
        getattr(out, "isatty", lambda: False)()
        and getattr(sys.stdin, "isatty", lambda: False)()
    )


def stdin_reader() -> Callable[[float | None], str]:
    """A one-character reader over stdin's file descriptor.

    **Unbuffered, via ``os.read``, and that is the whole point.** The obvious
    implementation -- ``sys.stdin.read(1)`` plus ``select`` to peek -- does not
    work and fails in a way that looks like a decoding bug: ``read(1)`` fills
    Python's *own* buffer from the kernel, so after taking the ``ESC`` of an
    arrow key the following ``[B`` is sitting in userspace where ``select`` on
    the descriptor cannot see it. Every arrow key then reads as a lone Escape,
    which this module treats as quit -- so the first press exited instead of
    moving. Reading the descriptor directly keeps the poll and the read looking
    at the same buffer.
    """
    fd = sys.stdin.fileno()

    def readch(timeout: float | None = None) -> str:
        if timeout is not None:
            import select as _select

            ready, _, _ = _select.select([fd], [], [], timeout)
            if not ready:
                return ""
        try:
            return os.read(fd, 1).decode("utf-8", "replace")
        except OSError:  # pragma: no cover - descriptor closed under us
            return ""

    return readch


#: Frames a burst of keypresses may swallow before one is drawn anyway.
_MAX_SKIPPED_FRAMES = 24


def input_pending() -> bool:
    """Is another keystroke already queued?

    Used to coalesce a held key into one repaint. Polling the descriptor is
    only correct because :func:`stdin_reader` reads it directly -- with
    ``sys.stdin.read`` the bytes would sit in Python's own buffer where
    ``select`` cannot see them, which is the same trap documented there.

    False whenever stdin is not a terminal, so an injected key source in a test
    is never treated as an endless burst and starved of repaints.
    """
    try:
        if not sys.stdin.isatty():
            return False
        import select as _select

        ready, _, _ = _select.select([sys.stdin.fileno()], [], [], 0)
        return bool(ready)
    except Exception:  # pragma: no cover - closed or non-selectable descriptor
        return False


def read_key(readch: Callable[[float | None], str] | None = None) -> str:
    """One keypress, as a :class:`Key` name.

    An arrow key arrives as three bytes (``ESC [ A``) and a lone Escape as one,
    so the two are told apart by whether anything follows -- which is why the
    reads after ``ESC`` pass a timeout. Treating a bare Escape as the start of a
    sequence is what makes a TUI hang until the next keystroke.
    """
    read = readch or stdin_reader()
    ch = read(None)
    if not ch:
        return Key.QUIT
    if ch == "\x1b":
        if read(_ESCAPE_GRACE) != "[":
            return Key.ESCAPE        # a lone Escape
        return _ARROWS.get(read(_ESCAPE_GRACE), Key.OTHER)
    return _KEYS.get(ch, Key.OTHER)


#: Signals that end the process, and would otherwise end it with the terminal
#: still in raw mode.  ``SIGHUP`` is the one that matters most on a login node:
#: it is what a dropped ssh connection sends.
_FATAL_SIGNALS = ("SIGTERM", "SIGHUP")


class _RawMode:
    """Raw keystrokes for the duration of a block, restored unconditionally.

    A terminal left in raw mode has no echo and no line editing, so failing to
    restore it does not crash anything -- it hands the user a shell that appears
    dead, with no echo to tell them that typing ``reset`` is working.  Hence the
    bare ``finally`` and the cursor being shown again there too.

    **``finally`` is not enough on its own, and that is the interesting part.**
    A default-handled ``SIGTERM`` or ``SIGHUP`` ends the process without raising
    anything, so no ``finally`` runs.  Measured: after ``SIGTERM``, echo off,
    canonical mode off, cursor hidden.  ``SIGINT`` was fine only because Python
    turns it into ``KeyboardInterrupt``.  So the fatal signals are handled long
    enough to put the terminal back and then re-raised with the default
    disposition, which keeps the exit status honest -- a tool that swallows
    ``SIGTERM`` is worse than one that leaves a messy terminal.

    ``SIGTSTP`` is the same problem wearing Ctrl-Z: suspending restores the
    terminal for the shell, and resuming takes it back.
    """

    def __init__(self) -> None:
        # `list[Any]`, because termios.tcgetattr returns a heterogeneous list
        # whose exact stub type differs between platforms; `object` here made
        # the restore call untypeable.
        self._saved: list[Any] | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _RawMode:
        # A no-op when stdin is not a terminal, rather than an exception.
        # `supported()` already gates the real usage, so reaching here without a
        # terminal means a caller wrapped a block defensively -- and a context
        # manager that throws on entry forces every such caller to duplicate the
        # check. Nothing is saved, so nothing is restored.
        if not getattr(sys.stdin, "isatty", lambda: False)():
            return self
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        sys.stdout.write("\033[?25l")   # hide the cursor
        sys.stdout.flush()
        self._catch_signals()
        return self

    def _catch_signals(self) -> None:
        import signal

        for name in _FATAL_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:      # pragma: no cover - platform without it
                continue
            # Not the main thread: nothing to install, and nothing to fix.
            with contextlib.suppress(ValueError):
                self._previous[sig] = signal.signal(sig, self._die)
        stop = getattr(signal, "SIGTSTP", None)
        if stop is not None:
            with contextlib.suppress(ValueError):
                self._previous[stop] = signal.signal(stop, self._suspend)

    def _restore_signals(self) -> None:
        import signal

        for sig, handler in self._previous.items():
            with contextlib.suppress(ValueError):
                signal.signal(sig, handler)
        self._previous.clear()

    def _die(self, signum: int, _frame: object) -> None:
        """Put the terminal back, then die as we would have."""
        import signal

        self.__exit__()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _suspend(self, signum: int, _frame: object) -> None:
        """Ctrl-Z: hand the terminal back, stop, and take it again on resume."""
        import signal

        saved, previous = self._saved, dict(self._previous)
        self.__exit__()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        # Resumed. Retake the terminal exactly as it was.
        signal.signal(signum, self._suspend)
        self._saved, self._previous = saved, previous
        if saved is not None:
            import tty

            tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

    def __exit__(self, *exc: object) -> None:
        self._restore_signals()
        if self._saved is None:
            return
        import termios

        sys.stdout.write("\033[?25h")   # show it again
        sys.stdout.flush()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
        self._saved = None


def raw_session() -> _RawMode:
    """Raw keystrokes for a whole interaction, not just one prompt.

    Needed because an interaction is more than a single list. After opening a
    row the caller waits for one keystroke to come back -- and if raw mode ended
    with the list, that read is in canonical mode, where nothing arrives until
    the user presses Enter. "Press any key" silently became "press enter", which
    is the sort of thing only a real terminal shows you.

    Output processing is untouched by ``setcbreak`` -- only ECHO and ICANON go --
    so ordinary ``print`` still emits a proper newline in here.
    """
    return _RawMode()


def select(
    render: Callable[[int], Sequence[str]],
    count: int,
    *,
    keys: Callable[[], str] | None = None,
    write: Callable[[str], object] | None = None,
    raw: bool = True,
    initial: int = 0,
    erase: bool = True,
    rows: Sequence[int] | None = None,
    escapable: bool = True,
    pending: Callable[[], bool] | None = None,
) -> int | str:
    """Move a highlight over ``count`` rows; return the chosen index or ``None``.

    ``render(i)`` returns the whole block to display with row ``i`` highlighted,
    and is called again on every keypress.  Redrawing everything rather than
    patching the changed line is deliberate: the block contains totals that
    depend on the selection in principle, and a partial repaint that gets one
    cell wrong is much harder to notice than a redraw that is simply slower --
    at these sizes, imperceptibly so.

    Returns the chosen index, or :data:`Key.BACK` / :data:`Key.QUIT` -- three
    outcomes rather than two, because a nested view needs "out of here" to be
    different from "out of the program". The caller pops a level on ``BACK`` and
    unwinds entirely on ``QUIT``.

    ``initial`` restores the cursor where the caller left it, so stepping out of
    a nested view lands on the row you came from instead of the top.

    ``erase`` clears the block on the way out. That is what makes the whole
    interaction happen in one place: each level replaces the last rather than
    scrolling it away, so there is one screen and not a transcript of screens.

    ``rows`` gives the display row each entry belongs to, which is what makes
    the arrows mean what they look like. **Several entries can share a row** --
    the overview's funnel is one line carrying four counts -- and with a flat
    list, Down moved between two things on the same line while the cursor
    appeared not to move at all. With ``rows``, Up and Down step between *rows*
    and Left and Right move *within* one; Right at the end of a row opens, which
    is what it does everywhere else.

    ``escapable`` is False at the root of a stack, where there is nothing to step
    back to. Left there does nothing: it used to return, and returning at the
    root exits, so a stray press took the whole program down -- "when pressing
    the left arrow, the entire app is gone". Escape is not a movement, so at the
    root it means what a reader means by it and leaves.

    ``keys`` and ``write`` are injectable so the loop can be driven by a test
    without a terminal, which is the only way this path gets exercised at all.
    """
    if count <= 0:
        return Key.BACK
    keys = keys or read_key
    out = write or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
    pending = pending or input_pending
    index = max(0, min(initial, count - 1))
    painted = 0
    # Entry -> display row, and the entries on each row in order. A flat list
    # behaves as one entry per row, which is the old behaviour exactly.
    at_row = list(rows) if rows is not None else list(range(count))
    by_row: dict[int, list[int]] = {}
    for entry, row in enumerate(at_row[:count]):
        by_row.setdefault(row, []).append(entry)
    order = sorted(by_row)

    def step_row(delta: int) -> int:
        """The entry one row up or down, holding the column where possible."""
        here = at_row[index]
        column = by_row[here].index(index)
        target = order[(order.index(here) + delta) % len(order)]
        siblings = by_row[target]
        return siblings[min(column, len(siblings) - 1)]

    def paint() -> None:
        """Repaint the block in place, without ever blanking the screen.

        **Overwrite, do not erase-then-write.** This used to move to the top of
        the previous block, clear everything downward with `ESC[J`, and only
        then write the new lines -- two writes with the screen empty in
        between. A terminal that renders anything in that gap shows a blank
        box, and holding an arrow key down turns that into a strobe: "when
        pressing down arrow constantly, the app is flickering".

        Each line is written over the old one and cleared only to end-of-line
        as it goes, so no cell is ever empty and there is one write per frame.
        A block that shrank still has to have its tail wiped, and the cursor
        put back where the next repaint expects it: one line below the block.
        """
        nonlocal painted
        block = list(render(index))
        buf: list[str] = []
        if painted:
            buf.append(f"\033[{painted}F")
        buf.extend(f"{line}\033[K\n" for line in block)
        stale = painted - len(block)
        if stale > 0:
            buf.append("\033[K\n" * stale)
            buf.append(f"\033[{stale}F")
        out("".join(buf))
        painted = len(block)

    mode = _RawMode() if raw else None
    try:
        if mode is not None:
            mode.__enter__()
        paint()
        skipped = 0
        while True:
            try:
                key = keys()
            except KeyboardInterrupt:
                return Key.QUIT
            if key == Key.QUIT:
                return key
            if key == Key.ENTER:
                return index
            if key == Key.ESCAPE:
                # Out of this view, or out of the program when this is the only
                # view there is.
                return Key.BACK if escapable else Key.QUIT
            if key == Key.BACK:
                # Left inside a row moves left; only at the leftmost edge does it
                # mean "out of here", and at the root there is no out.
                siblings = by_row[at_row[index]]
                column = siblings.index(index)
                if column > 0:
                    index = siblings[column - 1]
                elif escapable:
                    return key
            elif key == Key.UP:
                index = step_row(-1)
            elif key == Key.DOWN:
                index = step_row(+1)
            elif key == Key.RIGHT:
                siblings = by_row[at_row[index]]
                column = siblings.index(index)
                if column + 1 < len(siblings):
                    index = siblings[column + 1]
                else:
                    return index          # nothing further right: open it
            elif key == Key.TOP:
                index = by_row[order[0]][0]
            elif key == Key.BOTTOM:
                index = by_row[order[-1]][0]
            # Coalesce a burst. A held arrow key arrives as a stream of escape
            # sequences and only the last position matters, so drawing a frame
            # per repeat is work whose only visible effect is flicker. The cap
            # is insurance: a screen that never updates would be worse than one
            # that updates too often.
            if pending() and skipped < _MAX_SKIPPED_FRAMES:
                skipped += 1
                continue
            skipped = 0
            paint()
    finally:
        if erase and painted:
            # Wind the cursor back over the block so whatever the caller draws
            # next occupies the same rows. Without this each level would append
            # to a growing transcript instead of replacing the one before it.
            out(f"\033[{painted}F\033[J")
        if mode is not None:
            mode.__exit__()
