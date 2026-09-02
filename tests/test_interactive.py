"""Keyboard selection: key decoding, the move loop, and the fallbacks.

The whole path is injectable -- `read_key` takes a character reader and `select`
takes a key source and a writer -- because otherwise none of it could be
exercised without a terminal, and an interactive mode that is only ever tested
by hand is an interactive mode that breaks silently.
"""

from __future__ import annotations

import pytest

from nodetop.interactive import Key, read_key, select, supported


def _reader(chars):
    """A character reader that hands back `chars` one at a time, then EOF."""
    it = iter(chars)
    return lambda _timeout=None: next(it, "")


class TestKeyDecoding:
    """An arrow key is three bytes and a lone Escape is one.

    Telling them apart is the whole difficulty: treating Escape as the start of
    a sequence makes the UI hang until the next keystroke, and treating a
    sequence as three separate keys makes every arrow press quit.
    """

    @pytest.mark.parametrize(
        "chars,expected",
        [
            (["\x1b", "[", "A"], Key.UP),
            (["\x1b", "[", "B"], Key.DOWN),
            (["\x1b", "[", "H"], Key.TOP),
            (["\x1b", "[", "F"], Key.BOTTOM),
            # SS3, not CSI: the same keys once the terminal is in application
            # cursor mode (DECCKM). Accepting only "[" read every one of these
            # as a bare Escape, so arrows dismissed the view instead of moving.
            # nodetop never sets that mode, but a full-screen program that dies
            # before restoring it leaves it set.
            (["\x1b", "O", "A"], Key.UP),
            (["\x1b", "O", "B"], Key.DOWN),
            (["\x1b", "O", "H"], Key.TOP),
            (["\x1b", "O", "F"], Key.BOTTOM),
            (["\r"], Key.ENTER),
            (["\n"], Key.ENTER),
            ([" "], Key.ENTER),
            (["k"], Key.UP),
            (["j"], Key.DOWN),
            (["g"], Key.TOP),
            (["G"], Key.BOTTOM),
            (["q"], Key.QUIT),
            (["\x03"], Key.QUIT),  # ctrl-c
            (["\x04"], Key.QUIT),  # ctrl-d
            # A lone Escape is its own key: it steps out of a nested view like
            # Left, and leaves the program at the root where Left cannot.
            (["\x1b"], Key.ESCAPE),
            (["z"], Key.OTHER),
            ([], Key.QUIT),  # EOF: the terminal went away
        ],
    )
    def test_it_decodes(self, chars, expected):
        assert read_key(_reader(chars)) == expected

    def test_an_unknown_csi_sequence_is_ignored_not_quit(self):
        # A mouse report or a function key must not exit the UI.
        assert read_key(_reader(["\x1b", "[", "Z"])) == Key.OTHER

    def test_an_unknown_ss3_sequence_is_ignored_not_quit(self):
        assert read_key(_reader(["\x1b", "O", "Z"])) == Key.OTHER

    def test_an_escape_followed_by_a_letter_steps_out(self):
        # Alt-x arrives as ESC then 'x'. There is no binding for it, and
        # treating it as the start of a CSI sequence would eat the next key.
        assert read_key(_reader(["\x1b", "x"])) == Key.ESCAPE


class TestTheMoveLoop:
    @staticmethod
    def _run(keys, count=4):
        frames = []
        seq = iter(keys)
        chosen = select(
            lambda i: [f"row{j}" + (" <" if j == i else "") for j in range(count)],
            count,
            keys=lambda: next(seq, Key.QUIT),
            write=frames.append,
            raw=False,
        )
        return chosen, frames

    def test_enter_returns_the_highlighted_index(self):
        assert self._run([Key.DOWN, Key.DOWN, Key.ENTER])[0] == 2

    def test_up_and_down_cancel_out(self):
        assert self._run([Key.DOWN, Key.DOWN, Key.UP, Key.ENTER])[0] == 1

    def test_quit_is_reported_as_quit(self):
        # Not None: with three levels, the caller has to tell "leave this view"
        # from "leave the program".
        assert self._run([Key.DOWN, Key.QUIT])[0] == Key.QUIT

    def test_movement_wraps_at_both_ends(self):
        # A list you cannot get to the bottom of in one keypress is a list you
        # scroll past. Wrapping is cheaper than a page-down binding.
        assert self._run([Key.UP, Key.ENTER])[0] == 3
        assert self._run([Key.DOWN] * 4 + [Key.ENTER])[0] == 0

    def test_home_and_end_jump(self):
        assert self._run([Key.DOWN, Key.DOWN, Key.TOP, Key.ENTER])[0] == 0
        assert self._run([Key.BOTTOM, Key.ENTER])[0] == 3

    def test_an_unbound_key_changes_nothing(self):
        assert self._run([Key.OTHER, Key.OTHER, Key.ENTER])[0] == 0

    def test_it_repaints_once_per_keypress(self):
        _, frames = self._run([Key.DOWN, Key.DOWN, Key.ENTER])
        # One initial paint plus one per move; the Enter does not repaint.
        assert len([f for f in frames if "row0" in f]) == 3

    def test_a_repaint_never_blanks_the_screen_first(self):
        """Overwrite, not erase-then-write.

        Moving to the top of the block and clearing everything downward before
        writing the new lines leaves the screen empty for one render, and
        holding an arrow key turns that into a strobe: "when pressing down
        arrow constantly, the app is flickering". Each line is written over the
        old one and cleared only to end-of-line as it goes.
        """
        _, frames = self._run([Key.DOWN, Key.DOWN, Key.ENTER])
        repaints = [f for f in frames if "row0" in f]
        assert len(repaints) == 3  # one write per frame, no gap
        for f in repaints:
            assert "\x1b[J" not in f  # no clear-to-end-of-screen
            assert "\x1b[K" in f  # cleared per line instead

    def test_a_block_that_shrinks_wipes_its_own_tail(self):
        # Row 0 draws five lines and row 1 draws two: without the wipe, three
        # lines of the old block stay on screen below the new one.
        seq = iter([Key.DOWN, Key.ENTER])
        frames: list[str] = []
        select(
            lambda i: ["x"] * (5 if i == 0 else 2),
            2,
            keys=lambda: next(seq, Key.QUIT),
            write=frames.append,
            raw=False,
        )
        shrunk = frames[1]
        assert shrunk.count("\x1b[K") == 5  # 2 content lines + 3 stale
        # And the cursor put back where the next repaint expects it.
        assert "\x1b[3F" in shrunk

    def test_a_burst_of_keypresses_repaints_once(self):
        """A held arrow key arrives as a stream; only the last position matters.

        Painting per repeat is work whose only visible effect is flicker.
        """
        seq = iter([Key.DOWN, Key.DOWN, Key.DOWN, Key.ENTER])
        queued = iter([True, True, False])
        frames: list[str] = []
        chosen = select(
            lambda i: [f"row{i}"],
            4,
            keys=lambda: next(seq, Key.QUIT),
            write=frames.append,
            raw=False,
            pending=lambda: next(queued, False),
        )
        assert chosen == 3  # every key still counted
        painted = [f for f in frames if "row" in f]
        assert len(painted) == 2  # initial, then one for the burst
        assert "row3" in painted[-1]  # and it shows where we ended

    def test_an_endless_burst_still_gets_repainted(self):
        # A screen that never updates would be worse than one that updates too
        # often, so the coalescing is capped.
        from nodetop.interactive import _MAX_SKIPPED_FRAMES

        keys = [Key.DOWN] * (_MAX_SKIPPED_FRAMES + 2) + [Key.ENTER]
        seq = iter(keys)
        frames: list[str] = []
        select(
            lambda i: [f"row{i}"],
            40,
            keys=lambda: next(seq, Key.QUIT),
            write=frames.append,
            raw=False,
            pending=lambda: True,
        )
        assert len([f for f in frames if "row" in f]) >= 2

    def test_an_empty_list_steps_out(self):
        assert (
            select(lambda _i: [], 0, keys=lambda: Key.ENTER, write=lambda _s: None, raw=False)
            == Key.BACK
        )

    def test_a_keyboard_interrupt_is_a_quit_not_a_traceback(self):
        def boom():
            raise KeyboardInterrupt

        assert select(lambda _i: ["x"], 2, keys=boom, write=lambda _s: None, raw=False) == Key.QUIT


class TestItDegradesToThePrintout:
    """Redirected, piped or dumb: the answer is the static report."""

    def test_a_non_tty_stdout_is_unsupported(self):
        class NotATty:
            @staticmethod
            def isatty():
                return False

        assert not supported(NotATty())

    def test_a_dumb_terminal_is_unsupported(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")

        class Tty:
            @staticmethod
            def isatty():
                return True

        assert not supported(Tty())

    def test_stdin_must_be_a_terminal_too(self, monkeypatch):
        # Output being a terminal is not enough: with stdin redirected from a
        # file, the UI would consume that file as keystrokes.
        monkeypatch.setenv("TERM", "xterm")

        class Tty:
            @staticmethod
            def isatty():
                return True

        class NotATty:
            @staticmethod
            def isatty():
                return False

        monkeypatch.setattr("sys.stdin", NotATty())
        assert not supported(Tty())

    def test_a_terminal_too_short_for_one_frame_is_unsupported(self, monkeypatch):
        """Below the chrome's own height the repaint is destructive.

        A frame is repainted by moving the cursor up by its own height, and the
        chrome that nothing is allowed to drop is nine rows. On a shorter
        screen the cursor clamps at the top, the clear lands in the wrong place
        and every keypress leaves another copy behind -- a stack of `╭────╮`
        lines. The static print merely scrolls.
        """
        import os
        import shutil

        from nodetop.interactive import MIN_LINES

        monkeypatch.setenv("TERM", "xterm")

        class Tty:
            @staticmethod
            def isatty():
                return True

        monkeypatch.setattr("sys.stdin", Tty())
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((100, MIN_LINES - 1))
        )
        assert not supported(Tty())
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((100, MIN_LINES))
        )
        assert supported(Tty())


class TestRawModeSpansTheWholeInteraction:
    """One raw-mode block per interaction, not one per prompt.

    An interaction is more than a single list: after opening a row the caller
    waits for one keystroke to come back. With raw mode scoped to the list, that
    read happened in canonical mode -- where nothing arrives until Enter -- so
    "any key returns" silently became "press enter". Only a real terminal shows
    you that, which is why the pty test exists.
    """

    def test_it_is_a_reusable_context_manager(self):
        from nodetop.interactive import raw_session

        with raw_session():
            pass
        with raw_session():
            pass

    def test_it_does_nothing_without_a_terminal(self, monkeypatch):
        # `supported()` gates the real usage, so reaching here without a
        # terminal means a caller wrapped a block defensively. Throwing on entry
        # would force every such caller to duplicate the check.
        from nodetop.interactive import raw_session

        class NotATty:
            @staticmethod
            def isatty():
                return False

        monkeypatch.setattr("sys.stdin", NotATty())
        with raw_session():
            pass  # must not raise, and must not touch termios

    def test_select_can_skip_its_own_raw_mode(self):
        # `raw=False` is what lets the caller own the session instead.
        from nodetop.interactive import Key, select

        seq = iter([Key.DOWN, Key.ENTER])
        assert (
            select(
                lambda _i: ["a", "b"], 2, keys=lambda: next(seq), write=lambda _s: None, raw=False
            )
            == 1
        )


class TestTheTerminalSurvivesASignal:
    """`finally` is not enough, and that is the whole point of this class.

    A default-handled SIGTERM or SIGHUP ends the process without raising
    anything, so no `finally` runs and the terminal is left with echo and
    canonical mode off -- a shell that appears dead, with no echo to tell the
    user that typing `reset` is working. Measured through a pty before the fix:
    after SIGTERM, `echo=False canonical=False`. SIGINT was fine only because
    Python turns it into KeyboardInterrupt.

    SIGHUP is the one that matters on a login node: it is what a dropped ssh
    connection sends.
    """

    @staticmethod
    def _tty(monkeypatch):
        class Tty:
            @staticmethod
            def isatty():
                return True

            @staticmethod
            def fileno():
                return 0

        monkeypatch.setattr("sys.stdin", Tty())

    def test_the_fatal_signals_are_handled_while_raw(self, monkeypatch):
        import signal

        from nodetop.interactive import raw_session

        seen = {}
        real = signal.signal

        def spy(sig, handler):
            seen[sig] = handler
            return real(sig, handler)

        monkeypatch.setattr(signal, "signal", spy)
        monkeypatch.setattr("termios.tcgetattr", lambda _fd: ["saved"])
        monkeypatch.setattr("termios.tcsetattr", lambda *_a: None)
        monkeypatch.setattr("tty.setcbreak", lambda _fd: None)
        self._tty(monkeypatch)

        with raw_session():
            installed = dict(seen)
        for name in ("SIGTERM", "SIGHUP", "SIGTSTP"):
            assert getattr(signal, name) in installed, name

    def test_the_previous_handlers_are_put_back(self, monkeypatch):
        import signal

        from nodetop.interactive import raw_session

        monkeypatch.setattr("termios.tcgetattr", lambda _fd: ["saved"])
        monkeypatch.setattr("termios.tcsetattr", lambda *_a: None)
        monkeypatch.setattr("tty.setcbreak", lambda _fd: None)
        self._tty(monkeypatch)

        marker = signal.getsignal(signal.SIGTERM)
        with raw_session():
            assert signal.getsignal(signal.SIGTERM) is not marker
        assert signal.getsignal(signal.SIGTERM) is marker

    def test_nothing_is_installed_without_a_terminal(self, monkeypatch):
        import signal

        from nodetop.interactive import raw_session

        class NotATty:
            @staticmethod
            def isatty():
                return False

        monkeypatch.setattr("sys.stdin", NotATty())
        marker = signal.getsignal(signal.SIGTERM)
        with raw_session():
            assert signal.getsignal(signal.SIGTERM) is marker

    def test_the_restore_is_idempotent(self, monkeypatch):
        # `_die` restores and then the enclosing `finally` restores again.
        # Doing it twice must not raise or double-write the cursor escape.
        from nodetop.interactive import raw_session

        calls = []
        monkeypatch.setattr("termios.tcgetattr", lambda _fd: ["saved"])
        monkeypatch.setattr("termios.tcsetattr", lambda *_a: calls.append(1))
        monkeypatch.setattr("tty.setcbreak", lambda _fd: None)
        self._tty(monkeypatch)

        session = raw_session()
        session.__enter__()
        session.__exit__()
        session.__exit__()
        assert len(calls) == 1

    @staticmethod
    def _suspendable(monkeypatch):
        """A session mid-`with`, with Ctrl-Z's actual stop stubbed out.

        `_suspend` ends in `os.kill(getpid(), SIGTSTP)` with the default
        disposition installed, which really would stop the test run, so the
        kill is the one thing replaced. Everything either side of it -- the
        restore on the way out and the retake on the way back -- runs for real.
        """
        import os

        from nodetop.interactive import raw_session

        retaken = []
        monkeypatch.setattr("termios.tcgetattr", lambda _fd: ["saved"])
        monkeypatch.setattr("termios.tcsetattr", lambda *_a: None)
        monkeypatch.setattr("tty.setcbreak", lambda _fd: retaken.append("cbreak"))
        monkeypatch.setattr(os, "kill", lambda *_a: None)
        TestTheTerminalSurvivesASignal._tty(monkeypatch)
        return raw_session(), retaken

    def test_the_fatal_signals_survive_a_ctrl_z_round_trip(self, monkeypatch):
        # Ctrl-Z hands the terminal back by calling `__exit__`, which puts the
        # ORIGINAL SIGTERM/SIGHUP dispositions back -- so resuming has to
        # reinstall the handlers, not merely remember what they replaced.
        # Without that, a browse that was suspended and resumed is exactly the
        # process `_FATAL_SIGNALS` exists for: SIGHUP from a dropped ssh
        # connection ends it with echo off and the cursor hidden.
        import signal

        session, _retaken = self._suspendable(monkeypatch)
        with session:
            session._suspend(signal.SIGTSTP, None)
            for name in ("SIGTERM", "SIGHUP"):
                sig = getattr(signal, name)
                assert signal.getsignal(sig) == session._die, name

    def test_a_ctrl_z_round_trip_retakes_the_terminal(self, monkeypatch):
        # CONTROL: the other half of "resuming takes it back" -- cbreak and the
        # hidden cursor -- and that leaving still hands the original handlers
        # back. True both before and after the fix above; it is here so that
        # test is not just a restatement of the change.
        import signal

        original = {
            name: signal.getsignal(getattr(signal, name))
            for name in ("SIGTERM", "SIGHUP", "SIGTSTP")
        }
        session, retaken = self._suspendable(monkeypatch)
        with session:
            session._suspend(signal.SIGTSTP, None)
            assert retaken == ["cbreak", "cbreak"]
            assert session._saved == ["saved"]
        for name, handler in original.items():
            assert signal.getsignal(getattr(signal, name)) is handler, name


class TestNavigationHasThreeOutcomes:
    """ "Out of here" and "out of the program" are different keys.

    With one level, `select` returning None for both was fine. With three, a key
    that leaves the program from the bottom of the stack is a key you press once
    and then stop trusting -- so Escape, Backspace, Left and `h` step out, while
    `q` and Ctrl-C unwind everything.
    """

    @pytest.mark.parametrize(
        "chars,expected",
        [
            (["\x1b"], Key.ESCAPE),  # a lone Escape
            (["\x7f"], Key.BACK),  # Backspace
            (["h"], Key.BACK),
            (["\x1b", "[", "D"], Key.BACK),  # Left
            # Right is its own key now: inside a row it moves right, and only at
            # the row's edge does it open. `select` turns it into a selection there.
            (["\x1b", "[", "C"], Key.RIGHT),
            (["q"], Key.QUIT),
            (["\x03"], Key.QUIT),
        ],
    )
    def test_the_keys_are_distinguished(self, chars, expected):
        it = iter(chars)
        assert read_key(lambda _t=None: next(it, "")) == expected

    def test_select_reports_back_and_quit_separately(self):
        for key, expected in ((Key.BACK, Key.BACK), (Key.QUIT, Key.QUIT)):
            got = select(
                lambda _i: ["a", "b"], 2, keys=lambda k=key: k, write=lambda _s: None, raw=False
            )
            assert got == expected

    def test_an_empty_level_steps_out_rather_than_quitting(self):
        # A node with no jobs must not close the whole browser.
        assert (
            select(lambda _i: [], 0, keys=lambda: Key.ENTER, write=lambda _s: None, raw=False)
            == Key.BACK
        )

    def test_escape_steps_out_of_a_nested_view(self):
        got = select(
            lambda _i: ["a", "b"], 2, keys=lambda: Key.ESCAPE, write=lambda _s: None, raw=False
        )
        assert got == Key.BACK

    def test_escape_leaves_the_program_at_the_root(self):
        """Where there is nothing to step out of, Escape means out.

        Left has to do nothing at the root -- it used to return, and returning
        there exits, so a stray press took the whole program down. Escape got
        the same treatment and should not have: it is the key a reader reaches
        for to get out, and doing nothing is indistinguishable from a hang.
        """
        got = select(
            lambda _i: ["a", "b"],
            2,
            keys=lambda: Key.ESCAPE,
            write=lambda _s: None,
            raw=False,
            escapable=False,
        )
        assert got == Key.QUIT

    def test_right_at_a_leaf_does_nothing(self):
        """The mirror of Left at the root, and it was missing.

        Right means "deeper" everywhere in this browser, and at the job detail
        there is nothing deeper. It returned the index anyway, which the caller
        could only read as "step back" -- so the key for going in jumped a
        level out, landing on the job list it had just come from: "pressing
        right arrow will go into the same interface ... this is very
        confusing."
        """
        # Right three times, then Left. Only the Left may end it.
        seq = iter([Key.RIGHT, Key.RIGHT, Key.RIGHT, Key.BACK])
        got = select(
            lambda _i: ["detail"],
            1,
            keys=lambda: next(seq, Key.QUIT),
            write=lambda _s: None,
            raw=False,
            openable=False,
        )
        assert got == Key.BACK

    def test_enter_at_a_leaf_does_nothing_either(self):
        # Enter is "open the selected thing", and at a leaf there is no such
        # thing. It must not stand in for going back.
        seq = iter([Key.ENTER, Key.ENTER, Key.ESCAPE])
        got = select(
            lambda _i: ["detail"],
            1,
            keys=lambda: next(seq, Key.QUIT),
            write=lambda _s: None,
            raw=False,
            openable=False,
        )
        assert got == Key.BACK

    def test_quit_still_leaves_from_a_leaf(self):
        assert (
            select(
                lambda _i: ["detail"],
                1,
                keys=lambda: Key.QUIT,
                write=lambda _s: None,
                raw=False,
                openable=False,
            )
            == Key.QUIT
        )

    def test_a_normal_level_still_opens_on_right(self):
        # The flag is off by default: every level above the leaf keeps working.
        got = select(
            lambda _i: ["a", "b"], 2, keys=lambda: Key.RIGHT, write=lambda _s: None, raw=False
        )
        assert got == 0

    def test_left_at_the_root_still_does_nothing(self):
        seq = iter([Key.BACK, Key.BACK, Key.ENTER])
        got = select(
            lambda _i: ["a", "b"],
            2,
            keys=lambda: next(seq, Key.QUIT),
            write=lambda _s: None,
            raw=False,
            escapable=False,
        )
        assert got == 0

    def test_the_cursor_starts_where_the_caller_left_it(self):
        # Stepping out of a nested view lands on the row you came from, not the
        # top of a list you have already read.
        seq = iter([Key.ENTER])
        assert (
            select(
                lambda _i: ["a", "b", "c"],
                3,
                keys=lambda: next(seq),
                write=lambda _s: None,
                raw=False,
                initial=2,
            )
            == 2
        )

    def test_an_out_of_range_initial_is_clamped(self):
        seq = iter([Key.ENTER])
        assert (
            select(
                lambda _i: ["a"],
                1,
                keys=lambda: next(seq),
                write=lambda _s: None,
                raw=False,
                initial=99,
            )
            == 0
        )

    def test_the_block_is_erased_on_the_way_out(self):
        # This is what makes each level replace the last instead of appending to
        # a transcript of screens.
        frames = []
        select(lambda _i: ["a", "b"], 2, keys=lambda: Key.QUIT, write=frames.append, raw=False)
        assert "J" in frames[-1] and "F" in frames[-1]

    def test_erase_can_be_declined(self):
        frames = []
        select(
            lambda _i: ["a"], 1, keys=lambda: Key.QUIT, write=frames.append, raw=False, erase=False
        )
        assert not any("J" in f and "F" in f for f in frames)


class TestTheHighlightSurvivesTheRowsOwnColours:
    """A rendered row is full of coloured cells, each ending in a reset.

    `ESC [ 0 m` clears reverse video along with the colour, so wrapping a row in
    `ESC [ 7 m` highlighted it as far as the first coloured cell and no further:
    the selected row read as a smudge on the left rather than as a selected row.
    """

    def test_inverse_is_rearmed_after_every_embedded_reset(self):
        from nodetop.render import Glyphs, Style

        st = Style(depth=24, glyphs=Glyphs())
        row = st.tint("name", 9) + "  " + st.muted("190") + "  " + st.tint("42", 7)
        painted = st.inverse(row)

        active, inside, total = False, 0, 0
        i = 0
        while i < len(painted):
            if painted[i] == "\x1b":
                j = painted.index("m", i)
                code = painted[i : j + 1]
                if code == "\x1b[7m":
                    active = True
                elif code == "\x1b[0m":
                    active = False
                i = j + 1
                continue
            total += 1
            inside += active
            i += 1
        assert total and inside == total, f"{inside}/{total} characters highlighted"

    def test_it_is_a_passthrough_when_colour_is_off(self):
        from nodetop.render import Glyphs, Style

        st = Style(depth=0, glyphs=Glyphs())
        assert st.inverse("plain") == "plain"

    def test_the_cursor_glyph_carries_the_selection_without_colour(self):
        # `inverse` is a no-op under NO_COLOR, so the glyph is the only thing
        # marking the row -- which is why it exists at all.
        from nodetop.render import Glyphs

        assert Glyphs().cursor and Glyphs.ascii().cursor
        assert Glyphs.ascii().cursor.isascii()


class TestTheTerminalIsReallyRestored:
    """The end-to-end half of `_RawMode`'s promise, in a real pty.

    The tests above verify the MECHANISM — that handlers are installed for
    `_FATAL_SIGNALS` and removed afterwards — with `termios.tcgetattr`
    monkeypatched. That is the right unit test and it would pass even if the
    handler restored the wrong thing, or restored termios and left the cursor
    hidden.

    So this drives the real interactive browse in a real pty, signals it, and asks
    the pty what state it was left in. Measured before the fix that
    `_FATAL_SIGNALS` exists for: after SIGTERM, `echo=False canonical=False`.

    Two sibling packages grew the same defect independently and were fixed the same
    way; each now has a pty test like this one, and this was the package whose
    correct behaviour nothing end-to-end actually checked.
    """

    @staticmethod
    def _drive(signame, settle=25.0):
        import contextlib
        import os
        import pathlib
        import pty
        import select
        import signal
        import sys
        import termios
        import time

        root = pathlib.Path(__file__).resolve().parent.parent
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs immediately
            os.environ.update(
                {
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TERM": "xterm-256color",
                    "COLUMNS": "120",
                    "LINES": "40",
                }
            )
            os.chdir(str(root))
            os.execv(sys.executable, [sys.executable, "-m", "nodetop"])

        seen = bytearray()

        def pump(seconds):
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                seen.extend(chunk)

        def lflags():
            try:
                bits = termios.tcgetattr(fd)[3]
            except Exception:
                return None
            return (bool(bits & termios.ECHO), bool(bits & termios.ICANON))

        # Wait until the pty actually reports raw mode rather than sleeping a
        # fixed interval: whether the browse has taken the terminal by then
        # depends on how long the cluster query takes, and a fixed sleep made the
        # test skip itself for some signals and not others on the same machine.
        deadline = time.time() + settle
        during = lflags()
        while time.time() < deadline and during != (False, False):
            pump(0.3)
            during = lflags()
        os.kill(pid, getattr(signal, f"SIG{signame}"))

        deadline = time.time() + 20
        while time.time() < deadline:
            pump(0.4)
            try:
                done, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done:
                break
        else:  # pragma: no cover - only on a hang
            os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            pytest.fail(f"nodetop never exited after SIG{signame}")
        text = seen.decode("utf-8", "replace")
        return during, lflags(), text

    @pytest.mark.parametrize("signame", ["TERM", "HUP", "INT"])
    def test_echo_and_canonical_mode_come_back(self, signame):
        during, after, text = self._drive(signame)
        if during != (False, False):
            pytest.skip(f"the browse never entered raw mode here (during={during})")
        assert after == (True, True), (
            f"SIG{signame} left the terminal raw: {after} — a shell with no echo, "
            f"where even typing `reset` gives no feedback"
        )
        assert "Traceback" not in text, text[-300:]

    @pytest.mark.parametrize("signame", ["TERM", "HUP", "INT"])
    def test_the_cursor_is_shown_again(self, signame):
        """`_RawMode` hides the cursor on entry, so it has to unhide it too.

        Restoring termios and leaving the cursor invisible is the half-fix that a
        mock-level test cannot see.
        """
        during, _after, text = self._drive(signame)
        if during != (False, False):
            pytest.skip("the browse never entered raw mode here")
        assert text.count("\033[?25l") <= text.count("\033[?25h"), (
            "the cursor was hidden more often than it was shown"
        )
