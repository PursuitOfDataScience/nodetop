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

    @pytest.mark.parametrize("chars,expected", [
        (["\x1b", "[", "A"], Key.UP),
        (["\x1b", "[", "B"], Key.DOWN),
        (["\x1b", "[", "H"], Key.TOP),
        (["\x1b", "[", "F"], Key.BOTTOM),
        (["\r"], Key.ENTER),
        (["\n"], Key.ENTER),
        ([" "], Key.ENTER),
        (["k"], Key.UP),
        (["j"], Key.DOWN),
        (["g"], Key.TOP),
        (["G"], Key.BOTTOM),
        (["q"], Key.QUIT),
        (["\x03"], Key.QUIT),      # ctrl-c
        (["\x04"], Key.QUIT),      # ctrl-d
        (["\x1b"], Key.BACK),      # a lone Escape steps out, it does not quit
        (["z"], Key.OTHER),
        ([], Key.QUIT),            # EOF: the terminal went away
    ])
    def test_it_decodes(self, chars, expected):
        assert read_key(_reader(chars)) == expected

    def test_an_unknown_csi_sequence_is_ignored_not_quit(self):
        # A mouse report or a function key must not exit the UI.
        assert read_key(_reader(["\x1b", "[", "Z"])) == Key.OTHER

    def test_an_escape_followed_by_a_letter_steps_out(self):
        # Alt-x arrives as ESC then 'x'. There is no binding for it, and
        # treating it as the start of a CSI sequence would eat the next key.
        assert read_key(_reader(["\x1b", "x"])) == Key.BACK


class TestTheMoveLoop:
    @staticmethod
    def _run(keys, count=4):
        frames = []
        seq = iter(keys)
        chosen = select(
            lambda i: [f"row{j}" + (" <" if j == i else "") for j in range(count)],
            count, keys=lambda: next(seq, Key.QUIT),
            write=frames.append, raw=False)
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

    def test_a_repaint_clears_the_previous_block(self):
        # Without the clear, a block that shrinks leaves its old tail on screen.
        _, frames = self._run([Key.DOWN, Key.ENTER])
        assert any("\x1b[" in f and "F" in f and "J" in f for f in frames)

    def test_an_empty_list_steps_out(self):
        assert select(lambda _i: [], 0, keys=lambda: Key.ENTER,
                      write=lambda _s: None, raw=False) == Key.BACK

    def test_a_keyboard_interrupt_is_a_quit_not_a_traceback(self):
        def boom():
            raise KeyboardInterrupt

        assert select(lambda _i: ["x"], 2, keys=boom,
                      write=lambda _s: None, raw=False) == Key.QUIT


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
            pass          # must not raise, and must not touch termios

    def test_select_can_skip_its_own_raw_mode(self):
        # `raw=False` is what lets the caller own the session instead.
        from nodetop.interactive import Key, select

        seq = iter([Key.DOWN, Key.ENTER])
        assert select(lambda _i: ["a", "b"], 2, keys=lambda: next(seq),
                      write=lambda _s: None, raw=False) == 1


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


class TestNavigationHasThreeOutcomes:
    """"Out of here" and "out of the program" are different keys.

    With one level, `select` returning None for both was fine. With three, a key
    that leaves the program from the bottom of the stack is a key you press once
    and then stop trusting -- so Escape, Backspace, Left and `h` step out, while
    `q` and Ctrl-C unwind everything.
    """

    @pytest.mark.parametrize("chars,expected", [
        (["\x1b"], Key.BACK),              # a lone Escape steps out
        (["\x7f"], Key.BACK),              # Backspace
        (["h"], Key.BACK),
        (["\x1b", "[", "D"], Key.BACK),    # Left
        (["\x1b", "[", "C"], Key.ENTER),   # Right opens, like Enter
        (["q"], Key.QUIT),
        (["\x03"], Key.QUIT),
    ])
    def test_the_keys_are_distinguished(self, chars, expected):
        it = iter(chars)
        assert read_key(lambda _t=None: next(it, "")) == expected

    def test_select_reports_back_and_quit_separately(self):
        for key, expected in ((Key.BACK, Key.BACK), (Key.QUIT, Key.QUIT)):
            got = select(lambda _i: ["a", "b"], 2, keys=lambda k=key: k,
                         write=lambda _s: None, raw=False)
            assert got == expected

    def test_an_empty_level_steps_out_rather_than_quitting(self):
        # A node with no jobs must not close the whole browser.
        assert select(lambda _i: [], 0, keys=lambda: Key.ENTER,
                      write=lambda _s: None, raw=False) == Key.BACK

    def test_the_cursor_starts_where_the_caller_left_it(self):
        # Stepping out of a nested view lands on the row you came from, not the
        # top of a list you have already read.
        seq = iter([Key.ENTER])
        assert select(lambda _i: ["a", "b", "c"], 3, keys=lambda: next(seq),
                      write=lambda _s: None, raw=False, initial=2) == 2

    def test_an_out_of_range_initial_is_clamped(self):
        seq = iter([Key.ENTER])
        assert select(lambda _i: ["a"], 1, keys=lambda: next(seq),
                      write=lambda _s: None, raw=False, initial=99) == 0

    def test_the_block_is_erased_on_the_way_out(self):
        # This is what makes each level replace the last instead of appending to
        # a transcript of screens.
        frames = []
        select(lambda _i: ["a", "b"], 2, keys=lambda: Key.QUIT,
               write=frames.append, raw=False)
        assert "J" in frames[-1] and "F" in frames[-1]

    def test_erase_can_be_declined(self):
        frames = []
        select(lambda _i: ["a"], 1, keys=lambda: Key.QUIT,
               write=frames.append, raw=False, erase=False)
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
                code = painted[i:j + 1]
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
