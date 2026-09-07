"""``FORCE_COLOR=0`` asked for no colour and got the most there is.

``_depth`` reads the variable's VALUE as a level: ``1``/``true`` mean 256 colours
and anything else means truecolor. So the one value that plainly names *zero*
selected the top of the ladder, measured on this tree before the fix::

    FORCE_COLOR='0'     depth=24  enabled=True
    FORCE_COLOR='false' depth=24  enabled=True

The function's own docstring defines that ladder -- "0 = no colour, 4 =
16-colour, 8 = 256-colour, 24 = truecolor" -- and `test_docstrings.py` pins it,
so a value of ``0`` mapping to 24 contradicts the line directly above it. It is
also the same distinction the same function already gets right one line earlier:
``NO_COLOR=""`` is falsy and correctly leaves colour alone, because an empty
value is *unset*, not *off*. ``0`` is not empty; it is off.

``false`` is included because the level mapping already treats ``true`` as a
synonym of ``1`` -- the vocabulary is the code's own, not one imported for this
fix. Any other value keeps its existing meaning, including the "unrecognised
value means truecolor" reading, which is what a caller writing ``FORCE_COLOR=3``
gets.

``TestControls`` pins every neighbouring branch of the same function, because
this fix inserts a return *before* the tty check: `NO_COLOR` still wins,
`FORCE_COLOR=1` still defeats a redirected stdout, `TERM=dumb` still disables,
and the four rungs of the ladder still map as the docstring says.
"""

import pytest

from nodetop.render import Style, _depth


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """A known environment: these four decide every branch of `_depth`."""
    for name in ("NO_COLOR", "FORCE_COLOR", "COLORTERM", "TERM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm")
    return monkeypatch


class TestAForcedZeroMeansNoColour:
    @pytest.mark.parametrize("value", ["0", "false"])
    def test_the_depth_is_zero(self, _clean_env, value):
        _clean_env.setenv("FORCE_COLOR", value)
        assert _depth() == 0

    @pytest.mark.parametrize("value", ["0", "false"])
    def test_the_style_is_disabled(self, _clean_env, value):
        _clean_env.setenv("FORCE_COLOR", value)
        assert Style().enabled is False

    @pytest.mark.parametrize("value", ["0", "false"])
    def test_nothing_is_painted(self, _clean_env, value):
        _clean_env.setenv("FORCE_COLOR", value)
        painted = Style().ok("x")
        assert painted == "x", painted
        assert "\033[" not in painted

    def test_it_holds_on_a_truecolor_terminal_too(self, _clean_env):
        # COLORTERM is checked after the forced branch, so it must not overrule it.
        _clean_env.setenv("FORCE_COLOR", "0")
        _clean_env.setenv("COLORTERM", "truecolor")
        assert _depth() == 0

    def test_it_holds_on_a_256_colour_terminal_too(self, _clean_env):
        _clean_env.setenv("FORCE_COLOR", "0")
        _clean_env.setenv("TERM", "xterm-256color")
        assert _depth() == 0


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_an_empty_value_is_unset_not_off(self, _clean_env):
        # The distinction this fix rests on, and which `_depth` already made for
        # `NO_COLOR`: empty means "say nothing", so the tty check decides.
        _clean_env.setenv("FORCE_COLOR", "")
        _clean_env.setattr("sys.stdout", _Tty(True))
        assert _depth() == 4  # a plain `xterm`, decided by the terminal

    def test_a_forced_one_still_means_256_colours(self, _clean_env):
        _clean_env.setenv("FORCE_COLOR", "1")
        assert _depth() == 8
        _clean_env.setenv("FORCE_COLOR", "true")
        assert _depth() == 8

    def test_an_unrecognised_value_still_means_truecolor(self, _clean_env):
        _clean_env.setenv("FORCE_COLOR", "3")
        assert _depth() == 24

    def test_forcing_still_defeats_a_redirected_stdout(self, _clean_env):
        _clean_env.setattr("sys.stdout", _Tty(False))
        _clean_env.setenv("FORCE_COLOR", "1")
        assert _depth() == 8

    def test_no_color_still_wins(self, _clean_env):
        _clean_env.setenv("NO_COLOR", "1")
        _clean_env.setenv("FORCE_COLOR", "3")
        assert _depth() == 0

    def test_a_dumb_terminal_still_disables(self, _clean_env):
        _clean_env.setenv("FORCE_COLOR", "1")
        _clean_env.setenv("TERM", "dumb")
        assert _depth() == 0

    def test_a_redirected_stdout_with_nothing_forced_is_still_plain(self, _clean_env):
        _clean_env.setattr("sys.stdout", _Tty(False))
        assert _depth() == 0

    @pytest.mark.parametrize(
        ("depth", "label"),
        [(0, "no colour"), (4, "16-colour"), (8, "256-colour"), (24, "truecolor")],
    )
    def test_the_docstring_ladder_still_holds(self, depth, label):
        assert Style(depth=depth).enabled is (depth > 0), label


class _Tty:
    """A stdout that is, or is not, a terminal."""

    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty

    def write(self, text):  # pragma: no cover - never written to here
        return len(text)
