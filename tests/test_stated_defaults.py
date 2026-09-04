"""A default written into help text must be the default the parser uses.

Sixteen options across five subcommands spell their default out in prose --
`(default: 20)`, `(default: 1)`, `(default: -)` -- rather than interpolating
`%(default)s`. Hardcoding is not wrong (it reads better than `%(default)s` when
the value needs a gloss, as `--top`'s "20; --all for every one" does), but it is
a copy, and a copy drifts silently: change the keyword and the sentence beside
it still says the old number. Nothing checked them.

A sibling package sidesteps this by interpolating everywhere, and two others
state only prose defaults, so this is the one package where the check has
anything to bite on. Verified at the time of writing: 13 value defaults agree,
and the three prose ones (`--accounts`, "all of yours") are skipped deliberately
-- see `_stated_value`.
"""

from __future__ import annotations

import argparse
import re

import pytest

from nodetop.cli import build_parser

#: `(default: X)` anywhere in a help string. Bounded so a long sentence
#: containing the word cannot be mistaken for a value.
_STATED = re.compile(r"\(\s*default[:\s]+([^)]{1,24})\)", re.I)

#: Stated defaults that are a DESCRIPTION rather than a value: `--accounts` says
#: "all of yours" for an empty string that means "resolve at run time". Those are
#: honest and cannot be compared mechanically, so they are skipped -- and listed
#: here rather than pattern-matched, so adding one is a deliberate act.
_PROSE = frozenset({"all of yours", "the current directory", "you", "none"})


def _stated_value(text: str) -> str | None:
    """The default this help string claims, or None if it states prose or nothing."""
    found = _STATED.search(text)
    if not found:
        return None
    stated = found.group(1).strip().strip("`'\"")
    # `--top` glosses its number: "20; --all for every one".
    stated = stated.split(";")[0].strip()
    return None if stated.lower() in _PROSE else stated


def _options() -> list[tuple[str, argparse.Action, str]]:
    """Every option whose help text states a comparable default value."""
    out: list[tuple[str, argparse.Action, str]] = []

    def walk(parser: argparse.ArgumentParser, prefix: str = "") -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    walk(sub, f"{prefix}{name} ")
                continue
            if "%(default)" in (action.help or ""):
                continue  # interpolated: cannot drift
            stated = _stated_value(action.help or "")
            if stated is not None:
                label = prefix + ("/".join(action.option_strings) or action.dest)
                out.append((label, action, stated))

    walk(build_parser())
    return out


def test_there_are_stated_defaults_to_check():
    """Guards against the scan going quiet and every assertion below passing.

    If the package moves to `%(default)s` throughout, this fails and says to
    delete the file rather than leaving a test that checks nothing.
    """
    found = _options()
    assert len(found) >= 10, (
        f"only {len(found)} hardcoded default(s) found; either they were "
        f"converted to %(default)s -- in which case this file has no job left -- "
        f"or the help wording changed shape and the scan no longer sees them"
    )


@pytest.mark.parametrize("case", _options(), ids=lambda c: c[0].replace(" ", ":"))
def test_the_stated_default_is_the_real_one(case):
    label, action, stated = case
    actual = action.default
    assert str(actual) == stated, (
        f"`nodetop {label}` help says the default is {stated!r}, but the parser "
        f"uses {actual!r}. One of the two moved; the help text is the copy."
    )


class TestTheComparisonItself:
    """Controls -- a check that passes because it compares nothing is worse than none."""

    def test_a_disagreement_would_be_caught(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--wrong", type=int, default=8, help="how many (default: 16)")
        action = next(a for a in parser._actions if a.dest == "wrong")
        stated = _stated_value(action.help or "")
        assert stated == "16" and str(action.default) != stated

    def test_prose_is_skipped_rather_than_failed(self):
        assert _stated_value("which accounts (default: all of yours)") is None

    def test_a_glossed_number_is_still_compared(self):
        # `--top`'s form: the number, then a semicolon and an explanation.
        assert _stated_value("how many rows (default: 20; --all for every one)") == "20"

    def test_help_without_a_default_is_ignored(self):
        assert _stated_value("show every queue, not the busiest ones") is None
