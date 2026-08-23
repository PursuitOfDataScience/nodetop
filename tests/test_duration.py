"""Walltime grammar, and the ambiguity it has to pick a side on."""

from __future__ import annotations

import pytest

from nodetop.core.duration import (
    format_age,
    format_duration,
    format_wait,
    parse_duration,
    parse_timestamp,
)


class TestUnambiguousForms:
    @pytest.mark.parametrize("text,seconds", [
        ("90m", 5400), ("2h", 7200), ("3d", 259200), ("1d12h", 129600),
        ("36h30m", 131400), ("45s", 45), ("2h30m15s", 9015), ("10min", 600),
    ])
    def test_suffixed(self, text, seconds):
        assert parse_duration(text) == seconds

    @pytest.mark.parametrize("text,seconds", [
        ("1:00:00", 3600), ("2:00:00", 7200), ("00:05:00", 300),
        ("2-00:00:00", 172800), ("7-00:00:00", 604800), ("1-12:00:00", 129600),
        ("2-12", 216000), ("1-00:30", 88200),
    ])
    def test_colon(self, text, seconds):
        assert parse_duration(text) == seconds

    def test_int_is_seconds(self):
        # An int came from code, not a command line, so there is no unit
        # convention to guess at.
        assert parse_duration(3600) == 3600


class TestTheAmbiguity:
    """The one convention that has to be chosen, chosen loudly."""

    def test_bare_number_is_minutes(self):
        # Matches Slurm (--time=60) and LSF (-W 60).
        assert parse_duration("60") == 3600
        assert parse_duration("30") == 1800

    def test_two_colon_fields_are_minutes_and_seconds(self):
        # Reading "2:00" as two hours is a 60x error.
        assert parse_duration("2:00") == 120
        assert parse_duration("2:30") == 150

    def test_a_day_part_promotes_the_leading_field_to_hours(self):
        # Same two fields, different meaning, because of the "1-".
        assert parse_duration("1-2:00") == 86400 + 2 * 3600
        assert parse_duration("2:00") == 120


class TestSentinelsAndJunk:
    @pytest.mark.parametrize(
        "text", ["UNLIMITED", "INFINITE", "NONE", "n/a", "", None, "-", "0"]
    )
    def test_no_limit(self, text):
        assert parse_duration(text) is None

    @pytest.mark.parametrize("text", ["tomorrow", "soon", "1:2:3:4:5", "abc"])
    def test_garbage_returns_none_rather_than_raising(self, text):
        assert parse_duration(text) is None


class TestFormatting:
    @pytest.mark.parametrize("seconds,text", [
        (3600, "1:00:00"), (300, "0:05:00"), (172800, "2-00:00:00"),
        (None, "unlimited"),
    ])
    def test_render(self, seconds, text):
        assert format_duration(seconds) == text

    @pytest.mark.parametrize("text", ["60", "2:00", "2-12", "1-00:30", "7-00:00:00", "90m"])
    def test_round_trip(self, text):
        seconds = parse_duration(text)
        assert parse_duration(format_duration(seconds)) == seconds

    @pytest.mark.parametrize("seconds,text", [
        (0, "now"), (30, "now"), (600, "10m"), (3600, "1h"),
        (5400, "1h 30m"), (172800, "2d"), (183600, "2d 3h"), (None, "?"),
    ])
    def test_wait(self, seconds, text):
        assert format_wait(seconds) == text


class TestTimestamps:
    def test_slurm_format(self):
        got = parse_timestamp("2026-08-21T17:00:12")
        assert (got.year, got.month, got.hour) == (2026, 8, 17)

    def test_kubernetes_iso_with_z_is_converted_to_local(self):
        # This used to assert hour == 17, i.e. that the zone was merely
        # stripped. That left a UTC wall-clock reading pretending to be local,
        # so every estimate built on it was off by the host's offset.
        from datetime import datetime, timezone

        got = parse_timestamp("2026-08-21T17:00:12Z")
        expected = (
            datetime(2026, 8, 21, 17, 0, 12, tzinfo=timezone.utc)
            .astimezone().replace(tzinfo=None)
        )
        assert got == expected

    def test_pbs_long_form(self):
        got = parse_timestamp("Thu Aug 21 17:00:12 2026")
        assert got is not None and got.year == 2026

    @pytest.mark.parametrize("text", ["Unknown", "N/A", "", None, "-"])
    def test_unset(self, text):
        assert parse_timestamp(text) is None


class TestFormatAge:
    """Elapsed time, which is not the same problem as a future wait."""

    @pytest.mark.parametrize("seconds,expected", [
        (0, "<1m"),
        (59, "<1m"),
        (60, "1m"),
        (90, "1m"),
        (3600, "1h"),
        (3600 + 20 * 60, "1h 20m"),
        (86400, "1d"),
        (86400 + 3600, "1d 1h"),
        (37 * 86400 + 3600, "37d 1h"),
    ])
    def test_magnitudes(self, seconds, expected):
        assert format_age(seconds) == expected

    def test_none_in_none_out(self):
        assert format_age(None) is None

    def test_a_future_instant_has_no_age(self):
        # A recording made on a host whose clock runs fast. format_wait would
        # have called this "overdue", which read as "captured overdue ago".
        assert format_age(-600) is None

    def test_small_negatives_are_tolerated_as_now(self):
        # Clock jitter, not skew.
        assert format_age(-5) == "<1m"

    def test_the_tolerance_is_adjustable(self):
        assert format_age(-120, tolerance=300) == "<1m"
        assert format_age(-120, tolerance=60) is None

    def test_it_never_says_now_or_overdue(self):
        # Those are format_wait's vocabulary and both are wrong for an age.
        for s in (-30, 0, 30, 10**7):
            assert format_age(s) not in {"now", "overdue"}
