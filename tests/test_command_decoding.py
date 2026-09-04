"""Reading a scheduler's output must not depend on the reader's locale.

`subprocess.run(..., text=True)` with no `encoding=` decodes using the PARENT
process's preferred encoding. Under `LC_ALL=C` with coercion off -- an ordinary
cron or batch environment, and the default on older cluster images -- that is
`ANSI_X3.4-1968`, pure ASCII. The bytes being decoded are not ours: a job name, a
node's `Reason`, a fileset name, an account description are all free text somebody
typed. One accented character was enough.

Measured in nodetop: a node whose `Reason` read `disk caf\\xc3\\xa9 replaced` made all
four queries raise `UnicodeDecodeError`, and `nodetop status` reported "no data: every
query failed, so there is nothing to report" with exit 3 -- on a cluster that was
answering perfectly well. The verdict was not merely noisy, it was the wrong verdict:
after the fix the same input reports "the queries answered, but nothing could be read
from them", which is what had actually happened.

The two kwargs are pinned through an *invalid* UTF-8 byte rather than by reading
the source, because that one input discriminates all three states: a locale-ASCII
decode raises, `encoding="utf-8"` alone raises, and only the pair returns text.
"""

from __future__ import annotations

from nodetop.runner import SubprocessRunner


def _read(cmd: list[str]) -> str:
    """Through the one runner every Slurm query in this package goes through."""
    _rc, out, _err = SubprocessRunner().run_full(cmd, timeout=30)
    return out


#: Valid UTF-8, the measured case: a fileset or a node Reason with an accent.
ACCENTED = ["/bin/sh", "-c", "printf 'caf\\303\\251 ok'"]

#: NOT valid UTF-8 -- a lone 0xff. Slurm will not normally emit this, but a
#: multi-byte sequence truncated at a buffer boundary has the same shape, and
#: `errors="replace"` is what makes both survivable.
INVALID = ["/bin/sh", "-c", "printf 'a\\377b'"]


class TestOutputIsDecodedAsUtf8RegardlessOfLocale:
    def test_accented_output_reads_back_intact(self):
        assert "caf\u00e9 ok" in _read(ACCENTED)

    def test_an_invalid_byte_is_replaced_rather_than_raised(self):
        # Fails with no `encoding=` (the locale may be ASCII) and also with
        # `encoding="utf-8"` but no `errors=` (a strict UTF-8 decode rejects 0xff).
        text = _read(INVALID)
        assert text.startswith("a") and text.endswith("b"), repr(text)
        assert "\ufffd" in text, f"0xff was not replaced: {text!r}"

    def test_plain_ascii_is_unaffected(self):
        # The control: the ordinary case must read back byte for byte.
        assert _read(["/bin/sh", "-c", "printf 'PARTITION AVAIL'"]) == "PARTITION AVAIL"

    def test_the_byte_this_rests_on_really_is_undecodable(self):
        """Pins the premise rather than assuming it.

        If a future Python decodes 0xff as UTF-8, the test above would pass for
        the wrong reason; this one fails instead and says so.
        """
        try:
            b"a\xffb".decode("utf-8")
        except UnicodeDecodeError:
            return
        raise AssertionError("0xff decodes cleanly as UTF-8; this test needs a new byte")
