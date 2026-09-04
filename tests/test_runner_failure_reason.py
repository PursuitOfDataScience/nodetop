"""A failed query has to say why, whichever stream the client said it on.

`CommandError` reports stderr, and `Runner.run_full`'s own docstring explains
why that is the right default: a dry-run writes its verdict there and exits
non-zero, so stderr is where a scheduler's diagnosis lives.

`scontrol` is the exception, and it is the client this tool leans on hardest.
Measured on live Slurm 20.11.8 (607 nodes, 87 partitions)::

    $ scontrol show node nosuchnode ; echo "rc=$?"
    Node nosuchnode not found            <-- STDOUT
    rc=1                                 <-- and stderr is EMPTY

    $ scontrol show partition nosuchpart ; echo "rc=$?"
    Partition nosuchpart not found       <-- STDOUT
    rc=1

So every funnel in `runner` raised ``CommandError(cmd, rc, err)`` with `err` the
empty string, and the reader was shown

    CommandError: scontrol show node --all --oneliner exited 1:

-- a reported failure whose reason field is blank. `Cluster.load` stores that
same string under ``errors["nodes"]`` and it is printed verbatim when a query
dies, so the one thing the report exists to carry was the one thing missing,
while stdout held the only copy of it.

The control below is the other half: stderr must still win where it has anything
at all. A fix that concatenated the streams, or preferred stdout, would drop a
partial data dump into an error message -- `scontrol show node --all` prints
423,902 bytes on this cluster.
"""

from __future__ import annotations

import sys

import pytest

from nodetop.exceptions import CommandError
from nodetop.runner import CapturingRunner, RecordedRunner, SubprocessRunner


def _script(stdout: str, stderr: str, rc: int) -> list[str]:
    """A real child process that writes exactly this and exits with `rc`."""
    return [
        sys.executable,
        "-c",
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({rc})\n",
    ]


#: The three funnels, each with a way to make it fail exactly like live
#: `scontrol` does. All three built the error the same way, so all three were
#: blank; a fix to one of them is not a fix.
def _subprocess_case(stdout, stderr, rc):
    return SubprocessRunner(), _script(stdout, stderr, rc)


def _capturing_case(stdout, stderr, rc):
    return CapturingRunner(RecordedRunner({"q": (rc, stdout, stderr)})), ["q"]


def _recorded_case(stdout, stderr, rc):
    return RecordedRunner({"q": (rc, stdout, stderr)}), ["q"]


FUNNELS = [
    pytest.param(_subprocess_case, id="SubprocessRunner"),
    pytest.param(_capturing_case, id="CapturingRunner"),
    pytest.param(_recorded_case, id="RecordedRunner"),
]

# Verbatim from the live cluster.
SCONTROL_STDOUT = "Node nosuchnode not found\n"


@pytest.mark.parametrize("make", FUNNELS)
def test_a_reason_only_on_stdout_is_still_reported(make):
    """The bug: `scontrol`'s explanation was thrown away for being on stdout."""
    runner, cmd = make(SCONTROL_STDOUT, "", 1)

    with pytest.raises(CommandError) as caught:
        runner.run(cmd)

    exc = caught.value
    assert exc.returncode == 1
    assert exc.stderr == "Node nosuchnode not found", (
        "the failure reported no reason at all; stdout held the only copy of it"
    )
    # And it reaches the reader, which is the whole point: this string is what
    # `Cluster.load` files under `errors[label]` and prints when a query dies.
    assert "Node nosuchnode not found" in str(exc)
    assert not str(exc).endswith("exited 1: "), "blank reason"


@pytest.mark.parametrize("make", FUNNELS)
def test_stderr_still_wins_when_the_client_uses_both_streams(make):
    """The control: stdout is a fallback, not an addition.

    A dry-run exits non-zero with its verdict on stderr *and* informative
    output on stdout. Appending or preferring stdout would bury the diagnosis
    under however many bytes of records the query had already emitted.
    """
    runner, cmd = make("PartitionName=alpha\nPartitionName=beta\n", "fatal: bad flag\n", 2)

    with pytest.raises(CommandError) as caught:
        runner.run(cmd)

    exc = caught.value
    assert exc.returncode == 2
    assert exc.stderr == "fatal: bad flag"
    # The reason as rendered, i.e. the tail of the message after the argv. Split
    # rather than searching the whole string: `SubprocessRunner`'s argv is the
    # child's own source here, so the payload legitimately appears in it.
    reason = str(exc).rsplit("exited 2: ", 1)[1]
    assert reason == "fatal: bad flag", "stdout displaced or padded the diagnosis"


@pytest.mark.parametrize("make", FUNNELS)
def test_a_successful_command_is_untouched(make):
    """The other control: nothing about the rc==0 path changed."""
    runner, cmd = make("PartitionName=alpha\n", "warning: ignored\n", 0)
    assert runner.run(cmd) == "PartitionName=alpha\n"


@pytest.mark.parametrize("make", FUNNELS)
def test_run_full_still_hands_back_both_streams_unmerged(make):
    """`run_full` is the unfiltered triple and stays that way.

    The probe paths parse stdout and stderr separately -- `parse_probe` reads a
    verdict out of both -- so the fallback belongs in the error, not in the
    data.
    """
    runner, cmd = make("on-out\n", "on-err\n", 1)
    assert runner.run_full(cmd) == (1, "on-out\n", "on-err\n")
