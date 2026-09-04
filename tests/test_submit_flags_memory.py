"""The flags handed over to be pasted must not ask for less memory than the shape.

`--mem` takes a fractional size -- `--mem 1.5G` is accepted and the Slurm backend
emits an exact `--mem=1536M` for it. Two backends built their memory flag with
`int(shape.memory_gb)` instead, which throws the remainder away *downwards*:

    asked        slurm          pbs (before)     sge (before)
    0.5 GiB      --mem=512M     mem=0gb          h_vmem=0G
    0.9 GiB      --mem=921M     mem=0gb          h_vmem=0G
    1.5 GiB      --mem=1536M    mem=1gb          h_vmem=1G
    3.7 GiB      --mem=3788M    mem=3gb          h_vmem=3G

Two different failures. Above 1 GiB the request is short by up to a gigabyte, and
under-asking for memory is what gets a job killed -- the tool would have computed
a placement for 3.7 GiB and then handed over a command asking for 3. Below 1 GiB
it collapses to **zero**, which is not a smaller request but a meaningless one.

All four backends now derive the figure from `shape.memory_mb_per_node`. The
gigabyte spelling is kept where it is exact -- `mem=64gb` reads better than
`mem=65536mb` in a command someone is about to paste, and that is what almost
every request is -- and megabytes appear only where gigabytes would lose
something. Two existing backend tests asserted the gigabyte spelling, and a first
version of this fix switched everything to megabytes and broke both of them for
no gain; they were right to complain.

These two functions were among the ones in this package carrying neither a
docstring nor a comment, which is how the difference from the Slurm path went
unremarked.
"""

from __future__ import annotations

import re

import pytest

from nodetop.backends.lsf import LsfBackend
from nodetop.backends.pbs import PbsBackend
from nodetop.backends.sge import SgeBackend
from nodetop.backends.slurm import SlurmBackend
from nodetop.core.model import JobShape

#: Sizes a user can actually type. The sub-gigabyte pair is the one that used to
#: produce a request for nothing.
SIZES = [0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5, 3.7, 15.5, 64.25, 500.75]


def _slurm_mb(flags: list[str]) -> int | None:
    return next((int(a.split("=")[1].rstrip("M")) for a in flags if a.startswith("--mem=")), None)


def _pbs_mb(flags: list[str]) -> int | None:
    """Accepts either spelling: `mem=64gb` when exact, `mem=1536mb` when not."""
    text = " ".join(flags)
    found = re.search(r"mem=(\d+)(gb|mb)", text)
    if not found:
        return None
    return int(found.group(1)) * (1024 if found.group(2) == "gb" else 1)


def _sge_mb(flags: list[str]) -> int | None:
    """Accepts either spelling: `h_vmem=64G` when exact, `h_vmem=1536M` when not."""
    found = re.search(r"h_vmem=(\d+)([GM])", " ".join(flags))
    if not found:
        return None
    return int(found.group(1)) * (1024 if found.group(2) == "G" else 1)


def _lsf_mb(flags: list[str]) -> int | None:
    """LSF spells it as two argv tokens, `-M <megabytes>`.

    Read positionally rather than by pattern: a bare-number regex over the whole
    flag list also matches `-W 60` and the core count, which is how a first
    version of this file reported four failures against correct code.
    """
    for index, token in enumerate(flags):
        if token == "-M" and index + 1 < len(flags):
            return int(flags[index + 1])
    return None


#: ``(name, backend class, megabyte extractor)``.
BACKENDS = [
    ("slurm", SlurmBackend, _slurm_mb),
    ("pbs", PbsBackend, _pbs_mb),
    ("sge", SgeBackend, _sge_mb),
    ("lsf", LsfBackend, _lsf_mb),
]

IDS = [name for name, _cls, _fn in BACKENDS]


def _emitted_mb(backend_cls, extract, shape):
    """The megabyte figure a backend put in its flags, or None if it emitted none."""
    return extract(backend_cls().submit_flags("q", shape))


class TestNoBackendAsksForLessThanTheShape:
    @pytest.mark.parametrize("gb", SIZES)
    @pytest.mark.parametrize("name,cls,extract", BACKENDS, ids=IDS)
    def test_the_request_is_exact(self, name, cls, extract, gb):
        shape = JobShape(nodes=1, cpus_per_task=2, memory_gb=gb, walltime="01:00:00")
        want = shape.memory_mb_per_node
        got = _emitted_mb(cls, extract, shape)
        assert got is not None, f"{name} emitted no memory flag for {gb} GiB"
        assert got == want, (
            f"{name} asks for {got} MiB where the shape is {want} MiB "
            f"({gb} GiB) -- a job sized for the shape would be killed"
        )

    @pytest.mark.parametrize("name,cls,extract", BACKENDS, ids=IDS)
    def test_a_sub_gigabyte_request_is_not_zero(self, name, cls, extract):
        # The worst of the two failures: `int(0.5)` is 0, and a flag asking for
        # no memory is not a smaller request, it is a broken one.
        shape = JobShape(nodes=1, cpus_per_task=1, memory_gb=0.5, walltime="01:00:00")
        assert _emitted_mb(cls, extract, shape) == 512

    def test_all_four_backends_agree_on_the_number(self):
        # The property that keeps them from drifting again: one shape, one figure,
        # whatever the scheduler's spelling of it.
        shape = JobShape(nodes=1, cpus_per_task=2, memory_gb=1.5, walltime="01:00:00")
        seen = {name: _emitted_mb(cls, fn, shape) for name, cls, fn in BACKENDS}
        assert len(set(seen.values())) == 1, seen
        assert set(seen.values()) == {1536}, seen


class TestTheZeroAndUnsetCasesAreUnchanged:
    """Controls: an absent request must stay absent, not become `mem=0`."""

    @pytest.mark.parametrize("name,cls,extract", BACKENDS, ids=IDS)
    def test_no_memory_asked_means_no_memory_flag(self, name, cls, extract):
        shape = JobShape(nodes=1, cpus_per_task=2, memory_gb=0.0, walltime="01:00:00")
        assert _emitted_mb(cls, extract, shape) is None, (
            f"{name} emitted a memory flag for a shape that asked for none"
        )

    @pytest.mark.parametrize("name,cls,extract", BACKENDS, ids=IDS)
    def test_the_rest_of_the_flags_still_appear(self, name, cls, extract):
        # A change to the memory branch must not disturb its neighbours.
        shape = JobShape(nodes=1, cpus_per_task=4, memory_gb=2.0, walltime="02:00:00")
        flags = " ".join(cls().submit_flags("myqueue", shape))
        assert "myqueue" in flags
        assert "4" in flags, flags
