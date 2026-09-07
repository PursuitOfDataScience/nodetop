"""An unmeasured ROCm card was counted as free, which is the documented collision.

`SshPoolBackend.parse_host` decides how many of a host's GPUs are in use, and the
comment on that line states the rule:

    A GPU counts as busy when it holds memory, not when its utilisation is high:
    a job between kernels reads 0% util while still owning the card, and
    **treating that as free is how two jobs collide.**

The NVIDIA branch has three real numbers to work with -- the probe asks
`nvidia-smi` for `name,memory.total,memory.used,utilization.gpu`. The ROCm branch
has none: the probe runs `rocm-smi --showproductname --csv`, which reports a name
and nothing else, so **no memory reading for a ROCm card exists**. It was
recorded as `(name, 0, 0, 0)`, and `busy` reads that zero as "holds no memory".

Measured before the change:

    GPU=NVIDIA A100, 81920, 40000, 0   -> gpus_total=1 gpus_alloc=1
    GPU=NVIDIA A100, 81920, 12, 0      -> gpus_total=1 gpus_alloc=0
    ROCM=... x2                        -> gpus_total=2 gpus_alloc=0    <-- always free

So on an AMD pool every card reported free however much was running on it. That
is the collision the comment names, and it is the opposite of the direction this
module states *twice*: `cpus_alloc` rounds load UP because "understating
occupancy overstates free capacity, which is the direction that gets two jobs
started on the same cores".

`-1` now marks "never measured" and `busy` counts it as occupied -- the only safe
reading of an unknown being the one that does not hand the card to a second job.
Nothing else reads those slots: `busy` is the sole consumer, and the tuple never
leaves `parse_host` (`model` and `labels` take element 0, `gpus_total` takes the
length).

The discrimination matters as much as the direction, so it is pinned: a host with
an **idle NVIDIA card and a ROCm card** must report one busy, not two.
`TestControls` holds the NVIDIA arithmetic, which does not move.
"""

import pytest

from nodetop.backends.sshpool import SshPoolBackend

HEAD = "CPUS=8\nLOAD=1\nMEMTOTAL=64000\nMEMAVAIL=32000\n"


def _node(*gpu_lines):
    backend = SshPoolBackend.__new__(SshPoolBackend)
    return backend.parse_host("h1", HEAD + "".join(line + "\n" for line in gpu_lines))


class TestAnUnmeasuredCardIsNotOffered:
    def test_a_single_rocm_card_is_not_reported_free(self):
        node = _node("ROCM=AMD Instinct MI250X")
        assert node.gpus_total == 1
        assert node.gpus_alloc == 1, "an unmeasured card was offered as free"

    @pytest.mark.parametrize("count", [1, 2, 4, 8])
    def test_every_rocm_card_counts_as_occupied(self, count):
        node = _node(*["ROCM=AMD Instinct MI250X"] * count)
        assert node.gpus_total == count
        assert node.gpus_alloc == count

    def test_an_idle_nvidia_card_beside_a_rocm_one_is_still_free(self):
        """The discrimination: only the UNMEASURED card is assumed occupied."""
        node = _node("GPU=NVIDIA A100, 81920, 12, 0", "ROCM=AMD MI210")
        assert node.gpus_total == 2
        assert node.gpus_alloc == 1, node.gpus_alloc

    def test_a_busy_nvidia_card_beside_a_rocm_one_makes_two(self):
        node = _node("GPU=NVIDIA A100, 81920, 40000, 0", "ROCM=AMD MI210")
        assert node.gpus_alloc == 2

    def test_the_probe_really_asks_rocm_smi_for_no_memory(self):
        """Vacuity guard: if the probe supplied a memory figure, the right fix
        would be to parse it, not to assume the card is busy."""
        from nodetop.backends.sshpool import _PROBE_SCRIPT

        rocm = [ln for ln in _PROBE_SCRIPT.splitlines() if "rocm-smi" in ln]
        assert rocm, _PROBE_SCRIPT
        joined = " ".join(rocm)
        assert "--showproductname" in joined, joined
        for asks_memory in ("--showmeminfo", "--showmemuse", "memory.used"):
            assert asks_memory not in joined, joined


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_the_rocm_name_still_reaches_the_labels(self):
        """Passes in BOTH states -- verified by neutering, so it is a control.

        The name is the ONE figure `rocm-smi --showproductname` does supply, and
        the fix rewrites the tuple it sits in. That it survived is exactly what a
        control is for: element 0 is what `model` and `labels` read.
        """
        node = _node("ROCM=AMD Instinct MI250X")
        assert node.labels == ("gpu0=AMD Instinct MI250X",)

    def test_a_busy_nvidia_card_is_still_busy(self):
        assert _node("GPU=NVIDIA A100, 81920, 40000, 0").gpus_alloc == 1

    def test_an_idle_nvidia_card_is_still_free(self):
        assert _node("GPU=NVIDIA A100, 81920, 12, 0").gpus_alloc == 0

    def test_the_512_mib_threshold_is_unmoved(self):
        # The line between "holds memory" and "idle" is the same one.
        assert _node("GPU=NVIDIA A100, 81920, 513, 0").gpus_alloc == 1
        assert _node("GPU=NVIDIA A100, 81920, 512, 0").gpus_alloc == 0

    def test_a_host_with_no_gpus_is_unchanged(self):
        node = _node()
        assert node.gpus_total == 0 and node.gpus_alloc == 0
        assert node.labels == ()

    def test_a_malformed_gpu_line_is_still_ignored(self):
        # `GPU=` needs four fields; three is not a card.
        node = _node("GPU=NVIDIA A100, 81920, 40000")
        assert node.gpus_total == 0

    def test_the_nvidia_model_still_identifies_an_accelerator(self):
        node = _node("GPU=NVIDIA A100, 81920, 12, 0")
        assert node.accelerator_label == "NVIDIA A100"

    def test_the_host_facts_are_untouched(self):
        node = _node("ROCM=AMD MI210")
        assert node.cpus_total == 8
        assert node.memory_mb == 64000
        assert node.memory_alloc_mb == 32000
        assert node.queues == ("pool",)
        assert node.unreachable is False

    def test_an_unreachable_host_is_still_unreachable(self):
        backend = SshPoolBackend.__new__(SshPoolBackend)
        node = backend.parse_host("h1", "")
        assert node.unreachable is True
        assert node.gpus_total == 0 and node.gpus_alloc == 0
