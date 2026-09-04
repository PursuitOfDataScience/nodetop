"""The product label says how much HBM the card has, and it was thrown away.

`DESIGN.md` argues that accelerator memory is an inference because "no
scheduler records it", and that is true of every scheduler's *memory field* --
which is host RAM.  It is not true of the product string, which the same
paragraph quotes verbatim: ``nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB``.
Kubernetes writes the size into the label.  Sites write it into a Slurm
feature as ``a100-80gb``.  `hardware.py`'s own alias table already lists
``a10080gb`` and ``h10080gb`` because that is how the token arrives.

Every one of those matched the ``A100`` row and inherited its *conservative*
40 with ``memory_certain=False``, so on a cluster of 80 GiB A100s::

    $ nodetop where -g 1 --gpu-mem 80
    A100 has 40 GiB, need 80 (inferred from model)

...ruled out every node, off a label that said 80.  A false negative is the
expensive direction here: the conservative default exists to turn an OOM into a
warning, and instead it turned a fit into a refusal.

**The unit is settled, which is what makes the comparison sound.**  The table
stores the vendor's figure and renders it "GiB", and the vendor's figure is
binary.  NVIDIA's MIG user guide shows an ``A100-SXM4-40GB`` reporting
``0MiB / 40537MiB`` of framebuffer.  Decimal 40 GB is 38146 MiB (37.25 GiB) --
2390 MiB *less* than the card reports -- while 40 x 2**30 is 40960 MiB, of which
40537 is all but a ~1% ECC and reserved carve-out.  The same guide's
``mig -lgip`` table heads its Memory column ``GiB`` and lists the
eighth-of-a-card ``1g.5gb`` profile as ``4.75``, so NVIDIA's own "5gb" is 5 GiB
less reserve rather than 4.66.  That is the JEDEC reading (JESD100B.01: ``G`` is
2**30 in front of a semiconductor memory capacity), the same convention that
makes Slurm's ``G`` binary.  ``--gpu-mem 80`` and a table entry of ``80`` are
therefore the same quantity, and pinning one from the other is like for like.

Selecting is not guessing.  A label may only choose a size the table already
declares in ``memory_variants``; it may never add one.
"""

from __future__ import annotations

import pytest

from nodetop.core.capacity import hardware_ok
from nodetop.core.hardware import ACCELERATORS, identify_accelerator
from nodetop.core.model import JobShape, Node

K8S = "nvidia.com/gpu.product="


def _node(label: str, name: str = "n1") -> Node:
    return Node(
        name=name,
        cpus_total=128,
        memory_mb=1024 * 1024,
        gpus_total=8,
        accelerator=identify_accelerator(None, label),
    )


class TestTheLabelPinsTheSize:
    @pytest.mark.parametrize(
        ("label", "size"),
        [
            (K8S + "NVIDIA-A100-SXM4-80GB", 80),
            (K8S + "NVIDIA-A100-SXM4-40GB", 40),
            (K8S + "NVIDIA-A100-PCIE-40GB", 40),
            (K8S + "NVIDIA-H100-PCIe-80GB", 80),
            # The size is NOT always last. ``nvidia-smi -L`` names the 80 GiB
            # PCIe part "NVIDIA A100 80GB PCIe" and the feature-discovery label
            # follows, so an end-anchored read would miss the flagship card.
            (K8S + "NVIDIA-A100-80GB-PCIe", 80),
            (K8S + "NVIDIA-H100-80GB-HBM3", 80),
            (K8S + "Tesla-V100-SXM2-32GB", 32),
            (K8S + "Tesla-V100-SXM2-16GB", 16),
            # A Slurm feature, in the three spellings admins actually type.
            ("a100-80gb", 80),
            ("a100_80gb", 80),
            ("A100-80GB", 80),
            ("gold-6346,512g,a100-80gb,ib", 80),
        ],
    )
    def test_a_declared_size_in_the_label_is_read_back(self, label, size):
        spec = identify_accelerator(None, label)
        assert spec is not None, label
        assert spec.memory_gb == size, label
        assert spec.memory_certain is True, label

    def test_the_eighty_gigabyte_node_now_fits_an_eighty_gigabyte_job(self):
        """The reported symptom, end to end through the gate that refused it."""
        ok, why = hardware_ok(
            _node(K8S + "NVIDIA-A100-SXM4-80GB"),
            JobShape(gpus_per_node=1, gpu_memory_gb=80, nodes=1),
        )
        assert ok is True, why

    def test_a_forty_gigabyte_node_still_refuses_and_no_longer_hedges(self):
        """The refusal must survive -- and stop calling itself an inference.

        "(inferred from model)" was the honest thing to say while 40 was an
        assumption.  Once the label states 40 it is a measurement, and the
        hedge invites the reader to override a correct answer.
        """
        ok, why = hardware_ok(
            _node(K8S + "NVIDIA-A100-SXM4-40GB"),
            JobShape(gpus_per_node=1, gpu_memory_gb=80, nodes=1),
        )
        assert ok is False
        assert any("A100 has 40 GiB, need 80" in w for w in why), why
        assert not any("inferred from model" in w for w in why), why

    def test_the_size_is_read_off_the_raw_label_not_the_normalised_one(self):
        """``_normalise`` would make ``SXM4-40GB`` read as **440**.

        Stripping the separators runs the form factor into the size, so a plain
        ``(\\d+)gb$`` over the normalised token borrows the ``4`` from ``SXM4``.
        Nothing in ``memory_variants`` is 440, so the bug would have shown up
        as a silent no-op -- which is why it is pinned here explicitly.
        """
        spec = identify_accelerator(None, K8S + "NVIDIA-A100-SXM4-40GB")
        assert spec is not None
        assert spec.memory_gb == 40
        assert spec.memory_gb != 440


class TestSelectingIsNotGuessing:
    def test_an_undeclared_size_leaves_the_conservative_default_alone(self):
        """A 96 GiB A100 does not exist, so the table must not invent one.

        The failure mode that matters: a label the vocabulary half-recognises
        must not become a capacity claim.  Unknown stays unknown, and stays
        marked unknown.
        """
        spec = identify_accelerator(None, K8S + "NVIDIA-A100-SXM4-96GB")
        assert spec is not None
        assert spec.model == "A100"
        assert spec.memory_gb == 40
        assert spec.memory_certain is False

    def test_a_part_that_ships_in_one_size_is_never_rewritten(self):
        """No ``memory_variants`` means the model name pins it; hands off."""
        spec = identify_accelerator(None, K8S + "NVIDIA-L40S")
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (48, True)

    def test_the_shared_table_is_not_mutated(self):
        """`ACCELERATORS` is process-wide; pinning one node must not edit it.

        `AcceleratorSpec` is a frozen dataclass and the pin goes through
        `replace`, so this holds by construction -- and it is asserted anyway,
        because the alternative is one k8s label silently redefining the A100
        for every other backend in the same process.
        """
        identify_accelerator(None, K8S + "NVIDIA-A100-SXM4-80GB")
        assert ACCELERATORS["A100"].memory_gb == 40
        assert ACCELERATORS["A100"].memory_certain is False


class TestControls:
    """Pass identically with the change present or absent."""

    def test_control_a_bare_model_token_is_still_an_inference(self):
        """CONTROL. ``a100`` says nothing about the size, and must not start to.

        This is the case `DESIGN.md` is about, and 90 of 91 GPU nodes on the
        reference cluster look like this. The conservative variant and the
        ``memory_certain=False`` flag both have to survive untouched.
        """
        spec = identify_accelerator(None, "gold-6346,256g,a100")
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_a_typed_resource_carries_no_size(self):
        """CONTROL. ``gpu:a100:4`` is authoritative about the model only."""
        spec = identify_accelerator("gpu:a100:4", None)
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_a_host_ram_label_is_not_an_accelerator(self):
        """CONTROL. ``256g`` is the node's RAM and identifies nothing.

        The guard that matters most: this change reads sizes out of labels, and
        a label list is mostly sizes that have nothing to do with the GPU.
        """
        assert identify_accelerator(None, "256g") is None
        assert identify_accelerator(None, "1.5T,768g,gold-6346") is None

    def test_control_the_hbm_label_reads_gibibytes(self):
        """CONTROL. The unit on the rendered figure, unchanged by the pin."""
        text = JobShape(gpus_per_node=1, gpu_memory_gb=40, nodes=1).describe()
        assert ">=40 GiB HBM" in text, text

    def test_control_the_comparison_arithmetic_is_untouched(self):
        """CONTROL. This fixed which number is compared, not how.

        40 GiB against a 40 GiB ask fits; 41 does not. Both hold in either
        state, so a passing pin cannot be hiding a changed threshold.
        """
        node = _node(K8S + "NVIDIA-A100-SXM4-40GB")
        assert hardware_ok(node, JobShape(gpus_per_node=1, gpu_memory_gb=40))[0] is True
        assert hardware_ok(node, JobShape(gpus_per_node=1, gpu_memory_gb=41))[0] is False


class TestTheMixedRowDoesNotLie:
    """`accelerators` keys its rows on the model, so both A100 sizes share one.

    The pin above is what makes this reachable: while every A100 carried the
    same conservative 40, reading the row off ``group[0]`` was harmless. Now
    ``group[0]`` is whichever node sorted first, and a row holding both sizes
    could print ``80`` with no ``>=`` -- claiming 80 GiB cards on nodes that
    have 40.
    """

    def test_a_mixed_row_reports_the_smaller_size_as_a_floor(self):
        from nodetop.cli import _group_memory

        group = [_node(K8S + "NVIDIA-A100-SXM4-80GB", "a"),
                 _node(K8S + "NVIDIA-A100-SXM4-40GB", "b")]
        assert _group_memory(group) == (40, False)
        # Order must not decide it.
        assert _group_memory(list(reversed(group))) == (40, False)

    def test_a_row_where_every_node_agrees_is_exact(self):
        from nodetop.cli import _group_memory

        group = [_node(K8S + "NVIDIA-A100-SXM4-80GB", "a"),
                 _node(K8S + "NVIDIA-A100-SXM4-80GB", "b")]
        assert _group_memory(group) == (80, True)

    def test_control_an_unpinned_row_is_exactly_what_it_was(self):
        """CONTROL. Bare ``a100`` features: ``>=40``, in both states."""
        from nodetop.cli import _group_memory

        group = [_node("a100", "a"), _node("a100", "b")]
        assert _group_memory(group) == (40, False)


class TestAMigSliceIsNotAWholeCard:
    """A MIG label names the card's size and hands out a fraction of it.

    ``NVIDIA-A100-SXM4-40GB-MIG-1g.5gb`` is a 40 GiB A100 partitioned into
    eighths: the allocatable unit has 4.75 GiB behind it (NVIDIA's own
    ``mig -lgip`` figure). Pinning 40 as CERTAIN would let a 40 GiB job onto a
    4.75 GiB allocation -- an error in the OOM direction, which is the one the
    conservative default exists to prevent. So the pin declines and the row goes
    back to being an inference.

    This is the guard that the end-anchor used to provide by accident; relaxing
    the anchor for ``A100-80GB-PCIe`` is what makes it load-bearing.
    """

    @pytest.mark.parametrize(
        "label",
        [
            K8S + "NVIDIA-A100-SXM4-40GB-MIG-1g.5gb",
            K8S + "NVIDIA-A100-SXM4-80GB-MIG-3g.40gb",
            K8S + "NVIDIA-A100-SXM4-40GB-MIG-7g.40gb",
        ],
    )
    def test_a_mig_label_is_never_pinned(self, label):
        spec = identify_accelerator(None, label)
        assert spec is not None, label
        assert spec.model == "A100", label
        assert spec.memory_certain is False, label
        assert spec.memory_gb == 40, label

    def test_a_mig_node_still_refuses_a_whole_card_job(self):
        """And says so as an inference, which is the honest hedge here."""
        ok, why = hardware_ok(
            _node(K8S + "NVIDIA-A100-SXM4-80GB-MIG-3g.40gb"),
            JobShape(gpus_per_node=1, gpu_memory_gb=80, nodes=1),
        )
        assert ok is False
        assert any("inferred from model" in w for w in why), why

    def test_a_digit_may_not_run_into_the_size(self):
        """``140GB`` must not read as ``40GB``.

        Same class of defect as ``SXM4-40GB`` reading as 440, from the other
        side: without the lookbehind, any variant that is a numeric suffix of
        the label's number would match.
        """
        spec = identify_accelerator(None, K8S + "NVIDIA-A100-140GB")
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (40, False)
