"""Two real product names were refused a name, one of them identifiable.

Measured against the strings `nvidia-smi --query-gpu=name`, `rocm-smi
--showproductname` and Intel's driver actually emit -- not synthesised
`"<vendor> <model>"` forms, which overstate the problem because no probe emits
them:

    NVIDIA GeForce RTX 4090            spec=None      name=None      <-- both
    Intel(R) Data Center GPU Max 1550  spec=PVC1550   name=None      <-- name only

Two separate causes, both in `_GPU_NAME_SHAPED`:

* the alphanumeric cap was `{1,12}` counted after the vendor word.
  `nvidiaa10080gbpcie` is exactly 12 and squeezed through;
  `nvidiageforcertx4090` is 14 and did not. The cap was cutting real cards off
  at a boundary nothing measured.
* there was no `intel` alternative at all, so the Intel card normalised to
  `intelrdatacentergpumax1550` and matched nothing -- while
  `identify_accelerator` resolved it to PVC1550. That is the companion
  disagreement `name_accelerator`'s docstring forbids: the name is "a fact the
  node record already carries and which a report has no reason to throw away".

**Intel is admitted only in two exact forms, never as a bare prefix**, for the
same reason `amd` is not: Intel ships CPUs, and `_NON_ACCELERATOR_LABEL` refuses
only the `intel[-_]xeon` spellings. Measured, `Intel Xeon Gold 6248` and
`Intel(R) Xeon(R) Platinum 8360Y` reach the shape test unrejected, so a bare
`intel[a-z0-9]+` shape would have named a CPU as the node's accelerator. Both are
pinned as negatives.

Widening the NVIDIA cap is safe for that alternation specifically because NVIDIA
ships no CPUs -- there is no product-name class there to confuse a card with --
and a driver string is still refused by `_DRIVER_LABEL`.

**Scope of the identifiable-implies-nameable invariant, stated because it is not
universal.** It holds for a *vendor-named product string*. It deliberately does
NOT hold for a bare part number: `identify_accelerator(None, "a100")` resolves
via exact alias lookup, while `name_accelerator` refuses it, because the bare
shapes were removed on purpose (`p9`/`p8` are POWER, `m1024` is memory, `a64` an
architecture). Last round's invariant test happened to avoid the contradiction by
listing only prefixed strings; the boundary is asserted here instead of implied.
"""

import pytest

from nodetop.core.hardware import identify_accelerator, name_accelerator

#: What the probes really emit.
REAL_PRODUCTS = [
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA L40S",
    "NVIDIA A40",
    "Tesla V100-SXM2-32GB",
    "Tesla T4",
    "AMD Instinct MI250X",
    "AMD Instinct MI210",
    "AMD Instinct MI300X",
    "Intel(R) Data Center GPU Max 1550",
    "Intel Arc A770",
]

#: CPUs and non-cards. The Intel pair is why `intel` is not a bare prefix.
NOT_CARDS = [
    "Intel Xeon Gold 6248",
    "Intel(R) Xeon(R) Platinum 8360Y",
    "intel-xeon-gold-6248",
    "AMD EPYC 7763",
    "AMD Opteron 6376",
    "nvidia-driver-470",
    "nvidia_driver_535",
    "amd64",
    "x86_64",
    "linux",
    "p9",
    "dlc",
    "ib",
    "256g",
]


class TestEveryRealProductNameIsReported:
    @pytest.mark.parametrize("text", REAL_PRODUCTS)
    def test_it_is_named_verbatim(self, text):
        assert name_accelerator(None, text) == text

    def test_the_geforce_name_survives_the_length_cap(self):
        # 14 alphanumerics after the vendor word; the cap was 12.
        assert name_accelerator(None, "NVIDIA GeForce RTX 4090") == (
            "NVIDIA GeForce RTX 4090"
        )

    def test_the_intel_data_center_gpu_is_named(self):
        text = "Intel(R) Data Center GPU Max 1550"
        assert name_accelerator(None, text) == text

    def test_the_intel_card_no_longer_disagrees_with_its_own_spec(self):
        """The companion disagreement, which is what made this a defect."""
        text = "Intel(R) Data Center GPU Max 1550"
        spec = identify_accelerator(None, text)
        assert spec is not None and spec.model == "PVC1550"
        assert name_accelerator(None, text), "identified but unnameable"

    @pytest.mark.parametrize("text", REAL_PRODUCTS)
    def test_a_named_product_that_has_a_spec_is_never_unnameable(self, text):
        """The invariant, scoped to vendor-named product strings (see module
        docstring for why it is not universal)."""
        if identify_accelerator(None, text) is None:
            pytest.skip("no spec for this card; naming is asserted above")
        assert name_accelerator(None, text)

    def test_the_boundary_case_that_used_to_just_fit_still_fits(self):
        # Exactly 12 after the vendor -- it passed before and must still pass.
        assert name_accelerator(None, "NVIDIA A100 80GB PCIe") == (
            "NVIDIA A100 80GB PCIe"
        )


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize("text", NOT_CARDS)
    def test_a_non_card_is_still_refused(self, text):
        assert name_accelerator(None, text) is None, f"{text!r} was named"

    def test_the_intel_cpu_names_are_refused_by_the_shape_not_the_reject_list(self):
        """Why `intel` is not a bare prefix, asserted rather than argued."""
        from nodetop.core.hardware import _NON_ACCELERATOR_LABEL

        for cpu in ("Intel Xeon Gold 6248", "Intel(R) Xeon(R) Platinum 8360Y"):
            assert not _NON_ACCELERATOR_LABEL.match(cpu), cpu
            assert name_accelerator(None, cpu) is None, cpu

    def test_a_bare_part_number_is_still_refused(self):
        # Deliberate, and the reason the invariant above is scoped: these are
        # identifiable by exact alias and still must not be named by shape.
        for bare in ("a100", "h100"):
            assert identify_accelerator(None, bare) is not None, bare
            assert name_accelerator(None, bare) is None, bare

    def test_a_driver_string_is_still_not_a_card(self):
        # The widened cap must not let a driver label through.
        assert name_accelerator(None, "nvidia-driver-470") is None
        assert name_accelerator(None, "nvidia_driver_535") is None
        assert name_accelerator(None, "nvidia-firmware-12") is None

    def test_the_amd_prefix_from_the_previous_round_still_works(self):
        assert name_accelerator(None, "AMD Instinct MI250X") == "AMD Instinct MI250X"
        assert name_accelerator(None, "AMD MI210") == "AMD MI210"
        assert name_accelerator(None, "mi250x") == "mi250x"

    def test_a_typed_gres_still_bypasses_the_shape_test(self):
        for token in ("a100", "h100", "mi250x"):
            assert name_accelerator(f"gpu:{token}:4") == token

    def test_the_named_family_still_wins_over_a_bare_token(self):
        assert name_accelerator(None, ["p9", "ib", "NVIDIA A100"]) == "NVIDIA A100"

    def test_nothing_shaped_still_answers_none(self):
        assert name_accelerator(None, ["ib", "dlc", "256g"]) is None
        assert name_accelerator(None, "") is None
        assert name_accelerator(None, None) is None
