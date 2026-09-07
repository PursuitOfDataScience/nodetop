"""`name_accelerator` dropped a card that `identify_accelerator` recognised.

The two are companions and `name_accelerator`'s docstring sets out the division
of labour: `identify_accelerator` answers "what can this card do" and "must
return ``None`` rather than guess", while `name_accelerator` answers "what did
the scheduler call it, **vocabulary or not**", which is "a fact the node record
already carries and which a report has no reason to throw away."

They disagreed. `rocm-smi --showproductname` answers with a full product name, so
`SshPoolBackend.parse_host` handed both functions `AMD Instinct MI250X`:

    identify_accelerator(None, "AMD Instinct MI250X")
      -> AcceleratorSpec(model='MI250X', vendor='AMD', arch='gfx90a',
                         memory_gb=128, memory_certain=True, ...)
    name_accelerator(None, "AMD Instinct MI250X")   -> None      <-- dropped

so a ROCm host's `accelerator_label` came out empty while nodetop knew exactly
what the card was. `_GPU_NAME_SHAPED` is anchored and its alternation began at
`instinct`; `_normalise("AMD Instinct MI250X")` is `amdinstinctmi250x`, which
starts with the vendor. `nvidia` is accepted bare in the same alternation, so
`nvidiaa100` matched and `amdinstinctmi250x` did not.

**`amd` is admitted only as an optional prefix on a family already listed**, and
that narrowness is the point. NVIDIA ships no CPUs; AMD does, and
`_NON_ACCELERATOR_LABEL` refuses only the `amd[-_]epyc` spellings -- measured, it
passes the SPACE-separated product names. A bare `amd[a-z0-9]+` shape would
therefore have named `AMD EPYC 7763` and `AMD Opteron 6376` as this node's
accelerator, which is the "a wrong name is worse than the shrug it replaces"
failure the block was narrowed to avoid. Both are pinned as negatives below.

The invariant is stated as the pairing rather than a list of models, because the
pairing is what broke: anything the spec table can identify, the report must be
able to name. `a100` and `h100` stay refused on the *labels* path on purpose --
a typed GRES reaches `name_accelerator` through `resource`, where the scheduler
has already said the field is a GPU type and no shape test applies.
"""

import pytest

from nodetop.core.hardware import identify_accelerator, name_accelerator

#: Product strings a probe or a node feature list really produces.
PRODUCT_STRINGS = [
    "AMD Instinct MI250X",
    "AMD Instinct MI300X",
    "AMD MI210",
    "amd mi300x",
    "AMD Radeon Pro W6800",
    "Instinct MI100",
    "mi250x",
    "NVIDIA A100",
    "NVIDIA H100 80GB HBM3",
    "Tesla V100-SXM2-32GB",
]

#: Not accelerators. The first two are the reason `amd` is not a bare prefix.
NOT_ACCELERATORS = [
    "AMD EPYC 7763",
    "AMD Opteron 6376",
    "amd-epyc-7763",
    "amd_epyc_7452",
    "amd64",
    "nvidia-driver-470",
    "p9",
    "dlc",
    "ib",
    "256g",
]


class TestTheTwoCompanionsAgree:
    @pytest.mark.parametrize("text", PRODUCT_STRINGS)
    def test_anything_identifiable_can_also_be_named(self, text):
        """The invariant that broke: knowing the card but not its name."""
        if identify_accelerator(None, text) is None:
            pytest.skip("not in the spec vocabulary; naming is tested separately")
        assert name_accelerator(None, text), (
            f"{text!r} has a spec but no name -- a report that knows what the "
            f"card can do must print what it is called"
        )

    def test_the_reported_rocm_product_name_is_named(self):
        # The exact string `rocm-smi --showproductname` produced.
        assert name_accelerator(None, "AMD Instinct MI250X") == "AMD Instinct MI250X"

    @pytest.mark.parametrize("text", PRODUCT_STRINGS)
    def test_every_product_string_is_named_verbatim(self, text):
        # `name_accelerator` reports what it was given, not a normalised form.
        assert name_accelerator(None, text) == text

    def test_a_card_with_no_spec_is_still_named(self):
        """The division of labour, in the direction that already worked.

        `AMD MI210` is not resolvable to a spec from that string, and it is
        still named -- which is exactly what "vocabulary or not" means.
        """
        assert identify_accelerator(None, "AMD MI210") is None
        assert name_accelerator(None, "AMD MI210") == "AMD MI210"

    def test_the_rocm_host_carries_the_label_end_to_end(self):
        from nodetop.backends.sshpool import SshPoolBackend

        backend = SshPoolBackend.__new__(SshPoolBackend)
        node = backend.parse_host(
            "h1",
            "CPUS=8\nLOAD=1\nMEMTOTAL=64000\nMEMAVAIL=32000\n"
            "ROCM=AMD Instinct MI250X\nROCM=AMD Instinct MI250X\n",
        )
        assert node.accelerator_label == "AMD Instinct MI250X"
        assert node.accelerator is not None


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize("text", NOT_ACCELERATORS)
    def test_a_non_accelerator_is_still_refused(self, text):
        assert name_accelerator(None, text) is None, (
            f"{text!r} was named as an accelerator"
        )

    def test_a_cpu_product_name_is_refused_by_the_shape_not_the_reject_list(self):
        """Why the prefix is narrow, asserted rather than argued.

        `_NON_ACCELERATOR_LABEL` does NOT catch the space-separated CPU names --
        it only refuses `amd[-_]epyc`. So the shape test is the only thing
        standing between `AMD EPYC 7763` and a wrong accelerator label.
        """
        from nodetop.core.hardware import _NON_ACCELERATOR_LABEL

        assert not _NON_ACCELERATOR_LABEL.match("AMD EPYC 7763")
        assert not _NON_ACCELERATOR_LABEL.match("AMD Opteron 6376")
        assert name_accelerator(None, "AMD EPYC 7763") is None
        assert name_accelerator(None, "AMD Opteron 6376") is None

    @pytest.mark.parametrize("text", ["a100", "h100"])
    def test_a_bare_part_number_is_still_refused_on_the_labels_path(self, text):
        # Documented and deliberate: a typed GRES arrives via `resource`, where
        # no shape test applies, so the labels path need not accept these.
        assert name_accelerator(None, text) is None

    @pytest.mark.parametrize("text", ["a100", "h100", "mi250x"])
    def test_a_typed_gres_still_names_it_without_a_shape_test(self, text):
        assert name_accelerator(f"gpu:{text}:4") == text

    def test_the_nvidia_names_are_untouched(self):
        assert name_accelerator(None, "NVIDIA A100") == "NVIDIA A100"
        assert name_accelerator(None, "Tesla V100-SXM2-32GB") == "Tesla V100-SXM2-32GB"

    def test_a_driver_label_is_still_not_a_card(self):
        assert name_accelerator(None, "nvidia-driver-470") is None
        assert name_accelerator(None, "nvidia_driver_535") is None

    def test_the_named_family_still_wins_over_a_bare_token(self):
        # The tie-break the function documents, unchanged.
        assert name_accelerator(None, ["p9", "ib", "NVIDIA A100"]) == "NVIDIA A100"

    def test_nothing_shaped_still_answers_none(self):
        assert name_accelerator(None, ["ib", "dlc", "256g"]) is None
        assert name_accelerator(None, "") is None
        assert name_accelerator(None, None) is None
