"""Naming the memory size in a model token cost the card its identity.

The alias table carries size-suffixed entries for two parts only --
``a10040gb``/``a10080gb`` and ``h10080gb``, hand-written because sites spell it
that way -- and nothing generates the rest. So the moment a size appeared in the
string, every other multi-variant part resolved to nothing::

    identify_accelerator("gpu:v100-32gb:4")     -> None
    identify_accelerator("gpu:gh200-144gb:4")   -> None
    identify_accelerator("gpu:pvc-128gb:4")     -> None
    identify_accelerator("gpu:h100-94gb:4")     -> None

while the bare spelling answered ``V100 16``, ``GH200 96``, ``PVC 48``,
``H100 80``. ``None`` is not a more conservative answer than those: it is a
different one. `identify_accelerator` returning ``None`` means the row prints
``arch -``, ``mem -``, ``bf16 unknown`` and the node is counted in no capability
claim, so a fleet of 32 GiB V100s read as a fleet of unidentifiable
accelerators -- and the reader could not tell that from a cluster with no GPUs
at all.

Both surfaces were affected, and the sized spelling is the one the module
documents as typical: `_pin_memory_from_label` says "a Slurm feature is written
``a100-80gb``". It reached the memory pin for one vendor's one part.

`_spec_for_token` retries the lookup with the size removed, as a *fallback*
after the exact lookup has already missed, so nothing that resolved before
resolves differently. The size is still read off the raw value by
`_pin_memory_from_label` under all of its rules.
"""

from __future__ import annotations

import pytest

from nodetop.core.hardware import ACCELERATORS, identify_accelerator


def _answer(spec):
    return None if spec is None else (spec.model, spec.memory_gb, spec.memory_certain)


# Every part the table declares more than one size for, with the sizes it
# declares -- read off ACCELERATORS rather than typed out, so a part gaining a
# variant is covered without editing this list.
_SIZED = [
    (spec.model, variant)
    for spec in ACCELERATORS.values()
    for variant in spec.memory_variants
]


def test_the_table_declares_variants_for_more_than_the_two_that_had_aliases():
    """Guard for the parametrisation: the finding is about the parts BEYOND A100.

    If this list ever collapsed to A100 the cases below would still pass while
    testing nothing, which is this suite's named failure mode.
    """
    models = {model for model, _ in _SIZED}
    assert models >= {"V100", "A100", "H100", "GH200", "PVC"}, models
    assert len(_SIZED) >= 10


@pytest.mark.parametrize(("model", "size"), _SIZED)
def test_a_typed_gres_that_names_the_size_still_names_the_card(model, size):
    got = _answer(identify_accelerator(f"gpu:{model.lower()}-{size}gb:4"))
    assert got == (model, size, True), got


@pytest.mark.parametrize(("model", "size"), _SIZED)
def test_a_node_feature_that_names_the_size_still_names_the_card(model, size):
    got = _answer(identify_accelerator(None, f"{model.lower()}-{size}gb"))
    assert got == (model, size, True), got


@pytest.mark.parametrize("spelling", ["v100_32gb", "V100-32GB", "v100 32gb", "v100-32GiB"])
def test_the_separator_and_the_case_do_not_decide(spelling):
    """`_normalise` already made the model half spelling-blind; the size half is
    read off the raw value, so it is checked here in the forms a site writes."""
    assert _answer(identify_accelerator(f"gpu:{spelling}:4")) == ("V100", 32, True)


def test_an_undeclared_size_leaves_the_conservative_variant_standing():
    """The claim `_pin_memory_from_label` already made and could not deliver.

    Its docstring says a label naming a size the table has never heard of "leaves
    the conservative default AND ``memory_certain=False`` in place, so an unknown
    stays an honest unknown". Through the resource path the model lookup failed
    first, so the answer was ``None`` -- not a conservative reading of an A100 but
    no A100 at all.
    """
    assert _answer(identify_accelerator("gpu:a100-96gb:4")) == ("A100", 40, False)
    assert _answer(identify_accelerator(None, "a100-96gb")) == ("A100", 40, False)


def test_a_mig_slice_still_vetoes_the_pin_and_still_names_the_card():
    """The veto is about the SIZE, and it must not take the model with it."""
    got = _answer(
        identify_accelerator(
            "gpu:a100-80gb:4",
            "nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB-MIG-3g.40gb",
        )
    )
    assert got == ("A100", 40, False), got


# -- CONTROLS -------------------------------------------------------------


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        # The two parts whose sized aliases were hand-written: unchanged.
        ("gpu:a100-80gb:4", ("A100", 80, True)),
        ("gpu:a100-40gb:4", ("A100", 40, True)),
        ("gpu:h100-80gb:4", ("H100", 80, True)),
        ("gres/gpu:a100-80gb=4", ("A100", 80, True)),
        # Bare model names: the conservative inference, still flagged.
        ("gpu:a100:4", ("A100", 40, False)),
        ("gpu:v100:4", ("V100", 16, False)),
        ("gpu:h100:4", ("H100", 80, False)),
        ("gpu:gh200:4", ("GH200", 96, False)),
        ("gpu:pvc:4", ("PVC", 48, False)),
        # Single-size parts: pinned by the model name, no variants to select.
        ("gpu:a30:4", ("A30", 24, True)),
        ("gpu:t4:4", ("T4", 16, True)),
        # Not an accelerator resource at all.
        ("cpu:32", None),
        ("gpu:4", None),
    ],
)
def test_control_the_resource_path_answers_exactly_as_it_did(resource, expected):
    assert _answer(identify_accelerator(resource)) == expected


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ("nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB", ("A100", 40, True)),
        ("nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB", ("A100", 80, True)),
        ("nvidia.com/gpu.product=NVIDIA-A100-80GB-PCIe", ("A100", 80, True)),
        ("a100", ("A100", 40, False)),
        # The no-separator alias: `80` is preceded by a digit, so the size is not
        # read off it and the conservative 40 stands. The fallback must not
        # change this -- the exact lookup hits first, and the raw-value rule that
        # refuses `a10080gb` is the reason `A100-SXM4-40GB` is not read as 440.
        ("a10080gb", ("A100", 40, False)),
        ("a100,ib,256g", ("A100", 40, False)),
    ],
)
def test_control_the_label_path_answers_exactly_as_it_did(labels, expected):
    assert _answer(identify_accelerator(None, labels)) == expected


@pytest.mark.parametrize(
    "feature",
    ["256g", "mem-256gb", "nvme-1024gb", "ssd-960gb", "shm-64gb", "scratch-2gb", "16gb"],
)
def test_control_a_sized_feature_that_is_not_a_card_is_still_not_a_card(feature):
    """The cost of the fallback, bounded: what is left after the size comes off
    has to be an exact accelerator alias, and a host-RAM or disk size leaves a
    word that is not one. `256g` is refused one step earlier -- the suffix
    requires the ``b`` of ``GB``.
    """
    assert identify_accelerator(None, feature) is None
    assert identify_accelerator(f"gpu:{feature}:4") is None


def test_control_a_stale_feature_about_another_card_is_still_not_read():
    """`gpu:a100:4` beside a `v100-32gb` feature: 32 is not a fact about the A100.

    This case was safe before only because `v100-32gb` resolved to nothing at
    all. Now that it resolves, the cross-model guard in `identify_accelerator`
    is what keeps it safe, so it is worth pinning that the answer did not move.
    """
    got = _answer(identify_accelerator("gpu:a100:4", "v100-32gb"))
    assert got == ("A100", 40, False), got
