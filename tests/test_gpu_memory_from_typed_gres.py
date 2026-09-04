"""A typed `Gres` names the HBM size too, and the short-circuit skipped it.

The product-label pin landed two rounds ago: a k8s
`nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB` or a Slurm feature `a100-80gb`
now selects 80 out of `memory_variants` instead of inheriting the conservative
40.  `identify_accelerator` never reached it from the *resource* argument,
because the resource path returned the instant an alias matched::

    identify_accelerator("gpu:a100-80gb:4")   -> ('A100', 40, False)

The string says 80.  Worse, the early return also discarded the labels
entirely, so the form that a typed-GRES site actually presents -- a bare
`Gres=gpu:a100:4` on a node whose feature list carries `a100-80gb` -- answered
40/uncertain off a node that had stated 80 twice::

    identify_accelerator("gpu:a100:4", "a100-80gb")  -> ('A100', 40, False)

Both are the reported `where -g 1 --gpu-mem 80` symptom ("A100 has 40 GiB, need
80 (inferred from model)", every node ruled out), reached through the one field
Slurm actually allocates against.  A typed GRES is at least as authoritative
about the size as a node feature: `--gres=gpu:a100-80gb:1` *selects* on it,
while a feature is documentation.  So it pins.

**When both name a size and they disagree, nothing pins.**
`Gres=gpu:a100-80gb:4` on a node whose feature says `a100-40gb` is a site where
one of the two strings is stale, and nothing in either says which -- the same
admin typed both.  `hardware.py`'s rule is that an unidentifiable accelerator
is `None` rather than a guess, and that only a *known* negative excludes a
node; a contradiction is not a known anything.  Picking a winner would be a
guess wearing precedence as a costume, and the two ways of being wrong are not
symmetric: believing 80 on a 40 GiB card is an OOM ninety minutes into a run,
while falling back to the conservative variant is the needless warning the
default already exists to produce.  So a conflict lands exactly where an
untyped `a100` already sits -- 40, `memory_certain=False`, and the caveat that
"the smaller was assumed" pointing the reader at the labels.

That fallback is also a no-op against the old behaviour, which is what makes it
safe: a conflicting pair answered 40/uncertain before this change and answers
40/uncertain after.  The change only *adds* pins where the sources do not
contradict each other.
"""

from __future__ import annotations

import pytest

from nodetop.core.capacity import hardware_ok
from nodetop.core.hardware import ACCELERATORS, identify_accelerator
from nodetop.core.model import JobShape, Node

K8S = "nvidia.com/gpu.product="


def _node(resource: str | None, labels: str | None, name: str = "n1") -> Node:
    return Node(
        name=name,
        cpus_total=128,
        memory_mb=1024 * 1024,
        gpus_total=8,
        accelerator=identify_accelerator(resource, labels),
    )


class TestTheTypedGresPinsTheSize:
    @pytest.mark.parametrize(
        ("resource", "size"),
        [
            # The two spellings `identify_accelerator` documents for the
            # resource argument: `scontrol show node` writes the first,
            # `sinfo -O gres` and the TRES form write the second.
            ("gpu:a100-80gb:4", 80),
            ("gres/gpu:a100-80gb=4", 80),
            ("gpu:a100-40gb:4", 40),
            ("gpu:A100-80GB:4", 80),
            ("gpu:a100_80gb:4", 80),
            # Slurm prints socket affinity on the node's own field, and a node
            # may advertise more than one GRES kind.
            ("gpu:a100-80gb:4(S:0-1)", 80),
            ("mps:400,gpu:a100-80gb:4", 80),
        ],
    )
    def test_a_declared_size_in_the_resource_is_read_back(self, resource, size):
        spec = identify_accelerator(resource, None)
        assert spec is not None, resource
        assert spec.memory_gb == size, resource
        assert spec.memory_certain is True, resource

    @pytest.mark.parametrize(
        "labels",
        [
            "a100-80gb",
            "gold-6346,512g,a100-80gb,ib",
            K8S + "NVIDIA-A100-SXM4-80GB",
            K8S + "NVIDIA-A100-80GB-PCIe",
        ],
    )
    def test_a_bare_typed_gres_no_longer_swallows_the_label(self, labels):
        """The shape a real typed-GRES site presents.

        `Gres=gpu:a100:4` is authoritative about the MODEL and silent about the
        size; the feature list says 80.  Returning on the model alone threw the
        only statement of the size away unread.
        """
        spec = identify_accelerator("gpu:a100:4", labels)
        assert spec is not None, labels
        assert (spec.memory_gb, spec.memory_certain) == (80, True), labels

    def test_two_sources_that_agree_pin(self):
        """A typed GRES and a matching feature: the case typing GRES is for."""
        spec = identify_accelerator("gpu:a100-80gb:4", "a100-80gb")
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (80, True)

    def test_the_eighty_gigabyte_node_now_fits_through_the_resource_field(self):
        """The reported symptom, end to end through the gate that refused it."""
        ok, why = hardware_ok(
            _node("gpu:a100-80gb:4", None),
            JobShape(gpus_per_node=1, gpu_memory_gb=80, nodes=1),
        )
        assert ok is True, why

    def test_a_forty_gigabyte_typed_gres_refuses_without_hedging(self):
        """`gpu:a100-40gb:4` states 40, so the refusal is a fact, not a guess.

        "(inferred from model)" was honest while 40 was an assumption; once the
        GRES type says 40 the hedge invites a reader to override a right
        answer.
        """
        ok, why = hardware_ok(
            _node("gpu:a100-40gb:4", None),
            JobShape(gpus_per_node=1, gpu_memory_gb=80, nodes=1),
        )
        assert ok is False
        assert any("A100 has 40 GiB, need 80" in w for w in why), why
        assert not any("inferred from model" in w for w in why), why


class TestControls:
    """Pass identically with the change present or absent."""

    def test_control_a_bare_typed_gres_is_still_an_inference(self):
        """CONTROL. `gpu:a100:4` says nothing about the size, and must not start
        to.

        This is the case `DESIGN.md` is about -- 90 of 91 GPU nodes on the
        reference cluster report a bare GRES -- and it is the one an over-eager
        pin would break by reaching for a number no source stated.
        """
        spec = identify_accelerator("gpu:a100:4", None)
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_a_mig_shaped_resource_is_not_pinned(self):
        """CONTROL. `gpu:a100_1g.5gb:4` names a 40 GiB card handing out 4.75.

        The type token is not in the vocabulary, so the model is not resolved
        at all and the node stays an honest unknown rather than acquiring a
        size.  Either outcome is acceptable; a *pinned* one is not.
        """
        for resource in (
            "gpu:a100_1g.5gb:4",
            "gpu:a100-40gb_1g.5gb:7",
            "gpu:1g.5gb:7",
        ):
            spec = identify_accelerator(resource, None)
            assert spec is None or spec.memory_certain is False, resource

    def test_control_the_mig_veto_crosses_sources(self):
        """CONTROL. A MIG product label refuses a size the GRES type states.

        `Gres=gpu:a100-80gb:4` beside `...-80GB-MIG-3g.40gb` is an 80 GiB card
        handing out 40 GiB thirds.  The 80 is true of the card and false of the
        allocatable unit, so reading two sources must not let the un-vetoed one
        supply what the vetoed one refused -- the veto is over the node, not
        over one string.
        """
        for resource in ("gpu:a100:4", "gpu:a100-80gb:4"):
            spec = identify_accelerator(
                resource, K8S + "NVIDIA-A100-SXM4-80GB-MIG-3g.40gb"
            )
            assert spec is not None, resource
            assert (spec.memory_gb, spec.memory_certain) == (40, False), resource

    def test_control_a_conflict_pins_nothing(self):
        """CONTROL. Disagreeing sources answer what they answered before.

        Both orderings, because "which source wins" must not be decided by
        which one happens to be read first.
        """
        for resource, labels in (
            ("gpu:a100-80gb:4", "a100-40gb"),
            ("gpu:a100-40gb:4", K8S + "NVIDIA-A100-SXM4-80GB"),
        ):
            spec = identify_accelerator(resource, labels)
            assert spec is not None
            assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_a_size_is_only_read_off_the_token_that_named_the_card(self):
        """CONTROL. `80g` in the feature list is host RAM, not HBM.

        A node's label list is mostly sizes that have nothing to do with the
        GPU, which is why the size must come from the same string as the model.
        Reading the resource and the labels together is exactly where that rule
        could have been lost.
        """
        spec = identify_accelerator("gpu:a100:4", "gold-6346,80g,ib,nvlink")
        assert spec is not None
        assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_a_label_for_a_different_model_says_nothing(self):
        """CONTROL. A stale `v100-32gb` feature is not a fact about an A100.

        The typed resource decides the model, so a size attached to some other
        card is neither a pin nor a conflict -- it is simply irrelevant.
        """
        spec = identify_accelerator("gpu:a100:4", K8S + "Tesla-V100-SXM2-32GB")
        assert spec is not None
        assert spec.model == "A100"
        assert (spec.memory_gb, spec.memory_certain) == (40, False)

    def test_control_the_typed_resource_still_decides_the_model(self):
        """CONTROL. Precedence for the MODEL is unchanged: resource, then labels.

        Only the memory size started reading both.  A node whose GRES and
        features name different cards must still answer the GRES.
        """
        spec = identify_accelerator("gpu:a100:4", "v100")
        assert spec is not None and spec.model == "A100"
        # ...and a resource that names no model still falls through to them.
        spec = identify_accelerator("gpu:4", "a100-80gb")
        assert spec is not None and spec.model == "A100"

    def test_control_the_label_only_path_is_untouched(self):
        """CONTROL. No `resource` at all: the pin that already worked, working.

        The label scan was lifted into a helper to be shared with the resource
        path; this is the assertion that the lift changed none of its answers.
        """
        for labels, expected in (
            (K8S + "NVIDIA-A100-SXM4-80GB", (80, True)),
            ("a100-80gb", (80, True)),
            ("gold-6346,256g,a100", (40, False)),
            (K8S + "NVIDIA-A100-SXM4-96GB", (40, False)),
            (K8S + "NVIDIA-L40S", (48, True)),
        ):
            spec = identify_accelerator(None, labels)
            assert spec is not None, labels
            assert (spec.memory_gb, spec.memory_certain) == expected, labels
        assert identify_accelerator(None, "256g") is None
        assert identify_accelerator(None, None) is None

    def test_control_the_shared_table_is_not_mutated(self):
        """CONTROL. `ACCELERATORS` is process-wide; a pin must not edit it.

        `AcceleratorSpec` is frozen and the pin goes through `replace`, so this
        holds by construction -- and it is asserted anyway, because the
        alternative is one node's GRES silently redefining the A100 for every
        other backend in the same process.
        """
        for resource, labels in (
            ("gpu:a100-80gb:4", None),
            ("gpu:a100:4", "a100-80gb"),
            ("gpu:a100-80gb:4", "a100-40gb"),
        ):
            identify_accelerator(resource, labels)
            assert ACCELERATORS["A100"].memory_gb == 40
            assert ACCELERATORS["A100"].memory_certain is False
            assert ACCELERATORS["A100"].memory_variants == (40, 80)

    def test_control_the_comparison_arithmetic_is_untouched(self):
        """CONTROL. This fixed which number is compared, not how.

        Read off the node whose figure is 40 in *either* state -- a bare typed
        GRES -- so the threshold is checked without the pin in the way: 40 GiB
        against a 40 GiB ask fits, 41 does not.
        """
        node = _node("gpu:a100:4", None)
        assert hardware_ok(node, JobShape(gpus_per_node=1, gpu_memory_gb=40))[0] is True
        assert hardware_ok(node, JobShape(gpus_per_node=1, gpu_memory_gb=41))[0] is False
