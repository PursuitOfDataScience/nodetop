"""Accelerator identification and capability gating, across vendors."""

from __future__ import annotations

import pytest

from nodetop.core.hardware import ACCELERATORS, identify_accelerator, supports


class TestTypedResource:
    def test_typed_resource_is_authoritative(self):
        assert identify_accelerator("gpu:a30:4").model == "A30"

    def test_untyped_resource_alone_yields_nothing(self):
        # 90 of 91 GPU nodes on the reference cluster report a bare "gpu:4";
        # the model is simply not there and must not be invented.
        assert identify_accelerator("gpu:4") is None

    def test_typed_resource_beats_labels(self):
        assert identify_accelerator("gpu:a30:4", "a100").model == "A30"

    @pytest.mark.parametrize("resource", ["(null)", "none", "", None])
    def test_no_accelerator(self, resource):
        assert identify_accelerator(resource) is None


class TestSchedulerLabels:
    @pytest.mark.parametrize("labels,model", [
        # Real Slurm feature strings, with the case inconsistency admins produce.
        ("gold-6346,256g,a100", "A100"),
        ("gold-6346,512g,H100", "H100"),
        ("gold-6346,1024g,H200", "H200"),
        ("gold-5218,192g,v100", "V100"),
        ("gold-6248r,384g,rtx6000", "RTX6000"),
        ("Gold-6448Y,512g,L40S", "L40S"),
        ("gold-6346,256g,a40", "A40"),
    ])
    def test_slurm_features(self, labels, model):
        assert identify_accelerator("gpu:4", labels).model == model

    @pytest.mark.parametrize("labels,model", [
        # Kubernetes nvidia.com/gpu.product values, verbatim shapes.
        ("nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB", "A100"),
        ("nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3", "H100"),
        ("nvidia.com/gpu.product=Tesla-V100-SXM2-16GB", "V100"),
        ("nvidia.com/gpu.product=NVIDIA-L40S", "L40S"),
    ])
    def test_kubernetes_product_labels(self, labels, model):
        assert identify_accelerator("nvidia.com/gpu", labels).model == model

    def test_accepts_a_list(self):
        assert identify_accelerator(None, ["gold-6346", "256g", "a100"]).model == "A100"


class TestVendors:
    @pytest.mark.parametrize("labels,model,vendor", [
        ("AMD-Instinct-MI300X", "MI300X", "AMD"),
        ("AMD-Instinct-MI250X", "MI250X", "AMD"),
        ("Intel-Data-Center-GPU-Max-1550", "PVC1550", "Intel"),
    ])
    def test_non_nvidia(self, labels, model, vendor):
        spec = identify_accelerator(None, labels)
        assert (spec.model, spec.vendor) == (model, vendor)

    def test_amd_capabilities_are_not_derived_from_cuda(self):
        # MI250 has bf16 and no fp8; it has no compute capability at all, so
        # any sm-based derivation would be meaningless here.
        mi250 = ACCELERATORS["MI250"]
        assert mi250.sm is None
        assert (mi250.bf16, mi250.fp8) == (True, False)
        assert ACCELERATORS["MI300X"].fp8 is True


class TestNoFalsePositives:
    @pytest.mark.parametrize("labels", [
        "gold-6248r,192g", "AMD-EPYC-7713,1T", "epyc-9335,2048g",
        "Intel-9242,384g", "gold-6542Y,1.5T,DLC", "Gold-6448Y,512g",
        "kubernetes.io/arch=amd64,kubernetes.io/os=linux",
        "node.kubernetes.io/instance-type=m5.24xlarge",
    ])
    def test_cpu_and_platform_labels_are_not_accelerators(self, labels):
        # "gold-6448y" must not match the short alias "l4", and a RAM size
        # must not match anything at all.
        assert identify_accelerator("gpu:4", labels) is None


class TestCapabilities:
    @pytest.mark.parametrize("model,bf16,fp8", [
        ("V100", False, False),     # sm_70: no bf16 -- passes submission, dies at autocast
        ("RTX6000", False, False),  # sm_75
        ("A30", True, False),       # sm_80
        ("A100", True, False),      # sm_80: bf16 yes, fp8 NO
        ("A40", True, False),       # sm_86
        ("L40S", True, True),       # sm_89: first fp8
        ("H100", True, True),       # sm_90
        ("H200", True, True),
    ])
    def test_nvidia_dtype_gates(self, model, bf16, fp8):
        spec = ACCELERATORS[model]
        assert (spec.bf16, spec.fp8) == (bf16, fp8)

    def test_nvidia_boundaries(self):
        for spec in ACCELERATORS.values():
            if spec.vendor != "NVIDIA":
                continue
            assert spec.bf16 == (spec.sm >= 80)
            assert spec.fp8 == (spec.sm >= 89)


class TestMemoryHonesty:
    @pytest.mark.parametrize("model", ["A100", "H100", "V100", "GH200"])
    def test_variant_models_are_marked_uncertain(self, model):
        # An "A100" label could be 40 GB or 80 GB and no scheduler records
        # either, so the value must be labelled an inference.
        assert ACCELERATORS[model].memory_certain is False

    @pytest.mark.parametrize("model", ["A40", "H200", "L40S", "A30", "MI300X"])
    def test_single_capacity_models_are_certain(self, model):
        assert ACCELERATORS[model].memory_certain is True

    def test_uncertain_models_report_the_conservative_value(self):
        # Assuming the smaller variant fails safe: a needless warning rather
        # than an OOM ninety minutes into a run.
        spec = ACCELERATORS["A100"]
        assert spec.memory_gb == min(spec.memory_variants)


class TestSupports:
    def test_unknown_accelerator_is_none_not_false(self):
        # "We cannot identify this" must not collapse into "it cannot do
        # this" -- only the latter justifies excluding a node.
        assert supports(None, "bf16") is None

    def test_known_negative(self):
        assert supports(ACCELERATORS["V100"], "bf16") is False

    def test_known_positive(self):
        assert supports(ACCELERATORS["H100"], "fp8") is True

    def test_aliases(self):
        assert supports(ACCELERATORS["H100"], "float8") is True
        assert supports(ACCELERATORS["A100"], "bfloat16") is True

    def test_vendor_questions(self):
        assert supports(ACCELERATORS["MI300X"], "rocm") is True
        assert supports(ACCELERATORS["MI300X"], "cuda") is False

    def test_unrecognised_requirement(self):
        assert supports(ACCELERATORS["H100"], "quantum") is None


class TestACodenameIsNotASku:
    """`PVC` names two parts, 48 GB and 128 GB, and a scheduler hands over both.

    A 10,624-node PBS Pro cluster advertises `resources_available.gputype = PVC`
    and nothing else, where the hardware is the 128 GB Max 1550. The bare
    codename used to alias straight to the 1100, so nodetop named a SKU it had
    no evidence for and reported its 48 GB as certain -- and `where -g 6
    --gpu-mem 64` then ruled out every node on a machine whose every GPU has
    128. Same shape as a bare `A100` standing for 40 or 80.
    """

    def test_the_bare_codename_claims_no_sku(self):
        spec = identify_accelerator(None, "gputype=PVC")
        assert spec is not None
        assert spec.model == "PVC"
        assert spec.memory_certain is False
        assert spec.memory_variants == (48, 128)
        # The conservative variant, per the module's stated rule: a needless
        # warning beats an OOM ninety minutes in.
        assert spec.memory_gb == 48

    def test_an_explicit_part_still_pins_it(self):
        assert identify_accelerator(None, "gputype=max1550").model == "PVC1550"
        assert identify_accelerator(None, "gputype=max1550").memory_gb == 128
        assert identify_accelerator(None, "gputype=max1100").model == "PVC1100"
        assert identify_accelerator(None, "gpu_model=Intel-Data-Center-GPU-Max-1550"
                                    ).memory_gb == 128

    def test_it_is_still_an_intel_xe_hpc_part(self):
        spec = identify_accelerator(None, "gputype=pvc")
        assert (spec.vendor, spec.arch) == ("Intel", "Xe-HPC")
        assert spec.bf16 is True and spec.fp8 is False
        assert spec.cuda is False
