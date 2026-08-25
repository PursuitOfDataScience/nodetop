"""Accelerator identification and the capability facts that gate a job.

**No batch system models this.**  Slurm, PBS, LSF, SGE and Kubernetes all
treat an accelerator as an opaque countable resource -- ``gres/gpu=4``,
``ngpus=4``, ``nvidia.com/gpu: 4`` -- so every one of them will happily place
a bf16 job on a V100 and let it die at the first autocast, or resume an fp8
checkpoint on a card that has no fp8 at all.

The information that decides whether a job can *run* -- architecture, memory
size, which dtypes exist -- is in none of them.  This module supplies it, and
it is the one part of nodetop that is useful with no scheduler present.

Three facts about real clusters shape the design:

* **The resource name is usually untyped.**  On the cluster this was developed
  against, 90 of 91 GPU nodes report a bare ``Gres=gpu:4`` with no model; the
  model is only recoverable from the node's feature labels, where it appears
  in whatever case the admin typed that day (``a100``, ``A100``, ``H100``,
  ``L40S``, ``rtx6000``).  Kubernetes is the same story: ``nvidia.com/gpu``
  plus a ``nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB`` label.
* **Memory size is an inference, never a measurement.**  Every scheduler's
  memory field is host RAM; none record accelerator memory.  ``A100`` alone
  does not say 40 GB or 80 GB.  Values here are therefore reported as
  inferences with an explicit :attr:`AcceleratorSpec.memory_certain` flag, and
  the conservative variant is used for fit decisions so the failure mode is a
  needless warning rather than an OOM ninety minutes into a run.
* **Capability is not a function of one number.**  Deriving dtype support from
  a CUDA compute capability works until an AMD or Intel part shows up, and
  then it silently reports nonsense.  Each capability is therefore stored
  explicitly per model, per vendor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ACCELERATORS",
    "AcceleratorSpec",
    "identify_accelerator",
    "supports",
]


@dataclass(frozen=True)
class AcceleratorSpec:
    """What one accelerator model can do.

    Capabilities are explicit fields rather than derived from ``arch``, so a
    non-NVIDIA part cannot be described by accident.
    """

    model: str
    vendor: str
    #: Architecture string as the vendor names it: ``sm_80``, ``gfx90a``,
    #: ``Xe-HPC``.  For display and for matching a toolchain target.
    arch: str
    memory_gb: int
    memory_variants: tuple[int, ...] = ()
    #: False when the model name alone does not pin the memory size.
    memory_certain: bool = True
    bf16: bool = False
    fp8: bool = False
    #: NVIDIA-specific tensor float; False for other vendors by definition.
    tf32: bool = False
    #: FlashAttention-2 or later has a working kernel for this part.
    flash_attention: bool = False
    aliases: tuple[str, ...] = field(default=(), repr=False)

    @property
    def cuda(self) -> bool:
        return self.vendor == "NVIDIA"

    @property
    def sm(self) -> int | None:
        """CUDA compute capability as an integer (``sm_80`` -> 80), else None."""
        m = re.match(r"sm_(\d+)", self.arch)
        return int(m.group(1)) if m else None


def _nv(model, arch, mem, **kw) -> AcceleratorSpec:
    """NVIDIA part.  bf16/tf32 from Ampere (sm_80), fp8 from Ada (sm_89)."""
    sm = int(re.match(r"sm_(\d+)", arch).group(1))  # type: ignore[union-attr]
    kw.setdefault("bf16", sm >= 80)
    kw.setdefault("tf32", sm >= 80)
    kw.setdefault("fp8", sm >= 89)
    kw.setdefault("flash_attention", sm >= 80)
    return AcceleratorSpec(model=model, vendor="NVIDIA", arch=arch, memory_gb=mem, **kw)


_SPECS: tuple[AcceleratorSpec, ...] = (
    # -- NVIDIA -------------------------------------------------------------
    _nv("K80", "sm_37", 12, aliases=("k80", "teslak80")),
    _nv("P100", "sm_60", 16, aliases=("p100", "teslap100")),
    _nv("V100", "sm_70", 16, memory_variants=(16, 32), memory_certain=False,
        aliases=("v100", "v100s", "teslav100")),
    _nv("T4", "sm_75", 16, aliases=("t4", "teslat4")),
    _nv("RTX6000", "sm_75", 24, aliases=("rtx6000", "quadrortx6000", "rtx6000turing")),
    _nv("A30", "sm_80", 24, aliases=("a30",)),
    _nv("A100", "sm_80", 40, memory_variants=(40, 80), memory_certain=False,
        aliases=("a100", "a100pcie", "a100sxm4", "a10080gb", "a10040gb")),
    _nv("A10", "sm_86", 24, aliases=("a10",)),
    _nv("A40", "sm_86", 48, aliases=("a40",)),
    _nv("A6000", "sm_86", 48, aliases=("a6000", "rtxa6000")),
    _nv("L4", "sm_89", 24, aliases=("l4",)),
    _nv("L40", "sm_89", 48, aliases=("l40",)),
    _nv("L40S", "sm_89", 48, aliases=("l40s",)),
    _nv("RTX6000ADA", "sm_89", 48, aliases=("rtx6000ada", "rtx6000adageneration")),
    _nv("H100", "sm_90", 80, memory_variants=(80, 94), memory_certain=False,
        aliases=("h100", "h100pcie", "h100sxm5", "h10080gb", "hopper")),
    _nv("H200", "sm_90", 141, aliases=("h200",)),
    _nv("GH200", "sm_90", 96, memory_variants=(96, 144), memory_certain=False,
        aliases=("gh200",)),
    _nv("B200", "sm_100", 180, aliases=("b200",)),
    _nv("GB200", "sm_100", 186, aliases=("gb200",)),
    _nv("GB10", "sm_121", 128, memory_certain=False,
        aliases=("gb10", "grace blackwell", "gracesblackwell", "sparkgb10")),
    # -- AMD ----------------------------------------------------------------
    # CDNA has bf16 from MI100; fp8 arrives with CDNA3 (MI300).  Deriving any
    # of this from a CUDA compute capability would be meaningless.
    AcceleratorSpec("MI50", "AMD", "gfx906", 16, bf16=False, fp8=False,
                    aliases=("mi50", "instinctmi50")),
    AcceleratorSpec("MI100", "AMD", "gfx908", 32, bf16=True, fp8=False,
                    flash_attention=True, aliases=("mi100", "instinctmi100")),
    AcceleratorSpec("MI210", "AMD", "gfx90a", 64, bf16=True, fp8=False,
                    flash_attention=True, aliases=("mi210", "instinctmi210")),
    AcceleratorSpec("MI250", "AMD", "gfx90a", 128, bf16=True, fp8=False,
                    flash_attention=True, aliases=("mi250", "instinctmi250")),
    AcceleratorSpec("MI250X", "AMD", "gfx90a", 128, bf16=True, fp8=False,
                    flash_attention=True, aliases=("mi250x", "instinctmi250x")),
    AcceleratorSpec("MI300X", "AMD", "gfx942", 192, bf16=True, fp8=True,
                    flash_attention=True, aliases=("mi300", "mi300x", "instinctmi300x")),
    AcceleratorSpec("MI325X", "AMD", "gfx942", 256, bf16=True, fp8=True,
                    flash_attention=True, aliases=("mi325", "mi325x")),
    # -- Intel --------------------------------------------------------------
    # `PVC` is Ponte Vecchio, the codename BOTH parts share, so on its own it
    # pins the model no more than a bare `A100` pins 40 GB against 80 -- and it
    # is exactly what a scheduler hands over. A 10,624-node PBS Pro cluster
    # advertises `resources_available.gputype = PVC` and nothing else, where the
    # hardware is the 128 GB Max 1550; the bare codename used to alias to the
    # 1100, so nodetop named a specific SKU it had no evidence for and called
    # its 48 GB certain. `where -g 6 --gpu-mem 64` then ruled out every node on
    # a machine whose every GPU has 128.
    AcceleratorSpec("PVC", "Intel", "Xe-HPC", 48, memory_variants=(48, 128),
                    memory_certain=False, bf16=True, fp8=False,
                    aliases=("pvc", "pontevecchio")),
    AcceleratorSpec("PVC1100", "Intel", "Xe-HPC", 48, bf16=True, fp8=False,
                    aliases=("datacentergpumax1100", "max1100")),
    AcceleratorSpec("PVC1550", "Intel", "Xe-HPC", 128, bf16=True, fp8=False,
                    aliases=("datacentergpumax1550", "max1550")),
    AcceleratorSpec("GAUDI2", "Intel", "Gaudi2", 96, bf16=True, fp8=True,
                    aliases=("gaudi2", "habanagaudi2")),
)

ACCELERATORS: dict[str, AcceleratorSpec] = {s.model: s for s in _SPECS}


def _normalise(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


#: normalised alias -> spec
_BY_ALIAS: dict[str, AcceleratorSpec] = {}
for _s in _SPECS:
    _BY_ALIAS.setdefault(_normalise(_s.model), _s)
    for _a in _s.aliases:
        _BY_ALIAS.setdefault(_normalise(_a), _s)

#: Aliases sorted longest-first, for substring matching inside vendor product
#: strings.  Longest-first so "a100sxm4" is not shadowed by "a100", and
#: "mi300x" not by "mi300".
_ALIASES_BY_LENGTH: list[tuple[str, AcceleratorSpec]] = sorted(
    _BY_ALIAS.items(), key=lambda kv: -len(kv[0])
)

#: Substring matching is only attempted on labels that clearly name a vendor
#: product.  Without this guard a CPU model or a RAM size can collide with a
#: short alias, and a false capability claim is worse than no claim.
_VENDOR_MARKER = re.compile(
    r"nvidia|tesla|quadro|geforce|instinct|radeon|habana|gaudi|"
    r"datacenter|amd-mi|intel-max|xe-hpc",
    re.IGNORECASE,
)

#: Labels that describe something other than an accelerator.  Host RAM sizes
#: and CPU model names dominate the label list on most clusters.
_NON_ACCELERATOR_LABEL = re.compile(
    r"""^(
        \d+(\.\d+)?\s*(g|gb|t|tb|mb|mi|gi)   # 256g, 1.5T, 768g
        | (intel|amd)[-_](epyc|xeon|opteron).*
        | (gold|silver|platinum|epyc|xeon|bronze|opteron)[-_].*
        | dlc | ib | hdr | edr | omnipath | opa | nvlink | ssd | nvme | ssd\d*
        | linux | amd64 | arm64 | x86.64 | aarch64
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def identify_accelerator(
    resource: str | None = None,
    labels: str | list[str] | None = None,
) -> AcceleratorSpec | None:
    """Best-effort accelerator identification for one node.

    ``resource`` is the scheduler's typed resource string when it has one
    (Slurm ``gpu:a30:4``, ``gres/gpu:a100=4``); ``labels`` is the node's
    feature/label set, as a comma-separated string or a list -- Slurm
    features, PBS resources_available, LSF resource strings, or k8s labels
    such as ``nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB``.

    Returns ``None`` when the node has no accelerator, or has one whose model
    cannot be determined -- an honest "unknown" rather than a guess.
    """
    # 1. A typed resource is authoritative when present.
    if resource:
        for entry in resource.split(","):
            parts = entry.strip().replace("=", ":").split(":")
            if len(parts) >= 3 and parts[0].lower() in {"gpu", "gres/gpu"}:
                spec = _BY_ALIAS.get(_normalise(parts[1]))
                if spec is not None:
                    return spec

    if not labels:
        return None
    tokens = labels.split(",") if isinstance(labels, str) else list(labels)

    # 2. Exact alias match on a label token.  Tried for every token before any
    #    substring matching, so a clean "a100" always wins.
    candidates: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token or _NON_ACCELERATOR_LABEL.match(token):
            continue
        # A k8s label arrives as key=value; the model is in the value.
        value = token.split("=", 1)[1] if "=" in token else token
        spec = _BY_ALIAS.get(_normalise(value))
        if spec is not None:
            return spec
        candidates.append(value)

    # 3. Substring match, but only inside something that names a vendor.  The
    #    marker is tested against the *normalised* value: a label written
    #    "Intel-Data-Center-GPU-Max-1550" contains no literal "datacenter"
    #    until the punctuation is stripped.
    for value in candidates:
        flat = _normalise(value)
        if not (_VENDOR_MARKER.search(value) or _VENDOR_MARKER.search(flat)):
            continue
        for alias, spec in _ALIASES_BY_LENGTH:
            if len(alias) >= 3 and alias in flat:
                return spec
    return None


def supports(spec: AcceleratorSpec | None, requirement: str) -> bool | None:
    """Check one named capability, returning ``None`` when unknown.

    The tri-state matters: "this node cannot do fp8" and "we do not know what
    this node is" must not collapse into the same answer, because only the
    first justifies excluding the node.
    """
    if spec is None:
        return None
    table = {
        "bf16": spec.bf16,
        "bfloat16": spec.bf16,
        "fp8": spec.fp8,
        "float8": spec.fp8,
        "tf32": spec.tf32,
        "flash": spec.flash_attention,
        "flash_attention": spec.flash_attention,
        "cuda": spec.cuda,
        "rocm": spec.vendor == "AMD",
    }
    return table.get(requirement.strip().lower())
