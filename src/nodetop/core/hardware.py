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
* **Memory size is an inference until the label says otherwise.**  Every
  scheduler's *memory field* is host RAM; none of them record accelerator
  memory as a resource, and ``A100`` alone does not say 40 GiB or 80 GiB.
  Values here are therefore reported as inferences with an explicit
  :attr:`AcceleratorSpec.memory_certain` flag, and the conservative variant is
  used for fit decisions so the failure mode is a needless warning rather than
  an OOM ninety minutes into a run.  But the *product string* frequently does
  say: ``NVIDIA-A100-SXM4-80GB`` and ``a100-40gb`` name the size outright, and
  so does a typed ``Gres=gpu:a100-80gb:4``.  :func:`identify_accelerator` reads
  it back off either when it names one of the sizes this table already declares
  for that part, and pins nothing when the two contradict each other -- see
  :func:`_pin_memory_from_label`.  Selecting among declared variants is not
  guessing; discarding the one place a scheduler DOES record the size is what
  ruled out every 80 GiB node under ``--gpu-mem 80``.
* **Capability is not a function of one number.**  Deriving dtype support from
  a CUDA compute capability works until an AMD or Intel part shows up, and
  then it silently reports nonsense.  Each capability is therefore stored
  explicitly per model, per vendor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

__all__ = [
    "ACCELERATORS",
    "AcceleratorSpec",
    "identify_accelerator",
    "name_accelerator",
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
    #: Accelerator memory in **gibibytes** despite the field name.  This is the
    #: vendor's own figure, and the vendor's figure is binary -- checked against
    #: NVIDIA's MIG user guide rather than assumed, because if it were decimal
    #: every "GiB" this tool prints for HBM would be 7% high::
    #:
    #:     |   0  A100-SXM4-40GB      Off  | 00000000:36:00.0 Off |         0 |
    #:     | N/A   29C    P0    62W / 400W |      0MiB / 40537MiB | 6% Default|
    #:
    #: Decimal 40 GB is 40e9 B = **38146 MiB** (37.25 GiB).  The card reports
    #: **40537 MiB**, which is 2390 MiB MORE than the decimal reading allows and
    #: 423 MiB (~1%) less than 40 x 2**30 -- the ECC and reserved carve-out.  So
    #: "40GB" in the product name is 40 GiB.  The same guide's ``mig -lgip``
    #: table settles the unit outright: its Memory column is headed ``GiB`` and
    #: the eighth-of-a-card ``1g.5gb`` profile is listed as ``4.75``, so
    #: NVIDIA's own "5gb" is 5 GiB less reserve, not 4.66.
    #:
    #: This is the JEDEC convention (JESD100B.01 defines ``G`` as 2**30 in front
    #: of a semiconductor memory capacity), the same one that makes Slurm's
    #: ``--mem=244G`` binary -- so ``--gpu-mem`` and this field are the same
    #: quantity and compare like for like.  The field name is left alone: it is
    #: on the public :class:`AcceleratorSpec` and renaming it is not a
    #: labelling fix.
    memory_gb: int
    #: Every size this part ships in, when the model name does not pin one.
    #: A label may SELECT from this tuple; it may never add to it.
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
    # -- NVIDIA, pre-Ampere -------------------------------------------------
    # Old parts are not out of scope, and leaving them out is not the
    # conservative choice it looks like.  On a 1,614-node cluster **232 of 384
    # GPUs -- 60% -- rendered as `UNKNOWN`**, more than the V100s, K80s and
    # P100s put together, because eight tokens the scheduler names in node
    # features were missing from this table.  Three of the eight are Tesla
    # datacentre parts from the same product line as the K80 just below, so
    # "consumer cards are out of scope" did not explain the gap either.
    #
    # Every one of these is pre-Ampere, so `_nv` derives bf16/tf32/fp8/flash as
    # False from the compute capability -- which is the answer, not a guess.
    # The alternative was worse than a wrong label: an unidentified card is
    # counted in NO capability row, so a survey printed `bf16 0/384` off a
    # denominator that was 60% unknown.  That was right on this cluster by
    # luck; one A100 behind an unrecognised token makes it false and identical.
    _nv("M2090", "sm_20", 6, aliases=("m2090", "teslam2090")),
    _nv("K20M", "sm_35", 5, aliases=("k20m", "k20", "teslak20m", "teslak20")),
    _nv("K40M", "sm_35", 12, aliases=("k40m", "k40", "k40c", "teslak40m", "teslak40")),
    _nv("GTX780", "sm_35", 3, aliases=("gtx780", "geforcegtx780")),
    _nv("GTXTITANX", "sm_52", 12, aliases=("gtxtitanx", "titanx", "geforcegtxtitanx")),
    _nv("GTX1080", "sm_61", 8, aliases=("gtx1080", "geforcegtx1080")),
    _nv("GTX1080TI", "sm_61", 11, aliases=("gtx1080ti", "geforcegtx1080ti")),
    # Volta, so sm_70 -- same architecture as the V100 beside it, and like the
    # V100 it has no bf16.  The hyphen is why `_normalise` strips separators:
    # this token arrives as `titan-v` on one cluster and `titanv` on the next.
    _nv("TITANV", "sm_70", 12, aliases=("titanv", "titan-v", "nvidiatitanv")),
    _nv("RTX2080TI", "sm_75", 11, aliases=("rtx2080ti", "geforcertx2080ti", "2080ti")),
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


#: A label naming a MIG *slice* rather than a whole card.  Either spelling is
#: conclusive: ``-MIG-`` as its own token, or the ``<n>g.<m>gb`` profile shape.
_MIG_SLICE = re.compile(r"\bmig\b|\d+g\.\d+gb", re.IGNORECASE)


def _pin_memory_from_label(spec: AcceleratorSpec, *values: str) -> AcceleratorSpec:
    """``spec`` with its memory pinned, if ``values`` name a declared variant.

    **The one place a scheduler does record accelerator memory is the product
    string, and it was being thrown away.**  ``nvidia.com/gpu.product`` is
    ``NVIDIA-A100-SXM4-80GB``; a Slurm feature is written ``a100-80gb``; the
    alias table below already lists ``a10080gb`` and ``h10080gb`` precisely
    because sites write it that way.  All of them matched the ``A100`` row and
    inherited its *conservative* 40 with ``memory_certain=False``, so
    ``where -g 1 --gpu-mem 80`` answered "A100 has 40 GiB, need 80 (inferred
    from model)" and ruled out every node -- on a cluster that had said 80 in
    the label the tool was reading.

    Two rules keep this a selection rather than a guess, which is the whole
    difference between this and inventing a number:

    * **Only a size already in :attr:`~AcceleratorSpec.memory_variants` counts.**
      A part with no variants is pinned by its model name and is left alone; a
      label naming a size the table has never heard of (a future 96 GiB A100)
      leaves the conservative default AND ``memory_certain=False`` in place, so
      an unknown stays an honest unknown.
    * **The size is read off the RAW value, never the normalised one**, and a
      digit may not immediately precede it.  ``_normalise`` strips separators,
      which turns ``A100-SXM4-40GB`` into ``a100sxm440gb`` -- where a plain
      ``(\\d+)gb`` reads the size as **440**, borrowing the ``4`` from the form
      factor.  The lookbehind is what makes ``-40GB`` legible and ``sxm440gb``
      not.

    The size is deliberately NOT anchored to the end of the string, because the
    80 GiB PCIe part is labelled ``NVIDIA-A100-80GB-PCIe`` -- ``nvidia-smi -L``
    names it that way and the GPU feature discovery label follows -- so an
    end-anchored read would miss the flagship 80 GiB card and go on ruling it
    out.  ``NVIDIA-H100-80GB-HBM3`` is the same shape.

    **A MIG slice is refused outright.**  ``NVIDIA-A100-SXM4-40GB-MIG-1g.5gb``
    names a 40 GiB card, but the unit the scheduler hands out is a 1/8 slice
    with 4.75 GiB behind it, so pinning 40 as CERTAIN there would let a 40 GiB
    job onto a 4.75 GiB allocation -- an error in the OOM direction, which is
    the one this module exists to avoid.  Unpinned it stays: ``>=40``, flagged
    as an inference, exactly as before.

    The cost of the raw-value rule is a label written with no separator at all
    (``a10080gb``): ``80`` there is preceded by a digit, so the size is not
    read and the conservative 40 stands.  That is the safe direction, and it is
    the alias spelling rather than anything a scheduler emits.

    **More than one source may name the card, and then they must agree.**  A
    Slurm node can carry both a typed ``Gres=gpu:a100-80gb:4`` and a feature
    ``a100-80gb``; :func:`identify_accelerator` passes both here.  With one
    value this is the single-source rule above, unchanged.  With two:

    * Either naming a size while the other names none pins it -- a source that
      is silent about the size is not evidence against it.
    * Both naming the SAME size pins it, which is the common case and the whole
      reason a site types its GRES.
    * Both naming DIFFERENT sizes pins nothing.  ``Gres=gpu:a100-80gb:4`` on a
      node whose feature says ``a100-40gb`` is a site where one of the two
      strings is stale, and nothing here says which: both are hand-typed by the
      same admin, so ranking them would be a guess dressed as precedence.  The
      module's rule is that only a *known* fact decides, and a contradiction is
      not a known anything -- so the conservative variant and
      ``memory_certain=False`` stay, which is exactly the state an untyped
      ``a100`` is already in.  That keeps the error on the needless-warning
      side rather than admitting an 80 GiB job onto a 40 GiB card, and the
      caveat the flag already prints ("the smaller was assumed") points the
      reader at the labels.
    * A MIG spelling in ANY value refuses the pin outright, even one another
      value names a size for.  ``Gres=gpu:a100-80gb:4`` with a product label of
      ``NVIDIA-A100-SXM4-80GB-MIG-3g.40gb`` is an 80 GiB card handing out
      40 GiB thirds; pinning the 80 the GRES states would be the OOM-direction
      error the guard exists to prevent, so the veto crosses sources.
    """
    if not spec.memory_variants or any(_MIG_SLICE.search(v) for v in values):
        return spec
    named = set()
    for value in values:
        for variant in spec.memory_variants:
            if re.search(rf"(?<![0-9]){variant}\s*G(?:i?B)?(?![0-9A-Za-z])",
                         value, re.IGNORECASE):
                named.add(variant)
                break
    if len(named) != 1:
        return spec
    return replace(spec, memory_gb=named.pop(), memory_certain=True)


#: A memory size written as a suffix on a model name: the ``-80gb`` of
#: ``a100-80gb``, the ``-32GB`` of ``V100-32GB``.  The ``b`` is required, so a
#: host-RAM feature written ``256g`` is not read as an accelerator size.
_MEMORY_SUFFIX = re.compile(r"[-_ ]?\d+\s*gi?b$", re.IGNORECASE)


def _spec_for_token(value: str) -> AcceleratorSpec | None:
    """The spec one token names exactly, allowing a trailing memory size.

    **Naming the size cost a site the whole card.**  The alias table carries
    size-suffixed entries for exactly two parts -- ``a10040gb``/``a10080gb``
    and ``h10080gb``, hand-written because sites spell it that way -- and
    nothing generates the rest, so every other multi-variant part resolved to
    ``None`` the moment the size appeared in the string::

        identify_accelerator("gpu:v100-32gb:4")     -> None
        identify_accelerator("gpu:gh200-144gb:4")   -> None
        identify_accelerator("gpu:pvc-128gb:4")     -> None
        identify_accelerator("gpu:h100-94gb:4")     -> None

    while the bare form answered ``V100 16``, ``GH200 96``, ``PVC 48``,
    ``H100 80``.  ``None`` is not a smaller answer than those -- it is a
    different one: the row prints ``arch -``, ``mem -``, ``bf16 unknown`` and
    the card is counted in no capability claim at all, so a fleet of 32 GiB
    V100s reads as a fleet of unidentifiable accelerators.  The same string is
    the spelling :func:`_pin_memory_from_label` documents as typical for a
    Slurm feature, and it reached the memory pin for one vendor's one part.

    Retrying the lookup with the size removed is a *fallback*: it runs only
    where the exact lookup has already failed, so no token that resolves today
    resolves differently.  The size itself is still read off the raw value by
    :func:`_pin_memory_from_label` under all of its rules -- so a size the
    table declares pins it (``v100-32gb`` -> 32, certain), a size it does not
    leaves the conservative variant and ``memory_certain=False`` standing
    (``a100-96gb`` -> 40, uncertain, which is what that function's docstring
    already claimed and could not deliver), and a MIG spelling still vetoes
    the pin while the model comes back.

    The suffix requires the ``b`` of ``GB``/``GiB``.  A node feature list is
    full of bare-``g`` sizes that are host RAM -- ``256g`` -- and those must
    not be read as part of a model name.
    """
    spec = _BY_ALIAS.get(_normalise(value))
    if spec is not None:
        return spec
    stripped = _MEMORY_SUFFIX.sub("", value.strip())
    if not stripped or stripped == value.strip():
        return None
    return _BY_ALIAS.get(_normalise(stripped))


def _identify_from_labels(
    labels: str | list[str] | None,
) -> tuple[AcceleratorSpec, str] | None:
    """The spec a label set names, WITH the raw token value that named it.

    The value comes back alongside the spec because the memory size may only
    be read off the token that identified the card, never off a sibling label.
    A node's feature list is mostly sizes that have nothing to do with the GPU
    -- ``256g`` is host RAM -- and ``80g`` there would otherwise pin 80 GiB of
    HBM on a 40 GiB A100.  Pairing the two keeps the size and the model coming
    from one string, which is what made the single-label case safe.
    """
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
        spec = _spec_for_token(value)
        if spec is not None:
            return spec, value
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
                return spec, value
    return None


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

    **A typed resource decides the MODEL, not the memory size on its own.**
    The resource path used to return the moment it matched an alias, which
    meant the memory pin was unreachable from the one form a site that bothers
    to type its GRES actually emits: ``Gres=gpu:a100-80gb:4`` answered
    ``(40, uncertain)`` even though the string says 80, and on a node carrying
    both a bare ``Gres=gpu:a100:4`` and a feature ``a100-80gb`` the early
    return threw the feature away unread.  The typed form is at least as
    authoritative about the size as a node feature -- it is the string Slurm
    allocates against -- so both are read and reconciled by
    :func:`_pin_memory_from_label`, which pins only when they do not
    contradict each other.

    The label's size is consulted only when the label names the SAME model the
    resource does.  ``Gres=gpu:a100:4`` beside a ``v100-32gb`` feature is a
    stale feature about a different card, and 32 is not a fact about the A100.
    """
    from_labels = _identify_from_labels(labels)

    # 1. A typed resource is authoritative when present.
    if resource:
        for entry in resource.split(","):
            parts = entry.strip().replace("=", ":").split(":")
            if len(parts) >= 3 and parts[0].lower() in {"gpu", "gres/gpu"}:
                spec = _spec_for_token(parts[1])
                if spec is not None:
                    named = [parts[1]]
                    if from_labels is not None and from_labels[0].model == spec.model:
                        named.append(from_labels[1])
                    return _pin_memory_from_label(spec, *named)

    if from_labels is None:
        return None
    return _pin_memory_from_label(*from_labels)


#: Tokens that NAME a vendor accelerator product, for a card the vocabulary
#: above cannot identify.
#:
#: Used for one thing only: printing a name instead of ``UNKNOWN``.  No
#: capability, memory or architecture is ever derived from a match --
#: `identify_accelerator` still returns ``None``, so the row still reads `arch
#: -`, `mem -`, `bf16 unknown`, and the card is still counted in no capability
#: claim.  `rtx2080ti` tells a reader everything about the node; `UNKNOWN` tells
#: them nothing and hides the fact that the scheduler DID say what the card is.
#:
#: **An explicit vendor or family prefix is required.**  A first version also
#: accepted bare part-number shapes (`[aklpt]\d{1,3}`, `m\d{2,4}` and friends),
#: which is how a node feature list is actually written and therefore full of
#: false positives: `p9` and `p8` are POWER9/POWER8, a standard feature on every
#: POWER+V100 cluster; `m1024`/`m2048` are memory sizes; `a64` is an
#: architecture; `t2`, `k10`, `b100` are ordinary site labels.  Worse, none of
#: those carries a family prefix, so on `p9,ib,<unknown-gpu>,gpu` the first
#: match won and `cmd_accelerators` grouped the GPUs under a model called `p9`.
#:
#: A wrong name is worse than the shrug it replaces, so the shape test is gone
#: and only a token that says whose product it is qualifies.  That is the same
#: discipline `_VENDOR_MARKER` already applies to substring matching, and it
#: costs nothing here: every token this fallback would have caught by shape is
#: now in the vocabulary above.
_GPU_NAME_SHAPED = re.compile(
    r"""^(?:
        (?:nvidia|geforce|gtx|rtx|quadro|titan|tesla)[a-z0-9]{1,12}
      | (?:radeon|instinct|firepro)[a-z0-9]{1,12}
      | gaudi\d*
      | mi\d{2,3}x?
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

#: A driver or firmware version, which is not a card.
#:
#: ``nvidia_driver_535`` satisfies :data:`_GPU_NAME_SHAPED` exactly -- an
#: ``nvidia`` prefix followed by twelve or fewer alphanumerics, once
#: :func:`_normalise` has taken the underscores out -- so it became a model row
#: in ``nodetop accelerators``.  Worse on a node that carries both:
#: ``nvidia-driver-470,a100`` answered ``nvidia-driver-470``, because the driver
#: label came first in the label list and both are "named family" tokens.
#:
#: Requiring a vendor prefix was the right narrowing and it does not reach this,
#: because the driver label genuinely has one.  A driver version is a real fact
#: about the node and it is not what the card is called, so it is refused here
#: rather than promoted to a product name.  Tested against the NORMALISED value,
#: since that is where the punctuation between the words has gone.  (``driver``
#: is the form measured; ``firmware`` is the same class and costs nothing.)
_DRIVER_LABEL = re.compile(r"(driver|firmware)", re.IGNORECASE)

#: Prefixes that name a vendor product outright.  Every alternative in
#: :data:`_GPU_NAME_SHAPED` now carries one, so this only orders the candidates.
_NAMED_FAMILY = re.compile(
    r"^(nvidia|geforce|gtx|rtx|quadro|titan|tesla|radeon|instinct|firepro|gaudi|mi\d)",
    re.IGNORECASE,
)


def name_accelerator(
    resource: str | None = None,
    labels: str | list[str] | None = None,
) -> str | None:
    """The label that NAMES this node's accelerator, vocabulary or not.

    Companion to :func:`identify_accelerator`, and deliberately separate from
    it: that function answers "what can this card do", and must return ``None``
    rather than guess.  This one answers "what did the scheduler call it", which
    is a fact the node record already carries and which a report has no reason
    to throw away.

    Returns ``None`` when nothing in the labels is shaped like a product name.
    """
    if resource:
        for entry in resource.split(","):
            parts = entry.strip().replace("=", ":").split(":")
            # A typed resource names the model in its second field, and there
            # it needs no shape test: the scheduler has already said the field
            # is a GPU type.
            if len(parts) >= 3 and parts[0].lower() in {"gpu", "gres/gpu"} and parts[1]:
                return parts[1]
    if not labels:
        return None
    tokens = labels.split(",") if isinstance(labels, str) else list(labels)
    shaped: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token or _NON_ACCELERATOR_LABEL.match(token):
            continue
        value = token.split("=", 1)[1] if "=" in token else token
        flat = _normalise(value)
        if flat and _GPU_NAME_SHAPED.match(flat) and not _DRIVER_LABEL.search(flat):
            shaped.append(value)
    if not shaped:
        return None
    # A named family beats a bare part number: on a node labelled
    # `...,rtx2080ti,gpu,l16b` both could in principle fit a shape, and only one
    # of them is a GPU.
    named = [v for v in shaped if _NAMED_FAMILY.match(_normalise(v))]
    return (named or shaped)[0]


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
