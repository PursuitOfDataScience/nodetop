"""RAM is gibibytes, and now says so.

Every memory figure in this tool is binary end to end: ``cli`` documents its
parser as taking "a bare number meaning GiB", ``JobShape.memory_mb_per_node``
returns ``memory_gb * 1024``, and the backends read MB values that are
themselves 1024-based. Four render sites nonetheless labelled it "GB", while
GPU HBM in the same output was labelled "GiB" -- one binary division, two
different units, depending on which resource it was.

Measured against this controller rather than assumed: on a 250000 MB node,
``sbatch --test-only --mem=244G`` is accepted and ``--mem=245G`` is refused,
identically to ``249856M`` and ``250880M``. So Slurm's ``G`` is 1024-based, and
a "GB" label was understating what the reader typed by about 7% -- 8 GiB read as
8 GB is 8.59 GB, and at 250 GiB the gap is 19 GB.

All four sibling tools (``rapidu``, ``slurmpast``, ``slurmwatch`` byte
formatters, and ``slurmate``'s memory forms) print GiB/MiB for memory.

The ``memory_gb`` field keeps its name: it is on the public ``JobShape`` and
renaming it is a different change from labelling what it prints.
"""

from nodetop.core.capacity import hardware_ok
from nodetop.core.model import JobShape


def test_the_shape_summary_says_gibibytes():
    """``JobShape``'s own one-line description, which the report prints."""
    text = JobShape(cpus_per_task=4, memory_gb=8, nodes=1).describe()
    assert "8 GiB RAM/node" in text, text
    assert "GB RAM" not in text, text


def test_a_fractional_size_keeps_its_spelling():
    """``:g`` still trims, so the unit change did not reformat the number."""
    assert "0.5 GiB RAM/node" in JobShape(memory_gb=0.5, nodes=1).describe()
    assert "250 GiB RAM/node" in JobShape(memory_gb=250, nodes=1).describe()


def test_control_the_gpu_memory_label_is_left_alone():
    """CONTROL, passing with the change present or absent.

    HBM was labelled GiB before the RAM fix and still is. What this pins is that
    the RAM change did not touch it.

    **The open question this docstring used to carry is now closed, and the
    answer is that GiB was right.** The doubt was real: the hardware table
    stores the VENDOR's figure (``hardware.py`` has A100=40, H100=80, the
    numbers on the datasheets and in the Kubernetes product label
    ``NVIDIA-A100-SXM4-40GB``), and if those were decimal an "A100 40GB" would
    hold 37.25 GiB and this label would be high by 7%.

    They are not decimal. NVIDIA's MIG user guide shows an ``A100-SXM4-40GB``
    reporting ``0MiB / 40537MiB`` of framebuffer. Decimal 40 GB is 38146 MiB
    (37.25 GiB), so the card reports **2390 MiB more than the decimal reading
    allows**; 40 x 2**30 is 40960 MiB, of which 40537 is all but a ~1% ECC and
    reserved carve-out. The same guide's ``mig -lgip`` table heads its Memory
    column ``GiB`` and lists the eighth-of-a-card ``1g.5gb`` profile as
    ``4.75``, so NVIDIA's own "5gb" is 5 GiB less reserve, not 4.66. That is the
    JEDEC convention (JESD100B.01 defines ``G`` as 2**30 in front of a
    semiconductor memory capacity), the same one that makes Slurm's ``G``
    binary. This repository's own fixture agrees, and is where the claim that
    "no recorded fixture holds a device reading" went wrong:
    ``tests/backends/test_sshpool.py`` records ``GPU=NVIDIA A100-SXM4-40GB,
    40960`` off ``nvidia-smi --query-gpu=memory.total`` with ``nounits``, whose
    unit is MiB -- 40 GiB exactly.

    Still not measurable from this host, and it did not need to be: no
    accelerator is reachable without a billed allocation, ``nvidia-smi`` is
    absent on a CPU node, and Slurm advertises only ``Gres=gpu:4`` -- which is
    what ``DESIGN.md`` means by "no scheduler records it" about the memory
    *resource*. The product *label* is a different matter and is now read:
    see ``test_gpu_memory_from_label.py``.

    The RAM side is measured independently: ``--mem=244G`` accepted and
    ``--mem=245G`` refused on a 250000 MB node, and a GPU node reporting
    ``RealMemory=256000`` with ``CfgTRES=mem=250G`` (250 x 1024). Two units in
    one output line, binary for the same reason.
    """
    text = JobShape(gpus_per_node=1, gpu_memory_gb=40, nodes=1).describe()
    assert ">=40 GiB HBM" in text, text


def test_control_the_numbers_themselves_are_untouched():
    """CONTROL, in both states. A label fix must not move a value.

    ``memory_mb_per_node`` is the binary conversion the label now matches, and
    it is asserted off the arithmetic rather than off the renderer.
    """
    assert JobShape(memory_gb=8, nodes=1).memory_mb_per_node == 8 * 1024
    assert JobShape(memory_gb=244, nodes=1).memory_mb_per_node == 249856
    # 244 GiB is what Slurm accepted on a 250000 MB node; 245 GiB is not
    assert JobShape(memory_gb=245, nodes=1).memory_mb_per_node == 250880


def test_the_hardware_gate_says_gibibytes_too():
    """The other surface that names a RAM size, so the two cannot drift apart."""
    from nodetop.core.model import Node

    node = Node(name="n1", cpus_total=8, memory_mb=4 * 1024)
    ok, why = hardware_ok(node, JobShape(memory_gb=8, nodes=1))
    assert ok is False
    assert any("only 4 GiB RAM installed" in w for w in why), why
