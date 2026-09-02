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
    the RAM change did not touch it -- NOT that GiB is the right unit for HBM,
    which is unverified and may well be wrong in the other direction.

    The hardware table stores the VENDOR's figure (``hardware.py`` has A100=40,
    H100=80, the numbers on the datasheets and in the Kubernetes product label
    ``NVIDIA-A100-SXM4-40GB``), and ``DESIGN.md`` writes them "40 GB or 80 GB".
    If those are decimal, an "A100 40GB" holds 37.25 GiB and this label is high
    by 7%. It could not be settled from this cluster: no accelerator is reachable
    without a billed allocation, ``nvidia-smi`` is absent on a CPU node, no
    recorded fixture holds a device reading, and Slurm advertises only
    ``Gres=gpu:4`` -- which is exactly what ``DESIGN.md`` means by "no scheduler
    records it".

    Contrast the RAM side, which IS measured: ``--mem=244G`` accepted and
    ``--mem=245G`` refused on a 250000 MB node, and a GPU node reporting
    ``RealMemory=256000`` with ``CfgTRES=mem=250G`` (250 x 1024). Two units in one
    output line, correct for different reasons -- so do not "fix" one to match
    the other without a device reading.
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
