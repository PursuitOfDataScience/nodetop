"""A figure the table QUALIFIES must not reach ``--json`` bare.

Every view renders twice -- once for a person, once for a program -- and only
the first one has anywhere to put a caveat.  So the text form grew prefixes,
markers and empty cells that mean something (``>=`` for an inferred size,
``group-only:`` for hardware that is not yours, a blank cell for "nowhere
open", a denominator beside a count), and the document beside it published the
bare number.  A consumer filtering ``--json`` then cannot see the hedge the
human reader is given, which is worse than not being told at all: the number
looks measured.

Both surfaces are rendered from ONE ``Cluster`` here, so nothing can be blamed
on the cluster having moved between the two calls.

The controls are the point of the file as much as the findings.  An ordinary,
fully-known cluster -- every partition shared and usable, every accelerator
size pinned, nothing excluded -- must render byte-for-byte what it renders
today in both surfaces, because a qualification that fires when there is
nothing to qualify is just a second bug.
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

import nodetop.cli as cli
from nodetop.cli import build_parser
from nodetop.core.capacity import EXCLUDED_REASON
from nodetop.core.cluster import Cluster
from nodetop.core.hardware import identify_accelerator
from nodetop.core.model import Identity, Node, Queue
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _run(cluster, fn, argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(cluster, _args(argv), PLAIN)
    return buf.getvalue()


def _both(cluster, fn, argv: list[str]) -> tuple[str, object]:
    """``(table, document)`` for one command on one cluster state."""
    table = _run(cluster, fn, argv)
    doc = json.loads(_run(cluster, fn, ["--json", *argv]))
    return table, doc


def _node(name, *, gpus=0, label="", queues=(), state="IDLE", cpus=8, mem=16384):
    return Node(
        name=name, state_raw=state, cpus_total=cpus, memory_mb=mem,
        gpus_total=gpus, queues=tuple(queues),
        accelerator=identify_accelerator(None, label) if label else None,
        accelerator_label=label,
        labels=(label,) if label else (),
    )


def _queue(name, nodes, **kw):
    return Queue(name=name, node_names=tuple(n.name for n in nodes),
                 declared_nodes=len(nodes), nodes=list(nodes), **kw)


def _cluster(nodes, queues) -> Cluster:
    return Cluster(
        backend_name="synthetic", queue_term="partition",
        nodes=list(nodes), queues={q.name: q for q in queues},
        identity=Identity(user="me", accounts=("mine",), qos=("normal",)),
    )


# ---------------------------------------------------------------------------
# where / fit: a WRONG HW verdict the table qualifies and the document did not
# ---------------------------------------------------------------------------
@pytest.fixture
def all_excluded():
    """A partition whose every node the CALLER kept out with ``--exclude``.

    The nodes are perfectly capable.  ``Capacity`` records that, and the table
    refuses to blame the hardware for it -- because ``WRONG HW`` carries the
    legend "go elsewhere; waiting will not help", which is a false statement
    about a machine nobody looked at.
    """
    ns = [_node(f"gn{i}", gpus=4, label="a100-80gb", queues=("gpuq",))
          for i in (1, 2)]
    return _cluster(ns, [_queue("gpuq", ns)])


WHERE_EXCLUDED = ["where", "-q", "gpuq", "-g", "1", "--exclude", "gn1,gn2",
                  "--declared", "--all"]


class TestAnExcludedNodeIsNotWrongHardware:
    def test_the_table_says_so_outright(self, all_excluded):
        table, _ = _both(all_excluded, cli.cmd_where, WHERE_EXCLUDED)
        assert "WRONG HW" in table
        # ... and immediately takes it back, naming the real cause.
        assert "excluded by request, not by hardware" in table
        assert f"2 nodes: {EXCLUDED_REASON}" in table

    def test_the_document_carries_the_reason_too(self, all_excluded):
        """THE FINDING. Before the fix there was no such key at all.

        `EXCLUDED_REASON` is exported so a consumer can make this test
        "without matching on a sentence" -- and `--json`, the only surface a
        consumer reads, was the one that never published a reason to test.
        """
        _, doc = _both(all_excluded, cli.cmd_where, WHERE_EXCLUDED)
        assert doc[0]["hardware_incompatible"] is True
        assert doc[0]["hardware_reasons"] == {EXCLUDED_REASON: 2}

    def test_an_empty_histogram_is_not_a_missing_one(self):
        """`{}` and "no such key" are different claims.

        Not a control -- it reads the added key, so it can only pass after the
        fix -- but it is the case that keeps the fix from being a special
        report for bad news: `{}` says "every node considered was capable",
        which is a measurement.  The absence of the key said nothing at all.
        """
        ns = [_node(f"ok{i}", gpus=4, label="a100-80gb", queues=("fine",))
              for i in (1, 2)]
        cluster = _cluster(ns, [_queue("fine", ns)])
        _, doc = _both(cluster, cli.cmd_where,
                       ["where", "-q", "fine", "-g", "1", "--gpu-mem", "80",
                        "--declared", "--all"])
        assert doc[0]["hardware_reasons"] == {}
        assert doc[0]["nodes_considered"] == 2

    def test_the_histogram_comes_with_its_denominator(self, all_excluded):
        """`hardware_reasons` cannot be summed into a node count.

        A node contributes several reasons, so `Capacity.considered` is
        recorded rather than derived -- and shipping the histogram without it
        invites exactly the overcount that field exists to prevent.  It is
        also the number the table prints: `right hw  0/2`.
        """
        table, doc = _both(all_excluded, cli.cmd_where, WHERE_EXCLUDED)
        assert doc[0]["nodes_considered"] == 2
        assert doc[0]["nodes_capable"] == 0
        assert "0/2" in table


class TestTheCapableCountKeepsItsDenominator:
    """`right hw 1/6` and `nodes_capable: 1` are not the same claim.

    "With a denominator: `1` next to reasons accounting for 10 other nodes is
    arithmetic the reader cannot close" -- the table's own reason for growing
    the fraction.  The document published the numerator alone.
    """

    @staticmethod
    def _mixed():
        pinned = _node("big1", gpus=4, label="a100-80gb", queues=("mix",))
        small = [_node(f"sm{i}", gpus=4, label="a100", queues=("mix",))
                 for i in (1, 2)]
        plain = [_node(f"cpu{i}", queues=("mix",)) for i in (1, 2, 3)]
        ns = [pinned, *small, *plain]
        return _cluster(ns, [_queue("mix", ns)])

    ARGV = ["where", "-q", "mix", "-g", "1", "--gpu-mem", "80", "--declared",
            "--all"]

    def test_both_surfaces_report_one_capable_node_of_six(self):
        cluster = self._mixed()
        table, doc = _both(cluster, cli.cmd_where, self.ARGV)
        assert "1/6" in table
        assert (doc[0]["nodes_capable"], doc[0]["nodes_considered"]) == (1, 6)

    def test_the_reasons_agree_with_the_text(self):
        cluster = self._mixed()
        table, doc = _both(cluster, cli.cmd_where, self.ARGV)
        # The inference rides on the shared caveat here, because the table
        # prints its hardware histogram only for a placement it ruled out
        # -- and this one runs now off the single pinned node.
        assert "accelerator memory is inferred from the model name" in table
        assert any("inferred from the model name" in c
                   for c in doc[0]["caveats"])
        assert doc[0]["hardware_reasons"] == {
            "no accelerator": 3,
            "A100 has 40 GiB, need 80 (inferred from model)": 2,
        }


# ---------------------------------------------------------------------------
# accelerators: WHERE the cards are vs WHERE YOU CAN SUBMIT
# ---------------------------------------------------------------------------
@pytest.fixture
def homes():
    """Three models, one per relationship the table distinguishes.

    * ``A100`` sits in a shared, usable partition -- an ordinary row.
    * ``RTX6000`` is in one group's private hardware, which the table prefixes
      ``group-only:`` because it is not somewhere the reader can submit.
    * ``V100`` is only in a DOWN partition, which the table renders as an
      EMPTY cell -- a state its docstring is careful about, since a blank must
      not read as "nowhere at all".
    """
    shared = _node("a1", gpus=4, label="a100-80gb", queues=("shared",))
    priv = _node("p1", gpus=2, label="rtx6000", queues=("privq",))
    dead = _node("d1", gpus=2, label="v100", queues=("deadq",))
    return _cluster([shared, priv, dead], [
        _queue("shared", [shared]),
        _queue("privq", [priv], allow_accounts=("pi-other",)),
        _queue("deadq", [dead], state_raw="DOWN", enabled=False),
    ])


ACCEL = ["accelerators", "--all"]


class TestTheInventorySaysWhereYouCanSubmit:
    def test_the_table_marks_private_hardware(self, homes):
        table, _ = _both(homes, cli.cmd_accelerators, ACCEL)
        rtx = next(ln for ln in table.splitlines()
                   if ln.strip().startswith("RTX6000"))
        assert "group-only: privq" in rtx

    def test_the_table_leaves_the_cell_empty_when_nothing_is_open(self, homes):
        table, _ = _both(homes, cli.cmd_accelerators, ACCEL)
        v100 = next(ln for ln in table.splitlines()
                    if ln.strip().startswith("V100"))
        assert "deadq" not in v100, "a DOWN partition is not somewhere to go"

    def test_the_document_marks_them_too(self, homes):
        """THE FINDING. One flat list answered both questions, and only one
        of them correctly.

        `partitions` is every holder -- the right answer to "where is this
        hardware".  It was also the only answer, so a DOWN partition and one
        group's private share arrived under the same key as a submittable one,
        with nothing to tell them apart.  `status --json` has published
        `dedicated` per partition all along; this view was the outlier.
        """
        _, doc = _both(homes, cli.cmd_accelerators, ACCEL)
        models = doc["models"]
        assert models["A100"]["partitions_open"] == ["shared"]
        assert models["A100"]["partitions_group_only"] == []

        assert models["RTX6000"]["partitions_open"] == []
        assert models["RTX6000"]["partitions_group_only"] == ["privq"]

        # The empty cell, said out loud: the cards exist, nowhere open.
        assert models["V100"]["partitions"] == ["deadq"]
        assert models["V100"]["partitions_open"] == []
        assert models["V100"]["partitions_group_only"] == []

    def test_the_two_surfaces_name_the_same_partitions(self, homes):
        """Whatever the table's cell says, the document says it too.

        The invariant rather than a fixed string: the reading is shared
        (`_model_holders`), so the only way these can drift is if one of them
        stops using it.
        """
        table, doc = _both(homes, cli.cmd_accelerators, ACCEL)
        for model, g in doc["models"].items():
            row = next(ln for ln in table.splitlines()
                       if ln.strip().startswith(model))
            for name in g["partitions_open"] + g["partitions_group_only"]:
                assert name in row, (model, name)
            for name in set(g["partitions"]) - set(
                g["partitions_open"] + g["partitions_group_only"]
            ):
                assert name not in row, (model, name)


# ---------------------------------------------------------------------------
# CONTROL: an ordinary cluster renders exactly what it rendered before
# ---------------------------------------------------------------------------
@pytest.fixture
def ordinary():
    """Nothing to qualify: one shared usable partition, memory pinned by label.

    Deliberately ONE partition per model, so the table's busiest-first column
    and the document's alphabetical `partitions` list cannot be told apart --
    the control must not depend on which ordering either surface happens to
    use.
    """
    ns = [_node(f"n{i}", gpus=4, label="a100-80gb", queues=("work",))
          for i in (1, 2)]
    return _cluster(ns, [_queue("work", ns)])


class TestNothingChangesWhenThereIsNothingToQualify:
    """Passes before the fix and after it.  That is the whole job of it."""

    #: The `accelerators --all` rows, verbatim.  A literal, not a property:
    #: byte-identical is the claim being made.
    ROWS = (
        "  model  vendor  arch   mem  nodes  free           bf16  fp8  "
        "partitions",
        "  A100   NVIDIA  sm_80  80G      2  █████████ 8/8  yes   no   work",
    )

    def test_the_table_is_byte_identical(self, ordinary):
        table, _ = _both(ordinary, cli.cmd_accelerators, ACCEL)
        for line in self.ROWS:
            assert line in table.splitlines(), line

    def test_no_marker_appears_anywhere(self, ordinary):
        table, _ = _both(ordinary, cli.cmd_accelerators, ACCEL)
        # `>=` is the inferred-size prefix and `group-only:` the private-
        # hardware one. A fully-known, shared cluster earns neither.
        assert ">=" not in table
        assert "group-only" not in table

    def test_the_documents_pre_existing_keys_are_untouched(self, ordinary):
        """Only additions.  Every key a consumer already read still reads the
        same, which is why this passes on both sides of the fix.
        """
        _, doc = _both(ordinary, cli.cmd_accelerators, ACCEL)
        was = {
            "identified": True,
            "scheduler_label": "a100-80gb",
            "vendor": "NVIDIA",
            "arch": "sm_80",
            "memory_gb": 80,
            "memory_inferred": False,
            "installed": 8,
            "free": 8,
            "nodes": 2,
            "capabilities": {"bf16": True, "fp8": False, "tf32": True,
                             "flash_attention": True},
            "partitions": ["work"],
        }
        got = doc["models"]["A100"]
        assert {k: got[k] for k in was} == was

    WHERE = ["where", "-q", "work", "-g", "1", "--gpu-mem", "80",
             "--declared", "--all"]

    def test_where_still_says_run_now_with_no_caveat_about_hardware(
        self, ordinary
    ):
        table, doc = _both(ordinary, cli.cmd_where, self.WHERE)
        assert "RUN NOW" in table
        assert "WRONG HW" not in table
        assert "hardware" not in table
        was = {
            "queue": "work",
            "runnable_now": True,
            "starts_now": True,
            "reachable": True,
            "hardware_incompatible": False,
            "nodes_free": 2,
            "nodes_capable": 2,
            "nodes_unverified": 0,
        }
        assert {k: doc[0][k] for k in was} == was
