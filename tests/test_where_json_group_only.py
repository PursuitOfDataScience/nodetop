"""``where``'s ``--json`` erased the access distinction its own table makes.

Reproduced against a frozen snapshot of the cluster this was written on -- the
same input to both surfaces, because a live control plane shifts between two
calls -- for a one-node 8-CPU 32 GiB 4-hour shape::

    nodetop --no-color --replay SNAP where -c 8 --mem 32G -t 4:00:00
      ● RUN NOW  caslake   56/1  190/190  now  declared
      ● RUN NOW  ssd        18/1   18/18  now  group-only      <- 11 rows of 19
      ● RUN NOW  avieregg   10/1   10/10  now  group-only
    nodetop --replay SNAP --json where -c 8 --mem 32G -t 4:00:00
      every row, ssd and avieregg among them: "entitlement_source": "declared"

Diffed field by field, a ``ssd`` row and a ``caslake`` row differed only in
capacity numbers and ``submit_flags``. Nothing in the document carried the
distinction, so a consumer ranking partitions treated a PI's private hardware
exactly like a general queue.

**The JSON is the wrong surface, not the table.** The distinction is decided in
one place, :attr:`Queue.is_dedicated` -- a structural reading of the queue's own
allowlist, and the strongest thing sayable without a dry-run. The table consults
it. ``--json`` walked a *second* copy of the ladder that had no rung for it at
all, and it had the queue in hand the whole time: the very next key calls
``cluster.submit_flags(p.queue, shape)``. ``status --json`` has published
``"dedicated": q.is_dedicated`` per queue since long before this.

So the ladder now lives in ``cli._entitlement_source``, called once per row by
each surface, and the table keeps only the choice of word and colour. The
document also gains a ``dedicated`` key, because that line has ONE slot and a
verdict outranks the heuristic in it: measured live, ``ssd`` comes back
``entitlement_source: "refused"``, and without the extra key the row would again
stop saying whose hardware refused it. A cell has to choose; a document has no
width to run out of.

The ``verdict: null`` half of the report was **by design** -- see
``TestTheNullVerdictIsTheDryRunsAbsence`` -- and is unchanged.
"""

import dataclasses
import io
import json
import sys

from nodetop.cli import ENTITLEMENT_SOURCES, build_parser, cmd_where
from nodetop.core.cluster import Cluster, Node, Queue
from nodetop.core.model import BackendCapabilities, Identity, Verdict, VerdictCategory
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())

#: One or two accounts on the allowlist is a group's own hardware.
OWNED = ("pi-okafor",)

#: The fixture identity claims membership of that account, which is the whole
#: reason `is_dedicated` exists. Measured on the cluster this was reproduced on:
#: `sacctmgr` reports the user as associated with 34 accounts, `pi-avieregg` and
#: `ssd` among them, so `Queue.access_blockers` raises no `ACCOUNT_NOT_ALLOWED`
#: for any PI partition and all 11 of them come back RUN NOW. Give the fixture a
#: narrower identity and the two rows would differ on `reachable`, the access
#: fact would be recoverable from a blocker, and every assertion below would
#: pass for the wrong reason.
ACCOUNTS = ("mine", *OWNED)


def _nodes():
    return [
        Node(name=f"n{i}", state_raw="IDLE", cpus_total=8, memory_mb=16000,
             queues=("shared", "owned"))
        for i in range(2)
    ]


class _Backend:
    """A dry-run that accepts ``shared`` and refuses ``owned``."""

    def __init__(self, caps, answers=None):
        self._caps = caps
        self._answers = answers or {}

    def probe(self, q, shape, account=None):
        return self._answers.get(q)

    def capabilities(self):
        return self._caps

    def submit_flags(self, q, shape):
        return [f"--partition={q}"]

    def format_nodelist(self, names):
        return ",".join(sorted(names))


def _cluster(*, replayed=False, answers=None):
    """Two partitions, identical hardware; one of them somebody's own.

    Identical on purpose: every capacity number in the two rows matches, so the
    only thing that can tell them apart is the access fact.
    """
    nodes = _nodes()
    queues = {
        "shared": Queue(name="shared", node_names=("n0", "n1"), declared_nodes=2,
                        nodes=nodes),
        "owned": Queue(name="owned", node_names=("n0", "n1"), declared_nodes=2,
                       nodes=nodes, allow_accounts=OWNED),
    }
    caps = BackendCapabilities(
        probe=answers is not None, probe_supported=answers is not None,
        probe_command="sbatch --test-only",
    )
    return dataclasses.replace(
        Cluster(
            backend_name="slurm", queue_term="partition", nodes=nodes,
            queues=queues, identity=Identity(user="me", accounts=ACCOUNTS),
        ),
        capabilities=caps, replayed=replayed,
        _backend=_Backend(caps, answers),
    )


def _run(cluster, argv):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        cmd_where(cluster, build_parser().parse_args(argv), PLAIN)
    finally:
        sys.stdout = saved
    return buf.getvalue()


def _rows(cluster, argv):
    return {r["queue"]: r for r in json.loads(_run(cluster, argv))}


def _header(text):
    """The table's column-heading line, found by a heading only it carries."""
    return next(ln for ln in text.splitlines() if "fits/need" in ln)


ARGV_JSON = ["--json", "where", "-c", "1", "--all"]
ARGV_TEXT = ["--no-color", "where", "-c", "1", "--all"]


class TestTheDocumentMakesTheDistinctionTheTableDoes:
    def test_a_group_owned_partition_is_not_called_declared(self):
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        assert rows["owned"]["entitlement_source"] == "group-only", rows["owned"]

    def test_and_a_shared_one_still_is(self):
        # The contrast is the point: one word for both is what the bug was.
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        assert rows["shared"]["entitlement_source"] == "declared", rows["shared"]

    def test_the_two_surfaces_agree_row_for_row(self):
        """Driven through the real renderers, on one cluster, in one test.

        The table and the document are the two things that disagreed, so the
        assertion is about both of them at once rather than about either one's
        expected text. Every row is required to be FOUND in the table, because a
        loop that matched nothing would pass -- which is the failure mode this
        whole round is about.
        """
        cluster = _cluster(replayed=True)
        text = _run(cluster, ARGV_TEXT)
        rows = _rows(cluster, ARGV_JSON)
        seen = set()
        for line in text.splitlines():
            for name, row in rows.items():
                if f" {name} " not in line or "RUN NOW" not in line:
                    continue
                seen.add(name)
                marked = "group-only" in line
                assert marked is (row["entitlement_source"] == "group-only"), (
                    line, row["entitlement_source"])
                assert marked is row["dedicated"], (line, row["dedicated"])
        assert seen == set(rows) == {"shared", "owned"}, (seen, text)

    def test_every_row_carries_the_ownership_key(self):
        # A schema promise, not a happens-to-be: a consumer may read it blind.
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        assert all("dedicated" in r for r in rows.values()), rows
        assert rows["owned"]["dedicated"] is True
        assert rows["shared"]["dedicated"] is False

    def test_the_capacity_numbers_really_are_identical(self):
        """Otherwise the assertions above could pass on a difference of shape.

        This is the fixture's own control, and the condition the live
        reproduction had too: `ssd` and `caslake` differed only in capacity, and
        a consumer cannot rank on capacity to recover an access fact.
        """
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        keys = ("nodes_free", "nodes_capable", "nodes_considered",
                "runnable_now", "starts_now", "reachable")
        assert ({k: rows["owned"][k] for k in keys}
                == {k: rows["shared"][k] for k in keys}), rows


class TestAVerdictOutranksTheHeuristicInTheOneSlot:
    """And the separate key is why that costs the document nothing."""

    ANSWERS = {
        "shared": Verdict(queue="shared", allowed=True,
                          category=VerdictCategory.OK, reason="ok"),
        "owned": Verdict(queue="owned", allowed=False,
                         category=VerdictCategory.NOT_ENTITLED,
                         reason="Invalid account"),
    }

    def test_the_refusal_wins_the_source_line(self):
        rows = _rows(_cluster(answers=self.ANSWERS), ARGV_JSON)
        assert rows["owned"]["entitlement_source"] == "refused", rows["owned"]

    def test_but_the_row_still_says_whose_hardware_it_was(self):
        rows = _rows(_cluster(answers=self.ANSWERS), ARGV_JSON)
        assert rows["owned"]["dedicated"] is True, rows["owned"]

    def test_the_table_agrees_that_the_verdict_wins(self):
        # Measured live before the change and after it: `ssd` reads `refused`,
        # not `group-only`, once a dry-run has answered for it. The precedence
        # is the table's, and unifying the ladder had to keep it.
        text = _run(_cluster(answers=self.ANSWERS), ARGV_TEXT)
        line = next(ln for ln in text.splitlines() if " owned " in ln
                    and ("BLOCKED" in ln or "RUN NOW" in ln or "QUEUE" in ln))
        assert "refused" in line and "group-only" not in line, line

    def test_a_confirmed_partition_says_so_on_both_keys(self):
        rows = _rows(_cluster(answers=self.ANSWERS), ARGV_JSON)
        assert rows["shared"]["entitlement_source"] == "confirmed"
        assert rows["shared"]["confirmed"] is True
        assert rows["shared"]["dedicated"] is False


class TestTheNullVerdictIsTheDryRunsAbsence:
    """Settled first, and the answer is *by design*: nothing changed here.

    The report also noted ``"verdict": null`` on all 19 rows of the replay while
    the table's verdict column was populated. Two different things share the
    word:

    * The JSON ``verdict`` key publishes the :class:`Verdict` dataclass -- the
      control plane's own answer -- and its five sub-keys are that dataclass's
      fields verbatim. Its docstring is explicit that ``None`` is what a backend
      *without* a dry-run yields, "an absence the report must state rather than
      paper over". A replay has no dry-run (``Cluster.can_probe`` is False for
      one by construction), so every ``p.verdict`` is None and ``null`` is the
      honest report of that -- corroborated in the same row by
      ``confirmed: false`` and ``entitlement_unconfirmed: true``.
    * The table's column is ``cli._verdict_label(p)``, a PLACEMENT label reading
      five inputs, of which ``p.verdict`` is only one. It is populated because
      the other four are, and every one of them is published.

    So the document does not withhold the table's verdict; it publishes the
    ingredients and leaves the label to the renderer, exactly as it does for the
    ``fits/need`` and ``right hw`` cells.
    """

    def test_a_replay_has_no_dry_run_to_report(self):
        cluster = _cluster(replayed=True)
        assert cluster.can_probe is False
        rows = _rows(cluster, ARGV_JSON)
        for row in rows.values():
            assert row["verdict"] is None, row
            assert row["confirmed"] is False
            assert row["entitlement_unconfirmed"] is True

    def test_the_placement_verdict_is_published_as_its_ingredients(self):
        # `_verdict_label` reads starts_now, fatal_blockers, verdict,
        # hardware_incompatible and soft_blockers. Each is in the row, and
        # `blockers` carries the `fatal` flag that splits the last two.
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        row = rows["shared"]
        for key in ("starts_now", "runnable_now", "reachable",
                    "hardware_incompatible", "blockers", "verdict"):
            assert key in row, row
        assert all("fatal" in b for b in row["blockers"]), row["blockers"]

    def test_the_table_still_prints_a_label_on_a_replay(self):
        # The half of the observation that was true and correct.
        assert "RUN NOW" in _run(_cluster(replayed=True), ARGV_TEXT)


class TestControls:
    """Properties of surfaces the change did not touch.

    Each holds with the ladder unified and with the second copy restored, so
    none of them is what detects the fix.
    """

    def test_the_access_column_is_dropped_when_it_would_not_vary(self):
        # No probe and no group-owned partition in the list: one word on every
        # row is noise, and the footnote says it once instead.
        cluster = _cluster(replayed=True)
        del cluster.queues["owned"]
        text = _run(cluster, ["--no-color", "where", "-c", "1", "--all"])
        # The HEADER row, not the whole page: the footnote says the word too,
        # and asserting on the page would pass for that reason instead.
        assert "access" not in _header(text), text

    def test_and_kept_when_a_group_owned_partition_is_listed(self):
        text = _run(_cluster(replayed=True), ARGV_TEXT)
        assert "access" in _header(text), text

    def test_a_spent_probe_budget_is_still_its_own_answer(self):
        # `probe budget spent` is a rung of the same ladder and a different
        # fact from `declared`; unifying must not have collapsed it.
        assert "probe budget spent" in ENTITLEMENT_SOURCES
        assert "declared" in ENTITLEMENT_SOURCES
        assert "not asked" in ENTITLEMENT_SOURCES

    def test_the_shared_partition_is_still_the_one_you_can_use(self):
        # The ranking nudge that puts a group's hardware second among equals.
        rows = list(json.loads(_run(_cluster(replayed=True), ARGV_JSON)))
        assert [r["queue"] for r in rows] == ["shared", "owned"], rows

    def test_submit_flags_and_caveats_survive(self):
        rows = _rows(_cluster(replayed=True), ARGV_JSON)
        assert rows["owned"]["submit_flags"] == ["--partition=owned"]
        assert any("DECLARED" in c for c in rows["owned"]["caveats"]), rows

    def test_every_source_the_ladder_can_return_is_published(self):
        from nodetop.cli import _entitlement_source

        cluster = _cluster(answers=TestAVerdictOutranksTheHeuristicInTheOneSlot.ANSWERS)
        seen = set()
        for c in (cluster, _cluster(replayed=True)):
            for row in _rows(c, ARGV_JSON).values():
                seen.add(row["entitlement_source"])
        assert seen <= set(ENTITLEMENT_SOURCES), seen
        assert callable(_entitlement_source)
