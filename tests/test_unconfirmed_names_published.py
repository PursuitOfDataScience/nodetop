"""The funnel hedges 25 of 30 rows and never said which 25.

`status` pretests entitlement with a dry-run per candidate partition and sorts the
answers into three buckets. Two of them are findings: a durable refusal drops the
partition and `excluded` names it with reason `refused`; an acceptance keeps it. The
third is the hedge -- `access.MAYBE`, "the control plane did not settle this" -- and
`ask()`'s own docstring says why it is kept apart: "so the funnel does not present a
guess as a finding".

The partitions in that third bucket are collected into `unsettled_names`
(`cli.py:1561` at HEAD) and the very next line reads it:

    unsettled = len(unsettled_names)

That was the only read in `src`. The count then reaches both surfaces -- the printed
funnel as `30 open to you (25 unconfirmed)`, `--json` as `funnel.unconfirmed` -- and
the names reached neither.

**Why that is a loss and not just an omission.** The count exists to say the rows are
not all verified. So a consumer reading `{"shown": 30, "unconfirmed": 25}` learns that
5 of the 30 `listed` rows were measured and 25 were assumed, and cannot learn which 5
-- which makes the one decision the hedge is for, preferring a confirmed partition
over an assumed one, the one decision it cannot support. Nothing else in the payload
separates them: `listed` draws a measured acceptance and an unasked guess as the same
row, and the unsettled partitions are deliberately *not* in `excluded`, because they
are shown and putting them there would break the property that the funnel's terms sum
to the total.

**The repo had already committed to naming them.** `cli.py` dropped the DEAD block and
the funnel footer on an explicit condition -- "Nothing is *hidden* by dropping them,
which is the condition for dropping them at all" -- satisfied by the names being
recoverable elsewhere. Every other term honours it: "no access" (`cli.py:1449`),
"refused" (`:1562`), "no nodes" (`:1717`) and "down" (`:1716`) all append to
`excluded`. `unconfirmed` was the single term that did not, and `status`'s own
docstring is what it fails against: "The README promises `--json` carries what the
text does."

Proved by rendering the surface twice. Two clusters identical in every respect except
which partitions the dry-run settled -- `{alpha, bravo}` confirmed against
`{charlie, delta}` confirmed, so the unconfirmed sets are disjoint and both hold two
names -- emitted `status --json` documents of 2209 bytes with the same sha256.

Fixed in `--json` only, following `unverified_node_names`: additive, no layout risk,
and the payload was where the asymmetry lived. The printed funnel keeps its bare count
because it renders into a panel that truncates rather than wraps, and this repo has
already shipped one table addition that turned out to be dead code behind a gate.
Published as a top-level list rather than a term inside `funnel`, because `funnel`'s
values are counts that sum to the total and a list there would break both properties.

`--all` gets an empty list on purpose: that flag replaces the head term with "N with
nodes", which makes no claim about access at all, so there is no hedge to itemise. The
`--declared` and no-dry-run paths get every shown partition, which is what the count
already says there -- "Nothing asked means nothing settled".
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import sys

from nodetop.cli import build_parser, cmd_status
from nodetop.core.cluster import Cluster
from nodetop.core.model import (
    BackendCapabilities,
    Identity,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())

#: Four interchangeable partitions: same node count, same emptiness, no allowlist. So
#: the only thing a scenario can vary is what the dry-run said, which is the point.
FOUR = ("alpha", "bravo", "charlie", "delta")


def _cluster(settled, *, names=FOUR, refuses=(), probe=True):
    """A cluster whose dry-run accepts ``settled`` and hedges everything else.

    ``refuses`` answers with a DURABLE refusal instead, which is the other branch of
    `ask()` -- those partitions leave the table and are named in `excluded`.
    """
    nodes, queues = [], {}
    for name in names:
        mine = [
            Node(name=f"{name}{i}", state_raw="IDLE", cpus_total=8,
                 memory_mb=16000, queues=(name,))
            for i in range(2)
        ]
        nodes += mine
        queues[name] = Queue(name=name, node_names=tuple(n.name for n in mine),
                             declared_nodes=2, nodes=mine)

    class _Backend:
        name = "synthetic"
        queue_term = "partition"

        def capabilities(self):
            return BackendCapabilities(probe=probe, probe_supported=probe,
                                       probe_command="stub")

        def probe(self, q, shape, account=None):
            if q in settled:
                return Verdict(queue=q, account=account, allowed=True,
                               category=VerdictCategory.OK, reason="")
            if q in refuses:
                return Verdict(queue=q, account=account, allowed=False,
                               category=VerdictCategory.NOT_ENTITLED,
                               reason="not in the account list")
            # Transient, so `Verdict.durable` is False: the question went
            # unanswered, the partition stays on screen, and the funnel counts it
            # as unconfirmed rather than dropping it.
            return Verdict(queue=q, account=account, allowed=False,
                           category=VerdictCategory.CONTROL_PLANE_DOWN,
                           reason="controller unreachable")

        def submit_flags(self, q, shape):
            return []

    cluster = Cluster(backend_name="synthetic", queue_term="partition",
                      nodes=nodes, queues=queues,
                      identity=Identity(user="me", accounts=("mine",), qos=("x",)))
    if probe:
        cluster = dataclasses.replace(cluster, capabilities=_Backend().capabilities(),
                                      _backend=_Backend())
    return cluster


def _render(cluster, argv=("status", "--json")):
    buf, saved = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        cmd_status(cluster, build_parser().parse_args(list(argv)), PLAIN)
    finally:
        sys.stdout = saved
    return buf.getvalue()


def _payload(cluster, argv=("status", "--json")):
    return json.loads(_render(cluster, argv))


class TestTheHedgeNamesItsMembers:
    def test_disjoint_unsettled_sets_no_longer_render_identically(self):
        """The rendering that proved the loss: same count, different names.

        Before the fix both documents were 2209 bytes with an identical sha256.
        """
        a = _render(_cluster({"alpha", "bravo"}))
        b = _render(_cluster({"charlie", "delta"}))
        assert json.loads(a)["funnel"]["unconfirmed"] == 2
        assert json.loads(b)["funnel"]["unconfirmed"] == 2
        assert a != b, "the unconfirmed set is disjoint; the documents must differ"

    def test_the_names_are_the_partitions_the_control_plane_did_not_settle(self):
        got = _payload(_cluster({"alpha", "bravo"}))
        assert got["unconfirmed_names"] == ["charlie", "delta"]
        assert _payload(_cluster({"charlie", "delta"}))["unconfirmed_names"] == [
            "alpha", "bravo",
        ]

    def test_a_confirmed_partition_is_not_named(self):
        got = _payload(_cluster(set(FOUR)))
        assert got["funnel"]["unconfirmed"] == 0
        assert got["unconfirmed_names"] == []

    def test_the_count_and_the_names_agree_on_every_path(self):
        """The invariant the fix adds: the list IS the count, itemised."""
        cases = [
            (("status", "--json"), _cluster({"alpha"})),
            (("status", "--json"), _cluster(set(FOUR))),
            (("status", "--json"), _cluster(set())),
            (("status", "--json", "--declared"), _cluster({"alpha"})),
            (("status", "--json", "--all"), _cluster({"alpha"})),
            (("status", "--json"), _cluster({"alpha"}, probe=False)),
        ]
        for argv, cluster in cases:
            got = _payload(cluster, argv)
            assert got["funnel"]["unconfirmed"] == len(got["unconfirmed_names"]), argv

    def test_every_named_partition_is_one_of_the_listed_rows(self):
        """It qualifies a row in `listed`; a name outside it could qualify nothing."""
        got = _payload(_cluster({"alpha"}))
        listed = {row["name"] for row in got["listed"]}
        assert got["unconfirmed_names"], "this scenario has a hedge to itemise"
        assert set(got["unconfirmed_names"]) <= listed

    def test_nothing_asked_means_every_shown_partition_is_named(self):
        """`--declared` and a backend with no dry-run both settle nothing."""
        for cluster, argv in (
            (_cluster({"alpha"}), ("status", "--json", "--declared")),
            (_cluster(set(), probe=False), ("status", "--json")),
        ):
            got = _payload(cluster, argv)
            assert got["funnel"]["unconfirmed"] == got["funnel"]["shown"]
            assert sorted(got["unconfirmed_names"]) == sorted(
                row["name"] for row in got["listed"]
            )

    def test_all_makes_no_access_claim_so_it_names_nobody(self):
        got = _payload(_cluster({"alpha"}), ("status", "--json", "--all"))
        assert got["funnel"]["unconfirmed"] == 0
        assert got["unconfirmed_names"] == []

    def test_the_degenerate_path_carries_the_key_too(self):
        """One schema, whichever path reaches it -- including the empty cluster."""
        empty = Cluster(backend_name="synthetic", queue_term="partition",
                        nodes=[], queues={})
        assert _payload(empty)["unconfirmed_names"] == []

    def test_a_refusal_is_a_finding_not_a_hedge(self):
        """The bucket next door must not leak into this list.

        `charlie` is refused durably, so it leaves the table and `excluded` names it;
        `delta` is unsettled, so it stays and this list names it. Reporting one as the
        other is the conflation `ask()` is written to prevent.
        """
        got = _payload(_cluster({"alpha", "bravo"}, refuses={"charlie"}))
        assert got["unconfirmed_names"] == ["delta"]
        assert {"name": "charlie", "reason": "refused"} in got["excluded"]
        assert "charlie" not in {row["name"] for row in got["listed"]}


class TestWhatTheFixMustNotDisturb:
    """Controls. Every one of these passes with the fix in and with it out.

    Deliberately built from inputs the tests above do not use -- a cluster with a
    refused and a dead partition, and the printed rather than the JSON form -- so a
    control cannot go red for the same reason a finding does.
    """

    @staticmethod
    def _mixed():
        """Two accepted, one durably refused, one dead. Nothing unsettled."""
        cluster = _cluster({"alpha", "bravo"}, names=("alpha", "bravo", "charlie"),
                           refuses={"charlie"})
        dead = Queue(name="zulu", node_names=("zulu0",), declared_nodes=1,
                     nodes=[Node(name="zulu0", state_raw="IDLE", cpus_total=8,
                                 memory_mb=16000, queues=("zulu",))],
                     state_raw="DOWN", enabled=False)
        queues = {**cluster.queues, "zulu": dead}
        return dataclasses.replace(cluster, queues=queues,
                                   nodes=[*cluster.nodes, *dead.nodes])

    def test_the_funnel_terms_still_sum_to_the_total(self):
        got = _payload(self._mixed())["funnel"]
        total = got.pop("total")
        shown = got.pop("shown")
        got.pop("unconfirmed")
        assert shown + sum(got.values()) == total, got

    def test_excluded_still_names_every_partition_it_dropped(self):
        got = _payload(self._mixed())
        by_name = {row["name"]: row["reason"] for row in got["excluded"]}
        assert by_name["charlie"] == "refused"
        assert by_name["zulu"] == "down"
        assert set(by_name).isdisjoint(row["name"] for row in got["listed"])

    def test_the_printed_funnel_still_carries_the_count(self):
        out = _render(_cluster({"alpha"}), ("status", "--static"))
        line = next(ln for ln in out.splitlines() if "partitions" in ln)
        assert re.search(r"\(\d+ unconfirmed\)", line), line

    def test_the_printed_form_still_lists_a_row_per_open_partition(self):
        out = _render(self._mixed(), ("status", "--static"))
        assert "alpha" in out and "bravo" in out

    def test_listed_still_reports_the_room_it_always_did(self):
        rows = {row["name"]: row for row in _payload(self._mixed())["listed"]}
        assert rows["alpha"]["nodes"] == 2
        assert rows["alpha"]["cpus_total"] == 16
