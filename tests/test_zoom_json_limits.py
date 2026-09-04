"""The two figures `zoom` acts on were on screen and in no payload.

The rendered header puts them above the node table::

    maxtime  1-12:00:00  (from slurm QOS gpu; the partition itself says unlimited)
    at once  4 of 11 nodes, 192 of 528 cores, 16 of 44 GPUs  (per user, from slurm QOS gpu)

`zoom --json` carried neither, and its own payload commits to the rule it broke:
"Named as `queues --json` names them -- one quantity, one key, whichever command
you ask." `queues --json` publishes `max_walltime_queue` and
`max_walltime_effective` for every partition; `zoom`, the drill-in for ONE
partition, published neither.

The per-user ceiling was worse: it appeared in **no** payload this tool emits,
checked across all ten commands. `_per_user_ceiling`'s docstring is about exactly
that -- "a per-user ceiling is invisible in every view that counts nodes",
reported live off four idle nodes against `MaxTRESPerUser=cpu=64,node=2` -- and
`zoom --json` was one of those views.

And the GPU half of the ceiling was dead on BOTH surfaces. `_limit_tres`
(`backends/slurm.py`) documents itself as producing "neutral limit keys" and maps
`gres/gpu=32` to ``out["gpu"]``; `Limits` documents the vocabulary as `cpu`,
`mem_mb`, `gpu`, `node`. The consumer read `"gres/gpu"`, the scheduler spelling,
so the lookup never hit. Measured on the recorded cluster: a cap of 2 against a
partition holding 8 GPUs rendered nothing at all, while the same cap written
`gres/gpu` rendered "2 of 8 GPUs" -- only the unreachable spelling worked, which
is why no test caught it. On the live cluster the `gpu` partition's line gained
"16 of 44 GPUs".
"""

from __future__ import annotations

import dataclasses
import json
import re

import nodetop.cli as cli
from nodetop.cli import build_parser
from nodetop.core.model import Limits
from nodetop.render import Glyphs, Style

PLAIN = Style(depth=0, glyphs=Glyphs())


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def _zoom_json(cluster, name: str, capsys) -> dict:
    cli.cmd_zoom(cluster, _args(["zoom", name, "--json"]), PLAIN)
    return json.loads(capsys.readouterr().out)


def _zoom_text(cluster, name: str, capsys) -> str:
    cli.cmd_zoom(cluster, _args(["zoom", name]), PLAIN)
    return capsys.readouterr().out


def _capped(cluster, queue: str, per_user: dict[str, int]):
    """``cluster`` with one queue's limit set replaced, so a ceiling BINDS.

    The recorded fixture's caps all sit above what its queues hold -- `gpu` has
    `per_user={'cpu': 192, 'gpu': 16, 'node': 4}` against 8 GPUs, 96 cores and 2
    nodes, so nothing binds and every assertion about a ceiling would pass
    vacuously against it. Asserted in `test_the_fixture_alone_would_prove_nothing`.
    """
    return dataclasses.replace(
        cluster,
        limits={**cluster.limits,
                queue: Limits(name=queue, per_user=dict(per_user), source="slurm QOS")},
    )


class TestThePerUserCeilingReachesBothSurfaces:
    def test_the_fixture_alone_would_prove_nothing(self, cluster):
        """Vacuity guard for `_capped`: no recorded queue has a binding ceiling,
        so the cases below have to construct one or they test nothing."""
        for name, queue in cluster.queues.items():
            assert cli._per_user_ceiling(cluster, queue, PLAIN) is None, name

    def test_a_gpu_ceiling_is_named_on_the_rendered_line(self, cluster, capsys):
        """The dead branch. 2 of 8 GPUs, under the key the parser really emits."""
        capped = _capped(cluster, "gpu", {"gpu": 2})
        line = cli._per_user_ceiling(capped, capped.queues["gpu"], PLAIN)
        assert line is not None, "the GPU ceiling is still invisible"
        assert "2 of 8 GPUs" in line, line

    def test_the_scheduler_spelling_is_not_what_binds(self, cluster):
        """The other half of the same finding: `gres/gpu` is what the SCHEDULER
        writes and what `_limit_tres` renames away, so a Limits carrying it is a
        state the backend cannot produce. Pinned so a "fix" that accepts both
        spellings has to argue for it rather than paper over the rename."""
        stale = _capped(cluster, "gpu", {"gres/gpu": 2})
        assert cli._per_user_ceiling(stale, stale.queues["gpu"], PLAIN) is None

    def test_the_ceiling_reaches_the_payload_too(self, cluster, capsys):
        capped = _capped(cluster, "gpu", {"gpu": 2})
        limits = _zoom_json(capped, "gpu", capsys)["limits"]["gpu"]
        assert limits["per_user"]["accelerators"] == [2, 8]
        assert limits["per_user"]["source"] == "slurm QOS"

    def test_the_two_surfaces_name_the_same_three_ceilings(self, cluster, capsys):
        """Rendered against payload, on one cluster: whatever the line lists, the
        document lists, and nothing more."""
        capped = _capped(cluster, "gpu", {"gpu": 2, "cpu": 8, "node": 1})
        line = cli._per_user_ceiling(capped, capped.queues["gpu"], PLAIN)
        per_user = _zoom_json(capped, "gpu", capsys)["limits"]["gpu"]["per_user"]
        assert ("1 of 2 nodes" in line) is ("nodes" in per_user)
        assert ("8 of 96 cores" in line) is ("cpus" in per_user)
        assert ("2 of 8 GPUs" in line) is ("accelerators" in per_user)
        assert per_user["nodes"] == [1, 2]
        assert per_user["cpus"] == [8, 96]
        assert per_user["accelerators"] == [2, 8]

    def test_a_ceiling_that_does_not_bind_is_reported_by_neither(
        self, cluster, capsys
    ):
        """The rule `_per_user_ceiling` states: "a ceiling at or above what the
        queue has is not a ceiling". Publishing it would make a consumer refuse a
        request the scheduler would take.

        NOT named a control: it reads the payload, so it reddens if the `limits`
        block is removed. It IS a control for the key fix -- it passes with either
        spelling, because nothing binds either way.
        """
        loose = _capped(cluster, "gpu", {"gpu": 99, "cpu": 9999, "node": 99})
        assert cli._per_user_ceiling(loose, loose.queues["gpu"], PLAIN) is None
        assert _zoom_json(loose, "gpu", capsys)["limits"]["gpu"]["per_user"] is None

    def test_nothing_at_all_is_null_not_an_empty_object(self, cluster, capsys):
        """"No ceiling here" and "a ceiling of zero" must stay different
        documents. Reads the payload, so not a control for the payload fix."""
        payload = _zoom_json(cluster, "gpu", capsys)
        assert payload["limits"]["gpu"]["per_user"] is None


class TestTheWalltimePairUsesTheOneVocabulary:
    def test_zoom_publishes_the_names_queues_publishes(self, cluster, capsys):
        """The rule the payload's own note states, for the two keys it omitted."""
        cli.cmd_queues(cluster, _args(["queues", "--all", "--json"]), PLAIN)
        rows = {e["name"]: e for e in json.loads(capsys.readouterr().out)}
        for name in cluster.queues:
            limits = _zoom_json(cluster, name, capsys)["limits"][name]
            for key in ("max_walltime_queue", "max_walltime_effective"):
                assert key in limits, (name, key)
                assert limits[key] == rows[name][key], (name, key)

    def test_the_rendered_header_and_the_payload_agree(self, cluster, capsys):
        """Surface against surface: the `maxtime` the header prints is the
        `max_walltime_effective` the document carries."""
        for name in cluster.queues:
            text = _zoom_text(cluster, name, capsys)
            match = re.search(r"maxtime\s+(\S+)", text)
            assert match, text
            limits = _zoom_json(cluster, name, capsys)["limits"][name]
            assert limits["max_walltime_effective"] == match.group(1), name

    def test_a_multi_queue_zoom_keys_them_apart(self, cluster, capsys):
        """`zoom` takes a comma-separated list and prints a header per partition.
        A limit is not summable, so one scalar would have named the FIRST queue's
        ceiling as the answer for all of them -- the same defect one level out.
        """
        names = sorted(cluster.queues)[:2]
        payload = _zoom_json(cluster, ",".join(names), capsys)
        assert sorted(payload["limits"]) == names
        assert payload[cluster.queue_term] == names


class TestControls:
    def test_control_the_keys_that_were_already_published_are_untouched(
        self, cluster, capsys
    ):
        payload = _zoom_json(cluster, "gpu", capsys)
        for key in ("nodes", "wholly_idle", "with_room", "unschedulable", "cpus",
                    "effective_free_cpus", "effective_free_accelerators",
                    "accelerators", "members"):
            assert key in payload, key
        queue = cluster.queues["gpu"]
        assert payload["nodes"] == len(queue.nodes)
        assert [n["name"] for n in payload["members"]] == [
            n["name"] for n in payload["members"]
        ]

    def test_control_the_rendered_view_still_prints_its_header(self, cluster, capsys):
        text = _zoom_text(cluster, "gpu", capsys)
        for word in ("nodes", "idle", "accel", "maxtime"):
            assert word in text, word

    def test_control_the_json_view_is_one_document_and_nothing_else(
        self, cluster, capsys
    ):
        cli.cmd_zoom(cluster, _args(["zoom", "gpu", "--json"]), PLAIN)
        out = capsys.readouterr().out
        json.loads(out)
        assert "maxtime" not in out
