"""The MCP surface, pinned against the CLI it wraps.

The protocol is the easy half and it is not what this file is mostly about.
An MCP server's real failure mode is that its schema and the command it
shells into drift apart: the tool advertises ``gpus`` forever, the flag it
builds stops parsing, and the model is told "no capacity" by a usage error it
never sees.  So the load-bearing test here is
:class:`TestEverySchemaBuildsArgvTheParserAccepts`, which feeds every declared
property of every tool through the real ``build_parser()``.

It exists because the motivating bug is already in this package: ``--gpu`` is
an ambiguous prefix of ``--gpus`` and ``--gpu-mem``, so the obvious spelling of
the commonest question is a parse error.  A hand-written flag list would
reintroduce that at the first rename; this one reddens instead.
"""

from __future__ import annotations

import io
import json

import pytest

from nodetop import mcp
from nodetop.cli import _JSON_SINK, _JsonCollector, _print_json, build_parser


def _rpc(method: str, ident: object = 1, **params: object) -> dict:
    message: dict = {"jsonrpc": "2.0", "method": method}
    if ident is not None:
        message["id"] = ident
    if params:
        message["params"] = params
    return message


#: A realistic value for each free-text property.
#:
#: Named rather than generated, because several of these flags validate their
#: argument and a generic string is not a test: `--needs 2` is rejected as an
#: unknown capability, which is the parser working. The point of the sweep is
#: to catch a flag that stopped existing, so every value here has to be one the
#: flag would actually accept.
STRINGS = {
    "memory_gb": "40",
    "walltime": "02:00:00",
    "gpu_memory_gb": "40",
    "needs": "bf16",
    "exclude": "cn-0001",
    "queue": "gn",
    "account": "acct-a",
    "qos": "normal",
}


def _sample(spec: dict, name: str) -> object:
    """A plausible value of the type a schema property declares."""
    kind = spec.get("type")
    if kind == "integer":
        return max(1, int(spec.get("minimum", 1)))
    if kind == "boolean":
        return True
    return STRINGS[name]


# ---------------------------------------------------------------------------
# the schema and the parser
# ---------------------------------------------------------------------------
class TestEverySchemaBuildsArgvTheParserAccepts:
    """The one invariant that keeps the tools and the CLI from drifting."""

    @pytest.mark.parametrize("tool", mcp.TOOLS, ids=[t.name for t in mcp.TOOLS])
    def test_with_no_arguments_at_all(self, tool: mcp.Tool) -> None:
        # Nothing is required, so an empty call must still be a valid command.
        build_parser().parse_args(tool.argv({}))

    @pytest.mark.parametrize("tool", mcp.TOOLS, ids=[t.name for t in mcp.TOOLS])
    def test_with_every_declared_property_set(self, tool: mcp.Tool) -> None:
        arguments = {n: _sample(sp, n) for n, sp in tool.properties.items()}
        parsed = build_parser().parse_args(tool.argv(arguments))
        assert parsed.command == tool.command
        assert parsed.json is True

    @pytest.mark.parametrize("tool", mcp.TOOLS, ids=[t.name for t in mcp.TOOLS])
    def test_each_property_alone_is_accepted(self, tool: mcp.Tool) -> None:
        # One at a time as well as all together: a flag that only parses in the
        # company of another is a flag a caller cannot use on its own.
        for name, spec in tool.properties.items():
            build_parser().parse_args(tool.argv({name: _sample(spec, name)}))

    def test_every_free_text_property_has_a_named_sample(self) -> None:
        """Otherwise a new string property would be swept with a stale value.

        The table above is what makes this file a test rather than a
        smoke screen; a property missing from it has to be added deliberately.
        """
        for tool in mcp.TOOLS:
            for name, spec in tool.properties.items():
                if spec.get("type") == "string":
                    assert name in STRINGS, (tool.name, name)

    def test_every_tool_names_a_real_command(self) -> None:
        parser = build_parser()
        subs = next(a for a in parser._subparsers._group_actions
                    if hasattr(a, "choices"))
        for tool in mcp.TOOLS:
            assert tool.command in subs.choices, tool.name

    def test_a_tool_probes_exactly_when_its_command_can_skip_the_dry_run(self) -> None:
        """`probes` is measured against the parser, not asserted by hand.

        `--declared` exists on exactly the commands that spend a dry-run, so it
        is the parser's own record of which tools are expensive. Declaring that
        by hand is how the throttle would come to guard the wrong ones.
        """
        parser = build_parser()
        subs = next(a for a in parser._subparsers._group_actions
                    if hasattr(a, "choices"))
        for tool in mcp.TOOLS:
            takes_declared = any(
                "--declared" in (a.option_strings or [])
                for a in subs.choices[tool.command]._actions
            )
            assert tool.probes is takes_declared, tool.name


class TestTheSchemaIsWellFormed:
    def test_names_are_unique(self) -> None:
        names = [t.name for t in mcp.TOOLS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("tool", mcp.TOOLS, ids=[t.name for t in mcp.TOOLS])
    def test_every_property_has_a_type_and_a_description(self, tool: mcp.Tool) -> None:
        for name, spec in tool.properties.items():
            assert spec.get("type"), (tool.name, name)
            assert spec.get("description"), (tool.name, name)

    @pytest.mark.parametrize("tool", mcp.TOOLS, ids=[t.name for t in mcp.TOOLS])
    def test_the_description_says_when_to_reach_for_it(self, tool: mcp.Tool) -> None:
        # A description that only restates the name gives a model nothing to
        # choose on, and choosing is the only thing it does with this list.
        assert len(tool.description) > 80, tool.name

    def test_nothing_that_writes_or_asks_the_host_is_exposed(self) -> None:
        """`snapshot` writes a file, `exclude` emits shell input, `backends`
        answers about this host, and `check` exists only to spend dry-runs.
        None is a question an agent asks, and the first is not read-only."""
        exposed = {t.command for t in mcp.TOOLS}
        assert not (exposed & {"snapshot", "exclude", "backends", "check"})

    def test_the_schema_serialises(self) -> None:
        # It goes on the wire; a value json cannot encode is a dead session.
        json.dumps([t.schema() for t in mcp.TOOLS])


# ---------------------------------------------------------------------------
# the collector in cli.py
# ---------------------------------------------------------------------------
class TestTheJsonSink:
    def test_a_payload_is_collected_rather_than_printed(self, capsys) -> None:
        with _JsonCollector() as got:
            _print_json({"a": 1})
        assert got == [{"a": 1}]
        assert capsys.readouterr().out == ""

    def test_printing_resumes_afterwards(self, capsys) -> None:
        with _JsonCollector():
            _print_json({"a": 1})
        _print_json({"b": 2})
        assert json.loads(capsys.readouterr().out) == {"b": 2}

    def test_the_stack_is_left_clean_after_an_exception(self) -> None:
        depth = len(_JSON_SINK)
        with pytest.raises(RuntimeError), _JsonCollector():
            raise RuntimeError("boom")
        assert len(_JSON_SINK) == depth

    def test_an_inner_collection_does_not_steal_the_outer_one(self) -> None:
        with _JsonCollector() as outer:
            _print_json("out")
            with _JsonCollector() as inner:
                _print_json("in")
            _print_json("out again")
        assert outer == ["out", "out again"]
        assert inner == ["in"]


class TestStdoutIsNotCorrupted:
    def test_a_command_that_prints_lands_in_the_log_not_the_payload(self) -> None:
        # `health` with no `--json` writes a table to stdout. Nothing of it may
        # reach the caller as a payload, and none of it may reach the real
        # stdout, which in a session is the JSON-RPC channel.
        rc, payloads, log = mcp._run(["health"])
        assert rc == 0
        assert payloads == []
        assert log.strip()

    def test_serve_restores_stdout(self) -> None:
        import sys

        before = sys.stdout
        mcp.serve(io.StringIO(""), io.StringIO())
        assert sys.stdout is before

    def test_every_line_written_is_one_valid_json_object(self) -> None:
        source = io.StringIO("\n".join([
            json.dumps(_rpc("initialize", 1)),
            json.dumps(_rpc("notifications/initialized", None)),
            json.dumps(_rpc("tools/list", 2)),
            json.dumps(_rpc("tools/call", 3, name="cluster_health", arguments={})),
        ]) + "\n")
        sink = io.StringIO()
        assert mcp.serve(source, sink) == 0
        lines = [x for x in sink.getvalue().splitlines() if x]
        # Three requests, one notification, three replies.
        assert len(lines) == 3
        assert [json.loads(x)["id"] for x in lines] == [1, 2, 3]


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
class TestHandshake:
    def test_a_version_we_know_is_echoed(self) -> None:
        for version in mcp.PROTOCOL_VERSIONS:
            reply = mcp.handle(_rpc("initialize", 1, protocolVersion=version))
            assert reply["result"]["protocolVersion"] == version

    def test_a_version_we_do_not_know_falls_back_to_the_newest(self) -> None:
        reply = mcp.handle(_rpc("initialize", 1, protocolVersion="1999-01-01"))
        assert reply["result"]["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]

    def test_it_declares_tools_and_names_itself(self) -> None:
        result = mcp.handle(_rpc("initialize", 1))["result"]
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "nodetop"

    def test_tools_list_returns_every_tool(self) -> None:
        result = mcp.handle(_rpc("tools/list", 1))["result"]
        assert {t["name"] for t in result["tools"]} == {t.name for t in mcp.TOOLS}

    def test_ping_is_answered(self) -> None:
        assert mcp.handle(_rpc("ping", 1))["result"] == {}


class TestNotificationsAreNeverAnswered:
    @pytest.mark.parametrize("method", ["notifications/initialized",
                                        "notifications/cancelled",
                                        "tools/list"])
    def test_a_message_with_no_id_gets_no_reply(self, method: str) -> None:
        # Including one that would otherwise be a valid request: an id is what
        # makes it a request, and replying to a notification is a violation
        # some clients drop the session over.
        assert mcp.handle(_rpc(method, None)) is None


class TestMalformedInput:
    def test_an_unknown_method_is_a_protocol_error(self) -> None:
        reply = mcp.handle(_rpc("frobnicate", 1))
        assert reply["error"]["code"] == -32601

    def test_a_wrong_protocol_version_field_is_rejected(self) -> None:
        reply = mcp.handle({"jsonrpc": "1.0", "id": 1, "method": "ping"})
        assert reply["error"]["code"] == -32600

    def test_a_missing_method_is_rejected(self) -> None:
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1})
        assert reply["error"]["code"] == -32600

    def test_params_must_be_an_object(self) -> None:
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "ping",
                            "params": [1, 2]})
        assert reply["error"]["code"] == -32602

    def test_a_call_with_no_tool_name_is_rejected(self) -> None:
        reply = mcp.handle(_rpc("tools/call", 1, arguments={}))
        assert reply["error"]["code"] == -32602

    def test_unparseable_input_is_answered_and_the_session_continues(self) -> None:
        source = io.StringIO("{not json\n\n" + json.dumps(_rpc("ping", 7)) + "\n")
        sink = io.StringIO()
        mcp.serve(source, sink)
        replies = [json.loads(x) for x in sink.getvalue().splitlines() if x]
        # The blank line is skipped, the bad line is answered, ping still works.
        assert [r.get("id") for r in replies] == [None, 7]
        assert replies[0]["error"]["code"] == -32700

    def test_a_non_object_message_is_a_parse_error(self) -> None:
        sink = io.StringIO()
        mcp.serve(io.StringIO("[1, 2, 3]\n"), sink)
        assert json.loads(sink.getvalue())["error"]["code"] == -32700


class TestAToolFailureIsNotAProtocolFailure:
    """A command that ran and reported a problem is a successful call.

    The protocol reserves JSON-RPC errors for the call itself going wrong. A
    tool that ran and said "no batch system here" must reach the model as its
    own words, because that is a fact it can act on -- `-32603` is not.
    """

    def test_an_unknown_tool_is_an_error_result(self) -> None:
        reply = mcp.handle(_rpc("tools/call", 1, name="nope", arguments={}))
        assert "error" not in reply
        assert reply["result"]["isError"] is True
        assert "nope" in reply["result"]["content"][0]["text"]

    def test_a_rejected_argument_is_an_error_result_with_the_reason(self) -> None:
        result = mcp.call("zoom_queue", {"queue": "no-such-queue-here"})
        assert result["isError"] is True
        assert any("no-such-queue-here" in b["text"] for b in result["content"])

    def test_a_crash_in_a_tool_does_not_take_the_session_down(self, monkeypatch) -> None:
        monkeypatch.setattr(mcp, "call",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("gone")))
        reply = mcp.handle(_rpc("tools/call", 1, name="cluster_health", arguments={}))
        assert reply["error"]["code"] == -32603
        assert "gone" in reply["error"]["message"]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
class TestResultShape:
    def test_a_payload_comes_back_as_the_json_the_cli_would_print(self) -> None:
        result = mcp.call("list_nodes", {"top": 2})
        assert result["isError"] is False
        rows = json.loads(result["content"][0]["text"])
        assert rows and all("name" in r for r in rows)

    def test_an_object_payload_is_also_offered_structured(self) -> None:
        result = mcp.call("zoom_queue", {"queue": "gn"})
        assert isinstance(result["structuredContent"], dict)
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]

    def test_a_list_payload_carries_no_structured_content(self) -> None:
        # The field is specified to be an object; a list may not go there.
        assert "structuredContent" not in mcp.call("list_nodes", {"top": 1})


class TestTheDryRunThrottle:
    def test_a_second_probing_call_is_downgraded_and_says_so(self) -> None:
        throttle = mcp._Throttle()
        first = mcp.call("where_can_i_run", {"gpus": 1}, throttle=throttle)
        second = mcp.call("where_can_i_run", {"gpus": 1}, throttle=throttle)
        assert not any("Dry-run skipped" in b["text"] for b in first["content"])
        assert any("Dry-run skipped" in b["text"] for b in second["content"])

    def test_the_payload_itself_is_left_alone(self) -> None:
        """The note is its own content block, never folded into the answer.

        A caveat smuggled into the payload would make the MCP result a
        different shape from `--json`, which is the drift this whole module is
        arranged to avoid.
        """
        throttle = mcp._Throttle()
        mcp.call("where_can_i_run", {"gpus": 1}, throttle=throttle)
        second = mcp.call("where_can_i_run", {"gpus": 1}, throttle=throttle)
        payload = json.loads(second["content"][0]["text"])
        assert all("note" not in row for row in payload)

    def test_a_non_probing_tool_is_never_throttled(self) -> None:
        throttle = mcp._Throttle()
        for _ in range(3):
            result = mcp.call("list_nodes", {"top": 1}, throttle=throttle)
            assert not any("Dry-run skipped" in b["text"] for b in result["content"])

    def test_a_caller_who_already_asked_for_declared_is_not_told_it_was_skipped(
        self,
    ) -> None:
        throttle = mcp._Throttle()
        mcp.call("where_can_i_run", {"gpus": 1}, throttle=throttle)
        second = mcp.call("where_can_i_run", {"gpus": 1, "declared": True},
                          throttle=throttle)
        assert not any("Dry-run skipped" in b["text"] for b in second["content"])

    def test_zero_disables_the_limit(self, monkeypatch) -> None:
        monkeypatch.setenv("NODETOP_MCP_PROBE_INTERVAL", "0")
        throttle = mcp._Throttle()
        assert throttle.allow()
        assert throttle.allow()

    def test_a_typo_in_the_interval_is_the_default_not_a_crash(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("NODETOP_MCP_PROBE_INTERVAL", "soon")
        assert mcp._probe_interval() == mcp.PROBE_INTERVAL

    def test_a_negative_interval_disables_rather_than_reversing_time(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("NODETOP_MCP_PROBE_INTERVAL", "-5")
        assert mcp._probe_interval() == 0.0


# ---------------------------------------------------------------------------
# the subcommand
# ---------------------------------------------------------------------------
class TestTheSubcommand:
    def test_mcp_serves_and_exits_cleanly_on_a_closed_pipe(self, monkeypatch) -> None:
        from nodetop import cli

        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        monkeypatch.setattr("sys.stdout", io.StringIO())
        assert cli.main(["mcp"]) == 0

    def test_the_source_flags_are_forwarded_into_every_call(self, monkeypatch) -> None:
        from nodetop import cli

        seen: dict = {}
        monkeypatch.setattr("nodetop.mcp.serve",
                            lambda **kw: seen.update(kw) or 0)
        assert cli.main(["mcp", "--backend", "slurm"]) == 0
        assert seen["extra"] == ["--backend", "slurm"]

    def test_the_server_takes_no_rendering_flags(self) -> None:
        # There is nothing to colour and no `--json` to opt into; see
        # `_add_source_args`.
        parser = build_parser()
        subs = next(a for a in parser._subparsers._group_actions
                    if hasattr(a, "choices"))
        flags = {o for a in subs.choices["mcp"]._actions for o in a.option_strings}
        assert "--backend" in flags and "--replay" in flags
        assert not (flags & {"--json", "--no-color", "--ascii"})

    def test_extra_arguments_reach_the_command(self) -> None:
        # `--replay`/`--backend` are root flags, so they must lead the argv.
        rc, payloads, log = mcp._run(["--backend", "slurm", "nodes", "--json",
                                      "-n", "1"])
        assert rc == 0 and payloads
