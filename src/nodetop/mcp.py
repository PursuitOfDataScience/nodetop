"""MCP over stdio, so an agent asks this tool's question instead of ``sinfo``'s.

**Why, when ``--json`` already works.**  An agent with a shell can already run
``nodetop where -g 4 --json`` and get these exact bytes, so the protocol adds
no data.  What it adds is a *typed* interface, and that turns out to matter
more than it sounds: this parser has flags that are ambiguous prefixes of each
other, so ``--gpu`` is rejected with ``ambiguous option: --gpu could match
--gpus, --gpu-mem`` -- the obvious spelling of the commonest question is a
usage error.  A schema with ``gpus: integer`` deletes that whole class of
failure, and it is the only thing here the CLI could not do by itself.  The
second reason is clients with no shell at all.

**Nothing is re-derived.**  Each tool runs the real command with ``--json`` and
returns the payload :func:`nodetop.cli._print_json` was about to print.  A
second set of builders assembling the same dictionaries for the protocol is
exactly the drift this package keeps paying for elsewhere, and it would be
worse here than in a renderer: a report that disagrees with the CLI is
something a reader can see, while a tool result that disagrees is something a
model repeats with confidence.  Going through :func:`nodetop.cli.main` also
means every guard comes along unasked -- backend detection and its warnings,
the broken-snapshot refusal, the unknown-queue check, and the exit code.

**stdout belongs to the protocol, and nothing else may touch it.**  A single
stray ``print`` on the JSON-RPC channel corrupts the stream and the client
drops the connection with no useful diagnosis.  The package has 111 prints and
several background threads, so rather than audit them, :func:`serve` takes the
real stdout away at startup, hands it to the framing code alone, and points
``sys.stdout`` at stderr for the life of the process.  Anything that prints is
then merely logging, which is what the transport expects on that descriptor.

**One read per call, never a cache.**  The server outlives the cluster state it
describes -- that is the whole difference from a CLI run -- so each call takes
its own snapshot and pays the ~2 s.  Serving a remembered answer would be this
tool's cardinal sin committed by its newest surface.

**Dry-runs are rate-limited, and a downgrade is said out loud.**  ``status``,
``where`` and ``check`` ask the control plane with ``sbatch --test-only``.  A
person types those a few times an hour; an agent in a loop will call them fifty
times a minute at somebody else's controller.  Calls inside
:data:`PROBE_INTERVAL` of the last one are answered from the declared
allowlists instead, and the reply carries a second content block saying so --
the payload itself stays byte-identical to what ``--json`` prints, because a
caveat belongs beside the answer rather than smuggled into it.

The transport is newline-delimited JSON-RPC 2.0 on stdin/stdout, and a
tools-only server needs four methods, so this is standard library throughout.
Depending on the official SDK would pull in pydantic, anyio, httpx and
starlette and break the no-dependency rule that pyproject.toml states with a
reason: this must run on a login node with nothing but the system Python.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

__all__ = ["TOOLS", "Tool", "call", "handle", "serve"]

#: Protocol revisions this server knows how to speak, newest first.
#:
#: Negotiated rather than asserted: the spec says the client proposes a version
#: in ``initialize`` and the server answers with one it supports.  Echoing a
#: version we recognise keeps an older client working; falling back to the
#: newest we know is the honest answer to a version from the future, and lets
#: the client decide whether to continue.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

#: Seconds a dry-run buys before another is spent.  ``NODETOP_MCP_PROBE_INTERVAL``
#: overrides it; ``0`` disables the limit for a caller who knows their own
#: controller can take it.
#:
#: Ten seconds is not a tuned number.  It is "faster than a person asking twice
#: and slower than a loop", which is the only distinction the limiter needs to
#: draw -- the access cache in `nodetop.core.access` already holds a verdict for
#: a day, so a repeated *identical* question costs nothing either way, and what
#: this stops is a sweep of varying shapes.
PROBE_INTERVAL = 10.0

#: JSON-RPC error codes used here.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


def _probe_interval() -> float:
    """:data:`PROBE_INTERVAL`, honouring the environment override.

    A bad value is the default, not an error: this is a throttle knob, and
    refusing to start a server because someone exported a typo would trade a
    small misconfiguration for a total outage.
    """
    raw = os.environ.get("NODETOP_MCP_PROBE_INTERVAL")
    if raw is None:
        return PROBE_INTERVAL
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return PROBE_INTERVAL


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tool:
    """One exposed command: what to call it, and how its arguments spell out.

    ``build`` turns the schema's arguments into the argv the CLI already
    parses, which is deliberately the *only* translation layer in this module.
    Everything downstream of it is the real command.

    A dataclass rather than a hand-written ``__init__``, and not only for the
    brevity: ``self.name = name`` would trip the source scan in
    ``tests/test_model.py``, which flags any assignment to an attribute sharing
    a name with a ``Node`` field.  That guard cannot tell one class's ``name``
    from another's, and widening it to let this through would cost more than
    the generated constructor does.  Frozen because a tool table is a
    declaration.
    """

    name: str
    command: str
    description: str
    properties: dict[str, dict[str, Any]]
    build: Callable[[dict[str, Any]], list[str]]
    #: True when running this can spend a dry-run against the control plane.
    #: Read by the rate limiter, and measured against the parser by a test
    #: rather than trusted: `--declared` exists on exactly the commands that
    #: probe, so it is the parser's own record of which tools are expensive.
    probes: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.properties,
                "required": [],
                "additionalProperties": False,
            },
        }

    def argv(self, arguments: dict[str, Any]) -> list[str]:
        return [self.command, "--json", *self.build(arguments)]


def _flag(args: dict[str, Any], key: str, flag: str) -> list[str]:
    """``[flag]`` when the caller asked for it, nothing when they did not."""
    return [flag] if args.get(key) else []


def _value(args: dict[str, Any], key: str, flag: str) -> list[str]:
    """``[flag, value]`` for anything the caller actually set.

    Empty string and ``None`` both mean "not set" -- a client that spells an
    unused optional as ``""`` should not end up passing ``-q ''``, which the
    queue-name validator would reject as a name.  Zero is *not* treated that
    way: ``gpus: 0`` is a real request for a CPU job.
    """
    got = args.get(key)
    if got is None or got == "":
        return []
    return [flag, str(got)]


def _shape(args: dict[str, Any]) -> list[str]:
    """The job-shape flags shared by ``where`` and ``check``."""
    return [
        *_value(args, "nodes", "-N"),
        *_value(args, "gpus", "-g"),
        *_value(args, "cpus", "-c"),
        *_value(args, "memory_gb", "--mem"),
        *_value(args, "walltime", "-t"),
        *_value(args, "gpu_memory_gb", "--gpu-mem"),
        *_value(args, "needs", "--needs"),
        *_value(args, "exclude", "--exclude"),
        *_value(args, "queue", "-q"),
        *_value(args, "account", "-A"),
        *_value(args, "qos", "--qos"),
    ]


_QUEUE = {"type": "string",
          "description": "limit to these queues/partitions (comma-separated)"}
_ALL = {"type": "boolean", "description": "include what is normally filtered out"}
_DECLARED = {"type": "boolean",
             "description": "skip the dry-run and trust the declared allowlists "
                            "(faster, but they over-report on many clusters)"}

#: The exposed surface.
#:
#: Six of the eleven commands, and the omissions are not an oversight.
#: ``snapshot`` writes a file, ``exclude`` emits a node list for a shell to
#: interpolate, and ``backends`` answers a question about the local host rather
#: than about the cluster -- none of the three is a question an agent asks.
#: ``check`` is the one deliberate exclusion: it exists to spend dry-runs, and
#: ``where`` already reports the verdict it would return.
TOOLS: tuple[Tool, ...] = (
    Tool(
        "where_can_i_run",
        "where",
        "Rank the queues a job of this shape could actually run in, and say why "
        "each one is ruled out. Asks the control plane with a dry-run, so it "
        "reports refusals that the advertised limits do not. Returns each "
        "queue's verdict, free capacity, blocking reasons, earliest possible "
        "start, and ready-made submit flags. This is the tool to use for "
        "'where should I submit this'.",
        {
            "nodes": {"type": "integer", "minimum": 1, "description": "nodes wanted"},
            "gpus": {"type": "integer", "minimum": 0,
                     "description": "accelerators per node (0 for a CPU job)"},
            "cpus": {"type": "integer", "minimum": 1, "description": "CPUs per task"},
            "memory_gb": {"type": "string",
                          "description": "memory per node, e.g. '40' or '40G'"},
            "walltime": {"type": "string", "description": "e.g. '04:00:00'"},
            "gpu_memory_gb": {"type": "string",
                              "description": "minimum accelerator memory, e.g. '40'"},
            "needs": {"type": "string",
                      "description": "required accelerator capabilities, "
                                     "comma-separated (e.g. 'bf16')"},
            "exclude": {"type": "string", "description": "nodes to rule out"},
            "queue": _QUEUE,
            "account": {"type": "string", "description": "submit as this account"},
            "qos": {"type": "string", "description": "name this QOS in the dry-run"},
            "declared": _DECLARED,
            "all": {"type": "boolean", "description": "include ruled-out queues"},
        },
        lambda a: [*_shape(a), *_flag(a, "declared", "--declared"),
                   *_flag(a, "all", "--all")],
        probes=True,
    ),
    Tool(
        "cluster_status",
        "status",
        "Overview of what is usable right now: the queues that can start work, "
        "their free capacity, and a count of everything filtered out and why. "
        "Use this for 'how busy is the cluster' or 'what is free'.",
        {"declared": _DECLARED,
         "all": {"type": "boolean",
                 "description": "every queue, including those with nothing free"}},
        lambda a: [*_flag(a, "declared", "--declared"), *_flag(a, "all", "--all"),
                   "--static"],
        probes=True,
    ),
    Tool(
        "list_queues",
        "queues",
        "Per-queue state and the specific gates on each one -- disabled, closed "
        "to your account, closed to every QOS, hidden, time-capped. Use this "
        "for 'why will this partition not take my job'.",
        {"queue": _QUEUE,
         "unusable_only": {"type": "boolean",
                           "description": "only queues that cannot start work"},
         "all": {"type": "boolean", "description": "every queue, not only yours"}},
        lambda a: [*_value(a, "queue", "-q"),
                   *_flag(a, "unusable_only", "--unusable-only"),
                   *_flag(a, "all", "--all")],
    ),
    Tool(
        "list_nodes",
        "nodes",
        "Node inventory: state, free CPUs, free memory, free accelerators and "
        "the card model, per node. Use this for 'which nodes have a free GPU' "
        "or 'what hardware does this cluster have'.",
        {"queue": _QUEUE,
         "gpu": {"type": "boolean", "description": "GPU nodes only"},
         "cpu": {"type": "boolean", "description": "CPU-only nodes"},
         "free": {"type": "boolean", "description": "only nodes with something free"},
         "top": {"type": "integer", "minimum": 1, "description": "how many (default 20)"},
         "all": {"type": "boolean", "description": "every matching node"}},
        lambda a: [*_value(a, "queue", "-q"), *_flag(a, "gpu", "--gpu"),
                   *_flag(a, "cpu", "--cpu"), *_flag(a, "free", "--free"),
                   *_value(a, "top", "-n"), *_flag(a, "all", "--all")],
    ),
    Tool(
        "zoom_queue",
        "zoom",
        "Look inside one queue node by node: its gates, then each node's free "
        "capacity and per-user ceiling. Use this after cluster_status or "
        "where_can_i_run has named a queue worth a closer look.",
        {"queue": {"type": "string",
                   "description": "the queue to open (comma-separated for several)"},
         "gpu": {"type": "boolean", "description": "GPU nodes only"},
         "cpu": {"type": "boolean", "description": "CPU-only nodes"},
         "free": {"type": "boolean", "description": "only nodes with something free"},
         "top": {"type": "integer", "minimum": 1, "description": "how many (default 20)"},
         "all": {"type": "boolean", "description": "every node, however many"}},
        lambda a: [str(a.get("queue", "")), *_flag(a, "gpu", "--gpu"),
                   *_flag(a, "cpu", "--cpu"), *_flag(a, "free", "--free"),
                   *_value(a, "top", "-n"), *_flag(a, "all", "--all")],
    ),
    Tool(
        "cluster_health",
        "health",
        "Nodes that are down, drained or silently degraded, with the operator's "
        "reason for each. Use this for 'is anything broken' or to explain why a "
        "queue's advertised idle nodes cannot start anything.",
        {},
        lambda _a: [],
    ),
    Tool(
        "list_accelerators",
        "accelerators",
        "GPU inventory by model, with what each model can do (dtypes, memory) "
        "and how many are free. Use this for 'does this cluster have a card "
        "that supports bf16' or 'how many A100s are there'.",
        {"queue": _QUEUE,
         "all": {"type": "boolean",
                 "description": "every accelerator, not only in queues you may use"}},
        lambda a: [*_value(a, "queue", "-q"), *_flag(a, "all", "--all")],
    ),
)

_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# running a tool
# ---------------------------------------------------------------------------
class _Throttle:
    """When the last dry-run was spent, and whether another may be.

    Deliberately a wall-clock stamp rather than a token bucket: the thing being
    protected is somebody else's scheduler, and the only property that matters
    is that a tight loop cannot make back-to-back probe runs.
    """

    def __init__(self) -> None:
        self.last = float("-inf")

    def allow(self, now: float | None = None) -> bool:
        interval = _probe_interval()
        if interval <= 0:
            return True
        current = time.monotonic() if now is None else now
        if current - self.last < interval:
            return False
        self.last = current
        return True


#: Wording of the downgrade, in one place so the note and the test agree.
DOWNGRADED = (
    "Dry-run skipped: another probing call was made less than {interval:.0f}s ago, "
    "so this answer comes from the declared allowlists instead of from the control "
    "plane. Declared access over-reports on many clusters -- treat any 'allowed' "
    "here as unconfirmed. Ask again in a few seconds for the probed answer."
)


def call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    throttle: _Throttle | None = None,
    extra: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one tool and shape the result the way ``tools/call`` wants it.

    A failing command is **not** a JSON-RPC error.  The protocol reserves those
    for the call itself going wrong -- a method that does not exist, parameters
    that do not parse -- while a tool that ran and reported a problem is a
    successful call with ``isError`` set, so the model sees the diagnosis
    instead of a transport failure.  That distinction is the whole reason
    ``nodetop``'s exit codes are worth forwarding: exit 3 means *no batch system
    here*, which an agent can act on, and burying it in ``-32603`` would not.
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        return _error_result(f"no such tool: {name}")

    args = dict(arguments or {})
    notes: list[str] = []
    if (tool.probes and not args.get("declared")
            and throttle is not None and not throttle.allow()):
        args["declared"] = True
        notes.append(DOWNGRADED.format(interval=_probe_interval()))

    try:
        argv = [*extra, *tool.argv(args)]
    except (TypeError, ValueError) as exc:
        return _error_result(f"bad arguments for {name}: {exc}")

    rc, payloads, log = _run(argv)

    content: list[dict[str, Any]] = []
    if payloads:
        body = payloads[0] if len(payloads) == 1 else payloads
        content.append({"type": "text",
                        "text": json.dumps(body, indent=2, default=str)})
    elif not log:
        # Ran, said nothing, blamed nothing. Better to be explicit than to hand
        # back an empty result the model reads as "no capacity anywhere".
        content.append({"type": "text",
                        "text": f"{tool.command} produced no output (exit {rc})"})
    if log:
        # stderr carries the warnings that make an answer readable -- a forced
        # backend that does not detect, the named queries that failed in an
        # outage. Dropping them would hide exactly what the guard exists to say.
        content.append({"type": "text", "text": log.strip()})
    content.extend({"type": "text", "text": note} for note in notes)

    result: dict[str, Any] = {"content": content, "isError": rc != 0}
    if payloads and len(payloads) == 1 and isinstance(payloads[0], dict):
        # Only an object may go here; several commands answer with a list, and
        # those keep to the text block alone.
        result["structuredContent"] = json.loads(
            json.dumps(payloads[0], default=str))
    return result


def _error_result(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _run(argv: Sequence[str]) -> tuple[int, list[object], str]:
    """The real CLI, with its JSON collected and its streams caught.

    Both streams are captured for the same reason the server takes stdout away
    at startup: this runs with the JSON-RPC channel one descriptor over, and
    anything written by accident would corrupt a session rather than merely
    look untidy.  ``SystemExit`` is caught because ``argparse`` raises it for a
    bad argument, which here is a tool-call error and not a reason to stop
    serving.
    """
    import io

    from . import cli

    out, err = io.StringIO(), io.StringIO()
    stashed_out, stashed_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        with cli._JsonCollector() as payloads:
            try:
                rc = cli.main(list(argv))
            except SystemExit as exc:
                rc = exc.code if isinstance(exc.code, int) else 2
            except Exception as exc:  # noqa: BLE001 - a crash is a tool error
                rc = 1
                err.write(f"{type(exc).__name__}: {exc}\n")
    finally:
        sys.stdout, sys.stderr = stashed_out, stashed_err
    # Anything that reached stdout despite `--json` belongs with the log, not
    # with the payload: the caller asked for machine-readable output and this
    # is whatever was not.
    log = "\n".join(x for x in (err.getvalue(), out.getvalue()) if x.strip())
    return rc, list(payloads), log


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
def handle(
    message: dict[str, Any],
    *,
    throttle: _Throttle | None = None,
    extra: Sequence[str] = (),
) -> dict[str, Any] | None:
    """One request in, one response out -- or ``None`` for a notification.

    Separated from :func:`serve` so the whole protocol is testable without a
    pipe, which is also what keeps the framing honest: everything that decides
    *what* to answer lives here, and :func:`serve` only decides when to read
    and where to write.
    """
    if message.get("jsonrpc") != "2.0":
        return _rpc_error(message.get("id"), _INVALID_REQUEST,
                          "jsonrpc must be '2.0'")
    method = message.get("method")
    if not isinstance(method, str):
        return _rpc_error(message.get("id"), _INVALID_REQUEST, "missing method")
    ident = message.get("id")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _rpc_error(ident, _INVALID_PARAMS, "params must be an object")

    # A notification has no id and takes no response, ever -- including for an
    # error. `notifications/initialized` is the one every client sends, and
    # answering it is a protocol violation that some clients drop the session
    # over.
    if ident is None:
        return None

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        from ._version import VERSION

        return _rpc_result(ident, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "nodetop", "version": VERSION},
            "instructions":
                "Read-only cluster capacity. Every answer is a fresh reading, so "
                "a repeated call is not free. Prefer where_can_i_run for "
                "placement questions: it asks the scheduler directly, and the "
                "advertised limits it contradicts are the reason this exists.",
        })
    if method == "ping":
        return _rpc_result(ident, {})
    if method == "tools/list":
        return _rpc_result(ident, {"tools": [t.schema() for t in TOOLS]})
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            return _rpc_error(ident, _INVALID_PARAMS, "missing tool name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _rpc_error(ident, _INVALID_PARAMS, "arguments must be an object")
        try:
            return _rpc_result(ident, call(name, arguments, throttle=throttle,
                                           extra=extra))
        except Exception as exc:  # noqa: BLE001 - never take the session down
            return _rpc_error(ident, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
    return _rpc_error(ident, _METHOD_NOT_FOUND, f"unknown method: {method}")


def _rpc_result(ident: object, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _rpc_error(ident: object, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def _messages(stream: TextIO) -> Iterator[dict[str, Any] | None]:
    """Decode the stream, one line at a time.

    ``None`` is yielded for a line that is not JSON, so the caller can answer
    with a parse error rather than dying on somebody's blank line or a client
    that writes a partial frame.  Blank lines are skipped outright: the
    transport is newline-delimited, so an empty one carries nothing, and
    replying to it with an error would be noise in both directions.
    """
    for line in stream:
        text = line.strip()
        if not text:
            continue
        try:
            decoded = json.loads(text)
        except ValueError:
            yield None
            continue
        yield decoded if isinstance(decoded, dict) else None


def serve(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    *,
    extra: Sequence[str] = (),
) -> int:
    """Read requests until the client closes the pipe.

    The stream swap is the important line.  ``sys.stdout`` is pointed at stderr
    for the duration, so every ``print`` in the package -- and any thread still
    holding a reference to it -- lands in the client's log instead of in the
    middle of a JSON-RPC frame.  The real descriptor is kept here and written to
    nowhere else.
    """
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    throttle = _Throttle()

    stashed = sys.stdout
    sys.stdout = sys.stderr
    try:
        for message in _messages(source):
            if message is None:
                reply: dict[str, Any] | None = _rpc_error(
                    None, _PARSE_ERROR, "invalid JSON")
            else:
                reply = handle(message, throttle=throttle, extra=extra)
            if reply is None:
                continue
            # One line, flushed: the client is blocked on this read, and a
            # buffered reply is indistinguishable from a hung server.
            sink.write(json.dumps(reply, default=str) + "\n")
            sink.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        # The client went away. That is how an MCP session ends.
        return 0
    finally:
        sys.stdout = stashed
    return 0
