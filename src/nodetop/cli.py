"""Command-line interface.

Sub-commands map to the questions you actually ask, in the order you ask them:
what is the state of this cluster, why can I not run, and where should I go
instead.  Every command works on every supported batch system; the vocabulary
adapts (``partition`` on Slurm, ``queue`` on PBS/LSF/SGE, ``namespace`` on
Kubernetes) but the reasoning does not.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime

from . import backends as backend_registry
from . import interactive
from ._version import VERSION
from .backends.base import Backend
from .core.cluster import Cluster
from .core.duration import (
    format_age,
    format_duration,
    format_wait,
    parse_timestamp,
)
from .core.fit import (
    Placement,
    ProbeBudget,
    probe_accounts,
    probe_queue,
    rank,
    unsettled,
)
from .core.hardware import supports
from .core.model import Allocation, JobShape, Queue, split_reason
from .exceptions import NoBackendError
from .hostlist import expand
from .render import (
    Glyphs,
    Style,
    badge,
    bar,
    colorize_help,
    columns,
    flow,
    gauge,
    heat_steps,
    kv,
    panel,
    plural,
    section,
    table,
    term_height,
    term_width,
    tree,
    truncate,
    width,
    wrap_indent,
)

__all__ = ["build_parser", "main"]


# --------------------------------------------------------------------------
# argument plumbing
# --------------------------------------------------------------------------
def _why_no_probe(cluster: Cluster) -> str:
    """Why entitlement cannot be confirmed, distinguishing all three reasons.

    "this batch system has no dry-run", "its client is not installed here" and
    "this is a recording" are three different facts with three different
    remedies, and asserting the first when one of the others is true is simply
    false.  This function used to handle the recording case only, so on a host
    without ``kubectl`` it announced that *Kubernetes* has no verify-only
    submission -- the one backend here whose server-side dry-run runs real
    admission, and which says so in its own capability notes.  Same conflation
    on a Slurm login node with no ``sbatch``, which is exactly what the previous
    version of this docstring called out as false.
    """
    if cluster.replayed:
        return (
            "this is a replayed snapshot: it holds the answers to the queries "
            "that were made, not to a dry-run nobody ran"
        )
    caps = cluster.capabilities
    if caps is None:
        # Unknown is not "none". Falling through to the sentence below asserted
        # that the batch system has no dry-run, which is the same overclaim
        # this function exists to avoid -- just reached by a different path.
        return f"{cluster.backend_name} capabilities could not be read"
    if caps.probe_supported:
        # The system can; this machine cannot ask it.
        client = (caps.probe_command or "").split() or [cluster.backend_name]
        return (
            f"{cluster.backend_name} can dry-run with {caps.probe_command}, but "
            f"{client[0]} is not on PATH here"
        )
    return f"{cluster.backend_name} has no verify-only submission to ask"


def _entitled_queues(
    cluster: Cluster, args: argparse.Namespace
) -> tuple[set[str] | None, int]:
    """Queue names whose declared allowlist admits one of your accounts.

    ``(None, 0)`` means "do not filter" -- ``--all``, or a cluster with no
    identity to filter against.

    One helper, used by every command that enumerates partitions or the
    resources inside them, because getting this right in one place and not the
    others is a mistake this file has made three times: the entitlement logic
    lived in :func:`~nodetop.core.fit.evaluate` and only ``where`` called it, so
    ``status`` ranked partitions the caller could not submit to, and ``queues``,
    ``nodes`` and ``accelerators`` reported cluster-wide totals as though they
    were the caller's. Measured here: 84 usable partitions, 19 on this user's
    allowlist; 607 nodes, 330 of them inside those 19; 358 accelerators, 230.

    Free, and with no false negatives on this cluster -- a 10-partition sample
    of what it drops comes back ``ACCOUNT_MISMATCH`` from a real dry-run. It is
    not *sufficient*, which is why the commands whose output is a submission
    decision go on to probe; see :func:`cmd_status`.
    """
    if getattr(args, "all", False) or cluster.identity is None:
        return None, 0
    names = [q.name for q in cluster.usable_queues()]
    if not names:
        return None, 0
    reach = rank(
        cluster, JobShape(nodes=1, cpus_per_task=1), queues=names,
        accounts=list(cluster.identity.accounts) or None,
        include_unusable=True,
    )
    keep = {p.queue for p in reach if not p.fatal_blockers}
    if not keep:
        # Filtering to nothing means the entitlement data is unusable -- no
        # association rows, a sparse table, a backend that cannot enumerate
        # them -- not that the caller may use nothing. A blank screen is a
        # worse failure than an over-full one, and it hides the very data that
        # would explain it.
        return None, 0
    return keep, len(names) - len(keep)


def _grid(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    aligns: Sequence[str] | None = None,
    style: Style | None = None,
    indent: str = "  ",
    limits: Sequence[int] | None = None,
    size: int | None = None,
    fit: bool = True,
    rule_above: bool = True,
    head_paint: Sequence[Callable[[str], str] | None] | None = None,
) -> str:
    """A table in the house style: one rule, quiet headers, no underline.

    Every command routes through here rather than calling :func:`table`
    directly, because the style is three keyword arguments deep and the six
    commands that were written before it existed each kept the old look --
    bold capitals, a rule above AND below the header row. Consistency by
    construction rather than by remembering.

    Header names are lowercased here too. A column name is a label, not an
    announcement, and lowercase is what let the column glossary go.
    """
    st = style or Style()
    heads = [h.lower() for h in headers]
    out = table(heads, rows, aligns, st, indent=indent, limits=limits,
                size=size, fit=fit, header_role="dim", underline=False,
                header_paint=head_paint)
    lines = out.splitlines()
    if not lines or not rule_above:
        return out
    rule_width = max(width(x) for x in lines) - len(indent)
    return "\n".join([indent + st.dim(st.g.h * rule_width), *lines])


def _reject_broken_snapshot(cluster: Cluster, command: str) -> int:
    """Refuse to report numbers the control plane never supplied.

    This is the failure mode the whole tool is written against, and it was
    reachable in the tool itself. With every Slurm command failing, four
    commands printed clean, confident, wrong answers and exited 0:

    * ``queues`` -- "0 shown, 0 usable", which reads as a cluster with no
      partitions rather than a cluster we could not ask;
    * ``nodes`` -- "all 0 - 0 with GPUs - 0 out";
    * ``health`` -- "0 schedulable - 0 degraded - **0 out**", i.e. a perfectly
      healthy cluster, from the one command whose entire purpose is to find out
      whether anything is wrong;
    * ``exclude --unschedulable`` -- an empty nodelist and exit 0, so
      ``sbatch --exclude=$(nodetop exclude --unschedulable)`` submits with no
      exclusions while the script believes it has them.

    An empty answer and an unobtainable one are different claims. Exit 3 --
    the same code as "no batch system here" -- because both mean the tool could
    not do its job, and neither should be confused with 1 ("nothing fits"),
    which is a real answer.

    A *partial* failure still returns 0: the report is usable, and the missing
    query is named on stderr so stdout stays pipeable.
    """
    if not cluster.errors:
        return 0
    fatal = not cluster.nodes
    # `status` renders these in its own panel, so printing them again here
    # would say the same thing twice on a terminal where both streams land.
    if fatal or command != "status":
        for name, why in cluster.errors.items():
            print(f"query failed: {name}: {truncate(why, 120)}", file=sys.stderr)
    if fatal:
        print(
            "no data: every query failed, so there is nothing to report -- this "
            "is not an empty cluster",
            file=sys.stderr,
        )
        return 3
    return 0


def _reject_unknown_queues(cluster: Cluster, args: argparse.Namespace,
                           st: Style) -> int:
    """Fail loudly when ``-q`` names something the cluster does not have.

    A mistyped queue name used to be indistinguishable from an empty one.
    ``nodetop queues -q caslakke`` printed *nothing at all* and exited 0;
    ``nodes -q caslakke`` said "0 of 607 nodes" as though the partition existed
    and happened to be empty. Both are the failure mode this whole tool is
    written against -- a confident answer to a question that was never really
    asked -- and both were reachable by one dropped keystroke.

    Checked once here rather than in each of the five commands that take ``-q``,
    because five copies of a validation is four chances to omit it.
    """
    wanted = [x.strip() for x in getattr(args, "queue", "").split(",") if x.strip()]
    unknown = [w for w in wanted if w not in cluster.queues]
    if not unknown:
        return 0
    term = cluster.queue_term
    for name in unknown:
        # A near miss is worth naming: with 87 partitions the answer to "which
        # did I mean" is usually one edit away, and printing the whole list
        # instead is how a helpful error becomes an unreadable one.
        close = difflib.get_close_matches(name, list(cluster.queues), n=3, cutoff=0.6)
        hint = f" -- did you mean {', '.join(close)}?" if close else ""
        print(f"no such {term}: {name}{hint}", file=sys.stderr)
    if not any(difflib.get_close_matches(n, list(cluster.queues), n=1, cutoff=0.6)
               for n in unknown):
        print(f"run 'nodetop {term}s' to list them.", file=sys.stderr)
    return 2


#: Columns produced by :func:`_node_rows`, and their alignment.
#: The first column is the selection cursor and the second the status glyph.
#: Two separate columns on purpose: the cursor is written into column 0 by
#: `_browse`, and it can only do that safely if column 0 is a plain space. When
#: it wrote over the *glyph* column instead it was overwriting the first byte of
#: that cell's colour escape, and the row rendered as `❯[38;5;111m◐  node...`.
#: One `<resource> free` per column, and the meter unlabelled beside the number
#: it draws -- the same shape as the overview's table.
#:
#: It was `cpu | free | mem free | gpu`, with `cpu` over the meter and a bare
#: `free` over the fraction beside it: "what does 'free' mean here? and then
#: after that, you have 'mem free'. why so many frees?" The word was doing the
#: work of three different labels, and none of the three said which resource it
#: belonged to.
NODE_HEADS = ["", "", "node", "state", "cpu free", "", "mem free", "gpu free",
              "reason"]
NODE_ALIGNS = ["left", "left", "left", "left", "right", "left", "right", "left",
               "left"]
NODE_LIMITS = [0, 0, 24, 18, 0, 0, 0, 0, 34]


def _node_rows(nodes: Sequence, st: Style) -> list[list[object]]:
    """One table row per node, ranked-heat included.

    Extracted so `nodes` and `zoom` are the same table by construction rather
    than by imitation. Two renderers of the same thing drift -- this file has
    the scars, which is why `_grid` exists -- and a zoom view whose columns
    disagree with the listing it zooms out to is worse than no zoom view.
    """
    heat = dict(zip([n.name for n in nodes],
                    heat_steps([n.effective_free_cpus for n in nodes]),
                    strict=False))
    any_gpu = any(n.is_gpu_node for n in nodes)
    rows: list[list[object]] = []
    for n in nodes:
        if not n.schedulable:
            mark, state = st.dim(st.g.off), st.dim(n.state_raw)
        elif n.degraded:
            mark, state = st.warn(st.g.warn), st.warn(n.state_raw)
        elif not n.has_room:
            # Before `idle`, deliberately. A node with nothing running on it and
            # no allocatable memory left is idle by the core count and can host
            # nothing, and the green "free" mark is the one thing on the row a
            # reader trusts without checking the numbers beside it.
            mark, state = st.info(st.g.partial), n.state_raw
        elif n.idle:
            mark, state = st.ok(st.g.ok), n.state_raw
        else:
            mark, state = st.info(st.g.partial), n.state_raw

        # A dash, not the `·` used everywhere else for an empty cell. This
        # column says how many accelerators are free, and a node with none
        # installed is not answering that question at all -- "the . in the gpu
        # column is a very confusing thing ... putting a dot there means
        # nothing." A dash reads as not-applicable in any terminal, with colour
        # off, in ASCII.
        #
        # Empty rather than a dash when NOTHING in this listing has an
        # accelerator: `table` drops a column no row fills, so a CPU-only
        # partition spends no width at all on a question that does not arise.
        accel = st.dim(st.g.dash) if any_gpu else ""
        if n.is_gpu_node:
            if n.accelerator:
                # The MODEL, and not its memory.
                #
                # Accelerator memory is a property of the model, not of the
                # node, so it was the same string repeated down fifty-six rows
                # -- and for a part that ships in more than one size it carried
                # a `?` that this table has no room to explain: "why is there a
                # ton of 40G? why do they have '?'? that's very confusing."
                # `nodetop gpus` prints it once per model, which is once, with
                # the inference marked. Same argument that moved `maxtime` out
                # of the partition table.
                accel = (
                    st.heat(str(n.gpus_free),
                            n.gpus_free / n.gpus_total if n.gpus_total else 0)
                    + st.muted(f"/{n.gpus_total} ")
                    + st.accent(n.accelerator.model)
                )
            else:
                accel = f"{n.gpus_free}/{n.gpus_total} {st.warn('UNKNOWN')}"
        # A meter on CPU availability: the one resource every node has, and
        # the last table in the tool without one.
        #
        # Drawn flat and dim when the node is unschedulable, because a drained
        # node still reports its full complement free and a bright full-length
        # bar beside "8/8" reads as the roomiest row on the screen. It is
        # phantom capacity -- the mark and the state say so, but a meter shouts
        # louder than a glyph. The numbers are still shown: they are what the
        # scheduler claims, and hiding them would be its own kind of lie.
        # `effective_free_cpus`, not `cpus_free`: the same argument as the
        # unschedulable case below. A node with 44 of 48 cores idle and no
        # allocatable memory left drew a nearly-full bright meter beside
        # `0/180G`, which reads as the roomiest row on the screen and is the
        # single most misleading thing this table could say. The claimed core
        # count stays in the column beside it; the meter measures room.
        cpu_share = ((n.effective_free_cpus / n.cpus_total)
                     if n.cpus_total else 0.0)
        if n.schedulable:
            meter = bar(cpu_share, 8, st, step=heat[n.name])
            cpus = st.tint(str(n.cpus_free), heat[n.name])
            mem = st.heat(str(n.memory_free_mb // 1024),
                          (n.memory_free_mb / n.memory_mb) if n.memory_mb else 0)
        else:
            # Empty, not dim-but-full. Dimming is a colour, and with colour off
            # a full-length bar beside "8/8" still reads as the roomiest row on
            # the screen. The meter measures *room*, and an unschedulable node
            # has none -- which is exactly what `Queue.effective_free_cpus`
            # already says. The claimed counts stay in the column beside it.
            meter = bar(0.0, 8, st, role="dim")
            cpus = st.dim(str(n.cpus_free))
            mem = st.dim(str(n.memory_free_mb // 1024))
        rows.append([
            " ", mark, n.name, state,
            # The number, then the meter that draws it: same order as the
            # overview's table, so one habit reads both.
            cpus + st.muted(f"/{n.cpus_total}"),
            meter,
            mem + st.muted(f"/{n.memory_mb // 1024}G"),
            accel,
            st.dim(n.reason) if n.reason else "",
        ])
    return rows


#: `64G`, `64GB`, `64Gi`, `65536M`, `2T`, or a bare number meaning GiB.
#:
#: `Gi` is in there because this tool speaks Kubernetes too, and that is how
#: Kubernetes writes it. Both notations mean the same binary multiple, so
#: accepting each costs nothing and refusing one is a papercut for whichever
#: half of the audience is used to it.
_MEMORY = re.compile(
    r"^\s*(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]?)(?:iB?|B)?\s*$",
    re.IGNORECASE)
_MEMORY_SCALE = {"": 1.0, "K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1.0,
                 "T": 1024.0, "P": 1024.0 * 1024}


def memory_gb(text: str) -> float:
    """Parse a memory size to GiB, accepting the scheduler's own spellings.

    ``--mem 64G`` is what a Slurm user types, because it is what ``sbatch``
    takes. This argument used to be a bare ``float`` in GiB, so the natural
    thing to type was an argparse error -- a tool whose whole premise is
    scheduler fluency rejecting the scheduler's own notation.

    Bare numbers still mean GiB, so nothing that worked before changes meaning.
    Suffixes are binary (``K``/``M``/``G``/``T``/``P``, optional trailing ``B``),
    matching ``sbatch``: there is no second convention to guess between.
    """
    m = _MEMORY.match(text or "")
    if not m:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a memory size -- try 64, 64G, 64GB or 65536M")
    return float(m.group("n")) * _MEMORY_SCALE[m.group("unit").upper()]


def _window(data: list[str], index: int, st: Style,
            reserved: int = 6) -> list[str]:
    """``data`` trimmed to what fits, with a position line when it does not.

    **"14 above" was the wrong thing to say.** It answers a question nobody
    asked and raises one it does not answer -- "why can't we see the 14 above?
    what does that mean?" -- because a count of hidden rows describes the
    absence rather than where you are. `15-29 of 44` says which slice is on
    screen and implies both that more exists and that moving will reach it.

    One helper because the same slice arithmetic had been written out three
    times, once per level, and three copies of a scroll calculation is three
    chances for one of them to be off by one.
    """
    # Clamped, not abandoned. This used to give up when fewer than three rows
    # would fit and return the whole list -- which is the one case where doing
    # so is destructive: the redraw moves the cursor up by the height of the
    # last frame, so a 67-line frame on a 10-line terminal clamps at the top of
    # the screen, the clear-to-end lands in the wrong place, and every keypress
    # leaves another copy behind. One visible row and a position line is a
    # usable screen; forty are not, and nothing is hidden silently either way.
    #
    # **One spare line beyond the chrome, and it is load-bearing.** `reserved`
    # is what the caller draws around these rows, so without the spare the
    # frame comes out *exactly* as tall as the terminal -- and a frame that
    # tall does not fit: the last line's newline scrolls the screen by one, the
    # repaint's cursor-up lands one line low, and each keypress orphans the top
    # border. That is the stack of `╭────╮` lines a node listing used to grow,
    # one per repaint, at every terminal size. Subtracted here rather than
    # added to each caller's `reserved`, so no caller has to remember it.
    room = max(1, term_height() - reserved)
    if len(data) <= room:
        return data
    lo = max(0, min(index - room // 2, len(data) - room))
    shown = data[lo:lo + room]
    return [*shown, st.dim(f"{lo + 1}-{lo + len(shown)} of {len(data)}")]


def _note(text: str, st: Style, indent: str = "  ") -> str:
    """Dim explanatory prose, wrapped to the window.

    Every long sentence in the output goes through here. Printing prose raw is
    how a line ends up wider than the terminal, and a soft-wrapped explanation
    loses its indent and reads as new content.
    """
    return st.dim(wrap_indent(text, indent=indent))


def _add_global_args(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Register the global flags so they work on either side of the verb.

    Each call creates *fresh* action objects on purpose.  Sharing them via
    ``parents=[...]`` looks tidier and is a trap: ``set_defaults`` mutates the
    action's own ``default``, so giving the root a default also rewrites it in
    every sub-parser that shares the object -- and because a sub-parser parses
    into its own namespace and then copies every attribute over the parent's,
    that default lands on top of a value the root already parsed.  The symptom
    is ``nodetop --json status`` silently ignoring ``--json``.
    """
    default = argparse.SUPPRESS if suppress else False
    parser.add_argument("--json", action="store_true", default=default,
                        help="machine-readable output")
    parser.add_argument("--no-color", action="store_true", default=default,
                        help="disable colour even on a terminal that supports it")
    parser.add_argument(
        "--ascii", action="store_true", default=default,
        help="draw with ASCII only, for a terminal that cannot render UTF-8",
    )
    parser.add_argument(
        "--backend", default=argparse.SUPPRESS if suppress else None,
        metavar="NAME",
        help="force a batch system instead of autodetecting: "
             + ", ".join(backend_registry.names()),
    )
    parser.add_argument(
        "--replay", default=argparse.SUPPRESS if suppress else None,
        metavar="FILE",
        help="analyse a snapshot taken earlier instead of the live cluster",
    )


def _add_queue_selector(
    parser: argparse.ArgumentParser, term: str = "queue", what: str = "limit to these"
) -> None:
    """The queue selector, spelled the same way everywhere.

    Both vocabularies are accepted on every command so a Slurm user typing
    ``-p`` and a PBS user typing ``-q`` both work, whatever the cluster is.
    """
    parser.add_argument(
        "-q", "--queue", "-p", "--partition", default="", dest="queue",
        metavar=term.upper(),
        help=f"{what} (comma-separated; -p/--partition also accepted)",
    )


def _add_shape_args(p: argparse.ArgumentParser, *, dry_run_only: bool = False) -> None:
    """Register the job-shape flags.

    ``dry_run_only`` is for ``check``, which asks the control plane and nothing
    else.  Two of these flags describe hardware that no batch system can
    express, so ``check`` cannot act on them at all -- and saying they are
    "checked against the hardware model" there would promise a validation that
    does not happen.
    """
    g = p.add_argument_group("job shape")
    g.add_argument("-N", "--nodes", type=int, default=1, metavar="N",
                   help="nodes the job needs (default: 1)")
    g.add_argument("-g", "--gpus", type=int, default=0, metavar="N",
                   help="accelerators per node")
    g.add_argument("-c", "--cpus", type=int, default=1, metavar="N",
                   help="CPUs per task")
    g.add_argument("--tasks-per-node", type=int, default=1, metavar="N",
                   help="tasks per node (default: 1)")
    g.add_argument("--mem", type=memory_gb, default=0.0, metavar="SIZE",
                   help="host memory per node: 64, 64G, 64GB or 65536M")
    g.add_argument("-t", "--time", default="01:00:00", metavar="WALLTIME",
                   help="walltime: 4:00:00, 2-00:00:00, 90m, 36h "
                        "(a bare number is minutes)")
    if dry_run_only:
        gpu_mem_help = (
            "minimum per-GPU memory -- ACCEPTED BUT NOT CHECKED here, "
            "because no scheduler can express it; listed under 'not covered' "
            "and evaluated by 'nodetop where'"
        )
        needs_help = (
            "capabilities such as bf16,fp8 -- ACCEPTED BUT NOT CHECKED here, "
            "for the same reason; use 'nodetop where'"
        )
    else:
        gpu_mem_help = (
            "minimum per-GPU memory; no scheduler can express this, so "
            "it is checked against the hardware model"
        )
        needs_help = "comma-separated capabilities: bf16,fp8,tf32,flash"
    g.add_argument("--gpu-mem", type=memory_gb, default=0.0, metavar="SIZE",
                   help=gpu_mem_help)
    g.add_argument("--needs", default="", metavar="CAPS", help=needs_help)
    g.add_argument("--exclude", default="", metavar="NODES",
                   help="nodes to keep out of consideration")
    g.add_argument("--tolerates", default="", metavar="TAINTS",
                   help="node restrictions the job tolerates (Kubernetes taints)")


def _shape_from_args(a: argparse.Namespace) -> JobShape:
    return JobShape(
        nodes=a.nodes,
        gpus_per_node=a.gpus,
        cpus_per_task=a.cpus,
        tasks_per_node=a.tasks_per_node,
        memory_gb=a.mem,
        walltime=a.time,
        gpu_memory_gb=a.gpu_mem,
        requires=tuple(x.strip() for x in a.needs.split(",") if x.strip()),
        exclude=tuple(expand(a.exclude)),
        tolerates=tuple(x.strip() for x in a.tolerates.split(",") if x.strip()),
        account=getattr(a, "account", None),
    )


def _help_style() -> Style:
    """Colour for the help screen, resolved without a parsed namespace.

    ``-h`` fires *during* parsing, so there is no namespace to read yet and
    ``--no-color`` has to come off ``sys.argv`` directly. Everything else --
    ``NO_COLOR``, a redirected stdout, ``TERM=dumb`` -- Style already handles.
    """
    argv = sys.argv[1:]
    return Style(
        enabled=False if "--no-color" in argv else None,
        glyphs=Glyphs.ascii() if "--ascii" in argv else None,
    )


class _Parser(argparse.ArgumentParser):
    """argparse, with the finished help block painted on the way out.

    Colour goes on *after* formatting, never on the strings argparse is handed:
    argparse lays its columns out with ``len()``, so an escape sequence counted
    as visible text throws every column off by its own length.

    Sub-parsers inherit this class automatically -- ``add_subparsers`` defaults
    ``parser_class`` to the type of the parser it is called on -- so every
    ``nodetop <verb> --help`` is painted too, without a per-verb opt-in to
    forget.
    """

    def format_help(self) -> str:
        return colorize_help(super().format_help(), _help_style())


EXAMPLES = """examples:
  nodetop                            where you can run something, right now
  nodetop where -g 4 --gpu-mem 40    rank the queues that fit this job
  nodetop nodes --gpu --free         GPU nodes with something free now
  nodetop zoom gn               look inside one queue, node by node
  nodetop queues -q test             every gate on one queue, in detail
  nodetop check -q gpu -A myaccount  ask the control plane directly
  nodetop health                     down, drained and silently degraded nodes

  A queue that looks fine is not a queue that will take your job. `where` and
  `check` ask the control plane; everything else reads what it advertises.
"""


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="nodetop",
        description="See what your cluster actually has free -- and why a queue "
                    "that looks fine will not take your job. Works on Slurm, "
                    "PBS, LSF, Grid Engine, Kubernetes, or a bare pool of "
                    "machines.",
        epilog=EXAMPLES,
        # Raw, so the examples keep their columns. The description above is the
        # only prose in the block and it is short enough to wrap by hand.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_global_args(p, suppress=False)
    p.add_argument("--version", action="version", version=f"nodetop {VERSION}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("status", help="cluster overview: what is usable right now")
    _add_global_args(s, suppress=True)
    s.add_argument("--declared", action="store_true",
                   help="skip the dry-run and trust the declared allowlists "
                        "(faster, but they over-report on many clusters)")
    s.add_argument("--all", action="store_true",
                   help="show every queue, including those with no free capacity")
    # Interactive is the DEFAULT on a terminal, so the flag is the opt-out.
    #
    # A flag to switch it ON meant advertising the flag, and a line of the
    # overview spent telling the reader that a key exists is a line nobody
    # reads. The highlight is its own affordance: a row in inverse video is
    # something you try the arrow keys on.
    #
    # The opt-out exists for a terminal that is not a person -- `watch nodetop`
    # allocates a pty and would otherwise block on a keystroke forever -- and it
    # is not mentioned anywhere except `--help`, which is where a flag belongs.
    s.add_argument("--static", action="store_true",
                   help="print the report and exit instead of taking keystrokes")

    q = sub.add_parser("queues", aliases=["partitions"],
                       help="per-queue state and blockers")
    _add_global_args(q, suppress=True)
    _add_queue_selector(q)
    q.add_argument("--unusable-only", action="store_true",
                   help="show only the queues that cannot start work")
    q.add_argument("--detail", action="store_true",
                   help="one block per queue instead of a table (implied by -q)")
    q.add_argument("--all", action="store_true",
                   help="every queue, not only the ones you may use")

    z = sub.add_parser("zoom", aliases=["in"],
                       help=f"open one {'partition'} up: its gates, then its nodes")
    _add_global_args(z, suppress=True)
    # Stored as `queue` on purpose: the unknown-name guard in main() reads that
    # attribute, so naming it anything else would silently opt this command out
    # of the check every other -q command gets.
    z.add_argument("queue", metavar="NAME",
                   help="the queue to look inside (comma-separated for several)")
    z.add_argument("--gpu", action="store_true", help="GPU nodes only")
    z.add_argument("--cpu", action="store_true", help="CPU-only nodes")
    z.add_argument("--free", action="store_true",
                   help="only nodes with something free")
    z.add_argument("-n", "--top", type=int, default=20, metavar="N",
                   help="how many nodes to list (default: 20)")
    z.add_argument("--all", action="store_true", help="every node, however many")

    nd = sub.add_parser("nodes", help="node inventory and hardware")
    _add_global_args(nd, suppress=True)
    _add_queue_selector(nd, what="list only nodes serving these")
    nd.add_argument("--gpu", action="store_true", help="GPU nodes only")
    nd.add_argument("--cpu", action="store_true", help="CPU-only nodes")
    nd.add_argument("--free", action="store_true", help="only nodes with something free")
    nd.add_argument("-n", "--top", type=int, default=20, metavar="N",
                    help="how many rows to list (default: 20; --all for every one)")
    nd.add_argument("--all", action="store_true",
                    help="every matching node, however many")

    _add_global_args(
        sub.add_parser("health", help="down, drained and silently degraded nodes"),
        suppress=True,
    )

    w = sub.add_parser("where", aliases=["fit"],
                       help="rank the queues this job could run in")
    _add_global_args(w, suppress=True)
    _add_shape_args(w)
    _add_queue_selector(w, what="consider only these")
    w.add_argument("-A", "--account", default=None, metavar="NAME",
                   help="submit as this account instead of trying yours")
    w.add_argument("--declared", action="store_true",
                   help="skip the dry-run and trust the declared allowlists "
                        "(faster, but they over-report on many clusters)")
    w.add_argument("--accounts", default="", metavar="NAMES",
                   help="comma-separated accounts to try when checking "
                        "(default: all of yours)")
    w.add_argument("--all", action="store_true", help="show ruled-out queues too")

    c = sub.add_parser("check", aliases=["probe"],
                       help="ask the control plane directly, per queue and account")
    _add_global_args(c, suppress=True)
    _add_shape_args(c, dry_run_only=True)
    _add_queue_selector(c, what="ask about only these")
    c.add_argument("-A", "--account", default=None, metavar="NAME",
                   help="ask as this account instead of trying yours")
    c.add_argument("--accounts", default="", metavar="NAMES",
                   help="comma-separated accounts to try (default: all of yours)")

    ex = sub.add_parser("exclude", help="emit an exclusion node list")
    _add_global_args(ex, suppress=True)
    ex.add_argument("--gpu-nodes", action="store_true",
                    help="every GPU node, to keep CPU work off them")
    ex.add_argument("--unschedulable", action="store_true",
                    help="every down or drained node")
    ex.add_argument("--degraded", action="store_true",
                    help="schedulable nodes that look impaired")
    _add_queue_selector(ex, what="draw only from these")

    # "gpus" is the name people reach for; the other two are kept because the
    # neutral term is what the core and the JSON call it.
    ac = sub.add_parser("accelerators", aliases=["accel", "gpus"],
                        help="GPU inventory and what each model can do")
    _add_global_args(ac, suppress=True)
    _add_queue_selector(ac, what="limit the inventory to these")
    ac.add_argument("--all", action="store_true",
                    help="every accelerator on the cluster, not only the ones "
                         "in partitions you may use")

    b = sub.add_parser("backends", help="which batch systems are usable here")
    _add_global_args(b, suppress=True)

    sn = sub.add_parser("snapshot",
                        help="record this cluster's state for later analysis")
    _add_global_args(sn, suppress=True)
    sn.add_argument("-o", "--output", default="-", metavar="FILE",
                    help="file to write, or - for stdout (default: -)")
    return p


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_backends(cluster: Cluster | None, args: argparse.Namespace, st: Style) -> int:
    detected = set(backend_registry.available())
    if args.json:
        print(json.dumps({
            "detected": sorted(detected),
            "backends": {
                n: {
                    "detected": n in detected,
                    # Two separate facts: what the batch system can do, and
                    # whether it can be done from this host.
                    "dry_run_supported": bool(
                        (c := _caps(n)) and c.probe_supported),
                    "can_confirm_entitlement_here": bool(
                        (c := _caps(n)) and c.probe),
                    "probe_command": (c.probe_command if (c := _caps(n)) else ""),
                    "queue_term": backend_registry.get(n).queue_term,
                    "notes": list(c.notes) if (c := _caps(n)) else [],
                }
                for n in backend_registry.names()
            },
        }, indent=2))
        return 0

    rows = []
    for name in backend_registry.names():
        caps = _caps(name)
        here = name in detected
        backend = backend_registry.get(name)
        # Column 4 is the *system's* capability; column 1 already says whether
        # it is usable here.  Reading the capability off `probe` made a Slurm
        # login node report that SGE has no dry-run, when the truth was only
        # that qsub was not installed.
        if caps and caps.probe_supported:
            mechanism = (st.ok if caps.probe else st.dim)(
                caps.probe_command or "yes")
        else:
            mechanism = st.warn("none")
        rows.append([
            st.ok(st.g.ok) if here else st.dim(st.g.off),
            st.head(name) if here else st.dim(name),
            backend.queue_term,
            mechanism,
        ])
    print(panel([
        f"{st.ok(str(len(detected)))} of {len(backend_registry.names())} usable here"
        f"   {st.dim(st.g.sep)}   "
        f"{st.dim('--backend NAME to force one')}",
    ], "batch systems", st))
    print()
    print(_grid(["", "backend", "calls it", "dry-run the system offers"],
                rows, style=st))
    print()
    print(flow([
        f"{st.g.ok} usable here",
        f"{st.g.off} client not installed",
        f"{st.warn('none')} {st.dim('= no dry-run exists')}",
    ], st))
    print(_note(
        "Where no dry-run exists, entitlement can only be read from a queue's "
        "declared ACL, and nodetop labels it unconfirmed rather than verified.",
        st))
    return 0


def _caps(name: str):
    try:
        return backend_registry.get(name).capabilities()
    except Exception:
        return None


#: Rows the overview shows before deferring to --all. Enough to answer "where
#: can I go" without turning the dashboard into a listing.
STATUS_ROWS = 12


def cmd_status(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    """The overview: one box, no prose.

    Three rounds of feedback on this view, each blunter than the last, and all
    of it the same complaint: too much text. It had a three-meter block, a
    sentence-long note under every heading, a legend defining each column, a
    warning written as prose, and a footer suggesting other commands. None of
    that is the answer to "where can I run this", and none of it gets read.

    What is left is the answer: partitions you can submit to, ranked, with the
    group-owned ones and the dead ones tagged in a word. Sub-headings are one
    word in capitals; column names are lowercase and dim; nothing is explained.
    If a column needs a legend it needs a better name.
    """
    # `--json` is emitted from the SAME population as the panel, further down,
    # not from `cluster.summary()` here.
    #
    # It used to return early with the cluster digest, which answers a
    # different question than the text does: the panel said "222 of 358 GPUs,
    # 53 free" -- your partitions -- while the JSON said 358 total and 126
    # free, counting accelerators in partitions this account cannot submit to,
    # and carried neither the funnel nor a single partition row. The README
    # promises `--json` carries what the text does. Paying for the same
    # dry-runs the text pays for is the price of that being true; `--declared`
    # skips them for both forms alike.
    summary = cluster.summary()

    def status_json(
        yours: dict[str, int] | None = None,
        funnel: dict[str, int] | None = None,
        listed: Sequence[dict[str, object]] = (),
        excluded: Sequence[tuple[str, str]] = (),
    ) -> int:
        """One schema, whichever path reaches it.

        The degenerate paths -- no nodes at all, a control plane that answered
        nothing -- return before the population is known, and a `--json` that
        prints nothing there is worse than one that prints zeros: the caller
        cannot tell "no nodes" from "the command crashed".
        """
        print(json.dumps({
            **summary,
            "yours": yours or {
                "nodes": 0, "nodes_schedulable": 0, "accelerators_total": 0,
                "accelerators_free": 0, "cpus_total": 0,
                "cpus_free_advertised": 0, "effective_free_cpus": 0,
            },
            "funnel": funnel or {
                "total": len(cluster.queues), "shown": 0, "unconfirmed": 0,
            },
            "listed": list(listed),
            "excluded": [{"name": name, "reason": why} for name, why in excluded],
        }, indent=2, default=str))
        return 0

    term = cluster.queue_term
    nodes = len(cluster.nodes)
    gpu_total = int(str(summary["accelerators_total"]))
    usable = cluster.usable_queues()
    ident = cluster.identity
    body: list[str] = []

    # Built AFTER the filtering, then inserted at the top, so the numbers up
    # here describe the same population as the table below. They used to be
    # cluster-wide while the table was your slice, so the header said
    # "358 GPUs, 117 free" above five partitions holding 230 of them -- the
    # same mixing of two populations that made `gpus` report a total nobody
    # could use.
    # The tool and backend lead the facts line rather than sitting on a line of
    # their own: now that the title is content instead of part of the border, a
    # separate line for two words is a line spent on nothing.
    facts = [f"{st.head('nodetop')} {st.dim(st.g.sep)} "
             f"{st.accent(cluster.backend_name)}"]
    if ident is not None and ident.user:
        facts.append(st.head(ident.user))
    # Nodes first, GPUs after. Only 91 of this cluster's 607 nodes have an
    # accelerator, so leading with a GPU fraction frames the whole cluster by a
    # resource 85% of it does not have.
    # "4/32 GPUs" has the same which-side-is-free ambiguity the table columns
    # were split to remove, so the label says it.

    # The partition count moves out of this line and into the funnel below,
    # beside the numbers that explain it. On its own up here it was the source
    # of the only question this view has ever been asked twice: "it says 87
    # partitions -- why am I looking at five rows?"
    if not cluster.can_probe:
        facts.append(st.warn("no dry-run"))
    header_at = len(body)
    body.append("")  # placeholder; filled once the population is known

    if cluster.errors:
        for name, why in cluster.errors.items():
            body.append(f"{st.bad('FAILED')} {st.dim(name)}  {truncate(why, 60)}")

    if not cluster.nodes:
        if args.json:
            return status_json()
        body.append(st.warn(
            "no nodes -- wrong backend, or the control plane is down"
        ) + st.dim("  nodetop backends"))
        print(panel(body, "", st))
        return 0

    # Every partition that owns nodes -- NOT only the ones with room right now.
    #
    # This used to drop anything currently full, before the access filter ran,
    # and that understates access rather than capacity: `amd` and `build` are
    # partitions this account can submit to, and they were hidden because they
    # happened to be busy at the instant of the query. "Where can I run this"
    # includes "where can I queue", and 5 places you may use of which 2 are
    # full is a different and truer answer than 3 places.
    with_room = [q for q in usable if q.nodes]
    # Counted, not just filtered. Every partition dropped from here on has to
    # be accounted for by name in the funnel line below, or the funnel is just
    # another unexplained number: total = shown + refused + no-access + empty +
    # dead, exactly.
    no_nodes = len(usable) - len(with_room)

    # (index into `body`, kind, payload) for every row the cursor can land on.
    #
    # `kind` exists because not every selectable row is a partition. The funnel
    # line is one too: "65 no access" is a fact the reader can check, and the
    # goal is that anything on screen can be opened.
    selectable: list[tuple[int, str, str]] = []
    # (queue name, why it is not in the table) for everything the funnel counts.
    excluded: list[tuple[str, str]] = []

    # Filter by entitlement FIRST, before anything is shown.
    #
    # This is not a new feature -- `evaluate` already derives
    # ACCOUNT_NOT_ALLOWED / GROUP_NOT_ALLOWED / QOS_NOT_ALLOWED from each
    # queue's own allowlist, and `where` has always used it. `status` did not:
    # it filtered on *structural* usability only, so a partition that names
    # four accounts none of which are yours was listed as somewhere with room.
    # The allowlist-width heuristic that was supposed to catch those was
    # arbitrary and wrong -- it flagged queues naming <=2 accounts, so
    # `jfkfloor2` (4 accounts) and `voltron` (5) sailed through, while a plain
    # set intersection excludes both instantly and needs no dry-run.
    not_entitled = 0
    if not args.all:
        shape = JobShape(nodes=1, cpus_per_task=1)
        accounts = list(ident.accounts) if ident else None
        reach = {
            p.queue: p for p in rank(
                cluster, shape, queues=[q.name for q in with_room],
                accounts=accounts, include_unusable=True,
            )
        }
        before = len(with_room)
        reach_source = list(with_room)
        with_room = [
            q for q in with_room
            if (p := reach.get(q.name)) is None or not p.fatal_blockers
        ]
        not_entitled = before - len(with_room)
        kept_names = {q.name for q in with_room}
        excluded += [(q.name, "no access") for q in reach_source
                     if q.name not in kept_names]
    # The population every other listing reports as "yours", kept before the
    # dry-run narrows it further. See the header below for why the two differ.
    entitled_with_room = list(with_room)

    # Probing is the DEFAULT, not a flag.
    #
    # Measured on this cluster: the allowlist filter above takes 84 partitions
    # with room down to 19 and has no false negatives -- every one of an
    # 18-partition sample it drops is confirmed to accept none of this user's 34
    # accounts. But of the 19 it keeps, a dry-run accepts 8. The association
    # table lists this user in `grp-e`, `pi-okafor`, `pi-tanaka`, `pi-varga`,
    # `pi-ibrahim` and `pi-svensson`, and the submit plugin rejects every one of
    # them with "Invalid membership". No reading of any declared list can see
    # that, so an ACL-only answer is wrong eleven times out of nineteen here.
    # It is cheap precisely because the ACL filter runs first: nineteen queues
    # to ask about, not eighty-four -- and roughly one dry-run each, because
    # ProbeBudget learns which account the control plane accepts on the first
    # cheap queue and tries it first everywhere after.
    refused = 0
    unsettled = 0
    if not args.declared and not args.all and cluster.can_probe:
        # Pretest. A declared allowlist cannot answer this on a cluster whose
        # associations are templated -- the accounting database claims the same
        # entitlements for every account -- so the only way to show *only* the
        # partitions that will take your job is to ask the control plane about
        # each one. Read-only, and narrowed per queue to the accounts its own
        # allowlist admits, so this is a handful of calls rather than hundreds.
        accounts = list(cluster.identity.accounts) if cluster.identity else None
        verdicts = {
            p.queue: p for p in rank(
                cluster, JobShape(nodes=1, cpus_per_task=1),
                queues=[q.name for q in with_room],
                use_probe=True, accounts=accounts, include_unusable=True,
            )
        }
        # Dropped only on a DURABLE refusal. A verdict that could not settle
        # the question -- control plane down, or every account we asked about
        # refused while others went unasked -- keeps the partition on screen,
        # because "we do not know" is not "no" and this view exists to stop
        # exactly that conflation. Counted apart from the refusals so the
        # funnel does not present a guess as a finding.
        kept, refused_names, unsettled_names = [], [], []
        for q in with_room:
            p = verdicts.get(q.name)
            v = p.verdict if p is not None else None
            if p is not None and p.confirmed:
                kept.append(q)
            elif v is not None and not v.allowed and v.durable:
                refused_names.append(q.name)
            else:
                kept.append(q)
                unsettled_names.append(q.name)
        with_room = kept
        refused = len(refused_names)
        unsettled = len(unsettled_names)
        excluded += [(name, "refused") for name in refused_names]


    def order(q):
        # Free CORES first -- not free nodes, and still not accelerators.
        #
        # Nodes remain the spine of this view: ranking by accelerators sorted a
        # 607-node cluster by a property 85% of it lacks, and that stays fixed.
        # The defect was the denominator. `effective_free_nodes` counts only
        # *wholly* idle nodes, and on a busy cluster almost nothing is wholly
        # idle, so any partition with a single job running reported zero room.
        # It ranked `gn-bigmem` (128 free cores) above `amd` (2825) and
        # drew `amd` as an empty meter. See Queue.effective_free_cpus.
        return (-q.effective_free_cpus, -q.effective_free_gpus, q.name)

    def rows_for(queues):
        ranked = sorted(queues, key=order)
        # Two ramps, one per magnitude column, each ranked within this table.
        # The ramp means one thing in both places -- warmer is more of that
        # resource than the other rows have -- so a warm row is where the room
        # is, at a glance, without reading a single number.
        #
        # Tone and bar length now measure the SAME quantity, on purpose. They
        # used to differ (length was the row's own free share, tone was its
        # free cores against the rest), and a row could then be long and cold
        # or short and warm, which read as two contradictory answers to one
        # question. One quantity, drawn twice, agrees with itself -- and how
        # full a partition is in its own terms is what `idle` and `zoom` say.
        core_heat = heat_steps([q.effective_free_cpus for q in ranked])
        gpu_heat = heat_steps([q.effective_free_gpus for q in ranked])
        # The bar measures the column the table is SORTED by, which it did not.
        #
        # It drew each partition's free *share* of its own capacity while the
        # rows were ordered by absolute free cores, so the most prominent thing
        # on the screen ran 27%, 41%, 28%, 45%, 100% down a list that was in
        # perfect order -- "why rank the partitions in this way ... it looks a
        # bit confusing". The eye reads a bar as the ordering, so the bar has to
        # be the ordering. Share is still there, as its own number.
        peak_free = max((q.effective_free_cpus for q in ranked), default=0)
        out = []
        for i, q in enumerate(ranked):
            cores = q.cpus_total
            free = q.effective_free_cpus
            out.append([
                # A one-character mark column, empty until something selects
                # this row. The interactive cursor is written into it in place,
                # which is why it is a column and not a prefix bolted on later:
                # a prefix would shift the table under the highlight.
                " ",
                # The name leads. Seven numeric columns came first before, so
                # a row had to be read to its end before it said what it was
                # describing.
                st.tint(q.name, core_heat[i]),
                # `idle/total`, one column, same shape as the two beside it.
                # `idle` rather than `free` on purpose: a node counts here only
                # when it is WHOLLY free, which is a stricter claim than having
                # room -- this partition routinely shows 0 while carrying two
                # hundred free cores.
                (st.head(str(q.effective_free_nodes))
                 if q.effective_free_nodes else st.muted("0"))
                + st.muted(f"/{len(q.nodes)}"),
                # `free/total` in one column, under a header that names the
                # numerator.
                #
                # This was two columns, `cores` and `free`, on the reasoning
                # that a bare `4/4` cannot say which side is free -- and it
                # cannot: "4/4 100%" was once read as meaning all four are
                # BUSY. But splitting them put a second unqualified `free` on
                # the header row beside the accelerators' own, and one word
                # covering two resources is its own ambiguity: "why free
                # appears twice on the column title? what does it mean?" The
                # header carries the answer instead -- `cores free`, `gpu free`
                # -- which is what makes the same form readable in the node
                # table, and it costs two columns less width.
                (st.tint(str(free), core_heat[i]) + st.muted(f"/{cores}")
                 if cores else st.dim(st.g.dash)),
                bar(free / peak_free if peak_free else 0.0, 10, st,
                    step=core_heat[i]),
                (st.tint(str(q.effective_free_gpus), gpu_heat[i])
                 + st.muted(f"/{q.gpus_total}") if q.gpus_total
                 else st.dim(st.g.dash)),
                # Empty, not a dash: `gpu free` already said this partition
                # has none, and saying it twice on one row is noise.
                st.muted(", ".join(list(q.accelerator_models)[:2])),
            ])
        return out

    # Lowercase, dim: a column name should not shout, and "free gpu" needs no
    # legend the way "FREE" did. `maxtime` is gone -- it is a property of the
    # partition, not an answer to "where can I run this", and it lives one
    # command away in `queues -q NAME`.
    # One quantity per column, and the bar draws the column beside it.
    #
    # There used to be a `share` percentage as well -- free as a fraction of the
    # partition's own capacity -- next to a bar drawn from a different quantity,
    # and the question that kept coming back was "does the bar show share or
    # free?" It was unanswerable from the screen: two numbers for two different
    # notions of free, one of them also drawn. `share` is gone. `free` is the
    # count, the bar is that count against the largest on the list, and how full
    # a partition is in its own terms is what `zoom` is for.
    #
    # The sort marker went with it. `free ↓` was read as part of the value, not
    # as an ordering -- "what does free down arrow mean?" -- and the bar already
    # descends monotonically, which says the same thing without a glyph.
    heads = ["", term, "nodes idle", "cores free", "", "gpu free", "gpu model"]
    aligns = ["left", "left", "right", "right", "left", "right", "left"]
    # A hue per resource, so the eye groups the columns before it reads them:
    # the two accelerator columns are one colour, cores and the meter beside
    # them another, and the partition's own identity stays neutral. Drawn from
    # the same ramp every number in this table uses, at its cold and warm ends
    # -- a header in a colour the palette does not contain reads as decoration.
    head_paint = [None, st.head, st.muted,
                  lambda s: st.tint(s, 2), None,
                  lambda s: st.tint(s, 9), lambda s: st.tint(s, 9)]

    # The funnel. Every partition on the cluster is in exactly one of these
    # terms, so the line answers "why five rows" by arithmetic rather than by
    # asking the reader to trust it -- and it sits directly above the table it
    # accounts for, not up in the cluster facts where it read as trivia.
    # Now the population is settled: fill the header with it.
    #
    # Scoped to the partitions your ACCOUNTS are on -- the allowlist set -- and
    # deliberately not to the smaller set the dry-run confirmed. `nodes`,
    # `queues` and `gpus` all report the allowlist population, because a
    # listing command cannot afford a dry-run per queue, and this line was
    # reporting the probe-confirmed one: `status` said "220 of 358 GPUs" while
    # `gpus` said "230 of 358" from the same snapshot. Two numbers for one fact
    # is worse than either number, and "all four listings share the filter" is
    # a property this file has already had to fix three times.
    #
    # The narrower answer is not lost -- it is the funnel line immediately
    # below, which says how many of these partitions will actually take a job
    # and why the rest will not.
    mine = {n.name for q in entitled_with_room for n in q.nodes}
    my_nodes = [n for n in cluster.nodes if n.name in mine]
    my_up = sum(1 for n in my_nodes if n.schedulable)
    my_gpu = sum(n.gpus_total for n in my_nodes)
    # The same measure the table's `free` column uses, or the header and the
    # rows beneath it would disagree on one number.
    my_gpu_free = sum(n.effective_free_gpus for n in my_nodes if n.schedulable)
    scope = facts[:]
    if len(my_nodes) == nodes:
        scope.append(f"{plural(nodes, 'node')}, {my_up} up")
    else:
        scope.append(f"{len(my_nodes)} of {plural(nodes, 'node')}, {my_up} up")
    if gpu_total:
        seen = (f"{my_gpu} GPUs" if my_gpu == gpu_total
                else f"{my_gpu} of {gpu_total} GPUs")
        scope.append(st.muted(f"{seen}, {my_gpu_free} free"))
    body[header_at] = f"  {st.dim(st.g.sep)}  ".join(scope)

    dead_count = len(cluster.unusable_queues())
    # "down", not "dead": it is the word every other status line on a
    # cluster uses for a partition that can start nothing.
    excluded += [(q.name, "down") for q in cluster.unusable_queues()]
    excluded += [(q.name, "no nodes") for q in usable if not q.nodes]
    # Stable order, worst-explained last: the reader is looking for their own
    # partition, so the reason groups stay together and names sort inside them.
    order_of = {"no access": 0, "refused": 1, "no nodes": 2, "down": 3}
    excluded.sort(key=lambda e: (order_of.get(e[1], 9), e[0]))
    # The exclusion terms, each one its own selectable target.
    #
    # One entry per label rather than one for the whole line: "i want to hit
    # enter myself to each of the labels like no access and refused and dead".
    # They share a body row, so the cursor cannot distinguish them by position --
    # the selected term is the one drawn in the accent colour, and the row's
    # cursor column fills in as usual.
    terms: list[tuple[int, str]] = []
    if not args.all:
        for count, why in ((not_entitled, "no access"), (refused, "refused"),
                           (no_nodes, "no nodes"), (dead_count, "down")):
            if count:
                terms.append((count, why))
    elif dead_count:
        terms.append((dead_count, "down"))

    # The unsettled ones are *shown*, so they belong to the head term rather
    # than to a term of their own -- an exclusion count for a partition that is
    # on screen would break the one property this line has, which is that its
    # terms sum to the total.
    shown_note = f" ({unsettled} unconfirmed)" if unsettled else ""
    head_term = (f"{len(with_room)} open to you" if not args.all
                 else f"{len(with_room)} with nodes")

    if args.json:
        return status_json(
            # The header line: what of the cluster is yours.
            yours={
                "nodes": len(my_nodes),
                "nodes_schedulable": my_up,
                "accelerators_total": my_gpu,
                "accelerators_free": my_gpu_free,
                "cpus_total": sum(n.cpus_total for n in my_nodes),
                "cpus_free_advertised": sum(n.cpus_free for n in my_nodes
                                            if n.schedulable),
                "effective_free_cpus": sum(n.effective_free_cpus
                                           for n in my_nodes if n.schedulable),
            },
            # The funnel line: every partition in exactly one term, so these
            # sum to `queues` above exactly as the printed line does.
            funnel={
                "total": len(cluster.queues),
                "shown": len(with_room),
                "unconfirmed": unsettled,
                **{why.replace(" ", "_"): count for count, why in terms},
            },
            # The table. A fixed key, not `f"{term}s"`: on PBS and LSF the
            # queue term is "queue", and `summary` already has a `queues` key
            # holding the cluster's partition COUNT -- so an interpolated name
            # would have replaced a number with a list on half the backends.
            listed=[{
                "name": q.name,
                "dedicated": q.is_dedicated,
                "nodes": len(q.nodes),
                "nodes_idle": q.effective_free_nodes,
                "nodes_with_room": sum(1 for n in q.nodes if n.has_room),
                "cpus_total": q.cpus_total,
                "cpus_free_advertised": q.cpus_free,
                "effective_free_cpus": q.effective_free_cpus,
                "accelerators_total": q.gpus_total,
                "effective_free_accelerators": q.effective_free_gpus,
                "accelerator_models": q.accelerator_models,
            } for q in sorted(with_room, key=order)],
            # Why every other partition is not in that list, by name.
            excluded=excluded,
        )

    def render_funnel(selected: str | None = None) -> str:
        """The funnel line, with ``selected`` drawn as the active term.

        Three characters of lead, matching the table's cursor column plus its
        two separator spaces, so this line and the rows below it share one left
        edge and the cursor lands in the same place on both.
        """
        def mark(active: bool) -> str:
            """The cursor slot that every term on this line reserves.

            Reserved, not inserted. All of these terms share one row, so the
            selected one used to be shown by colour alone while the row cursor
            stayed pinned to the left margin -- pointing at the total, whatever
            was actually selected: "when pressing the right arrow, it doesn't
            land at ' 8 open to you'". The glyph moves instead.

            The slot is taken out of the separator's own padding rather than
            added to it, so the line is exactly as wide as it was before there
            was a cursor on it and nothing shifts as the selection moves.
            """
            return st.accent(st.g.cursor, bold=True) if active else " "

        # The total is a target like the rest, and the separator before the
        # next term is the same one used between all of them.
        #
        # It was `87 partitions  →  8 open to you`, with the total unselectable
        # and an arrow that carried the funnel's sense of narrowing. Neither
        # survived contact: "why can't we select the 87 partitions? also, why
        # there is a right arrow here? it makes no sense at all." Every term on
        # this line is now a peer -- a count you can open -- so they read as a
        # list and are punctuated as one.
        whole = plural(len(cluster.queues), term)
        head = (st.accent(head_term, bold=True) if selected == "open"
                else st.head(head_term)) + st.muted(shown_note)
        line = ("  " + mark(selected == "total")
                + (st.accent(whole, bold=True) if selected == "total"
                   else st.head(whole))
                + f"  {st.dim(st.g.sep)} " + mark(selected == "open") + head)
        for count, why in terms:
            text = f"{count} {why}"
            line += f"  {st.dim(st.g.sep)} " + mark(why == selected) + (
                st.accent(text, bold=True) if why == selected else st.muted(text))
        return line

    body.append("")
    body.append(render_funnel())
    funnel_at = len(body) - 1
    # The total leads, because it is the leftmost thing on the line: opening it
    # lists every partition on the cluster with the reason each is or is not in
    # the table below, which is the one view where the funnel's arithmetic can
    # be checked rather than trusted.
    selectable.append((funnel_at, "total", "total"))
    # The head term is selectable too: pressing Right from the total should land
    # on "8 open to you", which is a bucket like the others -- its members are
    # the rows below, so opening it moves the cursor into the table.
    selectable.append((funnel_at, "open", "open"))
    for _, why in terms:
        selectable.append((funnel_at, "excluded", why))

    def add_table(queues, cap, header=True):
        ranked = sorted(queues, key=order)
        rows = rows_for(queues)
        shown = rows if args.all else rows[:cap]
        out = table(heads, shown, aligns, st, indent="", size=200, fit=False,
                    header_role="dim", header_paint=head_paint,
                    show_header=header, underline=False).splitlines()
        if header:
            # One thin rule above the header, none beneath it: two separators
            # around one row of column names is a box drawn for its own sake.
            body.append(st.dim(st.g.h * max(width(x) for x in out)))
        # Where each data row landed in `body`, paired with the queue it draws.
        #
        # Recorded here rather than re-derived later because `table` is the only
        # thing that knows how many lines it emitted, and interactive selection
        # needs to paint one of THOSE lines -- not a second rendering of the same
        # row, which is how two views of one thing drift apart.
        first = len(body) + (1 if header else 0)
        for offset, queue in enumerate(ranked[:len(shown)]):
            selectable.append((first + offset, "queue", queue.name))
        body.extend(out)
        if len(shown) < len(rows):
            body.append(st.muted(f"+{len(rows) - len(shown)}")
                        + st.dim(f" more  {st.g.sep}  --all"))

    shared = [q for q in with_room if not q.is_dedicated]
    owned = [q for q in with_room if q.is_dedicated]
    # The names the table lists, in display order. Taken from here rather than
    # from `selectable`, which holds only the rows that fitted under the row cap
    # -- and the level that lists every queue has to list every queue.
    shown_names = [q.name for q in sorted(shared, key=order)] + [
        q.name for q in sorted(owned, key=order)]
    if shared:
        add_table(shared, STATUS_ROWS)
    if owned:
        # Same columns, so the header is not repeated: the tag is the only new
        # information and it belongs on one line.
        body.append("")
        body.append(st.warn("GROUP-ONLY")
                    + st.dim(f"  allows <={Queue.DEDICATED_ACCOUNT_LIMIT} accounts"))
        add_table(owned, 4, header=bool(not shared))

    # No DEAD block, and no footer glossing the funnel's terms.
    #
    # Both used to live here: a two-line list of the partitions that can start
    # nothing, with the phantom idle-node total, and under it a line naming the
    # dry-run that refused you. They are gone because on a healthy day they are
    # the only red on the screen and they say nothing you can act on. The reader
    # asked "where can I run this"; being told, every single run, that three
    # partitions they were never going to use are down is not an answer to it.
    #
    # Nothing is *hidden* by dropping them, which is the condition for dropping
    # them at all:
    #
    # * the counts are already in the funnel line above -- `3 dead`,
    #   `14 refused` -- so the accounting still closes and a reader who wants to
    #   know why 87 became 5 has it;
    # * WHICH partitions and WHY is `nodetop queues`, which prints a blocker
    #   code per row and is where someone looking for a fault goes;
    # * the phantom idle nodes are `nodetop health`, node by node, with the
    #   reason field and how long it has been that way.
    #
    # A bad-news line earns its place on this view only when it is about the
    # thing you asked for. `FAILED <query>` above stays, because a failed query
    # means the numbers on screen are incomplete. `DECLARED ONLY` below stays,
    # because you passed --declared and it is telling you what you gave up.
    if args.declared and cluster.can_probe:
        body.append("")
        body.append(st.warn(
            "DECLARED ONLY  allowlists over-report; drop --declared to dry-run"
        ))

    if (ident is not None and ident.entitlements_look_templated
            and args.declared):
        # Suppressed under --check, where it is moot: the point of that flag is
        # that the control plane, not the association table, decided what is
        # listed. The pointer to --check goes too -- the DECLARED line above
        # already carries it, and two lines naming the same flag is the kind of
        # repetition this view keeps being trimmed for.
        body.append("")
        body.append(
            st.warn("TEMPLATED")
            + st.dim(f"  {len(ident.accounts)} accounts, one entitlement list")
        )

    title = ""
    if (not getattr(args, "static", False)
            and selectable and interactive.supported()):
        return _browse(cluster, args, st, body, selectable, excluded,
                       shown_names, (funnel_at, render_funnel), title)
    print(panel(body, title, st))
    return 0


def _browse(cluster: Cluster, args: argparse.Namespace, st: Style,
            body: list[str], selectable: list[tuple[int, str, str]],
            excluded: list[tuple[str, str]],
            shown: Sequence[str],
            funnel: tuple[int, Callable[[str | None], str]],
            title: str) -> int:
    """Move a highlight over the overview's rows and open the chosen one.

    **Nothing is re-rendered here.** The panel is the one `cmd_status` just
    built; a row is highlighted by wrapping the finished line in inverse video
    and handing the same list back to `panel`. A second renderer for the
    interactive view is how it would start disagreeing with the printed one --
    the same reason `_grid` and `_node_rows` exist.

    Zooming prints below the list rather than replacing it, because the list is
    the context for what you just opened: scrolling back up to see which row you
    picked is worse than leaving it on screen.
    """
    # Where each selectable row sits in `body`, so the viewport below can tell a
    # partition row from a heading and keep every heading whatever it hides.
    at_position = {pos: k for k, (pos, _, _) in enumerate(selectable)}
    # Every queue on the cluster, in the order the funnel accounts for them: the
    # ones the table lists, then the excluded groups. Opening the total shows
    # this, which is the one view where the funnel's arithmetic can be checked
    # instead of trusted -- the rows are the terms, itemised.
    all_queues: list[tuple[str, str]] = (
        [(name, "open") for name in shown] + list(excluded))

    def visible(rows: list[str], index: int) -> list[str]:
        """``rows``, trimmed to a window around ``index`` that fits the screen.

        **Without this, `--all` is unusable.** The redraw moves the cursor up by
        the height of the previous frame, and 84 partitions is a 93-line frame:
        on a 24-row terminal the cursor clamps at the top of the screen, the
        clear-to-end lands in the wrong place, and every repaint leaves another
        copy of the listing behind. Measured before the fix -- 252 rows on
        screen for 84 partitions.

        Only *rows* are dropped. Headings, the funnel and the totals are kept
        whatever scrolls, because they are the frame of reference for the row
        you are looking at; a viewport that discards them to fit two more rows
        has thrown away the reason the rows mean anything.
        """
        # Counted over PARTITION rows only.
        #
        # `selectable` also holds the funnel's own buckets, which share one line
        # with each other and are never dropped -- so counting them as rows both
        # overstated the total ("of 61" for 60 partitions) and stole a line from
        # the window. The scrollable thing here is the table.
        scrollable = [k for k, (_, kind, _) in enumerate(selectable)
                      if kind == "queue"]
        total = len(scrollable)
        chrome = len(rows) - total
        # 2 borders, the title line panel now adds, and one spare.
        # Clamped to at least one row rather than bailing out below three.
        # Bailing returned the whole list on exactly the terminals least able
        # to draw it: 60 partitions on ten lines was a 67-line frame, and the
        # redraw arithmetic then destroys the screen on every keypress. See
        # `_window`, which had the same escape hatch.
        #
        # Three lines of the frame are not rows: its two borders and the
        # position line appended below.
        room = max(1, term_height() - chrome - 3)
        if room >= total:
            return rows
        # `index` is an entry in `selectable`; the window is over rows, so it is
        # the row's position in that list that has to stay on screen.
        here = scrollable.index(index) if index in scrollable else 0
        lo = max(0, min(here - room // 2, total - room))
        hi = lo + room
        keep = set(scrollable[lo:hi])
        # A selectable entry that is not a partition -- the funnel -- is kept
        # whatever scrolls, like the headings around it. It is the summary the
        # rows are a breakdown of, so dropping it to fit one more row would cost
        # the reader their bearings.
        out = [line for i, line in enumerate(rows)
               if at_position.get(i) is None
               or selectable[at_position[i]][1] != "queue"
               or at_position[i] in keep]
        # Which slice is on screen, not how many rows are missing from it.
        out.append(st.dim(f"{lo + 1}-{min(hi, total)} of {total}"))
        return out

    def highlight(rows: list[str], at: int) -> None:
        """Mark row ``at`` in place by writing a cursor into its first column.

        **The glyph alone, no reverse video.** Inverse across a row of coloured
        cells is both ugly and fragile: it fights the heat ramp underneath it,
        and it has to be re-armed after every embedded reset or it stops at the
        first coloured cell. An arrow at the left edge says which row, works
        identically with colour off, and cannot be broken by the row's contents.

        Written into a column reserved for it rather than over whatever is at
        index 0. Overwriting the status-glyph column removed the first byte of
        that cell's colour escape and the row rendered as literal
        `❯[38;5;111m◐  node...` -- the escape introducer eaten, the rest of the
        sequence printed as text.
        """
        # Bounds-checked because a crash here is disproportionate: this runs
        # inside a full-screen interaction, so an IndexError takes down the whole
        # browser to fail at drawing one row.
        if 0 <= at < len(rows):
            line = rows[at]
            rows[at] = st.accent(st.g.cursor, bold=True) + line[1:]

    funnel_at, render_funnel = funnel

    def framed(content: Sequence[str]) -> list[str]:
        """Every view in the same box, whatever it happens to hold.

        The frame used to size itself to its content, so stepping from an
        eighty-column overview into `3 down` shrank it to a third of the width
        and four rows: the window jumped on every keypress that changed level,
        and the eye had to find it again each time. "whatever we choose in the
        ui, the window should stay the same and the text and information getting
        displayed should dynamically get adjusted."

        So the box comes from :func:`term_width` and :func:`term_height` -- the
        same two numbers for every level -- and content is padded up to it
        rather than letting the border close early. Over-long content is
        truncated here as a backstop: the viewport helpers size themselves from
        the same two numbers, and a frame taller than its own box is what makes
        the repaint destructive.
        """
        rows = term_height()
        body = list(content)[: rows - 2]
        body += [""] * (rows - 2 - len(body))
        return panel(body, title, st, size=term_width(),
                     shrink=False).splitlines()

    def partition_frame(index: int) -> list[str]:
        rows = list(body)
        position, kind, payload = selectable[index]
        if position == funnel_at:
            # Several selectable terms share this one row, so which is active
            # cannot be shown by position: the line is redrawn with the cursor
            # sitting against that term. No left-margin cursor as well -- two
            # cursors on one row, one of them pointing at the wrong thing, is
            # what made Right look like it had done nothing.
            rows[position] = render_funnel(payload)
        else:
            highlight(rows, position)
        return framed(visible(rows, index))

    def node_frame(queue, nodes: list, index: int) -> list[str]:
        rows = _node_rows(nodes, st)
        lines = table(NODE_HEADS, rows, NODE_ALIGNS, st, indent="",
                      limits=NODE_LIMITS, header_role="dim",
                      underline=False).splitlines()
        head, data = lines[0], lines[1:]
        highlight(data, index)
        shown = _window(data, index, st)
        with_room = sum(1 for n in nodes if n.has_room)
        facts = [st.head(queue.name),
                 st.muted(f"{len(nodes)} nodes"),
                 st.muted(f"{with_room} with room")]
        return framed([f"  {st.dim(st.g.sep)}  ".join(facts), "", head, *shown])

    def node_detail(node, jobs: list) -> list[str]:
        """What this node is, above whatever is running on it.

        The level below a node listing used to be *only* the job table, so
        opening a drained node showed a four-line box saying "nothing running
        here" -- no state, no reason, and the reason is exactly what the reader
        opened it for. It is truncated in the listing (`maintenance [root@20…`)
        because the column has to hold forty of them; this is the one view with
        room to print it whole, so it does.
        """
        state = (st.bad(node.state_raw) if not node.schedulable
                 else st.warn(node.state_raw) if node.degraded
                 else st.muted(node.state_raw))
        facts = [st.head(node.name)]
        if node.state_raw:
            facts.append(state)
        facts.append(st.muted(f"{len(jobs)} running"))
        facts.append(st.muted(f"{node.cpus_free}/{node.cpus_total} cores free"))
        if node.gpus_total:
            # "gpu" rather than a bare "?" when the model is unidentifiable:
            # the question mark reads as a defect in the reader's own knowledge
            # rather than a gap in the scheduler's inventory.
            model = node.accelerator.model if node.accelerator else "gpu"
            facts.append(st.muted(
                f"{node.gpus_free}/{node.gpus_total} {model} free"))
        out = [f"  {st.dim(st.g.sep)}  ".join(facts)]

        text, who, when = split_reason(node.reason)
        if text:
            # Wrapped to the frame, never truncated: the whole point of being
            # here. The operator and timestamp follow it rather than sitting
            # inside it, because they are who to ask, not what is wrong.
            room = term_width() - 6
            body_text = wrap_indent(text, indent="", size=room)
            out.append("")
            for i, line in enumerate(body_text.splitlines()):
                out.append(st.warn(line) if i == 0 else st.muted(line))
            stamp = "  ".join(x for x in (who, when) if x)
            if stamp:
                out.append(st.dim(stamp))
        if node.unreachable:
            out.append(st.warn("the control plane has lost contact with this node"))
        return out

    def job_frame(node, jobs: list, index: int) -> list[str]:
        if node is None:                       # pragma: no cover - defensive
            return framed([st.warn("no such node")])
        if not jobs:
            # Three different reasons for an empty list, and saying the wrong
            # one is the phantom-capacity mistake in a new place: "no jobs here"
            # on a node that is plainly busy would have the reader conclude the
            # occupancy figures are wrong.
            if "jobs" in cluster.errors:
                note = st.warn("the job list could not be read")
            elif not cluster.jobs() and node.cpus_alloc:
                note = st.warn("this backend cannot list jobs")
            elif node.cpus_alloc or node.gpus_alloc:
                note = st.dim("busy, but no job claims these resources")
            elif not node.schedulable:
                # Not "nothing running here", which reads as "free".
                note = st.dim("nothing running here, and nothing can start")
            else:
                note = st.dim("nothing running here")
            return framed([*node_detail(node, jobs), "", note])
        heat = heat_steps([j.cpus for j in jobs])

        def here(job) -> Allocation | None:
            """This job's share of THIS node, or ``None`` if nothing can say.

            The column used to hold the job's total across every node it holds,
            which in a per-node table is a number the reader knows to be
            impossible: a 42-node job read as **512 cores on a 48-core
            machine**, with a `x42` marker that explained nothing -- "the cpu
            column doesn't make any sense. what do the column entries mean?"
            Its actual share of that node was seven cores and seven gigabytes.
            """
            return cluster.share_of(job, node.name)

        shares = [here(j) for j in jobs]
        spans = any(len(j.nodes) > 1 for j in jobs)

        def cell(value: int, i: int, tint: bool = False) -> str:
            if shares[i] is None:
                # Unknown, not zero. Only a multi-node job on a backend that
                # cannot report the split reaches this, and its totals are one
                # keypress away in the job's own view.
                return st.dim("?")
            if not value:
                # `0`, not the `·` this table used: the node has accelerators
                # and this job holds none of them, which is a count and not an
                # absence of one.
                return st.dim("0")
            return st.tint(str(value), heat[i]) if tint else str(value)

        # No accelerator column on a node that has none: `table` drops a column
        # no row fills, and a column of dashes costs width the job name needs.
        has_gpu = node.gpus_total > 0

        rows = [[
            " ",
            st.tint(j.id, heat[i]),
            j.user,
            st.muted(j.account),
            cell(share.cpus if (share := shares[i]) else 0, i, tint=True),
            cell((share.memory_mb // 1024) if share else 0, i),
            cell(share.gpus if share else 0, i) if has_gpu else "",
            # The span as its own column rather than a marker glued to a
            # number: "42" under `nodes` needs no decoding. The column appears
            # only when something actually spans, because on most nodes every
            # job is a single-node job and an all-`1` column costs width the
            # name needs -- but when it does appear it holds the count, `1`
            # included. A `·` there was the gpu column's mistake again: an
            # empty-cell mark standing in for a number the reader then has to
            # guess ("what does . mean in the node column?").
            *([st.muted(str(len(j.nodes)))] if spans else []),
            st.muted(j.elapsed),
            j.remaining,
            j.name,
        ] for i, j in enumerate(jobs)]
        lines = table(["", "job", "user", "account", "cpu", "mem gb", "gpu",
                       *(["nodes"] if spans else []), "used", "left", "name"],
                      rows,
                      ["left", "left", "left", "left", "right", "right",
                       "right", *(["right"] if spans else []), "right",
                       "right", "left"], st,
                      indent="",
                      limits=[0, 0, 14, 16, 0, 0, 0, *([0] if spans else []),
                              0, 0, 24],
                      header_role="dim", underline=False).splitlines()
        head, data = lines[0], lines[1:]
        highlight(data, index)
        detail = node_detail(node, jobs)
        shown = _window(data, index, st, reserved=5 + len(detail))
        return framed([*detail, "", head, *shown])

    def reason_frame(index: int, entries: Sequence[tuple[str, str]],
                     heading: str) -> list[str]:
        """A list of queues, each with the word that put it there.

        One renderer for three levels -- every partition, everything the funnel
        counted out, and one reason's worth -- because they differ only in which
        entries they are handed. Three tables of the same shape is three places
        for the columns to drift.
        """
        by_reason: dict[str, int] = {}
        for _, why in entries:
            by_reason[why] = by_reason.get(why, 0) + 1
        rows = []
        for name, why in entries:
            q = cluster.queues.get(name)
            tone = {"down": st.bad, "refused": st.warn,
                    "open": st.ok}.get(why, st.muted)
            rows.append([
                " ", name, tone(why),
                st.muted(str(len(q.nodes))) if q else "",
                st.muted(str(q.cpus_total)) if q and q.cpus_total else "",
                st.muted(", ".join(list(q.accelerator_models)[:2])) if q else "",
            ])
        lines = table(["", cluster.queue_term, "why", "nodes", "cores",
                       "models"], rows,
                      ["left", "left", "left", "right", "right", "left"], st,
                      indent="", limits=[0, 22, 12, 0, 0, 22],
                      header_role="dim", underline=False).splitlines()
        head, data = lines[0], lines[1:]
        highlight(data, index)
        shown = _window(data, index, st)
        facts = [st.head(heading)]
        if len(by_reason) > 1:
            facts += [st.muted(f"{n} {w}") for w, n in by_reason.items()]
        return framed([f"  {st.dim(st.g.sep)}  ".join(facts), "", head, *shown])

    def excluded_frame(index: int, only: str = "") -> list[str]:
        """The partitions the funnel counted out, optionally one reason's worth."""
        chosen = [e for e in excluded if not only or e[1] == only]
        heading = (f"{len(chosen)} {only}" if only
                   else f"{len(chosen)} not in the list")
        return reason_frame(index, chosen, heading)

    def every_frame(index: int) -> list[str]:
        """Every queue on the cluster, whether or not the table lists it."""
        return reason_frame(index, all_queues,
                            plural(len(all_queues), cluster.queue_term))

    def job_detail_frame(node, job, index: int) -> list[str]:
        """One job, in full.

        Enter on a job row used to pop straight back to the node listing --
        "when choosing any of the job here, it doesn't go into the job details
        but going back to the original node" -- so the row was the deepest the
        tool went, with its name truncated and its node list unseen. Nothing
        here is selectable: this is the leaf, and any key leaves it.
        """
        facts = [st.head(job.id)]
        if job.state:
            facts.append(st.muted(job.state))
        # Empty fields are dropped rather than joined: a backend that does not
        # report an account left a dangling separator at the end of the line.
        facts += [x for x in (job.user, st.muted(job.account) if job.account
                              else "") if x]
        out = [f"  {st.dim(st.g.sep)}  ".join(facts)]
        if job.name:
            # Untruncated, which the listing cannot afford.
            out += ["", st.accent(job.name)]
        share = cluster.share_of(job, node.name) if node is not None else None
        pairs: list[tuple[str, str]] = []
        if share is not None:
            bits = [f"{share.cpus} cores"]
            if share.memory_mb:
                bits.append(f"{share.memory_mb // 1024} GB")
            if share.gpus:
                bits.append(f"{share.gpus} gpu")
            pairs.append((f"on {node.name}",
                          st.head(f"  {st.dim(st.g.sep)}  ".join(bits))))
        elif node is not None:
            pairs.append((f"on {node.name}",
                          st.dim("this backend cannot report the per-node split")))
        # `job total` and `nodes` only earn their lines when the job spans
        # more than one machine. On a single-node job the share above IS the
        # total and the node is already named in its label, so both rows would
        # restate what the reader just read.
        if len(job.nodes) > 1:
            total = [f"{job.cpus} cores"]
            if job.gpus:
                total.append(f"{job.gpus} gpu")
            total.append(plural(len(job.nodes), "node"))
            pairs.append(("job total",
                          st.muted(f"  {st.dim(st.g.sep)}  ".join(total))))
        if job.queue:
            pairs.append((cluster.queue_term, st.muted(job.queue)))
        if job.elapsed or job.remaining:
            used = f"{job.elapsed} used" if job.elapsed else ""
            left = f"{job.remaining} left" if job.remaining else ""
            pairs.append(("time", st.muted(
                f"  {st.dim(st.g.sep)}  ".join(x for x in (used, left) if x))))
        if len(job.nodes) > 1:
            pairs.append(("nodes", st.muted(cluster.format_nodelist(job.nodes))))
        out += ["", *kv(pairs, st, indent="", size=term_width() - 6).splitlines()]
        return framed(out)

    # One raw-mode block, one screen, and an explicit stack.
    #
    # Each level REPLACES the one before it -- `select` erases its block on the
    # way out -- so partitions, the excluded list, nodes and jobs all occupy the
    # same rows. An earlier version printed each new view below the last, which
    # grew a transcript and left the reader scrolling to find what they had just
    # opened.
    #
    # A stack rather than a depth counter because the levels are no longer a
    # single chain: the overview leads to partitions OR to the excluded list, and
    # both lead to nodes. Cursor positions are kept per stack entry, so stepping
    # out lands on the row you came from.
    keyed = interactive.Key
    with interactive.raw_session():
        # Stack entries are (kind, key) with a STRING key, and cursors are
        # keyed by the same pair. Carrying the resolved object instead made the
        # entry unhashable -- a Node is a mutable dataclass -- so remembering the
        # cursor position raised TypeError the moment anyone opened a node.
        by_name = {n.name: n for n in cluster.nodes}
        stack: list[tuple[str, str]] = [("partitions", "")]
        cursors: dict[tuple[str, str], int] = {}
        while stack:
            where = stack[-1]
            kind, payload = where
            if kind == "partitions":
                # `rows` is what makes the arrows mean what they look like: the
                # funnel's counts share one display row, so Left and Right move
                # along them while Up and Down step between rows.
                #
                # `escapable=False` because this is the root. Left and Escape
                # here used to return, and returning at the root exits -- so a
                # stray Left took the whole program down.
                got = interactive.select(
                    partition_frame, len(selectable), raw=False,
                    initial=cursors.get(where, 0),
                    rows=[pos for pos, _, _ in selectable], escapable=False)
                if not isinstance(got, int):
                    return 0            # only `q` reaches here
                cursors[where] = got
                _, what, name = selectable[got]
                if what == "total":
                    stack.append(("all", ""))
                elif what == "excluded":
                    stack.append(("excluded", name))
                elif what == "open":
                    # Its members are the rows below, so opening the bucket
                    # means putting the cursor on the first of them.
                    first = next((i for i, (_, k, _) in enumerate(selectable)
                                  if k == "queue"), None)
                    if first is not None:
                        cursors[where] = first
                else:
                    stack.append(("nodes", name))
            elif kind == "all":
                got = interactive.select(every_frame, len(all_queues), raw=False,
                                         initial=cursors.get(where, 0))
                if got == keyed.QUIT:
                    return 0
                if not isinstance(got, int):
                    stack.pop()
                    continue
                cursors[where] = got
                stack.append(("nodes", all_queues[got][0]))
            elif kind == "excluded":
                subset = [e for e in excluded if not payload or e[1] == payload]

                def draw_excluded(i: int, only: str = payload) -> list[str]:
                    return excluded_frame(i, only)

                got = interactive.select(draw_excluded, len(subset), raw=False,
                                         initial=cursors.get(where, 0))
                if got == keyed.QUIT:
                    return 0
                if not isinstance(got, int):
                    stack.pop()
                    continue
                cursors[where] = got
                stack.append(("nodes", subset[got][0]))
            elif kind == "nodes":
                queue = cluster.queues.get(payload)
                nodes = sorted(
                    queue.nodes if queue else [],
                    key=lambda n: (not n.schedulable, -n.effective_free_gpus,
                                   -n.effective_free_cpus, n.name))

                def draw_nodes(i: int, q: object = queue,
                               ns: list = nodes) -> list[str]:
                    return node_frame(q, ns, i)

                got = interactive.select(draw_nodes, len(nodes), raw=False,
                                        initial=cursors.get(where, 0))
                if got == keyed.QUIT:
                    return 0
                if not isinstance(got, int):
                    stack.pop()
                    continue
                cursors[where] = got
                stack.append(("jobs", nodes[got].name))
            elif kind == "jobs":
                node = by_name.get(payload)
                jobs = cluster.jobs_on(payload)

                def draw_jobs(i: int, nd: object = node,
                              js: list = jobs) -> list[str]:
                    return job_frame(nd, js, i)

                got = interactive.select(draw_jobs, max(1, len(jobs)),
                                        raw=False,
                                        initial=cursors.get(where, 0))
                if got == keyed.QUIT:
                    return 0
                if isinstance(got, int) and got < len(jobs):
                    cursors[where] = got
                    stack.append(("job", f"{payload}|{jobs[got].id}"))
                    continue
                stack.pop()
            elif kind == "job":
                node_name, _, job_id = payload.partition("|")
                nd = by_name.get(node_name)
                job = next((j for j in cluster.jobs_on(node_name)
                            if j.id == job_id), None)
                if job is None:                # pragma: no cover - it vanished
                    stack.pop()
                    continue

                def draw_job(_i: int, n: object = nd, j: object = job) -> list[str]:
                    return job_detail_frame(n, j, 0)

                # One entry: there is nothing here to choose between, so any
                # key that is not `q` steps back out.
                got = interactive.select(draw_job, 1, raw=False)
                if got == keyed.QUIT:
                    return 0
                stack.pop()
            else:                          # pragma: no cover - unreachable
                stack.pop()
    return 0


def cmd_queues(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    wanted = [x.strip() for x in args.queue.split(",") if x.strip()]
    # Naming a queue is an explicit request for it, so the entitlement filter
    # steps aside -- "show me this one" should show it, blockers and all.
    entitled, hidden_queues = (None, 0) if wanted else _entitled_queues(cluster, args)
    queues = [
        q for q in cluster.queues.values()
        if (not wanted or q.name in wanted)
        and (not args.unusable_only or not q.usable)
        and (entitled is None or not q.usable or q.name in entitled)
    ]
    # Usable first, and within that the ones with room first. `(q.usable,
    # q.name)` put the *unusable* ones on top -- False sorts before True -- so
    # this listing opened with the three partitions that can start nothing,
    # which is the exact complaint that got the overview reordered and then got
    # its DEAD block deleted. A command whose rows are places you might submit
    # to should not lead with the places you cannot.
    #
    # Ranked by free cores inside each group, for the same reason `status` is: a
    # core is the unit of room, and it makes the top of the list the useful end.
    # Blocked queues keep name order -- there is no room to rank them by.
    queues.sort(key=lambda q: (
        not q.usable, -q.effective_free_cpus if q.usable else 0, q.name))

    if args.json:
        print(json.dumps([{
            "name": q.name,
            "state": q.state_raw,
            "enabled": q.enabled,
            "started": q.started,
            "hidden": q.hidden,
            "usable": q.usable,
            "routes": q.routes,
            "forwards_to": list(q.forwards_to),
            "blockers": [{"code": b.code, "detail": b.detail}
                         for b in q.structural_blockers()],
            "nodes": len(q.nodes),
            "nodes_declared": q.declared_nodes,
            "idle_nodes_advertised": len(q.idle_nodes),
            "effective_free_nodes": q.effective_free_nodes,
            # The text form prints "0 wholly free, 45 of 190 with something
            # spare" and a free-core count; `--json` carried neither, so the
            # two forms of the same command answered different questions.
            "nodes_with_room": sum(1 for n in q.nodes if n.has_room),
            "cpus_total": q.cpus_total,
            "cpus_free_advertised": q.cpus_free,
            "effective_free_cpus": q.effective_free_cpus,
            "accelerators_total": q.gpus_total,
            "accelerators_free_advertised": q.gpus_free,
            "effective_free_accelerators": q.effective_free_gpus,
            "accelerator_models": q.accelerator_models,
            "max_walltime_queue": format_duration(q.max_walltime_seconds),
            "max_walltime_effective": format_duration(
                cluster.effective_max_walltime(q.name)
            ),
        } for q in queues], indent=2))
        return 0

    # A block per queue is right for one queue and unreadable for eighty-seven:
    # this cluster would print 619 lines. So the table is the default and the
    # blocks are reserved for a named queue or an explicit --detail.
    if args.detail or args.queue:
        _queues_detail(cluster, queues, st)
    else:
        _queues_table(cluster, queues, st, hidden_queues)
    return 0


def _queue_mark(queue, st: Style) -> str:
    """Status glyph for one queue.

    A routing queue gets an arrow rather than a health mark: it is neither
    usable nor blocked in its own right, it forwards.
    """
    if queue.routes:
        return st.info(st.g.arrow)
    return st.ok(st.g.ok) if queue.usable else st.bad(st.g.off)


def _queues_table(cluster: Cluster, queues: list, st: Style,
                  hidden_queues: int = 0) -> None:
    term = cluster.queue_term
    # Ranked across the whole listing, not per row: a fixed set of colour bands
    # cannot know that three of eighty-seven queues happen to fall inside one of
    # them, and three queues an order of magnitude apart in the same tone is
    # what makes a ramp read as noise. See render.heat_steps.
    heat = dict(zip([q.name for q in queues],
                    heat_steps([q.effective_free_cpus for q in queues]),
                    strict=False))
    rows = []
    for q in queues:
        blockers = q.structural_blockers()
        summary = ""
        if blockers:
            summary = st.bad(blockers[0].code)
            if len(blockers) > 1:
                summary += st.dim(f" +{len(blockers) - 1}")
        schedulable = len([n for n in q.nodes if n.schedulable])
        mark = _queue_mark(q, st)
        if q.routes:
            # A routing queue owns no nodes, so there is nothing to meter; what
            # matters is where it sends work, which belongs in the note column
            # rather than in a capacity slot it has no value for.
            summary = st.info(f"{st.g.arrow} " + ",".join(q.forwards_to))
            capacity = ""
            nodes_up = st.dim(st.g.dash)
            idle: object = st.dim(st.g.dash)
        else:
            # The meter goes on free CORES. It measured GPU share first, so
            # seventy of eighty-seven rows drew an empty bar; it then measured
            # wholly-idle NODES, which is empty for anything running a single
            # job and ranked a 128-core partition above one with 2825 cores
            # free. A core is the unit of room -- see
            # Queue.effective_free_cpus. `idle` keeps the node count beside it,
            # because work wanting a whole node still needs that number.
            total = len(q.nodes)
            cores = q.cpus_total
            capacity = bar((q.effective_free_cpus / cores) if cores else 0.0,
                           8, st, step=heat[q.name])
            nodes_up = f"{schedulable}/{total}"
            # The number beside the meter must BE what the meter measures.
            # This held the wholly-idle node count while the bar showed core
            # share, so `amd` rendered a half-full bar next to a bare "0" --
            # the same self-contradiction the overview was carrying. Node-level
            # availability is still in `nodetop nodes` and in `-q <name>`.
            idle = (st.tint(str(q.effective_free_cpus), heat[q.name])
                    + st.muted(f"/{cores}"))
        gpus = (f"{q.effective_free_gpus}{st.muted('/' + str(q.gpus_total))}"
                if q.gpus_total else st.dim(st.g.dash))
        rows.append([
            # Number then meter, and `free/total` in one cell: the same shape
            # the overview uses, so one habit reads both tables.
            mark, q.name, st.muted(q.state_raw), nodes_up, idle, capacity, gpus,
            format_duration(cluster.effective_max_walltime(q.name)),
            summary,
        ])
    facts = [f"{len(queues)} shown",
             f"{sum(1 for q in queues if q.usable)} usable"]
    if hidden_queues:
        facts.append(f"{hidden_queues} not on your allowlist")
    print(section(f"{term}s", st, ", ".join(facts)))
    print(_grid(
        ["", term, "state", "nodes up", "cores free", "", "gpu free",
         "maxtime", "blocked by"],
        rows,
        ["left", "left", "left", "right", "right", "left", "right", "left",
         "left"],
        st, indent="  ", limits=[0, 22, 12, 0, 0, 0, 0, 0, 26],
        head_paint=[None, st.head, st.muted, st.muted,
                    lambda s: st.tint(s, 2), None,
                    lambda s: st.tint(s, 9), st.muted, st.muted],
    ))
    print()
    print(_note(f"zoom <{term}> lists the nodes inside one", st))


def _queues_detail(cluster: Cluster, queues: list, st: Style) -> None:
    """One block per queue: every gate and every number, for a close look."""
    for q in queues:
        mark = _queue_mark(q, st)
        head = st.head(q.name) if q.usable else st.bad(q.name, bold=True)
        flags = [f for f, on in (("hidden", q.hidden), ("default", q.is_default)) if on]
        tag = q.state_raw + (" " + " ".join(flags) if flags else "")
        # Clipped like every other line: a long queue name is still a name, and
        # letting the heading run past the window breaks the block it opens.
        room = term_width() - width(mark) - width(tag) - 5
        print(f"{mark} {truncate(head, max(8, room), st.g.ellipsis)} "
              f"{st.dim('[' + tag + ']')}")

        schedulable = len([n for n in q.nodes if n.schedulable])
        node_note = gauge(schedulable, len(q.nodes), 14, st, "schedulable")
        if q.unresolved_nodes:
            node_note += st.warn(
                f"  {st.g.warn} +{q.unresolved_nodes} claimed but unresolved"
            )
        # Both counts on one line, because the smaller one alone gets read as
        # the whole answer. `idle` counts *wholly* idle nodes and on a busy
        # cluster almost nothing is wholly idle -- this partition routinely
        # shows `idle 0` while carrying a couple of hundred free cores spread
        # over nodes that are each running something. "0" then reads as "nothing
        # here for me", which is wrong and costs the reader the partition.
        with_room = sum(1 for n in q.nodes if n.has_room)
        idle = str(q.effective_free_nodes)
        if not q.usable and q.idle_nodes:
            idle = st.bad(idle) + st.dim(f"  ({len(q.idle_nodes)} advertised)")
        idle += st.dim("  wholly free, ") + st.head(f"{with_room}") + st.dim(
            f" of {len(q.nodes)} with something spare")
        if q.routes:
            print(kv([
                ("type", st.info("routing queue") + st.dim(
                    "  -- forwards work; owns no nodes of its own")),
                ("forwards to", ", ".join(q.forwards_to) or st.dim("nothing")),
                ("maxtime", _wall_detail(cluster, q, st)),
            ], st, indent="    "))
            blockers = q.structural_blockers()
            if blockers:
                print(tree([(st.bad(b.code), b.detail) for b in blockers], st,
                           indent="    "))
            print()
            continue
        print(kv([
            ("nodes", node_note),
            ("idle", idle),
            ("accel", gauge(q.effective_free_gpus, q.gpus_total, 14, st, "free")
             if q.gpus_total else st.dim("none")),
            ("models", ", ".join(f"{k}x{v}" for k, v in q.accelerator_models.items())
             or st.dim("none")),
            ("maxtime", _wall_detail(cluster, q, st)),
        ], st, indent="    "))
        blockers = q.structural_blockers()
        if blockers:
            print(tree([(st.bad(b.code), b.detail) for b in blockers], st, indent="    "))
        print()


def _wall_detail(cluster: Cluster, queue, st: Style) -> str:
    """Render the wall limit, naming the limit set when it is the binding one."""
    eff = cluster.effective_max_walltime(queue.name)
    text = format_duration(eff)
    limits = cluster.limits_for(queue.name)
    if (
        limits is not None
        and limits.max_walltime_seconds is not None
        and eff == limits.max_walltime_seconds
        and queue.max_walltime_seconds != limits.max_walltime_seconds
    ):
        text += st.dim(
            f"  (from {limits.source or 'limits'} {limits.name}; the "
            f"{cluster.queue_term} itself says "
            f"{format_duration(queue.max_walltime_seconds)})"
        )
    return text


def cmd_zoom(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    """One partition, opened up: its gates, then the nodes inside it.

    **This exists because ``idle 0`` was being read as "nothing here for me".**
    ``idle`` counts *wholly* idle nodes, and on a busy cluster almost nothing is
    wholly idle -- `gn` routinely shows `idle 0` while carrying a couple of
    hundred free cores and twenty-odd free accelerators, spread thinly over
    nodes that are each running something. The overview cannot show that without
    becoming a node listing, so the number that fits there is the one most
    easily misread.

    So this view leads with both counts side by side -- how many nodes are
    entirely free, and how many have *any* room -- and then lists the nodes
    themselves, roomiest first. The header is the same block ``queues -q NAME``
    prints and the table is the same one ``nodes`` prints, from the same
    builders: a zoom view whose columns disagree with the listing it zooms out
    to is worse than no zoom view.
    """
    wanted = [x.strip() for x in args.queue.split(",") if x.strip()]
    queues = [cluster.queues[n] for n in wanted if n in cluster.queues]
    if not queues:
        print(f"name a {cluster.queue_term} to look inside", file=sys.stderr)
        return 2

    nodes = [n for q in queues for n in q.nodes]
    # De-duplicated by name: two named partitions can share hardware, and a node
    # listed twice would be counted twice in every figure below.
    nodes = list({n.name: n for n in nodes}.values())
    if args.gpu:
        nodes = [n for n in nodes if n.is_gpu_node]
    if args.cpu:
        nodes = [n for n in nodes if not n.is_gpu_node]
    if args.free:
        nodes = [n for n in nodes if n.has_room]

    with_room = [n for n in nodes if n.has_room]
    wholly_idle = [n for n in nodes if n.idle]
    out_of_service = [n for n in nodes if not n.schedulable]

    if args.json:
        print(json.dumps({
            cluster.queue_term: [q.name for q in queues],
            "nodes": len(nodes),
            "wholly_idle": len(wholly_idle),
            "with_room": len(with_room),
            "unschedulable": len(out_of_service),
            "cpus": [sum(n.cpus_free for n in nodes if n.schedulable),
                     sum(n.cpus_total for n in nodes)],
            # What the scheduler claims is free, and what could actually be
            # allocated: they differ by every core sitting on a node whose
            # memory is spoken for. `with_room` above counts those nodes out,
            # so these figures have to as well or the two disagree. Named as
            # `queues --json` names them -- one quantity, one key, whichever
            # command you ask.
            "effective_free_cpus": sum(n.effective_free_cpus for n in nodes
                                       if n.schedulable),
            "effective_free_accelerators": sum(
                n.effective_free_gpus for n in nodes if n.schedulable),
            "accelerators": [sum(n.gpus_free for n in nodes if n.schedulable),
                             sum(n.gpus_total for n in nodes)],
            "members": [{
                "name": n.name,
                "state": n.state_raw,
                "schedulable": n.schedulable,
                "idle": n.idle,
                "cpus": [n.cpus_free, n.cpus_total],
                "memory_gb": [n.memory_free_mb // 1024, n.memory_mb // 1024],
                "accelerators": [n.gpus_free, n.gpus_total],
                "accelerator_model": n.accelerator.model if n.accelerator else None,
                "reason": n.reason,
            } for n in nodes],
        }, indent=2))
        return 0

    _queues_detail(cluster, queues, st)

    # The line this command exists for. Both counts, adjacent, so the smaller
    # one cannot be read as the whole answer.
    # The counts are on the `idle` line of the block above; repeating them here
    # would be the third place the same number appears on one screen.
    facts = [plural(len(nodes), "node")]
    if out_of_service:
        facts.append(st.dim(f"{len(out_of_service)} out"))
    if len(nodes) > 1:
        facts.append(st.dim("roomiest first"))
    # A routing queue forwards work and owns no hardware, which the block above
    # has just said in those words. An empty node listing under it is a second
    # way of saying the same nothing.
    if not nodes and all(q.routes for q in queues):
        return 0

    dropped_by = [flag for flag, on in
                  (("--gpu", args.gpu), ("--cpu", args.cpu), ("--free", args.free))
                  if on]
    if not nodes and dropped_by:
        # "(nothing to show)" alone reads as "this queue is empty", which is a
        # different claim from "your filter excluded all of it".
        print(section("inside", st,
                      st.dim(f"nothing matches {' '.join(dropped_by)}")))
        return 0

    print(section("inside", st, f"  {st.g.sep}  ".join(facts)))
    # Same rule as `nodes`: a drained node reports everything free and none of
    # it is reachable, so it cannot lead a list of where the room is.
    nodes = sorted(nodes, key=lambda n: (not n.schedulable,
                                         -n.effective_free_gpus,
                                         -n.effective_free_cpus, n.name))
    rows = _node_rows(nodes, st)
    limit = None if args.all else max(1, args.top)
    visible = rows if limit is None else rows[:limit]
    print(_grid(NODE_HEADS, visible, NODE_ALIGNS, st, indent="  ",
                limits=NODE_LIMITS))
    if len(visible) < len(rows):
        print(st.muted(f"  +{len(rows) - len(visible)}")
              + st.dim(f" more  {st.g.sep}  --all"))
    return 0


def cmd_nodes(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    nodes = cluster.nodes
    hidden_nodes = 0
    if not args.queue:
        entitled, _ = _entitled_queues(cluster, args)
        if entitled is not None:
            reachable = {n.name for name in entitled
                         for n in cluster.queues[name].nodes}
            before = len(nodes)
            nodes = [n for n in nodes if n.name in reachable]
            hidden_nodes = before - len(nodes)
    if args.queue:
        wanted = {x.strip() for x in args.queue.split(",") if x.strip()}
        by_queue = {
            n.name for q in cluster.queues.values() if q.name in wanted for n in q.nodes
        }
        nodes = [n for n in nodes if n.name in by_queue or wanted & set(n.queues)]
    if args.gpu:
        nodes = [n for n in nodes if n.is_gpu_node]
    if args.cpu:
        nodes = [n for n in nodes if not n.is_gpu_node]
    if args.free:
        nodes = [n for n in nodes if n.has_room]

    # Most room first, because the listing is capped.
    #
    # It was in the scheduler's own order, which is node-name order, and that
    # makes a capped window arbitrary: `nodes --free` found 127 nodes with
    # something free, then showed the alphabetically first 20 -- the top row
    # having 0 of 32 cores free while 107 roomier nodes sat behind `--all`.
    # A cap is only defensible if what it keeps is what you wanted.
    #
    # Free accelerators lead, free cores break the tie. One rule, correct at
    # both ends rather than switched on a flag: on CPU-only nodes every
    # `gpus_free` is 0, so it degrades exactly to cores-descending, and on GPU
    # nodes it answers the question those rows are read for. Name last, so the
    # order is stable between runs on an idle cluster.
    # Unschedulable nodes last, whatever their counters say.
    #
    # A drained node still reports its full complement free, and ranking on that
    # put a DOWN+DRAIN node with "32/32 cores, 4/4 GPUs" at the top of the list
    # -- phantom capacity, at the head of the answer to "where is there room".
    # `Queue.effective_free_*` has always excluded these; the ordering had not.
    nodes = sorted(nodes, key=lambda n: (not n.schedulable,
                                         -n.effective_free_gpus,
                                         -n.effective_free_cpus, n.name))

    if args.json:
        print(json.dumps([{
            "name": n.name,
            "state": n.state_raw,
            "conditions": sorted(n.conditions),
            "schedulable": n.schedulable,
            "degraded": n.degraded,
            "unreachable": n.unreachable,
            "cpus": [n.cpus_free, n.cpus_total],
            "memory_gb": [n.memory_free_mb // 1024, n.memory_mb // 1024],
            "accelerators": [n.gpus_free, n.gpus_total],
            "accelerator_model": n.accelerator.model if n.accelerator else None,
            "accelerator_vendor": n.accelerator.vendor if n.accelerator else None,
            "accelerator_arch": n.accelerator.arch if n.accelerator else None,
            "accelerator_memory_gb": n.accelerator.memory_gb if n.accelerator else None,
            "accelerator_memory_inferred": (
                (not n.accelerator.memory_certain) if n.accelerator else None
            ),
            "taints": list(n.taints),
            "reason": n.reason,
        } for n in nodes], indent=2))
        return 0

    rows = _node_rows(nodes, st)
    matched = len(nodes)
    total = len(cluster.nodes)
    # Not `accel`: that name is the GPU *cell* in the row loop above, and
    # reusing it here for a count made one identifier mean a string and an int
    # in the same function.
    gpu_nodes = sum(1 for n in nodes if n.is_gpu_node)
    down = sum(1 for n in nodes if not n.schedulable)
    # Capped by default. This was the one command that answered "how are my 607
    # nodes doing" with 607 rows, which is not an answer, it is the raw data
    # again. `rdu` sets the precedent with its own `--top`, defaulting to ten.
    limit = None if args.all else max(1, args.top)
    visible = rows if limit is None else rows[:limit]
    note = f"{matched} of {total}" if matched != total else f"all {total}"
    facts = [note, f"{gpu_nodes} with GPUs", f"{down} out"]
    if hidden_nodes:
        facts.append(st.dim(f"{hidden_nodes} not on your allowlist"))
    print(section("nodes", st, f"  {st.g.sep}  ".join(facts)))
    print(_grid(NODE_HEADS, visible, NODE_ALIGNS, st, indent="  ",
                limits=NODE_LIMITS))
    if len(visible) < len(rows):
        hidden = len(rows) - len(visible)
        # Terse, like every other withheld-count in the tool. Listing four
        # ways to narrow the listing was 66 columns of flags -- the exact habit
        # the rest of this file spent several rounds removing.
        print("  " + st.dim(f"+{hidden} more {st.g.sep} --all"))
    return 0


def _reason_fields(reason: str | None) -> dict[str, str | None]:
    """The parsed halves of a scheduler reason, for the JSON view."""
    text, who, when = split_reason(reason)
    return {
        "reason_text": text or None,
        "reason_set_by": who or None,
        "reason_set_at": when or None,
    }


def cmd_health(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    degraded = cluster.degraded_nodes
    down = cluster.unschedulable_nodes
    if args.json:
        print(json.dumps({
            # `reason` is verbatim; `reason_text` / `reason_set_by` /
            # `reason_set_at` are the same string parsed, so a consumer can
            # group by cause without reimplementing the [who@when] split -- and
            # so the text view and a script agree on what one cause is.
            "degraded": [{"name": n.name, "state": n.state_raw, "reason": n.reason,
                          **_reason_fields(n.reason),
                          "accelerators": n.gpus_total,
                          "accelerator_model": (
                              n.accelerator.model if n.accelerator else None)}
                         for n in degraded],
            "unschedulable": [{"name": n.name, "state": n.state_raw,
                               "conditions": sorted(n.conditions), "reason": n.reason,
                               **_reason_fields(n.reason)}
                              for n in down],
            "unschedulable_nodelist": cluster.format_nodelist([n.name for n in down]),
        }, indent=2))
        return 0

    total = len(cluster.nodes)
    print(panel([
        f"{st.ok(str(total - len(down)))} schedulable   "
        f"{st.dim(st.g.sep)}   {st.warn(str(len(degraded)))} degraded   "
        f"{st.dim(st.g.sep)}   {st.bad(str(len(down)))} out   "
        f"{st.dim(st.g.sep)}   {total} total",
    ], "node health", st))

    print()
    # The caveat belongs in the heading, not under the result: it qualifies
    # what this section can ever find, so repeating it as a footnote under
    # "none detected" was a paragraph explaining an empty list.
    # "keyword match on the reason field" is the caveat, and it is three words
    # in parentheses rather than a clause.
    print(section("degraded", st, "reason-field keywords only"))
    if degraded:
        print(_note("scheduler still hands these out; they run slower", st))
        print()
        print(_grid(
            ["", "NODE", "STATE", "GPUS", "REASON"],
            [[st.warn(st.g.warn), n.name, n.state_raw,
              f"{n.gpus_total}x{n.accelerator.model if n.accelerator else '?'}"
              if n.is_gpu_node else st.dim(st.g.dash),
              st.dim(n.reason)] for n in degraded],
            style=st, indent="  ", limits=[0, 24, 16, 0, 46],
        ))
    else:
        print("  " + st.ok("none detected"))

    print()
    print(section("unschedulable", st, f"{len(down)} nodes"))
    if down:
        # Grouped on the reason TEXT, not the raw string: Slurm stamps each
        # reason with [who@when], so keying on the whole thing turns one
        # maintenance window into a row per second the operator spent typing.
        # The stamp is not discarded -- the oldest becomes the group's age,
        # which is the part worth knowing (out since July reads very
        # differently from out since this morning).
        groups: dict[str, list[str]] = {}
        oldest: dict[str, datetime] = {}
        for n in down:
            text, _who, when = split_reason(n.reason)
            key = text or f"state {n.state_raw}"
            groups.setdefault(key, []).append(n.name)
            stamp = parse_timestamp(when)
            if stamp is not None and (key not in oldest or stamp < oldest[key]):
                oldest[key] = stamp
        now = cluster.taken_at or datetime.now()
        items = []
        for reason, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            label = f"{st.bad(str(len(names)).rjust(3))}  {reason}"
            if reason in oldest:
                # An elapsed time, so format_age: a reason stamped ahead of the
                # clock has no age, and format_wait would have called it
                # "overdue".
                elapsed = format_age((now - oldest[reason]).total_seconds())
                if elapsed:
                    label += st.dim(f"  for {elapsed}")
            items.append((label, cluster.format_nodelist(names)))
        print(tree(items, st))
    else:
        print("  " + st.ok("every node is schedulable"))
    return 0


#: label -> (glyph name on Glyphs, Style role, what the reader should do).
#:
#: One table so the row glyph and the legend glyph cannot drift, and so the
#: legend can be built from the labels actually present.  The legend used to be
#: a fixed list of four, which meant `where` advertised "wrong hardware" over a
#: table containing no such row.
_VERDICT_LEGEND: dict[str, tuple[str, str, str]] = {
    "RUN NOW": ("ok", "ok", "runs now"),
    "BLOCKED": ("off", "dim", "not permitted"),
    "NO ANSWER": ("warn", "warn", "the control plane did not answer"),
    "WRONG HW": ("bad", "bad", "no node of the right kind"),
    "TOO FEW": ("bad", "bad", "right nodes, never enough of them"),
    "LIMIT": ("warn", "warn", "over a declared ceiling"),
    "QUEUE": ("partial", "warn", "would queue"),
}


def _verdict_paint(label: str, st: Style) -> tuple[str, str]:
    """(glyph, label) for one verdict label, coloured once and consistently."""
    glyph_name, role, _ = _VERDICT_LEGEND[label]
    paint = getattr(st, role)
    return paint(getattr(st.g, glyph_name)), paint(label)


def _verdict_label(p: Placement) -> str:
    """The single most actionable fact about one placement.

    Each label implies a different next move, so collapsing two of them sends
    the reader somewhere useless:

    ==========  ==================================================
    RUN NOW     submit
    BLOCKED     ask for access -- no job of any shape runs here
    WRONG HW    go elsewhere; waiting will not help
    LIMIT       resize or shorten; a smaller request would clear it
    QUEUE       submit and wait
    ==========  ==================================================

    ``Placement.reachable`` is deliberately *both* "permitted" and "the shape
    is legal", so testing it here is what conflated them: a queue whose only
    problem was a per-user accelerator ceiling rendered as BLOCKED / "not
    permitted", telling the reader to go request access they already had
    instead of to ask for fewer accelerators.  ``soft_blockers`` is documented
    as exactly the blockers a smaller request would clear -- the opposite of
    not being permitted -- so it gets its own label.

    Order is by what the reader cannot work around: no access at all, then
    hardware that cannot host this shape, then a ceiling, then mere queueing.
    """
    # `starts_now`, not `runnable_now`: free nodes are not a start time. Where
    # the scheduler offers its own estimate it outranks our arithmetic, and a
    # placement with room but a queue ahead of it falls through to QUEUE --
    # "submit and wait", which is the truth and a different next move.
    if p.starts_now:
        return "RUN NOW"
    if p.fatal_blockers:
        return "BLOCKED"
    if p.verdict is not None and not p.verdict.allowed:
        # A refusal we could not obtain is not a refusal. Calling a
        # control-plane outage "not permitted" is a false statement about
        # access, and it is the same conflation a ceiling had before it got its
        # own label -- `Placement.reachable` stopped treating these as denials,
        # so the label had to stop too or the two would disagree on the row.
        return "BLOCKED" if p.verdict.durable else "NO ANSWER"
    if p.hardware_incompatible:
        # Same outcome, different remedy: fewer nodes can fix one of these and
        # nothing can fix the other.
        return "TOO FEW" if (p.capacity and p.capacity.too_few_nodes) else "WRONG HW"
    if p.soft_blockers:
        return "LIMIT"
    return "QUEUE"


def _verdict_marks(p: Placement, st: Style) -> tuple[str, str]:
    """(glyph, label) for one placement.  See :func:`_verdict_label`."""
    return _verdict_paint(_verdict_label(p), st)


def _render_placements(
    cluster: Cluster, places: list[Placement], shape: JobShape, st: Style,
    show_all: bool, turned_away: int = 0,
) -> None:
    term = cluster.queue_term
    best = next((p for p in places if p.starts_now), None)
    verdict = (
        badge("RUN NOW", "ok", st) + "  " + st.head(best.queue)
        if best is not None
        else badge("NOWHERE NOW", "warn", st) + "  " +
        st.dim("nothing can start immediately")
    )
    print(panel([shape.describe(), verdict], "job", st))
    print()

    # The ACCESS column earns its width only when it varies. With no probe run
    # AND no group-owned partition in the list it is the same word on every
    # row, which is noise -- so it is dropped and stated once in the footnote
    # instead.
    #
    # Group ownership counts as variation, and this is the important half. A
    # partition allowing one or two accounts is a research group's own hardware,
    # and without a probe there is nothing to say you are in that group. The
    # table used to report RUN NOW for three such partitions out of five rows,
    # every one of which rejects this account with "Invalid membership" -- the
    # strongest claim the tool can make, about places the reader cannot go. See
    # Queue.is_dedicated for why the allowlist is read and the accounting
    # database is not.
    probed = any(p.verdict is not None for p in places)
    dedicated = {
        name for name, q in cluster.queues.items() if q.is_dedicated
    }
    show_access = probed or any(p.queue in dedicated for p in places)

    rows = []
    for p in places:
        glyph, label = _verdict_marks(p, st)
        # A dash: there is no start time because nothing can start, which is a
        # different claim from an empty cell.
        when = st.dim(st.g.dash)
        if p.starts_now:
            when = st.ok("now")
        elif p.earliest_start:
            # Measured from when the data was taken, not when it is read: on a
            # replay those differ by however old the snapshot is.
            delta = (p.earliest_start - (cluster.taken_at or datetime.now())
                     ).total_seconds()
            when = format_wait(delta)
            if not p.start_estimate_from_scheduler:
                when += st.dim("*")
        if p.verdict is not None:
            if p.verdict.allowed:
                access = st.ok("confirmed")
            else:
                # warn, not bad, when the answer never arrived: red on a
                # control-plane outage reads as "you are denied".
                paint = st.bad if p.verdict.durable else st.warn
                access = paint(p.verdict.category)
        elif p.queue in dedicated:
            # Somebody's own hardware. Not a refusal -- we genuinely do not know
            # -- but presenting it beside a shared partition with no distinction
            # is what made three unreachable rows read as RUN NOW.
            access = st.warn("group-only")
        elif p.entitlement_unconfirmed:
            # The backend has no dry-run at all: nothing could be confirmed.
            #
            # Checked AFTER group ownership, not before. Both are true, the
            # column has one slot, and "group-only" is the far more specific
            # and actionable of the two -- while "no dry-run exists anywhere"
            # is already stated once in the footer. Getting this order wrong
            # silently dropped the marker on every backend without a dry-run
            # (PBS, LSF, the ssh pool), which is precisely where there is no
            # probe to fall back on.
            access = st.dim("declared")
        else:
            # A dry-run exists and was not run. "unchecked" is the honest word:
            # a bare dash reads as "no information available" when in fact the
            # information is one flag away.
            access = st.dim("unchecked")
        cap = p.capacity
        # With a denominator: "1" next to reasons accounting for 10 other nodes
        # is arithmetic the reader cannot close, and reads as though the ten
        # were unexplained.
        # The denominator is muted here as everywhere else: it is the total the
        # number in front of it divides, not a second measurement.
        capable = (
            f"{len(cap.hardware_nodes)}" + st.muted(f"/{cap.considered}")
            if cap else "?"
        )
        if cap and cap.unverified_nodes:
            # Set aside, not incapable: worth showing next to the count it is
            # missing from, so the number is not read as the whole story.
            capable += st.warn(f"+{len(cap.unverified_nodes)}?")
        accel = ", ".join(
            f"{k}x{v}" for k, v in list(p.accelerator_models.items())[:2]
        ) or st.dim(st.g.dash)
        row = [
            glyph, label, p.queue,
            f"{p.nodes_available}" + st.muted(f"/{shape.nodes}"),
            capable,
            when,
        ]
        if show_access:
            row.append(access)
        row.append(accel)
        rows.append(row)

    # Renamed to be self-explanatory, which is what let the glossary go:
    # "FREE 3/1" needed a legend to say "fits now / needed"; "fits 3/1" does
    # not. "CAPABLE" needed one to say "right hardware at all"; "right hw"
    # says it. The start column carries its own asterisk.
    headers = ["", "", term, "fits/need", "right hw", "start"]
    aligns = ["left", "left", "left", "right", "right", "right"]
    limits = [0, 0, 26, 0, 0, 0]
    if show_access:
        headers.append("access")
        aligns.append("left")
        limits.append(0)
    headers.append("gpus")
    aligns.append("left")
    limits.append(30)

    note = f"{plural(len(places), term)} considered"
    if turned_away:
        note += f"  {st.g.sep}  {turned_away} refused you"
    print(section("placements", st, note))
    print(_grid(headers, rows, aligns, st, indent="  ", limits=limits))
    print()
    shown = {_verdict_label(p) for p in places}
    legend = [
        f"{_verdict_paint(label, st)[0]} {gloss}"
        for label, (_g, _r, gloss) in _VERDICT_LEGEND.items()
        if label in shown
    ]
    # One item on the line that already exists, not a line of its own. A glyph
    # cannot name itself, so the marks get a key; the columns were renamed so
    # they do not need one.
    if any("*" in str(r) for row in rows for r in row):
        legend.append(f"* {st.dim('our estimate, not the scheduler' + chr(39) + 's')}")
    # columns() flows the legend onto more rows when the window is narrow, and
    # measures display width so the coloured glyphs do not throw the layout.
    print(columns(legend, st, indent="  "))
    # No column legend. Three lines defining FREE, CAPABLE and START* is a
    # glossary for a five-row table, and a column that needs a definition needs
    # a better name -- so they were renamed instead. The verdict glyphs keep
    # their one-line key above, because a glyph genuinely cannot name itself.
    if not cluster.can_probe:
        # wrap_indent, not flow: this is prose, and flow clips an item too wide
        # for one line rather than wrapping it -- which would drop the
        # explanation and leave a bare "not confirmed". The line ran to 148
        # columns hand-joined, and it renders only on a replay or a backend with
        # no dry-run, which is why every live width sweep missed it.
        print(st.warn(wrap_indent(
            f"{st.g.warn} access is DECLARED, not confirmed "
            f"{st.g.sep} {_why_no_probe(cluster)}",
            indent="  ",
        )))
    # Nothing printed when a dry-run exists and was not run: the `access`
    # column already says "unchecked" on every row, and a sentence repeating it
    # is the kind of line that made this view unreadable.

    # Per-queue detail; caveats shared by every row are hoisted to a footnote,
    # because the same sentence printed nine times is not nine facts.
    seen: dict[str, int] = {}
    for p in places:
        for c in p.caveats:
            seen[c] = seen.get(c, 0) + 1
    shared = {c for c, n in seen.items() if n > 1}

    for p in places:
        own = [c for c in p.caveats if c not in shared]
        hw_note = p.hardware_incompatible or (
            p.capacity and p.capacity.capable_but_all_unavailable
        )
        bad_verdict = p.verdict is not None and not p.verdict.allowed
        if not (p.blockers or own or hw_note or bad_verdict):
            continue
        if not show_all and p.starts_now and not own:
            continue
        items: list[tuple[str, str]] = []
        for b in p.blockers:
            tag = st.bad("fatal") if b.fatal else st.warn("shape")
            items.append((f"{tag} {st.dim(b.code)}", b.detail))
        if bad_verdict and p.verdict is not None:
            durable = p.verdict.durable
            tag = st.bad("checked") if durable else st.warn("unanswered")
            items.append((f"{tag} {st.dim(p.verdict.category)}", p.verdict.reason))
        if hw_note and p.capacity:
            label = ("no node here has the right hardware"
                     if p.hardware_incompatible
                     else "the capable nodes are all down or drained")
            detail = "; ".join(
                f"{plural(n, 'node')}: {r}"
                for r, n in list(p.capacity.hardware_reasons.items())[:4]
            )
            items.append((f"{st.bad('hardware')} {st.dim(label)}", detail))
        for c in own:
            items.append((st.dim("note"), c))
        print()
        print(f"  {st.head(p.queue)}")
        print(tree(items, st, indent="    "))

    if shared:
        # A tag in the default view, the prose behind --all -- the same split
        # rdu uses for its audit. These are true and worth having, and they are
        # four lines of explanation nobody reads while looking for a partition
        # name.
        print()
        if show_all:
            print(section(f"applies to every {term} above", st))
            print(tree([(st.dim("note"), c) for c in sorted(shared)], st))
        else:
            print("  " + st.warn("NOTES") + st.dim(
                f"  {len(shared)} {st.g.sep} --all"))


def cmd_where(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    shape = _shape_from_args(args)
    wanted = [x.strip() for x in args.queue.split(",") if x.strip()] or None
    accounts = [x.strip() for x in args.accounts.split(",") if x.strip()]
    if not accounts:
        if args.account:
            accounts = [args.account]
        elif cluster.identity:
            accounts = list(cluster.identity.accounts)
    # Probing by default, exactly as `status` does, and for the same measured
    # reason: without it this command reported four partitions as RUN NOW and a
    # dry-run accepted one. Three of the four rows were the strongest claim the
    # tool can make, about places that refuse this account.
    probe = not args.declared and cluster.can_probe
    places = rank(
        cluster, shape, queues=wanted, use_probe=probe,
        accounts=accounts or None, include_unusable=args.all,
    )
    turned_away = 0
    if probe and not args.all:
        before = len(places)
        # Only a SETTLED refusal hides a placement. A verdict whose category is
        # transient -- the control plane unreachable, the client missing, an
        # answer we could not parse -- means the question was not answered, and
        # treating that as "no" would empty the screen every time the scheduler
        # hiccups, exactly when the reader most needs to see their options.
        places = [
            p for p in places
            if p.verdict is None or p.verdict.allowed or not p.verdict.durable
        ]
        turned_away = before - len(places)
    # A stable nudge, applied after ranking rather than inside it: among
    # placements that are equally good on capacity, a shared partition beats a
    # group's private hardware, because one of them you can actually submit to.
    # A confirmed verdict overrides the guess -- if a probe says you are in the
    # group, the partition is not second-class.
    def _second_class(place):
        q = cluster.queues.get(place.queue)
        # Only an ACCEPTANCE is consulted here. Refusals are already handled
        # twice over -- `Placement.score` orders confirmed placements ahead of
        # unconfirmed ones, and the default view filters durable refusals out
        # entirely -- so re-checking them here duplicated `rank` and made this
        # branch untestable in isolation.
        if place.verdict is not None and place.verdict.allowed:
            return False
        return bool(q is not None and q.is_dedicated)

    places.sort(key=lambda pl: (not pl.starts_now, _second_class(pl)))
    exit_ok = any(p.reachable and not p.hardware_incompatible for p in places)

    if args.json:
        print(json.dumps([{
            "queue": p.queue,
            # Two different facts, and they disagree often enough to be worth
            # both: `runnable_now` is "nodes of this shape are free", while
            # `starts_now` also asks the scheduler whether anything is ahead of
            # you. On this cluster `amd` was runnable_now with 1687 free cores
            # and four and a half hours from starting a four-core job.
            "runnable_now": p.runnable_now,
            "starts_now": p.starts_now,
            "reachable": p.reachable,
            "confirmed": p.confirmed,
            "entitlement_unconfirmed": p.entitlement_unconfirmed,
            "hardware_incompatible": p.hardware_incompatible,
            "nodes_free": p.nodes_available,
            "nodes_capable": len(p.capacity.hardware_nodes) if p.capacity else 0,
            "nodes_unverified": (
                len(p.capacity.unverified_nodes) if p.capacity else 0
            ),
            "earliest_start": p.earliest_start.isoformat() if p.earliest_start else None,
            "start_from_scheduler": p.start_estimate_from_scheduler,
            "accelerator_models": p.accelerator_models,
            "blockers": [{"code": b.code, "detail": b.detail, "fatal": b.fatal}
                         for b in p.blockers],
            "verdict": None if p.verdict is None else {
                "allowed": p.verdict.allowed,
                "category": p.verdict.category,
                "reason": p.verdict.reason,
                "filter_verdict": p.verdict.filter_verdict,
                "effective_qos": p.verdict.effective_qos,
            },
            "caveats": p.caveats,
            "submit_flags": cluster.submit_flags(p.queue, shape),
        } for p in places], indent=2, default=str))
        return 0 if exit_ok else 1

    if not places:
        print(st.bad(f"no {cluster.queue_term} can run this shape."))
        print(st.dim(wrap_indent(
            "re-run with --all to see every one and its blockers.", indent="  ")))
        return 1
    _render_placements(cluster, places, shape, st, args.all, turned_away)
    # Show the flags for the best option there is, not only for a run-now one:
    # "it would queue on gn" is still the answer you act on, and having to
    # reconstruct the flags by hand is where a mismatch with what was actually
    # checked creeps in.
    best = next((p for p in places if p.starts_now), None)
    note = "starts now"
    if best is None:
        best = next((p for p in places if p.reachable), None)
        note = "queues"
    if best is not None:
        flags = cluster.submit_flags(best.queue, shape)
        if flags:
            print()
            print("  " + st.dim(f"submit ({note})"))
            # Deliberately NOT wrapped or truncated. This line exists to be
            # copied, and both an ellipsis and a hanging indent would hand the
            # reader a broken command. A narrow terminal soft-wraps it, which
            # still pastes correctly.
            print("  " + st.accent(" ".join(flags)))
    return 0 if exit_ok else 1


def _check_notes(cluster: Cluster, shape: JobShape) -> list[str]:
    """What a dry-run cannot answer for, whatever the verdicts say.

    The hardware-capability flags are not part of any submission -- no batch
    system can express "I need bf16" or "I need 40 GiB of HBM" -- so the
    control plane was never asked about them. Accepting the flag and silently
    ignoring it would answer a different question than the one typed.
    """
    caps = cluster.capabilities
    notes = list(caps.notes) if caps and caps.notes else []
    ignored = []
    if shape.requires:
        ignored.append("--needs " + ",".join(shape.requires))
    if shape.gpu_memory_gb:
        ignored.append(f"--gpu-mem {shape.gpu_memory_gb:g}")
    if ignored:
        notes.insert(0, (
            f"{' and '.join(ignored)} took no part in this check: no scheduler "
            f"can express a dtype or a GPU memory size, so the control "
            f"plane was never asked about them. Use 'nodetop where' for those."
        ))
    return notes


def cmd_check(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    shape = _shape_from_args(args)
    if not cluster.can_probe:
        msg = (
            f"nothing to ask: {_why_no_probe(cluster)}. Entitlement can only be "
            f"read from declared ACLs -- use 'nodetop where' for that."
        )
        if args.json:
            print(json.dumps({"can_probe": False, "reason": msg}, indent=2))
        else:
            print(f"{st.warn(st.g.warn)} {st.warn(msg)}")
        return 2

    queues = [x.strip() for x in args.queue.split(",") if x.strip()]
    if not queues:
        queues = [
            q.name for q in cluster.usable_queues()
            if not shape.needs_gpu or any(n.is_gpu_node for n in q.nodes)
        ]
    accounts: list = [x.strip() for x in args.accounts.split(",") if x.strip()]
    if not accounts:
        accounts = (
            [args.account] if args.account
            else list(cluster.identity.accounts if cluster.identity else [None]) or [None]
        )

    # The same probe loop `where` uses, from the same helper -- narrowed to the
    # accounts each queue's allowlist admits, ordered so an account already
    # known to work is tried first, and bounded by one budget for the whole
    # question. This loop used to be a second copy with no ceiling at all,
    # which was survivable only while `probe_accounts` truncated to four
    # candidates: the moment that truncation went (it was hiding usable
    # partitions), a bare `nodetop check` could have fired 34 accounts against
    # 84 queues. Cheapest queues first, for the same reason as in `rank`.
    budget = ProbeBudget(queues=len(queues))
    results = {}
    queues = sorted(queues, key=lambda n: (
        len(probe_accounts(cluster.queues[n], accounts, shape))
        if n in cluster.queues else 0, n))
    for name in queues:
        best, tried, of = probe_queue(
            cluster, cluster.queues.get(name), name, shape, accounts, budget
        )
        best = unsettled(best, tried, of)
        if best is not None:
            results[name] = best

    # Exit status has to be usable in `nodetop check ... && sbatch ...`: zero
    # only when the control plane actually accepted something. Returning zero
    # after a total refusal would wave the caller through.
    accepted = sum(1 for v in results.values() if v.allowed)
    # Counted apart from the refusals: "1 of 3 accepted" hides that one of the
    # other two was never actually asked. The exit status still treats it as
    # not-accepted -- waving a caller through on an unanswered probe is the one
    # outcome worth being strict about.
    unanswered = sum(1 for v in results.values()
                     if not v.allowed and not v.durable)
    status = 0 if accepted else 1

    # Built once and shared by both renderers: a caveat that appears only in
    # the text is a caveat a script never learns, and a script is the consumer
    # most likely to act on the answer.
    notes = _check_notes(cluster, shape)
    disagreements = [
        q for q, v in results.items()
        if v.filter_verdict == "PASSED" and not v.allowed
    ]

    if args.json:
        print(json.dumps({
            "accepted": accepted,
            "unanswered": unanswered,
            "asked": len(results),
            "not_covered": notes,
            "filter_scheduler_disagreements": disagreements,
            "queues": {k: {
                "allowed": v.allowed, "category": v.category, "reason": v.reason,
                "account": v.account, "filter_verdict": v.filter_verdict,
                "effective_qos": v.effective_qos,
                "predicted_start": (
                    v.predicted_start.isoformat() if v.predicted_start else None),
                "predicted_nodes": list(v.predicted_nodes),
            } for k, v in results.items()},
        }, indent=2))
        return status

    caps = cluster.capabilities
    count = (
        st.ok(str(accepted)) if accepted else st.bad(str(accepted))
    )
    print(panel([
        shape.describe(),
        f"{st.dim(caps.probe_command if caps else 'dry-run')}   "
        f"{st.dim(st.g.sep)}   {count} of {len(results)} accepted"
        + (st.warn(f", {unanswered} unanswered") if unanswered else "")
        + "   "
        f"{st.dim(st.g.sep)}   {st.dim('read-only: nothing is created')}",
    ], "control-plane check", st))
    print()

    rows = []
    for name in sorted(results, key=lambda n: (not results[n].allowed, n)):
        r = results[name]
        rows.append([
            st.ok(st.g.ok) if r.allowed else st.bad(st.g.off),
            name, r.account or st.dim(st.g.dash),
            r.effective_qos or st.dim(st.g.dash),
            r.filter_verdict or st.dim(st.g.dash),
            "" if r.allowed else st.bad(r.category),
            r.predicted_start.strftime("%m-%d %H:%M")
            if r.predicted_start and r.allowed else "",
        ])
    print(_grid([
        "", cluster.queue_term.upper(), "ACCOUNT", "EFFECTIVE QOS",
        "FILTER", "CATEGORY", "START",
    ], rows, style=st, indent="  ", limits=[0, 24, 20, 20, 0, 0, 0]))

    disagree = [results[q] for q in disagreements]
    if disagree:
        print()
        print(section("the submit filter and the scheduler disagree", st))
        print(_note(
            "The site filter reported PASSED for these and the scheduler refused "
            "them anyway. Reading only the filter's verdict gives the opposite of "
            "the truth.", st))
        print(tree([(st.bad(r.queue), r.reason) for r in disagree], st))
    if not results:
        print()
        print(st.warn(wrap_indent(
            f"{st.g.warn} nothing to check: no {cluster.queue_term} has hardware "
            f"for this shape", indent="  ")))
    if notes:
        # No per-item "note" tag: the heading already says these are notes, and
        # a tree of one-word tags is scaffolding around the actual sentence.
        print()
        print(section("not covered", st, "by the dry-run"))
        for n in notes:
            print(st.dim(wrap_indent(
                n, indent="    ", first=f"  {st.g.dash} ",
            )))
    return status


def cmd_exclude(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    nodes = cluster.nodes
    if args.queue:
        wanted = {x.strip() for x in args.queue.split(",") if x.strip()}
        keep = {n.name for q in cluster.queues.values() if q.name in wanted for n in q.nodes}
        nodes = [n for n in nodes if n.name in keep]
    picked: set[str] = set()
    if args.gpu_nodes:
        picked |= {n.name for n in nodes if n.is_gpu_node}
    if args.unschedulable:
        picked |= {n.name for n in nodes if not n.schedulable}
    if args.degraded:
        picked |= {n.name for n in nodes if n.degraded}
    if not (args.gpu_nodes or args.unschedulable or args.degraded):
        print("pick at least one of --gpu-nodes / --unschedulable / --degraded",
              file=sys.stderr)
        return 2
    nodelist = cluster.format_nodelist(sorted(picked))
    if args.json:
        print(json.dumps({"count": len(picked), "nodelist": nodelist,
                          "nodes": sorted(picked)}, indent=2))
        return 0
    if not nodelist:
        print(st.dim("(no matching nodes)"))
        return 0
    print(nodelist)
    # A scattered set does not compress, and a scheduler will reject an
    # argument this long before it ever reads the node names. Said on stderr so
    # the nodelist itself stays pipeable.
    if len(nodelist) > 4096:
        print(
            st.warn(
                f"{st.g.warn} {len(nodelist)} characters for {len(picked)} nodes: "
                f"this set does not compress into bracket notation and is likely "
                f"too long for a command line -- narrow it with -q, or write it "
                f"to a file"
            ),
            file=sys.stderr,
        )
    return 0


#: Capabilities worth reporting reach for, in the order a decision needs them.
_CAPABILITIES = ("bf16", "fp8", "tf32", "flash_attention")


def cmd_accelerators(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    """Inventory by model, and how much of the cluster can do what.

    This is the one question no scheduler can answer: "how many of these
    accelerators support the dtype my job needs?"  The count that matters is
    per *accelerator*, not per node, and it excludes unschedulable nodes --
    hardware behind a drained node is not reach.
    """
    nodes = [n for n in cluster.nodes if n.is_gpu_node]
    cluster_total = sum(n.gpus_total for n in nodes)
    if args.queue:
        wanted = {x.strip() for x in args.queue.split(",") if x.strip()}
        keep = {n.name for q in cluster.queues.values() if q.name in wanted
                for n in q.nodes}
        nodes = [n for n in nodes if n.name in keep]
    else:
        # "358 GPUs" is true of the cluster and false of what the reader can
        # use; 230 of them are in partitions this account is on.
        entitled, _ = _entitled_queues(cluster, args)
        if entitled is not None:
            reachable = {n.name for name in entitled
                         for n in cluster.queues[name].nodes}
            nodes = [n for n in nodes if n.name in reachable]

    # Group by model, keeping "unidentifiable" as its own honest bucket rather
    # than folding it into any capability claim.
    groups: dict[str, list] = {}
    for n in nodes:
        groups.setdefault(n.accelerator.model if n.accelerator else "UNKNOWN", []).append(n)

    # Free means reachable-and-free. A schedulable node whose only partition is
    # DOWN has no free accelerators, however idle it looks -- the same
    # correction `Cluster.summary` needed, and for the same reason: this view
    # computes its own totals rather than asking the queue.
    live = {n.name for n in cluster.reachable_nodes()}

    def totals(group: list) -> tuple[int, int, int]:
        return (
            sum(x.gpus_total for x in group),
            sum(x.effective_free_gpus for x in group
                if x.schedulable and x.name in live),
            len(group),
        )

    reach: dict[str, tuple[int, int]] = {}
    installed = sum(n.gpus_total for n in nodes)
    for cap in _CAPABILITIES:
        able = [
            n for n in nodes
            if n.accelerator is not None and supports(n.accelerator, cap) is True
        ]
        reach[cap] = (
            sum(n.gpus_total for n in able),
            sum(n.effective_free_gpus for n in able
                if n.schedulable and n.name in live),
        )
    unknown = sum(n.gpus_total for n in nodes if n.accelerator is None)

    if args.json:
        print(json.dumps({
            "accelerators_installed": installed,
            "accelerators_unidentifiable": unknown,
            "models": {
                model: {
                    "vendor": g[0].accelerator.vendor if g[0].accelerator else None,
                    "arch": g[0].accelerator.arch if g[0].accelerator else None,
                    "memory_gb": g[0].accelerator.memory_gb if g[0].accelerator else None,
                    "memory_inferred": (
                        not g[0].accelerator.memory_certain if g[0].accelerator else None
                    ),
                    "installed": totals(g)[0],
                    "free": totals(g)[1],
                    "nodes": totals(g)[2],
                    "capabilities": {
                        c: supports(g[0].accelerator, c) for c in _CAPABILITIES
                    },
                    # Where they are, not just how many: the text form grew
                    # this column because a correct inventory that cannot say
                    # which queue holds a model reads as a wrong one.
                    f"{cluster.queue_term}s": sorted(
                        q.name for q in cluster.queues.values()
                        if any(n.name in {x.name for x in g} for n in q.nodes)
                    ),
                }
                for model, g in groups.items()
            },
            "capability_reach": {
                c: {"installed": i, "free": f} for c, (i, f) in reach.items()
            },
        }, indent=2))
        return 0

    vendors = {
        n.accelerator.vendor for n in nodes if n.accelerator is not None
    }
    print(panel([
        f"{st.accent(str(installed), bold=True)} GPUs"
        + (st.dim(f" of {cluster_total} on the cluster")
           if cluster_total != installed else "")
        + f"   {st.dim(st.g.sep)}   "
        f"{len(groups)} models   {st.dim(st.g.sep)}   "
        f"{len(vendors)} vendor{'s' if len(vendors) != 1 else ''}   "
        f"{st.dim(st.g.sep)}   {len(nodes)} nodes",
    ], "GPU inventory", st))

    def homes(group: list) -> str:
        """Which queues hold this model, busiest first.

        The inventory could say a cluster has 92 A100s and not where a single
        one of them is, so "which partition has the A100s?" took a `scontrol`
        sweep to answer -- and being unable to answer it is what makes a
        correct listing look wrong. Queues the caller cannot use are left out:
        this column exists to be acted on.
        """
        held = {n.name for n in group}

        def holders(pick) -> list[str]:
            counted = [(sum(1 for n in q.nodes if n.name in held), q.name)
                       for q in cluster.queues.values() if pick(q)]
            return [name for n, name in sorted(counted, reverse=True) if n]

        named = holders(lambda q: q.usable and not q.is_dedicated)
        if named:
            return ", ".join(named[:3]) + (st.dim(f" +{len(named) - 3}")
                                           if len(named) > 3 else "")
        # Nowhere open: say where the hardware IS rather than leaving the cell
        # blank, which reads as "nowhere at all". Marked, because a group's own
        # partition is not somewhere the reader can submit.
        owned = holders(lambda q: q.is_dedicated and q.usable)
        if not owned:
            return ""
        return st.dim("group-only: ") + ", ".join(owned[:2]) + (
            st.dim(f" +{len(owned) - 2}") if len(owned) > 2 else "")

    rows = []
    for model, group in sorted(groups.items(), key=lambda kv: -totals(kv[1])[0]):
        total, free, count = totals(group)
        spec = group[0].accelerator
        if spec is None:
            rows.append([
                st.warn(model), st.dim(st.g.dash), st.dim(st.g.dash),
                st.dim(st.g.dash),
                count, gauge(free, total, 9, st),
                st.dim("unknown"), st.dim("unknown"), st.muted(homes(group)),
            ])
            continue
        # `>=`, not a bare `?`. The part ships in more than one size, the
        # smaller was assumed, and "at least 40G" is what that means -- a
        # question mark needs a legend this table has no room for.
        mem = (st.dim(">=") if not spec.memory_certain else "") + f"{spec.memory_gb}G"
        rows.append([
            st.accent(spec.model), spec.vendor, spec.arch, mem, count,
            gauge(free, total, 9, st),
            st.ok("yes") if spec.bf16 else st.dim("no"),
            st.ok("yes") if spec.fp8 else st.dim("no"),
            st.muted(homes(group)),
        ])
    print()
    print(section("by model", st))
    print(_grid(
        ["MODEL", "VENDOR", "ARCH", "MEM", "NODES", "FREE", "BF16", "FP8",
         f"{cluster.queue_term.upper()}S"],
        rows, ["left", "left", "left", "right", "right", "left", "left", "left",
               "left"],
        st, indent="  ", limits=[0, 0, 0, 0, 0, 0, 0, 0, 30],
    ))

    if not installed:
        print()
        # Two different answers wear the same empty table, and saying the wrong
        # one is worse than saying nothing. A cluster with no accelerators at
        # all has nothing to report; a cluster whose accelerators are all in
        # partitions this account is not on has plenty to report and one thing
        # to do about it. The header directly above already counted them ("0
        # GPUs of 8 on the cluster"), so "no GPUs found in this cluster"
        # contradicted the line above it -- observed verbatim where both H100
        # partitions were closed to the caller.
        if cluster_total and not args.queue:
            where = sorted(
                q.name for q in cluster.queues.values()
                if any(n.is_gpu_node for n in q.nodes)
            )
            print(_note(
                f"{cluster_total} {plural(cluster_total, 'GPU')} on this cluster, none of "
                f"them in a {cluster.queue_term} you may submit to"
                + (f" -- they are in {', '.join(where[:4])}" if where else "")
                + ".  --all inventories them anyway", st))
        elif cluster_total:
            print(_note(
                f"no GPUs in that {cluster.queue_term}; the cluster has "
                f"{cluster_total} elsewhere", st))
        else:
            print(_note(
                "no GPUs found in this cluster, so there is nothing to report", st))
        return 0

    print()
    print(section("features", st, f"share of all {installed}"))
    # The header says "installed", so the rows do not repeat it. The bar gives
    # ground first in a narrow window: the counts are the measurement, the bar
    # only shows their shape.
    labels = {}
    for cap in _CAPABILITIES:
        total, free = reach[cap]
        labels[cap] = f"{total}/{installed}" + (f" {st.g.sep} {free} free" if total else "")
    window = term_width()
    widest = max(width(v) for v in labels.values())
    bar_w = max(6, min(18, window - 2 - 6 - 2 - widest))
    for cap in _CAPABILITIES:
        total, _ = reach[cap]
        share = (total / installed) if installed else 0.0
        print(
            f"  {st.dim(cap.replace('_attention', '').ljust(6))} "
            f"{bar(share, bar_w, st)} {st.dim(labels[cap])}"
        )
    if unknown:
        print()
        print(st.warn(wrap_indent(
            f"{st.g.warn} {unknown} {plural(unknown, 'accelerator')} could not be "
            f"identified and are counted in NO capability row", indent="  ")))
    return 0


def cmd_snapshot(cluster: Cluster, args: argparse.Namespace, st: Style) -> int:
    """Write everything the queries returned, for analysis after the fact.

    The point is post-mortem: when a partition goes down mid-run, the evidence
    is gone by the time anyone looks. A snapshot keeps it, and every command
    works against it unchanged.
    """
    runner = cluster.capture
    if runner is None:  # pragma: no cover - set by main() before dispatch
        print("nothing was captured", file=sys.stderr)
        return 2
    if runner is None:  # pragma: no cover - main() always supplies one
        print("snapshot needs a capturing runner", file=sys.stderr)
        return 2
    payload = {
        "nodetop": VERSION,
        "backend": cluster.backend_name,
        "queue_term": cluster.queue_term,
        "captured_at": (cluster.taken_at or datetime.now()).isoformat(),
        "errors": cluster.errors,
        "commands": {
            key: {"rc": rc, "stdout": out, "stderr": err}
            for key, (rc, out, err) in sorted(runner.captured.items())
        },
    }
    text = json.dumps(payload, indent=1)
    if args.output == "-":
        print(text)
        return 0

    pathlib.Path(args.output).write_text(text)
    queries = len(payload["commands"])
    size = len(text) / 1024
    # To stderr, so `nodetop snapshot -o file` stays pipe-safe.
    print(
        f"{st.accent(st.g.bullet)} wrote {st.head(args.output)} "
        f"{st.dim(f'({size:.0f} KiB, {queries} queries)')}",
        file=sys.stderr,
    )
    print(
        st.dim(f"  replay with:  nodetop --replay {args.output} status"),
        file=sys.stderr,
    )
    return 0


def _load_replay(path: str) -> tuple[Backend, str, datetime | None]:
    """Rebuild a backend from a snapshot file, with the time it was captured.

    Replay needs no special code path in the analysis layer: the same backend
    runs against a recorded runner instead of a live one.  The capture time does
    have to come back out, though -- the snapshot writes ``captured_at`` and it
    was being ignored, so a replay dated itself to the moment it was read.
    """
    from .runner import RecordedRunner

    data = json.loads(pathlib.Path(path).read_text())
    commands = data.get("commands") or {}
    responses = {
        key: (entry.get("rc", 0), entry.get("stdout", ""), entry.get("stderr", ""))
        for key, entry in commands.items()
    }
    name = data.get("backend") or ""
    return (
        backend_registry.get(name, RecordedRunner(responses)),
        name,
        parse_timestamp(data.get("captured_at")),
    )


_COMMANDS = {
    "status": cmd_status,
    "queues": cmd_queues,
    "partitions": cmd_queues,
    "nodes": cmd_nodes,
    "zoom": cmd_zoom,
    "in": cmd_zoom,
    "health": cmd_health,
    "where": cmd_where,
    "fit": cmd_where,
    "check": cmd_check,
    "probe": cmd_check,
    "exclude": cmd_exclude,
    "accelerators": cmd_accelerators,
    "accel": cmd_accelerators,
    "gpus": cmd_accelerators,
    "snapshot": cmd_snapshot,
}


#: The floor `pyproject.toml` declares. Repeated here because the interpreter
#: that runs this file is often not the one that installed it.
_MIN_PYTHON = (3, 10)


def main(argv: Sequence[str] | None = None) -> int:
    # Said once, plainly, before anything else runs. `pip` enforces the floor at
    # install time, but the way this tool actually arrives on a login node is a
    # `git clone` and a `PYTHONPATH=src python3 -m nodetop` -- and there
    # `python3` is whatever the distribution ships, which on RHEL 9 is 3.9.
    # Nothing here fails to *import* on 3.9, so the tool ran, every query threw
    # `TypeError: zip() takes no keyword arguments`, and the report that came
    # out was "every query failed, so there is nothing to report" -- a sentence
    # about the cluster, printed on a healthy cluster, because of the
    # interpreter. Observed on a Slurm 25.11 site whose system python3 is 3.9.
    if sys.version_info < _MIN_PYTHON:
        have = ".".join(str(v) for v in sys.version_info[:3])
        want = ".".join(str(v) for v in _MIN_PYTHON)
        print(
            f"nodetop needs Python {want} or newer; this is {have} "
            f"({sys.executable}).\n"
            f"Nothing else is wrong -- try a newer interpreter, e.g. "
            f"python3.11 -m nodetop",
            file=sys.stderr,
        )
        return 2
    parser = build_parser()
    # `parse_known_args`, not `parse_args`, because the verb is optional.
    #
    # A flag belonging to the default verb is not a flag the ROOT parser knows,
    # so a strict first parse rejected it before the default-verb fallback below
    # could ever run: `nodetop --all` died with "unrecognized arguments: --all"
    # while `nodetop status --all` worked. That is the exact command the
    # overview's own footer tells you to run, so the tool was advising something
    # it then refused -- and the bare invocation is the one people actually type.
    args, unknown = parser.parse_known_args(argv)
    if args.command is None:
        # Re-parse with the default verb spelled out. The root parser has none
        # of the sub-command's own flags, so dispatching a root-parsed
        # namespace into cmd_status left it missing attributes the renderer
        # reads -- a bare `nodetop` crashed on args.all. This parse is strict,
        # so a genuinely misspelled flag still gets the usual error, now naming
        # `status` and listing its flags rather than the root's.
        argv = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(["status", *argv])
    elif unknown:
        # A verb WAS given, so leftovers are a mistake and must not be swallowed:
        # the whole hazard of parse_known_args is that it makes a typo look like
        # a successful run.
        parser.error("unrecognized arguments: " + " ".join(unknown))
    command = args.command
    st = Style(
        enabled=False if args.no_color else None,
        glyphs=Glyphs.ascii() if args.ascii else None,
    )

    if command == "backends":
        return cmd_backends(None, args, st)

    if args.replay:
        try:
            backend, name, captured = _load_replay(args.replay)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cannot replay {args.replay}: {exc}", file=sys.stderr)
            return 2
        cluster = Cluster.load(
            backend, with_free_times=True, replayed=True, taken_at=captured,
        )
        # Say how old it is. Every number below is as-of that moment, and a
        # snapshot read as if it were current is the whole hazard of a replay.
        age = ""
        if captured is not None:
            elapsed = format_age((datetime.now() - captured).total_seconds())
            age = (
                f", captured {elapsed} ago" if elapsed
                else f", stamped {captured:%Y-%m-%d %H:%M} -- ahead of this host's clock"
            )
        print(
            st.dim(f"replaying {args.replay} ({name}{age})"),
            file=sys.stderr,
        )
        # Same guard as the live path below. A snapshot holds whichever queues
        # existed when it was taken, so a name that is merely absent from the
        # recording is exactly as worth reporting as a typo.
        if bad := _reject_broken_snapshot(cluster, command):
            return bad
        if bad := _reject_unknown_queues(cluster, args, st):
            return bad
        return _COMMANDS[command](cluster, args, st)

    try:
        if args.backend:
            backend = backend_registry.get(args.backend)
            # Forcing a backend is allowed even when detection says no -- the
            # detector is deliberately conservative and a caller may know
            # better. But say so once, up front: otherwise a missing client
            # surfaces as five cryptic query failures further down.
            if not type(backend).detect():
                print(
                    f"warning: forced backend {args.backend!r} does not detect its "
                    f"system here; queries will probably fail",
                    file=sys.stderr,
                )
        else:
            backend = backend_registry.detect()
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except NoBackendError as exc:
        print(str(exc), file=sys.stderr)
        print("run 'nodetop backends' to see what was looked for.", file=sys.stderr)
        return 3

    # One snapshot serves the whole command, so every number in the output
    # describes the same instant.
    if command == "snapshot":
        # Wrap the real runner so every query is recorded as it happens; a
        # hand-maintained command list would drift from what is actually run.
        from .runner import CapturingRunner

        capture = CapturingRunner()
        backend = backend_registry.get(backend.name, capture)
        cluster = Cluster.load(backend, with_free_times=True)
        cluster.capture = capture
        return cmd_snapshot(cluster, args, st)

    cluster = Cluster.load(backend, with_free_times=command in {"where", "fit", "status"})
    if bad := _reject_broken_snapshot(cluster, command):
        return bad
    if bad := _reject_unknown_queues(cluster, args, st):
        return bad
    return _COMMANDS[command](cluster, args, st)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
