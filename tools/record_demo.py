"""Render the README's demo GIF by driving the real UI.

Not shipped in the wheel: a maintenance script. It builds a synthetic cluster --
plausible shapes, invented names -- runs the actual interactive browser over it,
captures every frame the terminal would have received, and draws them.

Synthetic on purpose. A recording of a live cluster puts other people's
usernames, accounts and job names into a public image, and there is no way to
take that back once it is pushed.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nodetop.interactive as interactive  # noqa: E402
from nodetop.cli import build_parser, cmd_status  # noqa: E402
from nodetop.core.cluster import Cluster  # noqa: E402
from nodetop.core.model import (  # noqa: E402
    Allocation,
    BackendCapabilities,
    Identity,
    Job,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)
from nodetop.render import Glyphs, Style  # noqa: E402

FONT = ("/software/python-miniforge-25.3.0-el8-x86_64/envs/AI/lib/python3.11/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf")
FONT_BOLD = FONT.replace("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")

BG = (13, 17, 23)
FG = (201, 209, 217)

# --- the cluster ------------------------------------------------------------
#: (partition, nodes, cores/node, gpus/node, model, busy fraction, accessible)
SHAPE = [
    ("compute",   40, 128, 0, None,      0.63, "open"),
    ("gpu-a",     44,  32, 4, "A100",    0.55, "open"),
    ("wide",     190,  48, 0, None,      0.95, "open"),
    ("gpu-b",      9,  48, 4, "V100",    0.61, "open"),
    ("bigmem",     2,  48, 0, None,      0.50, "open"),
    ("build",      1,  48, 0, None,      0.29, "open"),
    ("astro",     10,  64, 0, None,      0.40, "refused"),
    ("chem-gpu",   6,  48, 8, "H100",    0.20, "no access"),
    ("legacy",    12,  24, 0, None,      0.00, "down"),
]


def cluster() -> Cluster:
    from nodetop.core.hardware import ACCELERATORS

    nodes: list[Node] = []
    queues: dict[str, Queue] = {}
    for part, count, cores, gpus, model, busy, kind in SHAPE:
        mine = []
        for i in range(count):
            used = int(cores * busy) if i % 7 else 0
            # Accelerators follow the cores: an IDLE node with two of its four
            # GPUs allocated is a state no scheduler reports, and a demo must
            # not show one.
            gused = min(gpus, max(1, int(gpus * busy))) if gpus and used else 0
            down = kind == "down"
            mine.append(Node(
                name=f"{part}-{i:04d}",
                state_raw="DOWN+DRAIN" if down else ("MIXED" if used else "IDLE"),
                conditions=frozenset({"DOWN"}) if down else frozenset(),
                cpus_total=cores, cpus_alloc=0 if down else used,
                memory_mb=cores * 4000,
                memory_alloc_mb=0 if down else used * 3600,
                gpus_total=gpus, gpus_alloc=0 if down else gused,
                accelerator=ACCELERATORS.get(model) if model else None,
                labels=(model.lower(),) if model else (),
                queues=(part,),
                reason="scheduled maintenance [root@2026-01-06T09:00:00]"
                if down else "",
            ))
        nodes += mine
        queues[part] = Queue(
            name=part, node_names=tuple(n.name for n in mine),
            declared_nodes=count, nodes=mine,
            allow_accounts=("grp-chem",) if kind == "no access" else (),
            **({"state_raw": "DOWN", "enabled": False} if kind == "down" else {}),
        )

    refused = {s[0] for s in SHAPE if s[6] == "refused"}

    class Backend:
        name = "slurm"
        queue_term = "partition"

        def capabilities(self):
            return BackendCapabilities(probe=True, probe_supported=True,
                                       probe_command="sbatch --test-only")

        def probe(self, q, _shape, account=None):
            ok = q not in refused
            return Verdict(queue=q, account=account or "grp-a", allowed=ok,
                           category=VerdictCategory.OK if ok
                           else VerdictCategory.NOT_ENTITLED,
                           reason="" if ok else "Invalid membership")

        def submit_flags(self, q, _shape):
            return [f"--partition={q}"]

        def format_nodelist(self, names):
            return ",".join(sorted(names))

    cl = Cluster(
        backend_name="slurm", queue_term="partition", nodes=nodes, queues=queues,
        identity=Identity(user="ada", accounts=("grp-a", "grp-b"), qos=("normal",)),
        capabilities=Backend().capabilities(),
    )
    cl._backend = Backend()

    users = ["ada", "rmartin", "lgarcia", "jkim", "aweber", "nhaddad"]
    accts = ["grp-a", "grp-b", "pi-okafor", "pi-varga"]
    names = ["train_bf16", "sweep_r13", "matmul", "orbit_tide", "pipeline.stage3",
             "cx-a", "split_0", "batch_18"]
    jobs, allocs = [], []
    jid = 4210001
    for n in nodes:
        if not n.cpus_alloc:
            continue
        # The jobs on a node must ACCOUNT FOR what the node says is allocated,
        # cores and accelerators both. Anyone can add up a column, and a demo
        # whose rows do not sum to the header is worse than no demo.
        left, gleft = n.cpus_alloc, n.gpus_alloc
        k = 0
        while left > 0 and k < 4:
            take = left if k == 3 or left <= 8 else max(1, left // 2)
            gtake = gleft if take == left else min(gleft, 1)
            span = 6 if (jid % 11 == 0) else 1
            held = [n.name] + [x.name for x in nodes[:span - 1]] if span > 1 else [n.name]
            jobs.append(Job(
                id=str(jid), user=users[jid % len(users)],
                account=accts[jid % len(accts)], queue=n.queues[0],
                name=names[jid % len(names)], state="RUNNING",
                nodes=tuple(held), cpus=take * span, gpus=gtake * span,
                elapsed=f"{(jid % 30) + 1}:{jid % 60:02d}:00",
                remaining=f"{(jid % 12) + 1}:{jid % 60:02d}:00",
            ))
            allocs.append(Allocation(job=str(jid), node=n.name, cpus=take,
                                     memory_mb=take * 3600, gpus=gtake))
            left -= take
            gleft -= gtake
            jid += 1
            k += 1
    cl._jobs = jobs
    cl._allocations = {(a.job, a.node): a for a in allocs}
    return cl


# --- capture ----------------------------------------------------------------
def frames() -> list[list[str]]:
    """Every frame the storyboard walks through, ANSI intact."""
    out: list[list[str]] = []
    st = Style(depth=256, glyphs=Glyphs())
    step = {"n": 0}

    def marked(render, count, want: str) -> int:
        """The entry whose frame puts the cursor on a row containing `want`.

        Found rather than counted: the funnel contributes a variable number of
        entries in front of the rows, and the ranking decides which partition
        is where, so a hardcoded index records the wrong storyboard the moment
        either changes.
        """
        for i in range(count):
            for line in render(i):
                if "❯" in line and want in line:
                    return i
        raise SystemExit(f"no row for {want!r} in {count} entries")

    def scripted(render, count, **_kw):
        step["n"] += 1
        n = step["n"]
        if n == 1:                                   # the overview
            target = marked(render, count, "gpu-a")
            for i in (target - 1, target, target + 1, target):
                out.append(list(render(i)))
            return target                            # open gpu-a
        if n == 2:                                   # its nodes
            busy = marked(render, count, "MIXED")
            for i in (0, busy):
                out.append(list(render(i)))
            return busy                              # open a busy one
        if n == 3:                                   # the jobs on it
            for i in range(min(count, 2)):
                out.append(list(render(i)))
            return 0                                 # open a job
        out.append(list(render(0)))                  # the job itself
        return interactive.Key.QUIT

    import os
    import shutil

    real_size = shutil.get_terminal_size
    shutil.get_terminal_size = lambda *_a, **_k: os.terminal_size((96, 15))
    real_supported, real_select, real_read = (
        interactive.supported, interactive.select, interactive.read_key)
    interactive.supported = lambda *_a, **_k: True
    interactive.select = scripted
    interactive.read_key = lambda *_a, **_k: interactive.Key.QUIT
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_status(cluster(), build_parser().parse_args(["status"]), st)
    finally:
        interactive.supported, interactive.select, interactive.read_key = (
            real_supported, real_select, real_read)
        shutil.get_terminal_size = real_size
    return out


# --- ANSI -> pixels ---------------------------------------------------------
_SGR = re.compile(r"\033\[([0-9;]*)m")
_CUBE = (0, 95, 135, 175, 215, 255)
_BASE16 = [
    (0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16),
    (36, 114, 200), (188, 63, 188), (17, 168, 205), (229, 229, 229),
    (102, 102, 102), (241, 76, 76), (35, 209, 139), (245, 245, 67),
    (59, 142, 234), (214, 112, 214), (41, 184, 219), (255, 255, 255),
]


def xterm(n: int) -> tuple[int, int, int]:
    if n < 16:
        return _BASE16[n]
    if n < 232:
        n -= 16
        return (_CUBE[n // 36], _CUBE[(n // 6) % 6], _CUBE[n % 6])
    v = 8 + (n - 232) * 10
    return (v, v, v)


def spans(line: str):
    """(text, rgb, bold) runs, with the escapes resolved."""
    colour, bold, at, out = FG, False, 0, []
    for m in _SGR.finditer(line):
        if m.start() > at:
            out.append((line[at:m.start()], colour, bold))
        parts = [int(x or 0) for x in m.group(1).split(";")]
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == 0:
                colour, bold = FG, False
            elif p == 1:
                bold = True
            elif p == 22:
                bold = False
            elif p == 38 and i + 2 < len(parts) and parts[i + 1] == 5:
                colour = xterm(parts[i + 2])
                i += 2
            elif p == 38 and i + 4 < len(parts) and parts[i + 1] == 2:
                colour = (parts[i + 2], parts[i + 3], parts[i + 4])
                i += 4
            elif 30 <= p <= 37:
                colour = _BASE16[p - 30]
            elif 90 <= p <= 97:
                colour = _BASE16[p - 90 + 8]
            i += 1
        at = m.end()
    if at < len(line):
        out.append((line[at:], colour, bold))
    return out


def render_gif(shots: list[list[str]], path: Path, size: int = 15) -> None:
    from PIL import Image, ImageDraw, ImageFont

    plain = ImageFont.truetype(FONT, size)
    heavy = ImageFont.truetype(FONT_BOLD, size)
    # An INTEGER cell width. A fractional advance (9.03) accumulates across a
    # row, and Pillow rounds each draw independently, so a run of `─` picks up
    # a one-pixel gap wherever the fraction wraps -- a border broken every
    # thirty-odd columns.
    cw = round(plain.getlength("M"))
    lh = size + 5
    cols = max(len(_SGR.sub("", ln)) for s in shots for ln in s)
    rows = max(len(s) for s in shots)
    pad = 14
    w = cw * cols + pad * 2
    h = lh * rows + pad * 2

    images = []
    for shot in shots:
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        for r, line in enumerate(shot):
            col = 0
            for text, rgb, bold in spans(line):
                font = heavy if bold else plain
                for ch in text:
                    # One cell per character, placed on the grid rather than
                    # advanced by measured width: a block or box-drawing glyph
                    # does not share a letter's advance, and the error
                    # accumulates across a row until the border no longer lines
                    # up with the one above it.
                    d.text((pad + col * cw, pad + r * lh), ch, font=font,
                           fill=rgb)  # integer grid: no sub-pixel drift
                    col += 1
        images.append(img.quantize(colors=200, method=Image.MEDIANCUT))

    hold = [900] * len(images)
    hold[0] = 1600
    hold[-1] = 2600
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=hold, loop=0, optimize=True, disposal=2)


if __name__ == "__main__":
    shots = frames()
    print(f"{len(shots)} frames, {max(len(s) for s in shots)} rows")
    target = Path(__file__).resolve().parents[1] / "docs" / "demo.gif"
    target.parent.mkdir(exist_ok=True)
    render_gif(shots, target)
    print(target, target.stat().st_size // 1024, "KiB")
