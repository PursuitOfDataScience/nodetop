<h1 align="center">nodetop</h1>

<p align="center">
  <strong>See what your cluster actually has free — and why a queue that looks fine will not take your job.</strong>
</p>

<p align="center">
  Slurm · PBS Pro / OpenPBS / Torque · LSF · Grid Engine · Kubernetes · a bare pool of machines
</p>

<p align="center">
  <a href="https://github.com/PursuitOfDataScience/nodetop/actions/workflows/ci.yml"><img src="https://github.com/PursuitOfDataScience/nodetop/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/nodetop/"><img src="https://img.shields.io/pypi/v/nodetop.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/dependencies-none-brightgreen.svg" alt="No dependencies">
</p>

```bash
pip install nodetop
nodetop        # or: nt
```

Zero runtime dependencies, Python 3.10+. You reach for this when a cluster is
misbehaving, so it runs on a login node with nothing but the system Python.

## What you get

```
╭──────────────────────────────────────────────────────────────────────────────────────╮
│ nodetop · slurm  ·  ada  ·  328 of 607 nodes, 326 up  ·  222 of 358 GPUs, 53 free    │
│                                                                                      │
│    87 partitions  · ❯8 open to you  ·  65 no access  ·  11 refused  ·  3 down        │
│ ──────────────────────────────────────────────────────────────────────────────────── │
│    partition   nodes idle  cores free              gpu free  gpu model               │
│    compute           0/40   1695/5120  ██████████         —                          │
│    gpu-a            11/44    406/1408  ██▍░░░░░░░    48/176  A100, A40               │
│    wide             0/190    436/9120  ██▋░░░░░░░         —                          │
│    gpu-b              0/9     186/432  █▏░░░░░░░░      0/36  V100, RTX6000           │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

Every column is `free/total` under a header naming the numerator, and one hue per
resource. `wide` has 190 nodes and 2654 cores its scheduler calls free. **436** of them can
actually be allocated: the rest sit on nodes whose memory is entirely spoken for, so
nothing can land there. `free` is what you can have, not what is advertised.

*(Names and figures throughout are illustrative. Run it on your own cluster for yours.)*

Every partition on the cluster is in exactly one term of that second line, so the
counts reconcile with the total. **8 of 87** is the honest answer here — the rest
your accounts are not on, or the scheduler refuses, or are down.

## Navigating

On a terminal it is interactive by default. Arrow keys move the cursor, enter
opens what it points at, `q` leaves. Everything on screen can be opened:

- **a partition** → the nodes inside it, roomiest first
- **a node** → its state and drain reason in full, then the jobs on it
- **a job** → who owns it, its share of *this* node, and every node it holds
- **a count on the funnel line** → the partitions it counts, and why each is out
- **the partition total** → every partition on the cluster, each with the reason
  it is or is not in the table

Left and right move along the funnel line; up and down move between rows. Each
view replaces the last in the same place, so there is one screen rather than a
transcript of them.

`--static` prints and exits — for `watch nodetop`, or a pipe.

## Commands

```bash
nodetop                      # the overview
nodetop zoom gpu-a           # one partition: its gates, then its nodes
nodetop where -g 4 --gpu-mem 40G --needs bf16 -t 2-00:00:00
nodetop nodes --gpu --free   # GPU nodes with something free now
nodetop queues -q test       # every gate on one queue
nodetop check -q gpu         # ask the control plane directly
nodetop health               # down, drained, and silently degraded nodes
nodetop gpus                 # what each accelerator model can do
nodetop exclude --gpu-nodes  # an exclusion list for CPU-only work
nodetop snapshot -o snap.json && nodetop --replay snap.json status
```

### Flags worth knowing

`--json` on every command · `--all` widens a view from what you can use to the
whole cluster, `--detail` unfolds the reasoning behind a verdict · `--needs bf16`
requires a capability, `--tolerates` waives one · `--backend slurm` skips
autodetection and `--replay snap.json` works from a saved snapshot ·
`--static`, `--no-color`, `--ascii` for pipes and dumb terminals.

`--json` works on every command and carries everything the text does, caveats
included. Exit status is usable in a pipeline: `where` and `check` return 0 only
when somewhere could actually take the job, 1 when nothing can, 2 when the system
has no dry-run to ask, and 3 when the queries themselves failed.

The vocabulary follows the system — `partition` on Slurm, `queue` on PBS/LSF/SGE,
`namespace` on Kubernetes — and `-p` is accepted everywhere as an alias for `-q`.

## Why it exists

Your scheduler says the queue is up. The accounting table says you have access.
The dry-run says your job passed verification. All three can be true while your
job never runs — and **every batch system has the same disagreements**, in its own
vocabulary:

| The claim | What it hides |
|---|---|
| "There are idle nodes here" | the partition is `DOWN`; those nodes can start nothing |
| "You have access" | the association table says yes, the submit filter says `Invalid membership` |
| "Verification passed" | the site filter passed, the scheduler core refused |
| "No time limit" | the partition is unlimited, a QOS caps you at two days |
| "Admitted, so it will run" | it pends forever on a limit nobody published |
| "44 of 48 cores are idle" | every byte of memory is allocated; nothing can land |
| "There is room, so it starts now" | the scheduler's own estimate says 4h 24m |
| "That job is using 512 cores here" | it holds 7; the rest are on 41 other nodes |
| "4 GPUs available" | no scheduler knows whether they do bf16 |

So the reasoning is scheduler-independent and lives in `nodetop.core`, which
imports no backend. Only *acquiring* the facts differs, and that is one adapter
per system:

```
nodetop/
  core/       model · hardware · capacity · fit · duration   ← knows nothing about schedulers
  backends/   slurm · pbs · lsf · sge · kubernetes · sshpool ← knows exactly one
```

Where a fact is missing or ambiguous, the answer is the one that claims **less**:
an unidentifiable accelerator does not count toward a stated capability, a
truncated node record is unschedulable rather than idle, and a refusal that was
never obtained is not reported as one. The failure mode of that bias is a
needless warning. The other way round is a job sent somewhere it cannot run,
discovered ninety minutes later.

## Library

```python
from nodetop import Cluster, JobShape, rank

cluster = Cluster.load()                  # autodetects the batch system
cluster.can_probe                         # False on PBS and LSF -- check this
cluster.queues["test"].usable             # False
[b.code for b in cluster.queues["test"].structural_blockers()]
# ['QUEUE_DISABLED', 'NO_ACCOUNTS', 'NO_QOS']

for place in rank(cluster, JobShape(nodes=1, gpus_per_node=4, requires=("bf16",)),
                  use_probe=True):
    print(place.queue, place.runnable_now, place.confirmed, place.earliest_start)
```

Every query runs against one snapshot, so all the numbers in a report describe
the same instant.

## Development

```bash
git clone https://github.com/PursuitOfDataScience/nodetop
cd nodetop && pip install -e ".[dev]"
pytest          # ~3400 tests, no batch system required
ruff check src tests && mypy src
```

The suite is hermetic: every test drives recorded scheduler output, so it passes
on a laptop and in CI. Adding a backend means implementing one protocol —
`backends/base.py` — and answering `can_probe` honestly.

[**Design notes**](https://github.com/PursuitOfDataScience/nodetop/blob/main/DESIGN.md) covers why each behaviour is the way it is: the
defects that shaped it, and what each one cost.

## License

MIT
