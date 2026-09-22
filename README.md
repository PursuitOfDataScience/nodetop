<div align="center">

# 🖥️ nodetop

**See what your cluster actually has free, and why a queue that looks fine will not take your job.**

Slurm · PBS Pro / OpenPBS / Torque · LSF · Grid Engine · Kubernetes · a bare pool of machines

<a href="https://github.com/PursuitOfDataScience/nodetop/actions/workflows/ci.yml"><img src="https://github.com/PursuitOfDataScience/nodetop/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://pypi.org/project/nodetop/"><img src="https://img.shields.io/pypi/v/nodetop.svg" alt="PyPI"></a>
<a href="https://pypi.org/project/nodetop/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/PursuitOfDataScience/nodetop/badges/downloads.json" alt="PyPI downloads per month"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
<img src="https://img.shields.io/badge/dependencies-none-brightgreen.svg" alt="No dependencies">

</div>

```
 87 partitions  · ❯8 open to you  ·  76 no access  ·  3 down

    partition  nodes idle     mem free  cores free  gpus free  gpu model  usable
    gpu-a            0/44  1320/11000G    396/1408     44/176  A100       ██▍░░░░░░░░░░░░░░░░░
    compute          0/40  1400/10000G   1320/5120          -             ██▊░░░░░░░░░░░░░░░░░
    wide            0/190   190/34200G    380/9120          -             ▏░░░░░░░░░░░░░░░░░░░
```

`usable` is the smaller of free cores and free memory. That is why `wide` sorts last: 190G
across 190 nodes is 1G each. (Illustrative figures.)

## ✨ Install

```bash
pip install nodetop
nodetop        # or: nt
```

No dependencies and Python 3.10+, so the system Python on a login node is enough.

## 🧰 Use

```bash
nodetop                      # the overview: arrows move, enter opens, q quits
nodetop where -g 4 --gpu-mem 40G --needs bf16 -t 2-00:00:00
nodetop check -q gpu         # ask the scheduler directly
nodetop zoom gpu-a           # one partition: its gates, then its nodes
nodetop nodes --gpu --free   # GPU nodes with something free now
nodetop health               # down, drained, and silently degraded nodes
nodetop mcp                  # serve these reports to an AI agent
```

| Flag | Does |
|---|---|
| `--json` | everything the text shows, on every command |
| `--all` | the whole cluster, not just what you can use |
| `--detail` | the reasoning behind a verdict, on `queues` |
| `--needs bf16` · `--tolerates` | require or waive a capability |
| `--backend slurm` · `--replay snap.json` | skip autodetection, or read a saved snapshot |
| `--no-color` · `--ascii` | for pipes and plain terminals |

## 📌 Good to know

🧮 Exit status is usable in a pipeline: `where` and `check` return 0 if the job fits somewhere, and 1 if not.
`check` returns 2 when there is no dry-run to ask;
`where` answers from the declared ACLs instead, so it still returns 0 or 1.
A usage error is 2, no batch system is 3, and Ctrl-C is 130.

🗣️ **Words follow your scheduler**: `partition` on Slurm, `queue` on PBS, LSF and SGE,
`namespace` on Kubernetes, `pool` on an ssh pool. `-p` is an alias for `-q` wherever a
command takes a queue.

⚖️ **When a fact is missing, it claims less.** An unknown GPU counts for nothing and a
truncated node is unschedulable. You get a needless warning, not a job that never starts.

🤖 **`nodetop mcp` is read-only.** Each tool returns exactly what `--json` prints.

⚡ **Interactive mode opens on the last access check** and refreshes it in the background.
`NODETOP_ACCESS_TTL=0` turns that off.

[Design notes](DESIGN.md) explain why each behaviour is the way it is.

## License

MIT
