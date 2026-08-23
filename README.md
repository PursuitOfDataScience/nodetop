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

Your scheduler says the queue is up. The accounting table says you have access. The
dry-run says your job passed verification. All three can be true while your job never
runs.

nodetop exists because those sources disagree with reality in specific, repeatable ways —
and because **every batch system has the same disagreements**, in its own vocabulary.

```bash
pip install nodetop
nodetop                    # what is usable right now
nodetop where -g 4 --gpu-mem 40 --needs bf16 -t 2-00:00:00
```

Zero runtime dependencies, Python 3.10+. You reach for this when a cluster is
misbehaving, so it must work with nothing but the system Python and whatever scheduler
client is already there.

---

## The same six lies, everywhere

| The lie | Slurm | PBS / LSF / SGE | Kubernetes |
|---|---|---|---|
| “There are idle nodes here” | `State=DOWN` with all 610 nodes still listed | `started=False`, `Open:Inact`, every queue instance disabled | `Ready` but cordoned, or tainted and untolerated |
| “You have access” | association table lists 90 partitions per account | `acl_users` enabled and empty | RBAC allows, admission webhook refuses |
| “Verification passed” | site filter says PASSED, core refuses anyway | *no dry-run exists at all* | `--dry-run=server` (this one is honest) |
| “No time limit” | `MaxTime=UNLIMITED`, QOS caps at 2 days | queue unlimited, `max_run_res` caps | no walltime concept |
| “Admitted, so it will run” | pends forever on `QOSMaxGRESPerJob` | pends forever on a queue limit | `Pending` forever, within quota |
| “4 GPUs available” | `gres/gpu=4` | `ngpus=4` | `nvidia.com/gpu: 4` |

Not one of those rows is about Slurm. That's the point — the *reasoning* is
scheduler-independent, so it lives in `nodetop.core`, which imports no backend and knows
what no scheduler is. Only *acquiring* the facts differs, and that lives in one adapter
per system.

```
nodetop/
  core/       model · hardware · capacity · fit · duration   ← knows nothing about schedulers
  backends/   slurm · pbs · lsf · sge · kubernetes · sshpool ← knows exactly one
```

---

## What it actually catches

### 1. A queue that advertises idle nodes and can start nothing

```
$ nodetop queues -q test
● test [DOWN]
    nodes    ·············· 549/610 schedulable
    idle     107
    blockers
      ├─ QUEUE_DISABLED   state=DOWN (accepts nothing)
      ├─ NO_ACCOUNTS      account allowlist is empty (nobody may submit)
      ╰─ NO_QOS           QOS allowlist is empty (no QOS is permitted)
```

One real partition, in one real state: down, closed to every account, closed to every
QOS, and hidden — **four independent kill switches** — while still reporting all 610 of
its nodes, 107 of them idle. Anything counting idle nodes concluded there were a hundred
machines waiting.

So `Queue.effective_free_nodes` is **zero** whenever the queue cannot start work, and
availability is a *list of blockers*, not a boolean — fixing one of four changes nothing.

Every system has this. PBS spells it `enabled=True, started=False`. LSF spells it
`Open:Inact` and reports 140 jobs pending against it. Kubernetes spells it
`Ready,SchedulingDisabled` with every capacity number intact.

### 1a. And a dashboard that answered a question nobody asked

The overview used to open with that finding, and then rank every partition together by
free capacity. Both were wrong, and a user said so:

> what does accelerator mean? it's very confusing. also why always first reports something
> ain't work? this makes no sense at all. also, the colors in the bars don't make any
> sense and they have a lot of private nodes that aren't open to all users.

All four land. Taking them in order of how much they mattered:

**The ranked list was mostly unreachable.** Of 84 usable partitions on that cluster, 73
allow one or two accounts each — they are individual research groups' cluster shares. A
single sort by free capacity put eleven of them in the top twelve rows, so the headline
answer to "where can I go" was almost entirely places the reader cannot go. `nodetop` now
reads each queue's own allowlist (`Queue.is_dedicated`) and separates the two, because the
accounting database *cannot* answer this here: it reports the user as associated with 34
accounts and gives every one an identical QOS list, so declared entitlement does not
distinguish a partition you may use from one that rejects you with `Invalid membership`.
An allowlist of `pi-depablo` alone does. That leaves **two** shared partitions with free
GPUs, which is the honest answer.

**Failures come last.** A tool asked "where can I run this" should not lead with a list of
things that do not work. The finding is still there, in one line, at the end — with the
phantom idle-node count intact, since that is the whole point of the tool.

**"Accelerator" was jargon.** It was chosen to stay neutral across six schedulers and
printed to someone looking at a rack of NVIDIA cards. Every backend calls the resource
`gpu` in its own syntax — `gres/gpu`, `nvidia.com/gpu`, `ngpus` — so the UI says GPU, and
`nodetop gpus` works. The JSON keys and the core model keep `accelerator`: a machine
consumer should not have its field names churned for a wording fix.

**One panel, one population.** The header reported cluster-wide totals while the table
below it was filtered to the caller's slice — `358 GPUs, 117 free` printed above five
partitions holding 230 of them. It is built after the filtering now and inserted at the
top, so it counts exactly the partitions shown: `90 of 607 nodes, 88 up · 176 of 358 GPUs,
31 free`. Your slice is the subject; the cluster size is the qualifier, and it disappears
under `--all` where nothing is hidden.

**Free means reachable-and-free.** `Queue.effective_free_gpus` has always returned 0 for an
unusable queue. `Cluster.summary` did not: it counted every *schedulable* node's free
resources, so an idle four-GPU node whose only partition was `DOWN` read as four free
accelerators — phantom capacity in the summary of a tool written to catch phantom capacity.
`accelerators` computed its own totals the same way. Both now count over
`Cluster.reachable_nodes()`, which also excludes a node in no queue at all: nothing can be
submitted to it, so its capacity is not capacity. Reachability changes what is *free*, never
what exists — the installed total still counts the whole cluster.

**Nodes are the spine; accelerators are a column.** The overview led with a GPU fraction,
ranked by free GPUs, and gave its meter to GPU share. On this cluster that is 91 of 607
nodes — so five of seven shared partitions drew an empty bar and a dash, which reads as
missing data rather than as a CPU partition, and the ranking sorted the whole cluster by a
property 85% of it does not have. GPUs are a column, populated where they exist and blank
where they do not, and `where -g N` is the command for the question that is actually about
accelerators. `queues` had the identical defect in the view with the most rows of it — 70
of 87 partitions drawing an empty meter.

**A core is the unit of room; a node is the unit of shape.** Free *nodes* replaced free
GPUs as the meter, on the reasoning that every partition has nodes — and it was wrong in
its own way, because it counts only **wholly idle** nodes and on a busy cluster almost
nothing is wholly idle. Any partition running a single job read as zero room. Measured
here: `amd` had 2825 of 5120 cores free and drew a **2%** meter; `build` had 42 of 48 and
drew **0%**; `beagle3` had 200 cores and 27 GPUs free and drew **0%** — while
`beagle3-bigmem`, with 128 free cores, drew a full bar and took the top row. The meter was
inverted with respect to the quantity it claimed to show, and the ranking put a partition
above one with **22x** more free capacity. So the meter and the ranking are free *cores*.
`idle` stays as a column, because work that wants a node to itself still needs it, but it
is no longer the measure of room. `nodes` already metered `cpus_free / cpus_total` per
node; this is the same arithmetic one level up.

**No cell is a bare `free/total` fraction.** `4/4  100%` was read as plausibly meaning all
four are *busy* — a single fraction cannot say which side is free, and a percentage next to
it does not disambiguate. Free and total are separate columns under their own names now,
and the header line spells it out too (`607 nodes, 549 up` rather than `549/607 nodes`).

**Access is filtered by default, in two stages, and both were measured against the control
plane before being trusted.**

The first is free: intersect each queue's declared allowlist with the accounts you actually
hold. On this cluster that takes 84 partitions with room down to 19, and it has **no false
negatives** — an 18-partition sample of what it dropped was dry-run against all 34 of this
user's accounts, and not one of them accepted anything. It is also what a width heuristic
could never do: `jfkfloor2` names four accounts and `voltron` five, so a "fewer than three
accounts means private" rule let both through, while a set intersection excludes them
instantly.

The second is the dry-run, and it is not optional, because the first stage is nowhere near
sufficient: **of the 19 partitions the allowlist keeps, a dry-run accepts 8.** The
association table lists this user in `ssd`, `pi-sriesenfeld`, `pi-aaz`, `pi-blekhman`,
`pi-jjberg` and `pi-mclark`, and the submit plugin rejects every one with `Invalid
membership`. No reading of any declared list can see that. It costs ~2.8s rather than 30
precisely because the allowlist filter runs first — 19 queues to ask about, not 84.

**And the dry-run has to ask about the right account, which is where this went wrong.**
`probe_accounts` used to truncate its candidate list to four, on the reasoning that a user
with dozens of associations against dozens of queues would otherwise fire hundreds of
submissions. That is a real cost, but truncation is the wrong place to pay it: the general
partitions here set `AllowAccounts=ALL`, so the intersection is all 34 accounts and only the
first four were ever tried. `caslake` (190 nodes), `gpu` (44 accelerators) and `bigmem` are
each accepted with `rcc-staff` — 32nd in that list — and all three were reported **refused**
and hidden. Three partitions you can submit to, missing from the answer to "where can I run
this", with nothing on screen to suggest anything had been skipped.

Two changes, because either alone leaves the hole open:

- **The ceiling moved to the probe loop, and became global.** A per-queue cap bounds nothing
  worth bounding — fifty queues times four is still two hundred round trips — while the
  per-queue loop already stops at the first *accept*, so the expensive case is a queue that
  refuses everything, not a user with many accounts. `MAX_PROBES_TOTAL` bounds the whole
  question; the per-queue limit is now a backstop rather than the mechanism.
- **The accounts that work are learned and tried first.** An account accepted by one queue
  is overwhelmingly likely to be accepted by the next — here the single account that clears
  the SU check clears it for every shared partition — so queues are evaluated **cheapest
  first**, by how many candidate accounts they admit. A queue whose allowlist admits exactly
  one of your accounts costs one probe and proves that account works; the `AllowAccounts=ALL`
  queues then try it first instead of grinding through 34 in declaration order. In the
  scheduler's own order the expensive queues went first, spent the cap, and were written off
  — and then `beagle3` accepted that very account one queue later.

The result is both more correct and faster: 8 partitions instead of 5, in 2.8s instead of
3.5, because most queues now settle on their first probe.

**And a refusal that was not established is not reported as one.** If the per-queue ceiling
is reached with candidate accounts still unasked, the verdict becomes
`ACCOUNTS_UNTRIED` — a *transient* category, so `Placement.confirmed` stays false but
`durable` does too. The overview keeps that partition, marked `unconfirmed` in the funnel's
shown count rather than filed under `refused`, because "we did not ask" and "you are denied"
are the exact pair of claims this whole tool exists to keep apart.

```
 youzhi  ·  607 nodes, 548 up  ·  358 GPUs, 110 free

 87 partitions  →  5 open to you  ·  65 no access  ·  14 refused  ·  3 dead
 ───────────────────────────────────────────────────────────────────────────
 partition       nodes  idle  cores  free  share             gpu  free  models
 amd                40     0   5120  2825  █████▌░░░░   55%
 beagle3            44     0   1408   200  █▍░░░░░░░░   14%  176    27  A100, A40
 amd-hm              1     1    128   128  ██████████  100%
 beagle3-bigmem      4     4    128   128  ██████████  100%
 build               1     0     48    42  ████████▊░   88%

 DEAD  150 idle nodes advertised
       test DOWN, 150 idle  ·  climate all 48 nodes down  ·  climate-build all 2 nodes down

 refused by sbatch --test-only  ·  --all for every partition
```

**The filtering has to reconcile, on screen, with the total.** Both of those numbers
were already on the dashboard and they were four lines apart: a cluster-facts line
saying `87 partitions`, and a footer saying what two of the stages had hidden. The
question that came back — *"why is it showing me five rows?"* — is what you get when a
reader has to do that arithmetic themselves and one term is missing from it. So the
count moved down beside the table it describes, every stage that drops a partition is
named in the same line, and the terms sum to the total exactly: shown + no-access +
refused + no-nodes + dead == 87. A partition cannot leave the screen without appearing
in that line. The footer no longer repeats any of those counts — it says only what
`refused` *means*, since which dry-run refused you is the part that is not obvious, and
two sixty-fives four lines apart read as two different sixty-fives.

**All four listings share the filter**, through one helper, because getting this right in
one place and not the others is a mistake this file has made three times. Unfiltered, they
reported cluster-wide figures as though they were yours:

| command | says | yours |
|---|---|---|
| `queues` | 84 usable partitions | **19** |
| `nodes` | 607 nodes | **330** |
| `gpus` | 358 accelerators | **230** |

**Access is the filter; occupancy is a column.** The overview also dropped anything with no
free capacity, and it did so *before* applying the access filter — so a partition this
account can submit to disappeared because it was busy at the instant of the query.
Measured: 5 partitions accept a dry-run, 2 of them were full, and the screen said 3. That
understates access rather than capacity, and "where can I run this" includes "where can I
queue". A full partition you may use is now listed, honestly reporting `0` free, and the
ordering still puts the room first.

Each listing states what it hid (`277 not on your allowlist`), `--all` turns it off, and
naming a queue explicitly overrides it — "show me this one" should show it, blockers and all. If the
filter would leave *nothing*, it does not apply: an empty screen means the entitlement data
is unusable, not that you may use nothing, and it hides the very data that would explain
that.

**`where` probes by default too.** It was the last command still trusting declared
entitlement, and it was the worst place for it: `nodetop where -g 1` reported **four**
partitions as `RUN NOW` and a dry-run accepted **one**. Three of four rows were the
strongest claim the tool can make, about places that refuse this account. It costs 3.1s.

That change exposed three places where *"we could not ask"* was being read as *"no"*. A
probe that fails returns `allowed=False` with a category in `TRANSIENT_CATEGORIES`, and:

- `Placement.reachable` treated it as unreachable, so a scheduler hiccup flipped the exit
  code to "nothing fits anywhere";
- `where` filtered the row away, emptying the screen exactly when the caller most needs
  their options;
- the label said `BLOCKED` / "not permitted" for a control-plane outage — a false statement
  about access, and the same conflation a ceiling had before it got its own label.

All three now require the refusal to be **durable**, and a transient one gets its own label,
`NO ANSWER`. Having found those three by accident, I grepped every reader of
`verdict.allowed` — nine of them, and four more carried a consequence:

- the **probe loop** kept whichever verdict came *last*, so when accounts disagreed the
  result depended on iteration order. It now keeps the most informative: accepted beats
  unanswered beats refused, because one unanswered account means the *queue* is unknown
  even if another was durably refused;
- the **ordering key** demoted an unanswered row as though refused. That check turned out to
  duplicate `Placement.score`, which already orders confirmed ahead of unconfirmed, so it
  was deleted rather than fixed — the key now consults an *acceptance* only;
- the **`ACCESS` cell** painted `CONTROL_PLANE_DOWN` red, which reads as "you are denied";
- **`check`** folded unanswered probes into its refused count, so `1 of 3 accepted` hid that
  one of the other two was never asked. It is reported separately now, though the exit
  status still treats it as not-accepted: waving a `nodetop check && sbatch` caller through
  on an unanswered probe is the one outcome worth being strict about. A test asserts `reachable` and the label never disagree on a row, because they
are derived separately and had already drifted apart once.

`--declared` skips the dry-run and says so (`DECLARED ONLY  allowlists over-report`);
`--all` skips both. What is hidden is always counted, per stage, so the filtering is never
silent.

**And `nodes` is capped.** It answered "how are my 607 nodes doing" with 607 rows, which is
not an answer, it is the raw data again. Twenty by default, `-n N` to change it, `--all`
for every one, and a count of what was withheld — the same shape `rdu` uses for its own
`--top`.

**One helper owns the table style.** Six commands predated it and each kept the old look —
bold capitals, a rule above *and* below the header row. The style is three keyword
arguments deep, so relying on every call site to remember it did not work; `_grid` applies
it and a test asserts no caller bypasses it by looking for an all-caps word in any header
row. Column names are lowercased there too: a name is a label, not an announcement, and
lowercase is what let the column glossary go.

**Meter colour is a scale, not a verdict.** The fill was green above half and amber
below, which restated the number the bar already draws. Worse, three different quantities
sat in the same block — nodes schedulable, GPUs free, partitions usable — so one amber bar
meant "40% of GPUs are free", which is not a warning about anything. Two colours either
side of a threshold is a judgement.

The fix was a flat single-colour fill, and that overcorrected: it bought the honesty by
giving up on colour carrying any quantity at all. What it now uses is a twelve-step ramp
from deep blue through cyan and green to amber — ordered, so it reads as a scale the way a
heatmap legend does, and with **no red at either end**, so neither a full bar nor an empty
one can be mistaken for an alarm. Red stays reserved for the things that are actually
wrong, said with a glyph and a word.

Two more properties make it mean something rather than merely look like something:

- **A column is coloured as a set, not row by row.** A fixed number of bands cannot know
  that three of eighty-seven partitions happen to fall inside one of them, and three
  values an order of magnitude apart in the same tone is what makes a ramp read as noise.
  Rows are walked largest-first and one that is *measurably* smaller than the row above is
  forced at least one step cooler; rows that really are equal stay equal.
- **Tone and bar length are deliberately different quantities.** Length is the row's own
  free *share*; tone is how its free cores rank against the other rows. `amd-hm` draws a
  full bar in a cold tone — all of it free, and it is one node. `amd` draws a short bar in
  a warm one — mostly busy, and still the largest pool of free cores on the list.
  Collapsing the two would lose whichever was dropped, and both are answers someone came
  here for.

### 1c. An empty answer and an unobtainable one are different claims

The tool had this bug in itself, in the form it was written to catch elsewhere.
With every Slurm command failing — a controller outage, the exact moment you reach for
this — four commands printed clean, confident, wrong answers and exited **0**:

| command | said | means |
|---|---|---|
| `queues` | `0 shown, 0 usable` | a cluster with no partitions |
| `nodes` | `all 0 · 0 with GPUs · 0 out` | a cluster with no nodes |
| `health` | `0 schedulable · 0 degraded · 0 out` | **a perfectly healthy cluster** |
| `exclude --unschedulable` | *(empty nodelist)* | nothing to exclude |

`health` is the worst of them: it is the command whose entire purpose is to tell you
whether something is wrong, and during a total outage it said nothing was. And the
`exclude` case is actively dangerous — `sbatch --exclude=$(nodetop exclude
--unschedulable)` submits with no exclusions while the script believes it has them.
Only `status` mentioned the failures at all.

One guard at dispatch now covers every command: if queries failed and the snapshot is
empty, nothing is printed to stdout, each failed query is named on stderr, and the exit
status is **3** — the same code as "no batch system here", because both mean the tool
could not do its job. Deliberately not 1, which means "nothing fits" and is a real
answer. A *partial* failure still exits 0 with the report intact and the missing query
named, because that report is usable.

**The same conflation had a subtler form one layer down.** `load_identity` caught every
exception and substituted an empty string, so a failed association query produced an
identity holding zero accounts — indistinguishable from a user who genuinely holds none.
The account and QOS checks downstream are tri-state and read "nothing to compare
against" as "no verdict", so one dead `sacctmgr` silently disabled all of them. Measured:
all 34 accounts vanished, every dry-run ran with no `--account`, the control plane fell
back to a default and refused it, and the overview reported **`0 open to you · 83
refused`** — total loss of access, asserted confidently, during a database hiccup. The
function's own comment warned about precisely this hazard; the `except` clause below it
reintroduced it.

It raises now, so `Cluster.load` records it and leaves `identity` as `None`, which the
entitlement filter already treats as "cannot filter" rather than "entitled to nothing".
A refusal obtained with no account named, *when the association query is known to have
failed*, is downgraded to unsettled — keyed on the failure and not merely on the absence
of an identity, because a backend with no notion of accounts at all (an ssh pool) probes
without one legitimately and its refusals are real. The same screen now reads `84 open to
you (84 unconfirmed)` with `FAILED identity` naming the cause.

**Two schedulers wrap long attribute values, and both backends dropped the tail.**
This is the same defect as the record-splitting one below, in a different dialect, and it
was found by sweeping for that one. PBS breaks a value at 80 columns mid-value and indents
the remainder with a tab; LSF wraps a long `bqueues -l` value onto indented following
lines. Every PBS parser required an `=` before accepting a line and LSF's `_after` stopped
at the end of the label's own line, so in both the continuation was silently discarded —
losing the tail of exactly the values long enough to wrap, which are the ones that matter:

| field | wrapped | consequence |
|---|---|---|
| PBS `resources_available.Qlist` | 8 queues → **5** | the node goes missing from the queues whose names were cut, and its capacity with it |
| PBS `acl_users` | 8 users → **6** | truncated allowlist read as authoritative → **false denial** |
| PBS `exec_host` | 7 nodes → **5** | maps running work to nodes, so free-time estimates land on the wrong machines |
| LSF `USERS:` | 14 users → **10** | same false denial, other scheduler |

The two false-denial rows are the ones that matter most, because they are the failure this
tool exists to prevent pointed the wrong way: reporting no access to a queue that would
have taken the job. `exec_host` is the longest field PBS emits and *always* wraps for a
multi-node job.

A continuation is an indented line with no `=` (PBS) or one that does not begin a new
`LABEL:` (LSF). Record headers are never indented in either, so a header cannot be
mistaken for a continuation or the reverse.

**A parser that cannot tell "nothing" from "one merged record" cannot complain.**
`scontrol` emits either one record per line or one field per line, and the two are a
single flag apart — this backend passes `--oneliner` when listing nodes and not when
listing partitions. Each parser understood only the shape its own command happened to
produce, and both failed silently on the other, in opposite directions: partitions were
split on blank lines, so oneliner input became one record whose field map kept the last
value for each key (**2000 partitions collapsing to 1**), while nodes were read one per
line, so multi-line input gave a record per line and every node came back with **0 CPUs
and no state** — a cluster that appears to own no resources. Neither is reachable while
the argv here is fixed, which is exactly why it would go unnoticed; the way in is a
replayed snapshot recorded where `scontrol` behaved differently, a site wrapper, or a
version whose output changed shape. Records are now delimited by their own header
keyword, required at a line start, so layout is irrelevant.

Writing the test for that found an older one underneath it. The field map was built by
dict comprehension, which keeps the **last** match — and no field repeats in a real
record, so a second occurrence can only have come from free text. One field on every node
is operator-authored prose, so a node drained with `Reason=replacing NodeName=n2 per
ticket` was **renamed to `n2 per ticket`**: it vanished from the report under its own name
and reappeared under a mangled one. First occurrence wins now; the real header is at the
record start, which is what makes that the safe rule rather than merely a different one.

**And the same defect was in five other adapters.** Having found it once in the Slurm
backend, the obvious next question is whether the other five made the same choice. They
did — a `try/except` around a query, returning something empty and plausible:

| backend | swallowed | consequence |
|---|---|---|
| Kubernetes | `kubectl get pods` | **every node reports as idle** |
| Kubernetes | `auth whoami` | RBAC group restrictions silently ignored |
| Kubernetes | `get resourcequota` | quota ceilings silently absent |
| Grid Engine | `qconf -sul` sweep | *partial* userset list → **false denials** |
| Grid Engine | `qconf -srqs` per set | *partial* ceilings, applied as complete |
| PBS | `qstat -Qf` limits | ceilings absent — and PBS has no dry-run to fall back on |
| PBS / LSF | local group lookup | *partial* group list → false denials |

The Kubernetes pod query is the worst of them and worth spelling out. `allocatable` is a
capacity, not a free count, so pod requests *are* the occupancy — and with no pod data
every node parses as zero-allocated. A node running 40 of its 48 cores and all 4 of its
accelerators was reported as `48/48` and `4/4` free, `idle=True`. That is phantom
capacity, the failure this tool was written to catch, manufactured by the tool itself. It
was also reachable in ordinary use: `kubectl get pods --all-namespaces` is routinely
forbidden by RBAC for a namespaced user. The suppression even carried a comment claiming
the missing query "is recorded in `Cluster.errors`" and that a node would not be shown as
fully free — neither was true, because swallowing the error made `load_nodes` *succeed*.

Two failure directions, and a partial answer is the more dangerous of them. An **empty**
result reads as "cannot tell" in the tri-state membership check, so restrictions are
ignored and queues are claimed that will refuse the job. A **partial** result reads as
authoritative, so the tool returns a verdict from a scan it knows did not finish — "none
of your groups are permitted here" — which is a false denial that hides a queue you can
actually use. Every one of these now either builds its answer atomically or raises, so
`Cluster.load` records the failure and the caller sees "we could not ask" instead of an
answer. The distinction the code already had a name for — `Limits.unreadable` — was being
thrown away at the point it was needed.

### 1b. How well the guess actually does — measured

`where` had the same defect and it mattered more, because `where` is the command you act
on. `nodetop where -g 1` listed five partitions and called **four** of them `RUN NOW`. A
dry-run then refused all but one:

| partition | allowlist | marked `group-only`? | `sbatch --test-only` |
|---|---|---|---|
| `beagle3` | 28 accounts | no | **confirmed** |
| `gpu` | *none* (open) | no | `NOT_ENTITLED` |
| `sriesenfeld-gpu` | 1 account | yes | `NOT_ENTITLED` |
| `ssd-gpu` | 1 account | yes | `NOT_ENTITLED` |
| `pedramh-gpu` | 1 account | yes | `NOT_ENTITLED` |

Three of the four false positives are caught. The fourth is the honest limit, and it is
worth stating precisely rather than papering over: `gpu` declares an **empty account
allowlist** and a QOS allowlist (`gpu`, `debug`) that intersects the caller's — so it is
open on every axis a structural reading can see, and it refuses anyway. The lie lives in
the association dump, which claims the same 92 QOS entries for all 34 of the caller's
accounts. No allowlist reading can reach that; only the control plane can.

So the marker is a marker, never a filter. `group-only` means *"allows 1–2 accounts; you
may not be one"* — not "you cannot go here" — and a confirmed verdict overrides it
entirely, because if a probe says you are in the group then the partition is not
second-class. Shared partitions are merely sorted ahead of private ones at equal standing,
so the row you can act on comes first. `tests/test_check.py` records the blind spot as an
executable test, so the heuristic can never quietly be mistaken for a substitute for
`--check`.

### 2. Entitlement that is declared but never verified

Three of the six systems have a real verify-only mode. Three do not:

| system | dry-run | entitlement |
|---|---|---|
| Slurm | `sbatch --test-only` | **confirmed** |
| Grid Engine | `qsub -w v` | **confirmed** |
| Kubernetes | `auth can-i` + `--dry-run=server` | **confirmed** |
| PBS / Torque | none | declared only |
| LSF | none | declared only |
| ssh pool | no scheduler | n/a |

Where there is no dry-run, nodetop says so rather than presenting an ACL as a verified
right — `ACCESS` reads `declared`, and `nodetop backends` prints why. Silence would let a
declared entitlement read as a confirmed one, which is the failure this tool exists to
prevent.

Where there *is* one, read both layers. On Slurm:

```
sbatch: error: Verification: ***PASSED***                 <- the site's job_submit filter
allocation failure: Invalid account or account/partition combination specified
                                                          <- the scheduler, refusing anyway
```

Grep for `Verification:` and stop, and you conclude the opposite of the truth. nodetop
requires both layers clean, classifies the refusal, and reports the disagreement. It also
reads back the QOS the controller *actually chose* — a request on `beagle3` came back
running under `beagle3-prio`, and checking ceilings against the name you asked for checks
the wrong ceilings.

nodetop also notices when a claim is worthless on its face, and says so in `status`
rather than burying it under every queue:

```
╭─ nodetop ─────────────────────────────────────────────────────────────────╮
│ slurm  ·  607 nodes  ·  91 with accelerators  ·  87 partitions            │
│ entitlement  confirmable via sbatch --test-only                          │
│ you          youzhi  ·  34 accounts  ·  92 QOS                           │
╰──────────────────────────────────────────────────────────────────────────╯

  ▲ every one of your 34 accounts claims an identical list of 92 entitlements,
  so the scheduler's access claim carries no per-account information here --
  only a dry-run (--check) settles where you can actually submit
```

### 3. A dry-run passing does not mean the job will start

No HPC scheduler evaluates resource ceilings in its dry-run. A 40-node × 4-GPU request on
a QOS capped at 4 GPUs per job comes back `PASSED`, with a plausible start time attached —
then pends indefinitely under `QOSMaxGRESPerJob`. An 8-day walltime on a 2-day QOS does
the same. Nothing warns you.

```
$ nodetop where -g 4 -t 8-00:00:00
  gpu
    [shape] MAX_WALLTIME
       walltime 8-00:00:00 exceeds gpu limit 1-12:00:00 -- typically accepted
       at submit time and then queued indefinitely
```

Kubernetes is the exception: server-side dry-run runs real admission, so a
`ResourceQuota` breach *is* caught before submission. nodetop says which situation you are
in instead of assuming.

Related: `Cluster.effective_max_walltime()` returns the tighter of the queue's limit and
its limit set, and says where the binding number came from:

```
maxtime  7-00:00:00  (from slurm QOS test; the partition itself says unlimited)
```

### 4. No scheduler models the accelerator

Slurm, PBS, LSF, SGE and Kubernetes all treat a GPU as an opaque countable resource, so
all of them will place a bf16 job on a V100 and let it die at the first autocast, or
resume an fp8 checkpoint on a card with no fp8.

```
$ nodetop where -g 4 --gpu-mem 40 --needs bf16 -t 2-00:00:00
╭─ job ────────────────────────────────────────────────────────────────────╮
│ 1 node, 4 GPU/node (4 total), >=40 GiB HBM, 2-00:00:00, needs bf16       │
│ [NOWHERE NOW]  nothing can start immediately                             │
╰──────────────────────────────────────────────────────────────────────────╯

⏺ placements  19 partitions considered
              PARTITION           FREE  CAPABLE  START  ACCESS            ACCELERATORS
  ─  ───────  ──────────────────  ────  ───────  -----  ────────────────  ─────────────────
  ◐  QUEUE     beagle3             0/1    44/44    44m  confirmed         A100x22, A40x22
  ○  BLOCKED   gagalli-gpu         2/1     6/30      ·  ACCOUNT_MISMATCH  H200x4, A100x2
  ○  BLOCKED   hcn1-gpu            1/1      1/1      ·  INVALID_QOS       L40Sx1
  ▲  LIMIT     schmidt-gpu         3/1      3/3      ·  confirmed         A100x2, H100x1
  ✗  WRONG HW  aettinger-gpu       0/1      0/2      ·  confirmed         RTX6000x2

  ● runs now  ○ not permitted  ✗ no node of the right kind
  ▲ over a declared ceiling  ◐ would queue
```

Note `gagalli-gpu`: two nodes free, six capable, and you still cannot use it. Without
`--check` the `ACCESS` column is absent entirely and that row would read as a queue worth
waiting for.

Every label implies a different next move, so collapsing two of them sends you somewhere
useless — and the labels are picked in order of what you *cannot* work around:

| label | what it means | what to do |
|---|---|---|
| `RUN NOW` | room right now | submit |
| `BLOCKED` | no job of any shape runs here | ask for access |
| `WRONG HW` | no node of the right kind | go elsewhere; waiting will not help |
| `TOO FEW` | right nodes, never enough of them | ask for fewer nodes |
| `LIMIT` | over a declared ceiling | resize or shorten |
| `QUEUE` | permitted, capable, just full | submit and wait |

The renderer used to test `Placement.reachable`, which is deliberately *both* "permitted"
and "the shape is legal". So a queue whose only problem was a per-user accelerator ceiling
rendered as `BLOCKED` / "not permitted", telling you to go request access you already had.
On a live cluster a `-N 40` request made **all five** candidate partitions read "not
permitted"; the answer was `-N 8`.

`TOO FEW` exists for the same reason. "Could this queue ever host the shape" needs the
right *kind* of node **and enough of them**; checking only the kind made a one-node queue
asked for forty report possible, so it rendered `QUEUE` — a wait for capacity the queue
does not contain. A queue whose node list is incomplete is exempt: ruling it out on a
lookup failure would be the worse error.

`CAPABLE` carries a denominator because the reason histogram beneath it cannot be summed
back into one — a node can fail on several counts at once. `1/11` next to "5 nodes: V100
lacks bf16; 5 nodes: RTX6000 lacks bf16" closes; a bare `1` does not.

The legend lists only the states actually present. It was a fixed list of four, which meant
`where` explained "wrong hardware" over tables containing no such row.

`nodetop accelerators` turns that into a cluster-wide answer, which is the
question you actually have before committing to a run:

```
⏺ by model
  MODEL    VENDOR  ARCH    MEM  NODES  FREE              BF16  FP8
  A100     NVIDIA  sm_80  40G?     29  █········ 13/116  yes   no
  H100     NVIDIA  sm_90  80G?      5  ███▌····· 7/18    yes   yes
  RTX6000  NVIDIA  sm_75   24G     16  █████▍··· 38/64   no    no

⏺ capability reach  share of the cluster that can do this at all
  bf16   █████████████▌···· 268/358 installed, 47 free now
  fp8    ███··············· 60/358 installed, 18 free now
```

Free counts exclude unschedulable nodes on purpose: hardware behind a drained
node is installed, not reachable, and an unidentifiable accelerator is counted
in **no** capability row rather than being assumed capable.

On Kubernetes, occupancy follows the scheduler's real arithmetic rather than a
naive sum: a pod reserves `max(sum(containers), max(init containers)) +
sidecars + spec.overhead`, so a node held by a large init container is reported
full rather than free.

Four details make this trustworthy rather than merely clever:

- **Capability is stored per model, per vendor — never derived from one number.** Deriving
  dtype support from a CUDA compute capability works until an AMD or Intel part appears
  and then reports nonsense. NVIDIA, AMD CDNA and Intel Xe/Gaudi are all covered.
- **The model comes from the typed resource first, then labels.** On the reference cluster
  90 of 91 GPU nodes report a bare `Gres=gpu:4`; the model is only in the node features,
  in whatever case the admin typed (`a100`, `A100`, `H100`, `L40S`). Kubernetes is the
  same story with `nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB`.
- **An unidentifiable accelerator is `None`, never a guess** — and unknown is not treated
  as incapable. Only a *known* negative excludes a node.
- **Memory is an inference and is labelled one.** `A100` alone does not say 40 GB or
  80 GB, and no scheduler records it. The conservative variant is assumed, so the failure
  mode is a needless warning rather than an OOM ninety minutes into a run.

---

## Commands

```bash
nodetop                          # cluster overview; unusable queues first
nodetop backends                 # which systems are here, and which can confirm access
nodetop queues                   # compact table  (alias: partitions)
nodetop queues -q gpuq           # every gate for one queue  (--detail for all)
nodetop nodes --gpu              # inventory with model, vendor, arch and memory
nodetop health                   # down, drained, and silently degraded nodes
nodetop where -g 4 --gpu-mem 40 --needs bf16 -t 2-00:00:00
nodetop where -g 4 --declared  # trust the allowlists, skip the dry-run
nodetop where -g 4 --all         # include the ruled-out queues and why
nodetop where -c 8 --tolerates dedicated=inference:NoSchedule
nodetop check -q gpuq -g 1       # the dry-run, directly  (alias: probe)
nodetop gpus                     # inventory + what each model can do  (alias: accel)
nodetop snapshot -o snap.json    # record this cluster's state
nodetop --replay snap.json status   # ...and analyse it later
nodetop exclude --gpu-nodes      # exclusion list for CPU-only work
nodetop --backend kubernetes status
```

The `ACCESS` column appears only once a dry-run has answered — the same word on
every row is noise, so with no probe run it is dropped and stated once, along
with the flag that would confirm it. `check` likewise declares which of your
flags took no part: `--needs` and `--gpu-mem` cannot, because no scheduler can
express them, so the control plane was never asked.

`--check` narrows itself: a queue that publishes an account allowlist has already
answered most of the question, so only the intersection with your own accounts is
dry-run, capped per queue. On a cluster where one user holds 34 associations across 87
partitions that is the difference between a couple of seconds and several hundred
submissions.

Exit status is meaningful, so `nodetop check … && sbatch …` behaves: `where` and
`check` return 0 only when somewhere could actually take the job, 1 when
nothing can, and `check` returns 2 when the system has no dry-run to ask.

`--all` widens `status`, `queues` and `where` to include what they would otherwise
filter out — on `where` that means the ruled-out queues with their blockers attached,
which is what you want when the question is "why can nothing run anywhere?".
`--tolerates` declares the node restrictions a job accepts; on Kubernetes that is how a
tainted node becomes eligible, and nothing else can express it.

`--json` works on every command, on either side of the sub-command name, as do
`--no-color`, `--ascii`, `--backend` and `--replay`. The JSON carries everything the
text does, including the caveats — a note that only appears in prose is a note a script
never learns, and a script is the consumer most likely to act on the answer. `check`
therefore reports `not_covered` and `filter_scheduler_disagreements` alongside the
per-queue verdicts, and a test asserts the two renderers cannot diverge. The vocabulary follows the system — `partition` on Slurm, `queue` on PBS/LSF/SGE, `pool` with no
scheduler — and `-p/--partition` is accepted everywhere as an alias for `-q/--queue`.

Two worth knowing:

- **`health` finds the node the scheduler still hands out while it runs several times
  slower than its siblings** — a power-capped or thermally throttled accelerator reports a
  perfectly healthy state, and that shows up as a mysteriously slow job, not an error.
  Nodes are called `degraded` only when they are *schedulable* and carry a suspicious
  reason; an already-drained node is not impaired-but-usable, it is out.
- **The reason field is parsed before anything reads it.** Slurm stamps every drain
  reason with `[who@when]`, and that suffix breaks both things built on top of it.
  Grouping on the raw string splits one maintenance window into a row per second the
  operator spent typing — on a 607-node cluster, 52 nodes out for the same cause rendered
  as five findings differing only in a timestamp, and the actual answer ("52 nodes, out
  for five weeks") was nowhere on the screen. And the keyword list that finds impairment
  holds short words like `fan`, `slow` and `clock`, which against the whole string also
  match the *operator's username*: an admin called `fanl` marked every node they touched
  as thermally throttled. `split_reason` separates the two, the text view groups by cause
  and reports the age of the oldest stamp, and `--json` exposes the same parse so a script
  cannot disagree with the terminal about what one cause is.
- **`exclude --gpu-nodes` decides accelerator-ness from the resource count, never the
  hostname.** Clusters routinely have a `beagle3-bigmem1` with no GPU sitting among 44
  nodes that have four each. Filtering on the name prefix is how CPU work ends up
  squatting an accelerator.

### 1d. `idle 0` does not mean there is nothing there for you

> idle is 0 doesn't mean there is nothing from there we can't use.

Right, and the column invites that reading. `idle` counts **wholly** idle nodes, and on a
busy cluster almost nothing is wholly idle. Measured here: `amd` reports `idle 0` while
carrying **2105 free cores spread over 24 of its 40 nodes**, every one of them running
something. Any job that does not need a whole node can start there immediately. The
overview cannot show that without turning into a node listing, so the number that fits in
the column is precisely the one most easily misread.

Two changes. The count that was misleading now states both figures wherever a single
partition is expanded — `idle 0 wholly free, 24 of 40 with something spare` — so the
smaller number can no longer be read as the whole answer. And `zoom` opens one partition
up: the same gate-by-gate block `queues -q NAME` prints, then the nodes inside it, roomiest
first, in the same table `nodes` prints.

```
● amd [UP]
    nodes    █████████████▋ 39/40 schedulable
    idle     0  wholly free, 24 of 40 with something spare
    accel    none
    maxtime  1-12:00:00  (from slurm QOS amd; the partition itself says unlimited)

⏺ inside  40 nodes  ·  1 out  ·  roomiest first
  no node here is entirely free, but 24 nodes have something spare -- 2105 cores. A job
  that does not need a whole node can start now.
  ────────────────────────────────────────────────────────
     node          state  cpu          free  mem free  gpu
  ◐  midway3-0507  MIXED  ███████▏  114/128   29/244G  ·
  ◐  midway3-0519  MIXED  ███████▏  114/128   29/244G  ·
```

**The header and the table are the existing renderers, not lookalikes.** `_queues_detail`
draws the block and `_node_rows` builds the rows, both shared with the commands they came
from. Two renderers of the same thing drift — this file has the scars, which is why
`_grid` exists — and a zoom view whose columns disagree with the listing it zooms out to
is worse than no zoom view at all.

Building it surfaced a bug in the ordering added earlier. A drained node still reports its
full complement free, so ranking nodes by free capacity put a `DOWN+DRAIN` node advertising
`32/32 cores, 4/4 GPUs` at the **top** of the answer to "where is there room" — phantom
capacity leading the list. `Queue.effective_free_*` had always excluded those; the sort had
not. Unschedulable nodes now sort last in both views regardless of what their counters say.

### 1e. You cannot act on a printout

> you can't operate on the print out at all. i hope there is something like claude code
> where we you can move the cursor up and down to select things

**On a terminal this is the default.** A highlight sits on a row, the arrow keys move it,
enter opens that partition, and you land back on the list when you are done. There is no
flag to turn it on and nothing on screen explaining it.

That last part is deliberate. A flag to switch it on meant advertising the flag, and a line
of the overview spent telling the reader that a key exists is a line nobody reads — the
overview has now lost a column glossary, a legend, a footer of suggestions, a DEAD block
and, finally, its own key hint. The highlight is the affordance: a row in inverse video is
something you try the arrow keys on. Every line of the default view is numbers.

`--static` prints the report and exits. It exists for a terminal that is not a person --
`watch nodetop` allocates a pty and would otherwise block on a keystroke forever -- and it
is documented in `--help` and nowhere else, which is where a flag belongs.

Three constraints ruled out reaching for a TUI library, and between them they decided the
whole shape of it:

- **No dependencies.** This is a tool you run on a login node while the cluster is
  misbehaving, so it has to work with nothing but the system Python. `termios` and `tty`
  are standard library on every platform the package claims; a TUI library is not
  installable at the moment it is needed.
- **The same output.** Nothing in the interactive path renders anything. It takes the
  finished lines `status` already built and wraps one of them in inverse video. A second
  renderer is how the interactive view would start disagreeing with the printed one — the
  same reason `_grid` and `_node_rows` exist.
- **It degrades to the printout.** Redirected, piped, `TERM=dumb`, or a platform without
  `termios`: you get the static report, not an error. Both streams are checked, and for
  different reasons — stdout must be a terminal for a highlight to mean anything, and
  *stdin* must be one or a run with input redirected from a file would consume that file
  as keystrokes.

Arrows or `j`/`k`, `g`/`G` or Home/End to jump, enter or space to open, `q`/Escape/Ctrl-C
to leave. Movement wraps at both ends, which is cheaper than a page-down binding.

**Driving it through a real pty is what made it work.** The first version decoded every
arrow key as a quit, and the reason is worth writing down because the code looked right:
`sys.stdin.read(1)` fills *Python's* buffer from the kernel, so after taking the `ESC` of
an arrow key the following `[B` sits in userspace where `select()` on the descriptor cannot
see it. The peek came back empty, the sequence read as a lone Escape, and a lone Escape
means quit. Reading the descriptor directly with `os.read` keeps the poll and the read
looking at the same buffer. A unit test with a fake character reader would have passed
either way.

A second bug the pty found, and the same shape as the first -- a state assumption that a
unit test cannot see. Raw mode was scoped to the list, so the keystroke that dismisses a
zoomed view was read in *canonical* mode, where nothing arrives until Enter. "Any key
returns" had quietly become "press enter". Raw mode now spans the whole interaction.

**Two more that only a terminal could show, both found by making the default
interactive and then attacking what shipped.**

A frame taller than the window destroys the screen. The repaint moves the cursor up by the
height of the previous frame, and 84 partitions is a 93-line frame: on a 24-row terminal
the cursor clamps at the top, the clear-to-end lands in the wrong place, and every keypress
leaves another copy of the listing behind — 252 rows on screen for 84 partitions. There is
a viewport now, and only *rows* are dropped: headings, the funnel and the totals survive
whatever scrolls, because they are the frame of reference for the row you are looking at. A
short line says `3 above  67 below`, because silently dropping rows reads as "this is all
of them", which is the lie the funnel exists to prevent.

`finally` does not restore a terminal. A default-handled `SIGTERM` or `SIGHUP` ends the
process without raising anything, so nothing runs and the terminal is left with echo and
canonical mode off — a shell that appears dead, with no echo to tell you that typing
`reset` is working. `SIGINT` was fine only because Python turns it into an exception.
Measured through a pty: after `SIGTERM`, `echo=False canonical=False`. The fatal signals
are now caught long enough to put the terminal back and then re-raised with the default
disposition, because a tool that swallows `SIGTERM` is worse than one that leaves a messy
terminal. `SIGTSTP` is the same problem wearing Ctrl-Z: suspending hands the terminal back
and resuming takes it again. **`SIGHUP` is the one that matters on a login node — it is
what a dropped ssh connection sends.**

The loop itself is injectable — `read_key` takes a character reader and `select` takes a
key source and a writer — so the move logic, the wrap-around, the repaint and the
KeyboardInterrupt path are all tested without a terminal. An interactive mode that is only
ever tested by hand is one that breaks silently.

### 1f. One screen, three levels, and a cursor you can see

> the ui should be just one where i can go in and go out rather than print every interface
> on the terminal and select the new interface after printing the new one
>
> the cursor isn't clear at all. the users don't know if they can move the cursor up or down

Partitions, then the nodes inside one, then the jobs on one node — each level **replaces**
the last in the same rows rather than printing beneath it. `select` erases its block on the
way out, so there is one screen instead of a transcript of screens, and the cursor position
is remembered per level: stepping out lands you on the row you came from, not at the top of
a list you have already read. Enter or Right goes in, Escape/Left/Backspace comes out, `q`
quits from any depth.

**The highlight was not a matter of taste — it was broken.** A rendered row is full of
coloured cells and each one ends in `ESC [ 0 m`, which clears reverse video along with the
colour. Wrapping such a row in `ESC [ 7 m` therefore highlighted it as far as the first
coloured cell and no further, so the selected row really was a smudge on the left. Inverse
is now re-armed after every embedded reset — 19 of 19 visible characters inside the
highlight, where it had been about four. And there is a `❯` in a one-character mark column,
because `Style.inverse` is a **no-op** under `NO_COLOR`: without the glyph the selection was
invisible in that mode entirely, and a glyph also implies the axis you can move along.

Jobs come from a `squeue` query fetched **lazily and cached** — a deliberate exception to
the one-snapshot rule, because almost no invocation asks for jobs, and while browsing the
newest answer beats the consistent one.

**`squeue` reports a job's counts as totals across every node it holds**, which in a
per-node table is actively misleading: a nine-node job appeared as `431` cores on a
48-core node, a number the reader knows to be impossible, and one impossible cell
discredits the whole column. The exact per-node share needs a `scontrol show job` per row,
and dividing would be a guess dressed as a fact — a job need not be allocated uniformly.
So both exact numbers are shown, `431 x9`, and single-node jobs (most of them) are unmarked
and read directly. Verified the other way too: on the 161 nodes where every job is
single-node, the job cores sum **exactly** to the node's allocation.

An empty job list distinguishes four cases, because "no jobs here" on a visibly busy node
would be phantom capacity in a new place: the query failed, this backend cannot list jobs,
busy but nothing claims the resources, or genuinely idle.

## The terminal UI

Everything is drawn against the standard library — no curses, no rich, no
dependency at all — and three things it handles are correctness rather than
decoration:

- **Display width is not string length.** Cells are padded by what the terminal
  will actually show, so ANSI colour, East-Asian wide characters and combining
  marks do not shift a column. `width("\033[31mabc\033[0m") == 3`,
  `width("日本語") == 6`, `width("é") == 1`.
- **Not every terminal speaks UTF-8.** Every glyph has an ASCII twin, chosen
  automatically from the real stdout encoding and forceable with `--ascii` or
  `NODETOP_ASCII=1`. The test suite asserts the ASCII path emits **no non-ASCII
  bytes at all** — a `LANG=C` session gets `+-|`, `*`, `->` and `#`, not mojibake.
- **Colour support is a spectrum.** Truecolor, 256-colour and 16-colour are
  detected from `COLORTERM`/`TERM`, and every semantic role survives all three
  depths. `NO_COLOR`, `FORCE_COLOR` and a non-TTY pipe are all honoured.

- **Scheduler text cannot repaint your terminal.** A node's `Reason`, a
  Kubernetes condition message and a dry-run's stderr are operator- or
  controller-authored free text, and they go straight into a table cell. Left
  alone they do damage `width()` cannot see, because it measures what text
  *occupies* while these characters act instead: `ESC [ 2 J` clears the caller's
  terminal mid-report, `\r` returns to column zero so the rest of the row
  overwrites what was drawn — silently, hiding content rather than mangling it —
  `\n` splits one row into two, and `\t` measures as one column then expands to a
  tab stop. Control characters are replaced with spaces at the point the data
  enters the model, so all six backends are covered by one rule and a mangled
  field still reads as mangled instead of quietly closing up. Styling is applied
  afterwards, so the escapes nodetop emits deliberately are untouched. This is
  also the `--replay` boundary: a snapshot is a JSON file that may have come
  from someone else, and reading one must not repaint your terminal.
- **Wrapping never invents a different flag.** `textwrap` splits on hyphens by
  default, which turns `--test-only` into `--test-` / `only` and `beagle3-0001`
  into two node names. Disabled everywhere, and asserted across every width
  from 24 to 80 columns.
- **Nothing may be wider than the window.** Tables shrink their widest columns
  to fit (headers included — truncating only the data leaves the header row as
  the one line guaranteed to overflow), panels clip their content, and prose
  wraps. Truncation is ANSI-aware, so cutting a coloured cell preserves the
  escape sequence and appends a reset rather than silently eating visible
  characters. Verified as a test across every command at 60/80/100/120 columns.

Meters use eighth-block sub-cell resolution, which is not cosmetic either: at
14 cells a naive bar rounds 2-of-176 free accelerators down to empty, and the
difference between 0 and 2 free is the whole question.

Three more decisions in the same drawing code, each of which had a wrong version first:

- **A bar is a box with a level in it, not a stripe.** The unfilled remainder is drawn in
  a near-background grey rather than left blank, so the eye has a fixed reference to
  measure a short fill against — `14%` and `55%` are not comparable at a glance without
  one. That grey is a step *below* the grey used for de-emphasised prose: it is a
  reference mark, not content.
- **The fill is a darker twin of the tone its number wears.** A bar is a slab and text is
  a line; the tone that reads as bright in a four-digit number reads as shouting across
  sixteen filled cells, and ten shouting bars are a wall. Terminals have no alpha channel,
  so the wash is a genuinely darker colour of the same hue — what compositing that hue at
  ~60% over a dark background would have produced. At 16 colours there is no room for a
  second copy of every step, so the fill simply keeps the text tone.
- **Secondary numbers get their own grey, not the prose grey.** A total beside the free
  count it divides is still *content*; painting it the same grey as a hint or a caveat is
  what made whole numeric columns read as furniture.

Panel borders carry a diagonal colour sweep — hue advances with `x + y`, lightest at the
top-left, the way a highlight falls across a glossy surface. Every anchor colour in it is
a *light* one, on purpose: a frame that sweeps light-to-deep puts the darkest end at the
bottom-right, where on a dark terminal it disappears, and a border that fades out halfway
down reads as a rendering fault rather than as a gradient. The sweep moves in hue and
stays put in brightness, and it is drawn in runs of equal tone — about ten escape
sequences per border rather than one per column.

`--help` is coloured too, and it is coloured *after* argparse has formatted it. argparse
lays its columns out with `len()`, so painting the strings it is handed throws every
column off by the width of its own escape sequences. Four roles and no more — flags and
sub-commands in blue (what you type), placeholders in amber (what you substitute), section
headings bold, defaults and example notes dim — because a help screen wearing a dozen
colours is harder to read than one wearing none.

The one deliberate exemption is the `to request exactly what was checked` line.
It is neither wrapped nor truncated, because it exists to be copied and an
ellipsis or hanging indent there would hand you a broken command.

## Library

```python
from nodetop import Cluster, JobShape, rank

cluster = Cluster.load()                      # autodetects the batch system
cluster.backend_name                          # 'slurm'
cluster.can_probe                             # False on PBS and LSF -- check this
cluster.queues["test"].usable                 # False
[b.code for b in cluster.queues["test"].structural_blockers()]
# ['QUEUE_DISABLED', 'NO_ACCOUNTS', 'NO_QOS']

shape = JobShape(nodes=1, gpus_per_node=4, gpu_memory_gb=40,
                 requires=("bf16",), walltime="2-00:00:00")

for place in rank(cluster, shape, use_probe=True):
    print(place.queue, place.runnable_now, place.confirmed, place.earliest_start)
```

Every query runs against one snapshot, so all the numbers in a report describe the same
instant instead of drifting across a dozen independent calls — enforced, not just
intended: a test asserts no backend issues the same command twice, since fetching a
source again for a second consumer is how two instants end up in one report. A query that fails is
recorded in `Cluster.errors` rather than silently becoming an empty result.

### Post-mortem

When a partition goes down mid-run, the evidence is gone by the time anyone looks.
`nodetop snapshot` records what the queries returned and **every command replays against
it unchanged** — on a laptop, days later, with no cluster in sight:

```bash
nodetop snapshot -o outage.json          # ~500 KiB for a 607-node cluster
nodetop --replay outage.json status      # or where / accelerators / health / queues
```

Replay needs no special code path in the analysis layer: the same backend runs against a
recorded runner instead of a live one, so it works for every backend for free. And the
recording is honest about what it is — `can_probe` is **False** on a replay even for
Slurm, because a recording holds the answers to the queries that were made, not to a
dry-run nobody ran.

**A replay carries the data's own clock.** The banner says how old the snapshot is, and
every relative time below it — `START`, the age of a drain reason — is measured from the
capture instant, not from when you opened the file:

```
replaying outage.json (slurm, captured 6d 4h ago)
```

This was wrong for a while, and the interesting part is which half mattered. `captured_at`
was being written and never read, so a replay stamped itself with the moment it was
opened. Misdating a post-mortem is a nuisance; the real damage is that node free times are
*absolute instants*, so comparing them against `now()` inflates every wait by the
snapshot's age. A node recorded as free in three hours reads `overdue` the moment the
recording is older than three hours — an authoritative-looking number that is pure
arithmetic error. Fifteen snapshot tests passed throughout; none of them looked at the
time.

Elapsed time and future waits are also formatted by separate functions on purpose.
`format_wait` says `now` under a minute and `overdue` for anything negative, which is
right for a start estimate and wrong for an age: reusing it printed "captured now ago",
and "captured overdue ago" for a recording made on a host whose clock ran fast.
`format_age` returns `None` there instead, so the caller reports clock skew rather than
rendering a duration for it.

Malformed output is exercised as well — truncated records, CRLF line endings, a JSON body
cut mid-object, a node listed twice, garbage in a numeric field. A parser that *raises*
is fine: the failure lands in `Cluster.errors` and the report says it is partial. The
dangerous case is one that succeeds on garbage, and two of those were real: a truncated
Slurm record carried no state at all, which made it read as schedulable *and* empty —
the most attractive thing in the cluster to a placement search — and `CPUAlloc=-5`
against `CPUTot=0` reported five free CPUs that did not exist.

Degenerate shapes are exercised too — an empty cluster, a single node, every node down,
a queue nothing belongs to, a 200-character queue name, ten thousand nodes — against
every command in every output mode. An empty cluster is treated as a *finding* rather
than reported blandly as "0 nodes", since in practice that reading means the wrong
backend or an unreachable control plane far more often than an empty cluster.

Docstring examples are executed, not just written: every worked
`input -> output` in `src/` is a test case, because a docstring that has quietly become
false is a confident wrong answer at exactly the moment someone is trying to understand
the code. That check found `gauge`'s own example showing eight filled cells and a shaded
trough for a value that renders as one partial cell over dots.

Help text is held to the same standard: every flag must have a description and a
metavar, and every behavioural claim in it is asserted — `--gpu` really returns only
accelerator nodes, `exclude --degraded` really returns exactly the impaired-but-
schedulable set, and every walltime form the help spells out really parses that way.
That check found `check --gpu-mem` claiming a validation it does not perform.

**Every command is rendered against every backend, at every width.** The rest of the
render suite runs on the Slurm fixture, which left the other five adapters' output
unmeasured — and they differ in exactly the ways a layout cares about: the queue term is
a different length (`partition` / `queue` / `namespace` / `pool`), the capability notes
are different prose, and `probe_supported=False` on PBS, LSF and the ssh pool reaches the
"declared, not confirmed" branch with different text in it. Pointing the sweep at the
other four found an unwrapped `re-run with --all` line, and — more usefully — that the
sweep's own exemption for the copy-me submit line tested for a leading `--`, which is
what *Slurm* happens to emit. PBS opens that line with `-q` and Kubernetes with `-n`, so
on every other backend the exemption silently did nothing. It now asks the cluster for
the flags instead of matching a prefix.

The lesson generalises past width: **a rendering path gated on a mode the fixtures never
enter is invisible to a sweep that looks exhaustive.** So the gates are now enumerated
and each one gets its own sweep:

| gate | what it unlocks | what was hiding there |
|---|---|---|
| `replayed=True` | the "declared, not confirmed" explanation | a 148-column line |
| a non-Slurm backend | different queue terms, notes, flag syntax | an unwrapped line, and a Slurm-shaped test exemption |
| a probe that answered | the `ACCESS` column, the disagreement heading | `section` clipped its note but never its title |
| a degenerate shape | 200-character names, 100,000 accelerators | nothing — but it was only ever checked at one width |

Each of the first three hid a defect that no amount of running the existing suite would
have surfaced. The fourth is the useful negative result: the shapes were already sound,
and the coverage that proves it went from one width to four.

A backend's `BackendCapabilities` is a declaration about itself, and the reporting layer
trusts it when deciding whether to say *confirmed* or *declared*. So the suite holds each
one to it: a backend claiming it cannot dry-run must not produce a verdict, one claiming
it can must name the command, and one that cannot must explain why.

### Adding a backend

Implement the `Backend` protocol — return the neutral objects, and declare what you
cannot establish:

```python
class MyBackend:
    name = "mine"
    queue_term = "queue"

    @classmethod
    def detect(cls) -> bool: ...
    def capabilities(self) -> BackendCapabilities: ...   # see the two probe flags
    def load_nodes(self) -> list[Node]: ...
    def load_queues(self) -> list[Queue]: ...
    def load_limits(self) -> dict[str, Limits]: ...
    def probe(self, queue, shape, account=None) -> Verdict | None: ...
```

None of the reasoning needs touching. That is the test of whether the layering is real.

Two things to get right in `capabilities()`:

- **`probe_supported` and `probe` answer different questions.** The first is whether the
  batch *system* has a dry-run at all; the second is whether it can be run from *this*
  host, so it is normally `which("your-client")`. One field doing both got both wrong: a
  Slurm login node reported that SGE has no dry-run — the truth was only that `qsub` was
  not installed — while the Kubernetes adapter hardcoded `probe=True` and so advertised
  confirmable entitlement on a machine with no `kubectl`. The reference table reads the
  capability; every "declared vs confirmed" decision reads the local one.
- **Gate `probe()` on `capabilities().probe` and return `None`.** Not a refusal, not an
  error — no answer, so the caller falls back to the declared ACL. Running the command
  anyway turns a missing client into a `CONTROL_PLANE_DOWN` verdict, which is in
  `TRANSIENT_CATEGORIES`, so the report invites a retry for something no amount of waiting
  fixes and blames the cluster for a local problem. `TestProbeIsGatedOnItsClient` enforces
  this across the registry, because two of the three probe-capable backends had the guard
  and the third did not.

## The bias, stated

**Every inference fails toward less.** Where a fact is missing or ambiguous, nodetop
answers with the reading that claims *less* capacity and *less* access — never more:

| ambiguity | nodetop's answer |
|---|---|
| a truncated node record with no state | unschedulable, not idle |
| an unreadable resource count | zero, never negative |
| an accelerator whose model is unidentifiable | does not satisfy a stated capability — set aside and reported |
| a memory size that could be 40 or 80 GB | assume 40 |
| a host group that cannot be expanded | the members resolved, not the whole cluster |
| a consumable whose total is invisible | the amount known free, not an assumed total |
| a queue whose allowlist names nobody | nobody, not everybody |
| a ceiling that will not parse | check skipped **and said so**, no invented limit |
| a dry-run that could not be run | *declared*, never *confirmed* |
| a free-time already in the past | **overdue**, not "now" |
| a start time we computed ourselves | marked `*` as a lower bound, since it ignores the queue |
| a timestamp carrying a timezone | converted to local before comparison, not stripped |

On the time axis the same bias reads as *later*: never promise a resource sooner than it
will be free. A node whose job has overrun its walltime reads `overdue`, not `now` —
rounding a negative interval to zero sends someone at a node that is still busy.

The failure mode of that bias is a needless warning. The failure mode of the opposite
bias is a job sent somewhere it cannot run, discovered ninety minutes later. Sixteen
iterations of boundary testing found the same asymmetry repeatedly — nine of ten bugs
erred toward claiming capacity that was not there — so it is now written down in
`core/capacity.py` and enforced by `tests/test_fail_safe.py` rather than rediscovered.

## What it will not do

It is **read-only**. The only commands any backend may run are dry-runs that create
nothing — `sbatch --test-only`, `qsub -w v`, `kubectl --dry-run=server` — and each backend
hard-codes its dry-run flag so a caller cannot omit it. nodetop never submits, cancels,
holds or requeues.

Four honesty rules are enforced in code rather than left to the reader:

| Rule | Why |
|---|---|
| An unreachable queue reports **no** start estimate | Schedulers return a plausible start time next to a refusal; showing it reads as encouragement to wait for something that will never run. |
| A start time we computed ourselves is marked `*` | Counting when nodes free up ignores every pending job ahead of you, so it is a *lower bound*. Unmarked estimates came from the scheduler. |
| “Wrong hardware” and “the right nodes are all down” are different verdicts | The first is durable, the second is about today. Conflating them turns an outage into a permanent-looking answer. |
| “We could not check” never renders as “allowed” | On PBS, LSF and an ssh pool there is no dry-run, and that gap is printed, not papered over. |

### Known limits

- **Only the Slurm backend has been validated end-to-end against a live cluster**, plus
  the ssh pool against a real unscheduled GPU box. PBS, LSF, Grid Engine and Kubernetes
  are built from those systems' documented output formats and tested against
  format-faithful fixtures — solid, but not yet confirmed against a live control plane.
- **`health` only finds impairment somebody wrote down.** It keyword-matches the reason
  field, so a node throttled with no reason recorded is invisible from a login node. That
  needs per-node telemetry, which is a different tool's job.
- **PBS `Qlist` restricts a node, it does not enrol one.** A node declaring no
  `Qlist` is unrestricted and any execution queue may use it. Requiring an
  explicit mention orphans every unrestricted node — its capacity becomes
  invisible to all queues — and leaves a queue no node happens to name looking
  genuinely empty. Routing queues (`queue_type = Route`) are recognised as
  such: they forward, own no nodes, and are not offered as placement targets,
  since their capacity belongs to their destinations.
- **LSF host groups need `bmgroup`.** A queue scoped to a host group nodetop
  cannot expand reports the members it resolved plus an unresolved count — it
  does not fall back to "every host", which would hand a four-machine queue the
  free capacity of the whole cluster.
- **PBS free-time estimates are an upper bound.**  PBS records no end time, so
  it is computed as `stime + Resource_List.walltime`.  A job may finish early,
  so the node may free sooner than reported -- never later.
- **Grid Engine accelerator totals need `qconf -se`.** `qhost` reports only how much of
  a consumable is *available*, so a fully busy GPU host reads as `0/0` and vanishes from
  the inventory entirely. nodetop reads the configured total from each accelerator host's
  exec definition — capped at 96 hosts, one round trip each, since Grid Engine has no
  bulk equivalent. Where that is unavailable it falls back to the available count and
  says on each affected node that the count is unknown.
- **Slurm's submit-filter output is site-shaped.** Stock Slurm phrasings are covered too,
  but a site with an unusual `job_submit` plugin may land in `UNKNOWN` rather than a
  specific category. `UNKNOWN` is treated as non-durable, so it never hardens into a
  false claim about your access.

## Development

```bash
pip install -e ".[dev]"
pytest          # 3140 tests, no batch system required
ruff check src tests
```

The suite is checked by mutation rather than trusted: deliberate breaks to the decision
logic — inverting `Node.schedulable`, letting `effective_free_nodes` ignore usability,
making `MAX_WALLTIME` never fire, stripping timezones instead of converting them, letting
negative waits round to `now`, grouping node reasons by their `[who@when]` stamp, reading
a backend's dry-run capability off the local client — and **every one is caught**. That
matters because four tests have been found *certifying* bugs: a test written to describe
behaviour rather than demand it will happily pin a mistake in place. The most recent
asserted `kubernetes.can_confirm_entitlement is True` unconditionally, which is exactly
the over-claim the backend was making.

> When running mutants, check that tests actually **ran**. `addopts` here already carries
> `-q`, so adding another gives `-qq`, which suppresses the summary line — and a harness
> grepping for `N failed` then reads a fully-aborted collection as a clean pass. "0
> failed" and "0 ran" are indistinguishable unless you assert the count.

The suite runs entirely against recorded output — which is also the only way to exercise
the interesting states on demand: a disabled queue, a cordoned node, a submit filter that
disagrees with its scheduler, a quota that admits then refuses. The Slurm probe fixtures
in `tests/fixtures/slurm/probe_outputs.py` are verbatim captures, including the
inconsistent `sbatch: error:` prefixing.

## License

MIT
