# Design notes
Why nodetop behaves the way it does. Extracted from the README, which is
now a README. Each section is a defect that was found and what it cost;
the code comments carry the same reasoning where it applies.

## The same lies, everywhere

| The lie | Slurm | PBS / LSF / SGE | Kubernetes |
|---|---|---|---|
| “There are idle nodes here” | `State=DOWN` with all 610 nodes still listed | `started=False`, `Open:Inact`, every queue instance disabled | `Ready` but cordoned, or tainted and untolerated |
| “You have access” | association table lists 90 partitions per account | `acl_users` enabled and empty | RBAC allows, admission webhook refuses |
| “Verification passed” | site filter says PASSED, core refuses anyway | *no dry-run exists at all* | `--dry-run=server` (this one is honest) |
| “No time limit” | `MaxTime=UNLIMITED`, QOS caps at 2 days | queue unlimited, `max_run_res` caps | no walltime concept |
| “Admitted, so it will run” | pends forever on `QOSMaxGRESPerJob` | pends forever on a queue limit | `Pending` forever, within quota |
| “4 GPUs available” | `gres/gpu=4` | `ngpus=4` | `nvidia.com/gpu: 4` |
| “44 of 48 cores are idle” | `AllocMem=RealMemory`; nothing can land there | mem consumed, ncpus free | `requests.memory` exhausted |
| “There is room, so it starts now” | `--test-only` itself says 4h 24m | queue ahead of you, unreported | scheduler backoff |

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

PBS's two words do not *fit*, though, and that turned out to matter. The state column is
twelve characters -- sized for Slurm's `UP` -- so `enabled=True started=True` rendered as
`enabled=Tru…`: cut off exactly at the answer, and the same eleven characters whether the
queue was open or shut. Seen on a live 2026.1.0 cluster listing ten queues, two of them
closed, all of them indistinguishable. The switches are independent, so each of the four
combinations now gets its own word -- `UP`, `STOPPED` (accepts, never runs), `DISABLED`
(runs, accepts nothing), `DOWN` (neither) -- while `enabled` and `started` stay on the
queue as booleans, because they are what every decision reads.

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
An allowlist of `pi-okafor` alone does. That leaves **two** shared partitions with free
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
drew **0%**; `gpu-a` had 200 cores and 27 GPUs free and drew **0%** — while
`gpu-a-bigmem`, with 128 free cores, drew a full bar and took the top row. The meter was
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
could never do: `grp-h` names four accounts and `grp-i` five, so a "fewer than three
accounts means private" rule let both through, while a set intersection excludes them
instantly.

The second is the dry-run, and it is not optional, because the first stage is nowhere near
sufficient: **of the 19 partitions the allowlist keeps, a dry-run accepts 8.** The
association table lists this user in `grp-e`, `pi-okafor`, `pi-tanaka`, `pi-varga`,
`pi-ibrahim` and `pi-svensson`, and the submit plugin rejects every one with `Invalid
membership`. No reading of any declared list can see that. It costs ~2.8s rather than 30
precisely because the allowlist filter runs first — 19 queues to ask about, not 84.

**And the dry-run has to ask about the right account, which is where this went wrong.**
`probe_accounts` used to truncate its candidate list to four, on the reasoning that a user
with dozens of associations against dozens of queues would otherwise fire hundreds of
submissions. That is a real cost, but truncation is the wrong place to pay it: the general
partitions here set `AllowAccounts=ALL`, so the intersection is all 34 accounts and only the
first four were ever tried. `wide` (190 nodes), `gpu` (44 accelerators) and `bigmem` are
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
  — and then `gpu-a` accepted that very account one queue later.

The result is both more correct and faster: 8 partitions instead of 5, in 2.8s instead of
3.5, because most queues now settle on their first probe.

**And a refusal that was not established is not reported as one.** If the per-queue ceiling
is reached with candidate accounts still unasked, the verdict becomes
`ACCOUNTS_UNTRIED` — a *transient* category, so `Placement.confirmed` stays false but
`durable` does too. The overview keeps that partition, marked `unconfirmed` in the funnel's
shown count rather than filed under `refused`, because "we did not ask" and "you are denied"
are the exact pair of claims this whole tool exists to keep apart.

```
 ada  ·  607 nodes, 548 up  ·  358 GPUs, 110 free

 87 partitions  →  5 open to you  ·  65 no access  ·  14 refused  ·  3 dead
 ───────────────────────────────────────────────────────────────────────────
 partition       nodes  idle  cores  free  share             gpu  free  models
 amd                40     0   5120  2825  █████▌░░░░   55%
 gpu-a            44     0   1408   200  █▍░░░░░░░░   14%  176    27  A100, A40
 compute-hm              1     1    128   128  ██████████  100%
 gpu-a-bigmem      4     4    128   128  ██████████  100%
 build               1     0     48    42  ████████▊░   88%

 DEAD  150 idle nodes advertised
       test DOWN, 150 idle  ·  eng all 48 nodes down  ·  eng-build all 2 nodes down

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
  free *share*; tone is how its free cores rank against the other rows. `compute-hm` draws a
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
| `gpu-a` | 28 accounts | no | **confirmed** |
| `gpu` | *none* (open) | no | `NOT_ENTITLED` |
| `grp-d-gpu` | 1 account | yes | `NOT_ENTITLED` |
| `grp-e-gpu` | 1 account | yes | `NOT_ENTITLED` |
| `grp-f-gpu` | 1 account | yes | `NOT_ENTITLED` |

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
reads back the QOS the controller *actually chose* — a request on `gpu-a` came back
running under `gpu-a-prio`, and checking ceilings against the name you asked for checks
the wrong ceilings.

nodetop also notices when a claim is worthless on its face, and says so in `status`
rather than burying it under every queue:

```
╭─ nodetop ─────────────────────────────────────────────────────────────────╮
│ slurm  ·  607 nodes  ·  91 with accelerators  ·  87 partitions            │
│ entitlement  confirmable via sbatch --test-only                          │
│ you          ada  ·  34 accounts  ·  92 QOS                           │
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

### 3a. Idle cores with no memory behind them

`wide` advertised **2322 free cores**. 2035 of them were unusable, and the reason was
not in the core counts at all:

```
$ scontrol show node cn-0023
CPUAlloc=4  CPUTot=48  RealMemory=184320  AllocMem=184320
```

Four cores in use, forty-four idle, and every byte of memory allocated to the job holding
those four. The cluster runs `SelectTypeParameters=CR_CORE_MEMORY`, so memory is a
consumable resource: nothing more can land on that node. 47 of `wide`'s 190 nodes were
in exactly that state, and `DefMemPerNode=UNLIMITED` there means a job that names no
`--mem` asks for the *whole node* — so not even a one-core job with no memory request
would fit.

The old behaviour reported all 2322 as free, ranked `wide` first on the strength of it,
drew it a full-length meter, and told `where -c 4` that 79 nodes fitted. The honest
numbers are 287, second place, and 32.

Two things make this safe to apply rather than a new way to be wrong:

* **`memory_mb <= 0` means "not reported", not "none".** A backend that never mentions
  memory has the constraint skipped.
* **Not every Slurm cluster enforces memory.** Without `_MEMORY` in
  `SelectTypeParameters`, Slurm never decrements it, so `AllocMem` records what jobs asked
  for rather than a ceiling — and reading it as one would report a whole cluster as full.
  `SlurmBackend.memory_is_consumable()` asks, once, and stamps the answer onto every node.
  Unreadable config claims *less* capacity, which is the bias everywhere else here.

### 3a-i. Zero was the wrong floor

`3a` above stopped counting cores on a node whose memory was *entirely*
allocated. A fifth cluster showed the same defect one resolution finer, and it
took a reader noticing a row that looked wrong to find it:

```
◐  midway3-0512  MIXED    97/128  ██████░░   0/244G
```

97 idle cores, a three-quarters-full meter, and `0/244G` beside it. The node has
`RealMemory=250000` and `AllocMem=249750` — **250 MB free**, which rounds down to
0 GiB in the column and sat 250 MB above a test written as `free <= 0`. That
cluster's `DefMemPerCPU` is 3810, so a single core of a job that names no
`--mem` needs fifteen times what is left: nothing can start there, and the row
sorted to the top of the listing as the roomiest thing on screen.

The floor is now the scheduler's own — `DefMemPerCPU`, read from the config
query already being made — so the comparison is against what Slurm will actually
hand out rather than against zero. What it recovered on that cluster:

| | free cores reported |
|---|---|
| before | 14,352 |
| after | **10,891** |

**3,461 phantom cores across 153 nodes**, a quarter of the total the tool had
been publishing. And it is targeted rather than blunt: on two other Slurm
clusters, whose floors are 16384 and 2048 MB, not one node sat in the gap and
the numbers did not move at all.

Two limits stated rather than papered over. A job naming a small `--mem`
explicitly can still land on such a node, so this is the same "claims less
capacity" bias as everywhere else here and not a hard impossibility — the
claimed counts stay in the column beside the meter. And a *partition* may
override the cluster's `DefMemPerCPU`; that is not modelled, so a partition with
a higher floor is under-detected.

The node's state string is left exactly as the scheduler wrote it. Slurm derives
`MIXED` and `ALLOCATED` from CPU allocation alone and never folds memory in, so
a memory-full node reads `MIXED` there; rewriting the scheduler's own word would
be a different kind of lie. The mark, the meter and the ordering are nodetop's
to get right, and those are what changed.

### 3b. Free nodes are not a start time

`sbatch --test-only` will tell you when the scheduler expects to start your job. nodetop
had that number in hand and printed `now` instead, because the row was decided by free
hardware alone. For a four-core ten-minute job, with every one of these partitions
reporting free cores:

| partition | nodetop said | the scheduler said |
|---|---|---|
| `gpu-a` | RUN NOW | now |
| `bigmem` | RUN NOW | now |
| `amd` | RUN NOW | **in 4h 24m** |
| `build` | RUN NOW | **in 8h** |
| `wide` | RUN NOW | **in 18h** |

`amd` is the largest pool of free cores on the cluster, so it is the top row of every
listing — and it was four and a half hours from starting anything. `Placement.starts_now`
now asks both questions, and a placement with room but a queue ahead of it falls through
to `QUEUE`, which is a different next move: submit and wait. Where no dry-run exists the
two questions collapse into one, rather than answering "no".

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
  ◐  QUEUE     gpu-a             0/1    44/44    44m  confirmed         A100x22, A40x22
  ○  BLOCKED   grp-b-gpu         2/1     6/30      ·  ACCOUNT_MISMATCH  H200x4, A100x2
  ○  BLOCKED   hcn1-gpu            1/1      1/1      ·  INVALID_QOS       L40Sx1
  ▲  LIMIT     grp-z-gpu         3/1      3/3      ·  confirmed         A100x2, H100x1
  ✗  WRONG HW  grp-k-gpu       0/1      0/2      ·  confirmed         RTX6000x2

  ● runs now  ○ not permitted  ✗ no node of the right kind
  ▲ over a declared ceiling  ◐ would queue
```

Note `grp-b-gpu`: two nodes free, six capable, and you still cannot use it. Without
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
per-queue verdicts, and a test asserts the two renderers cannot diverge.

That property has been broken twice and both were the same shape: `status --json` returned
`Cluster.summary()` and returned it *early*, so it answered a different question than the
panel — 358 accelerators and 126 free, cluster-wide, where the panel said "222 of 358
GPUs, 53 free" for the partitions this account can submit to — and it carried neither the
funnel nor a single partition row. `queues --json` carried no core figures at all while
its text form printed two. Both now emit from the same population the text does, which
means `status --json` pays for the same dry-runs the panel pays for; `--declared` skips
them for both forms alike. One name per quantity, too: `effective_free_cpus` means the
same thing in `status`, `queues`, `nodes` and `zoom`, and sums across them. The vocabulary follows the system — `partition` on Slurm, `queue` on PBS/LSF/SGE, `pool` with no
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
  hostname.** Clusters routinely have a `gn-bigmem1` with no GPU sitting among 44
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
  ◐  cn-0507  MIXED  ███████▏  114/128   29/244G  ·
  ◐  cn-0519  MIXED  ███████▏  114/128   29/244G  ·
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

### 1h. A per-node view has to show the per-node share

The job table under a node showed each job's totals across every node it holds, because
that is what a job list reports. On a 48-core machine:

```
job         user   cpu       gpu  used        left     name
4210001    rmartin   512 x42   ·    1-06:13:04  5:46:56  _interactive
```

512 cores on a 48-core node, marked `x42` to mean "spread over 42 nodes" — a number the
reader knows to be impossible next to a marker nobody could decode: *"the cpu column
doesn't make any sense. what do the column entries mean?"* And no memory column at all,
on the resource that most often decides whether a node is usable.

Only the scheduler knows the split, and it will say:

```
$ scontrol show job -d 4210001
     Nodes=cn-0114 CPU_IDs=41-47 Mem=7168 GRES=
```

Seven cores and seven gigabytes, not 512. So `Allocation` is now fetched and the columns
are `cpu`, `mem`, `gpu` — this node's share — with the span in its own `nodes` column,
present only when something actually spans. Three details made it work:

* **One call for the whole cluster.** 0.6s and 4.7 MB for 2928 jobs, against 0.13s for a
  single job — so asking about five jobs already pays for asking about all of them, and a
  node with 49 array tasks on it would otherwise stall an interactive repaint for six
  seconds. Fetched lazily on the first per-node view and cached.
* **`squeue` and `scontrol` disagree on what a job is called.** `squeue` names a running
  array task `4210001_132`; `scontrol` gives it a JobId of its own and records the array
  separately. 1864 of 2928 jobs here are array tasks, so keying on `JobId` alone would
  have found a share for none of them. Each allocation is registered under both spellings.
* **`Nodes=` is a nodelist.** Slurm collapses consecutive nodes that got the same shape of
  allocation — `Nodes=cn-[0521-0522] CPU_IDs=78-94` — and the figures then apply to
  each of them.

A single-node job needs no lookup at all: its totals *are* its share, which is most jobs
and the whole answer on a backend that models a job as living on one machine. A multi-node
job whose share cannot be established prints `?` rather than substituting a total.

### 1i-a. Right means deeper, and at the bottom deeper is nowhere

At the job detail, Right went back to the job list. `select` returns the index
when Right is pressed at the end of a row -- "nothing further right: open it" --
and a caller with nothing deeper to open could only read that as "step back", so
the key for going in came out. "pressing right arrow will go into the same
interface ... this is very confusing."

`openable=False` is the mirror of `escapable=False` at the root, and for the
same reason: at an edge, the key that moves that way does nothing rather than
something surprising. Right and Enter are ignored at the leaf, Left and Escape
leave, `q` quits. A node running no jobs is a leaf too -- its single row is a
placeholder, not something to open.

### 1i. Everything is a level, including the leaf

Two dead ends, both found by using the thing:

* **Enter on a job did nothing.** The stack popped instead of pushing, so the row was the
  deepest the tool went — with the job name truncated and its node list never shown.
  *"when choosing any of the job here, it doesn't go into the job details but going back
  to the original node".* A job now has its own view: the name in full, its share of this
  node beside the job's totals, and the whole nodelist.
* **Enter on a drained node showed four lines saying "nothing running here".** No state, no
  reason — and the reason is what the reader opened it for, truncated in the listing at
  `maintenance [root@…`. *"after hitting this one, nothing shows up, even people wanting
  to see the reason why this node is down."* The node's own view now leads with its state,
  prints the reason whole with the operator and timestamp separated out, says when the
  control plane has lost contact, and — for an unschedulable node — says "nothing running
  here, and nothing can start" rather than the phrasing that reads as *free*.

The funnel's own total became a target for the same reason: *"why can't we select the 87
partitions?"* It opens every queue on the cluster with the word that put it there, which
is the one view where the funnel's arithmetic can be checked rather than trusted. The `→`
between the total and the first term went with it — *"why there is a right arrow here? it
makes no sense at all"* — because once every term is a peer they read as a list and are
punctuated as one.

### 1j. The redraw, twice broken

**A frame exactly as tall as the terminal does not fit.** The node listing reserved six
rows for its own chrome and filled the rest, so its frame came out at exactly `LINES` —
and the last line's newline scrolls the screen by one, which puts the repaint's cursor-up
one line low. Every keypress then orphaned a top border:

```
╭──────────────────────────────────────────────╮
╭──────────────────────────────────────────────╮
╭──────────────────────────────────────────────╮
   ... thirteen of them
│ gpu-a  ·  44 nodes  ·  28 with room        │
```

One spare line, subtracted inside the windowing helper so no caller has to remember it.
Below ten rows the chrome alone exceeds the screen, and there `interactive.supported()`
now returns False: the static print scrolls, which is merely inconvenient.

**And the repaint blanked the screen before drawing.** It moved to the top of the block,
cleared everything downward with `ESC[J`, and only then wrote the new lines — two writes
with the screen empty in between. Holding an arrow key turned that gap into a strobe:
*"when pressing down arrow constantly, the app is flickering"*. Two fixes, measured
through a pty against the real cluster with twelve Down presses:

| | before | after |
|---|---|---|
| frames drawn | 12 | **1** |
| screen-clearing escapes | 12 | **0** |

Each line is now written over the old one and cleared only to end-of-line as it goes, in a
single write per frame, so no cell is ever empty. And a burst of keypresses coalesces:
a held key arrives as a stream of escape sequences and only the last position matters, so
the repaint waits until the input queue is empty — capped, because a screen that never
updates would be worse than one that updates too often.

### 5. A device index read as a device count

The worst defect found so far, and it looked like a shuffled table rather than a parse
error. Three jobs sharing one node's four accelerators:

| job | `GRES=` | reported | actual |
|---|---|---|---|
| 4210001 | `gpu:2(IDX:0,3)` | **0** | 2 |
| 4210001_1 | `gpu:1(IDX:2)` | **2** | 1 |
| 4210001 | `gpu:1(IDX:1)` | 1 | 1 |

Slurm appends *which* devices, not just how many, and the suffix holds both colons and
commas — the two characters this field is split on. `gpu:2(IDX:0,3)` split on commas gives
`gpu:2(IDX:0` and `3)`; the first, split on colons, ends in `0`. So the parser read the
device index as the count, and the wrong number it produced was often another job's, which
is why it read as rows out of order. The invariant that would have caught it in one line:
0 + 2 + 1 = 3 on a node reporting all four allocated.

The same field on a node — `Gres=gpu:v100:4(S:0-1)`, Slurm printing socket affinity —
parsed as **zero accelerators**. This cluster does not print that suffix, so that half was
latent, and it would have made every GPU node on a cluster that does print it look like a
CPU node.

Fixed by removing every parenthesised group before splitting. Then verified rather than
assumed, against the scheduler's own answers on a live cluster:

| checked against | count | mismatches |
|---|---|---|
| `scontrol show node` — cores, memory, accelerators, allocated and total | 607 nodes × 6 fields | 0 |
| `squeue` — per-job totals | 1209 | 0 |
| `scontrol show job -d` — per-node shares | 1717 | 0 |
| `sinfo` — per-partition accelerator totals | 92 | 0 |

Two things that look like discrepancies and are not. Rerunning the sweep shifts a handful
of `*_alloc` fields on two or three nodes: that is a job starting between the snapshot and
the check, and the mismatches move rather than persisting. And a **cancelled** job's
allocation block keeps its CPU and memory lines while `GRES=` empties out, because Slurm has
already taken the accelerators back — so a node in teardown can show shares that sum to
less than it holds. Both are the cluster changing, not the parser.

### 5a. And a count that was not a number

Swept for the same defect elsewhere and found its opposite in PBS. Its node parser called
`int()` directly on the field:

```
resources_available.ngpus = unlimited     # PBS, for an uncapped resource
resources_available.ngpus = 4x            # a site script
```

`int()` raises on both, and the exception went straight through the node parser — so one
odd field on one node emptied the *entire* node list, and an empty node list is reported as
"no nodes -- wrong backend, or the control plane is down". A misdiagnosis rather than a gap,
which is the worse of the two failures. The same code let a *negative* count through, which
Slurm's own helper exists to prevent: `cpus_free` is `total - alloc`, so an allocation of -5
against a total of 0 reports five free CPUs that do not exist.

Both adapters now share one `count()` in `backends/base.py` rather than each keeping its own
— having two implementations of the same job is how they came to disagree in the first
place.

### 6. `·` cannot mean two things -- or a number

`·` is this tool's empty cell. In a column headed `gpu free` it was appearing both on a node
with no accelerator installed and on a job holding none of a node's four — *"putting a dot
there means nothing. you put the same sign in the gpu partition but there is gpus in those
nodes."* Then it turned up again in the `nodes` column, standing in for the number **1**:
*"what does . mean in the node column? why can't you put 1 there?"* A single-node job spans
one node, and a count column holds counts.

So the rule is written down on the glyph itself. `·` is a **separator between words** and
never the content of a cell. A measurement goes in as a number, `0` included. `—` is what a
question that does not arise looks like — a node with no accelerator under `gpu free`, a
partition that can start nothing under `start`, a field the control plane never reported —
and it reads as not-applicable with colour off and in ASCII. Six more cells were carrying a
`·` for that meaning: the routing-queue rows in `queues`, the accelerator column in
`health`, `start` and `gpus` in `where`, the unreported account/QOS/filter fields in
`check`, and the vendor/arch/memory of an unidentifiable accelerator in `gpus`. A test now
sweeps the grid rows of seven commands and fails on any cell that is a bare `·`.

Where a whole column would be dashes, it is dropped instead: `table` already omits a column
no row fills, so a CPU-only partition spends no width on a question that does not arise.

The headers went the same way. It was `cpu | free | mem free | gpu`, with `cpu` over the
meter and a bare `free` over the fraction beside it — *"what does 'free' mean here? and then
after that, you have 'mem free'. why so many frees?"* One word was doing the work of three
labels and none of them said which resource it belonged to. Now every column names its own:
`cpu free`, `mem free`, `gpu free`, with the meter unlabelled beside the number it draws, in
the same order the overview's table uses.

### 7. A scheduler's vocabulary is versioned, and an unknown word reads as "fine"

Everything above was found on one Slurm cluster. Taken to a second one -- same scheduler,
Slurm **25.11** against the first site's older build, a different site's conventions -- six
defects surfaced in an hour, and five were the same shape: a state or a sentence this
version spells differently, met by a parser that had no case for it and therefore said
nothing was wrong.

| what the cluster said | what nodetop made of it | cost |
|---|---|---|
| `State=MIXED+PLANNED` on two H100 nodes | "all 2 nodes are down, drained or unreachable" | both GPU partitions struck out, all **8 H100s erased** from the inventory, `where -g 4` answered "no partition can run this shape" |
| `State=DOWN+NOT_RESPONDING` (no `*` anywhere in the record) | merely DOWN | a node **23 days** out of contact, and "the control plane has lost contact" unsayable on that cluster |
| a hidden partition's node, absent from a node query without `--all` | "1 node claimed but unresolved" | 31 nodes where the cluster has 32; that partition reported no capacity at all |
| `to start at <time> a using 1 processors on nodes mcn53` | start time kept, nodelist dropped | `predicted_nodes: []` on **every** accepted dry-run -- the one thing only a dry-run knows |
| `QoS=N/A` on the partition, `DefaultQOS=clay` on the association, `MaxTRESPerJob=cpu=16` | no ceiling found | a 32-CPU job reported **RUN NOW**; `--test-only` agrees, because it does not check QOS ceilings either |

The direction of each failure is the point. An unrecognised *blocking* state contributes no
condition, so the node stays schedulable and its idle cores are counted as room -- phantom
capacity, from the tool written to find phantom capacity. An unrecognised *harmless* state
mapped onto a blocking one does the reverse and is louder: it deletes hardware that is
running other people's jobs right now.

So the node-state table is now read off `man sinfo`'s NODE STATE CODES rather than off
whatever one cluster happens to have set, and every code whose wording is "not usable" or
"not capable of running any jobs" is mapped even where no cluster to hand has one:
`POWERED_DOWN`, `BLOCKED`, `PERFCTRS`/`NPC`, `REBOOT_ISSUED`. `PLANNED` is carried as an
informational condition -- it is the backfill scheduler's *plan*, not an outage -- and
`POWERING_UP` is deliberately left schedulable, because that is the one powersave state
Slurm does place work on.

The same hour on two live PBS Pro clusters, one 10,624 nodes and one 24, produced the
same class of finding and the worst single number yet. **10,194 of those nodes were
`job-exclusive`, and not one node on that cluster recorded `ngpus` under
`resources_assigned`** — whole-node placement means the scheduler never has to account
for a GPU individually. `ncpus` happened to be assigned in full, so the CPU figures were
right and the accelerator figures were not: nodetop announced **62,886 of 63,744 GPUs
free** where 1,722 were, a 36x overstatement on the one axis people pick that machine
for. It is modelled as occupancy rather than as a condition, because those nodes are
healthy and working: calling them unschedulable would report 96% of the cluster as out of
service and bury it in `health`, trading one wrong answer for a louder one. Full is not
broken. The second cluster is what makes the rule safe rather than lucky — its jobs share
nodes and every `ngpus` is accounted, and not one `job-exclusive` node there was
partially assigned, so the rule changes nothing and the free count still matches
`available - assigned` exactly.

Scale came with it, since a 10,624-node answer is 14 MB. `load_queues` held a `set(...)`
inside a generator's `if` clause, so the set was rebuilt once per node: 51 queues took
**151 seconds** of a 3m10s run whose queries accounted for 11s. Queue attributes were
also fetched twice — `qstat -Qf -F json` for the queues and the plain `qstat -Qf` for the
limits, 38 seconds for the same 37 KB, at two different instants, which is also how one
report comes to describe two moments. And `resources_available.gputype = PVC` was resolved
to the 1100 SKU with its 48 GB stated as certain, on the largest Ponte Vecchio machine in
existence, whose parts are 128 GB: `where -g 6 --gpu-mem 64` ruled out all 10,624 nodes.
`PVC` is the codename both SKUs share, so it now carries the same "at least" uncertainty a
bare `A100` does.

A fourth and fifth Slurm cluster -- 24.11 and 25.11, at a site that runs both --
produced the first outright *crash* and the first case of a report that was
honest and still useless.

**`nodetop check` died with `AttributeError: 'NoneType' object has no attribute
'lower'`.** It took two conditions at once, and no cluster before had both: a
caller with NO accounts, and a queue naming specific ones. `None` is the
sentinel for "probe without naming an account", which is exactly what a caller
holds on a cluster that enforces associations and has no association row for
them -- and it was being fed to the pass that narrows accounts against a
queue's allowlist. Four clusters had all given the caller at least one account,
so the sentinel had never met an allowlist.

**And nothing said why.** That same cluster refuses every submission from this
caller, because `AccountingStorageEnforce=associations` and `sacctmgr show
assoc` returns nothing for them. Six partitions: four came back
ACCOUNT_MISMATCH, two were refused by a site `job_submit` plugin -- and the
overview reported "2 open to you (2 unconfirmed)" over a table of free GPUs,
never naming the cause the control plane had already given twice. An empty
account list is two different facts, and only the cluster's own config tells
them apart: without enforcement it means "this site does not use accounts", and
with it, "nothing you submit will run". The overview now says the second one in
four words, and `Identity.accounts_required` carries the distinction rather
than leaving every reader to infer it.

The site plugin's own sentence was being thrown away too. `sbatch` printed both
layers::

    sbatch: error: Job submission rejected: Batch jobs cannot use the
        `interactive_*` partitions.
    allocation failure: Unspecified error

and the parser kept the second -- so two partitions were filed under `UNKNOWN:
Unspecified error` while the cluster had explained itself in one line. The
plugin channel is now read, preferred as the reason, and fed to the classifier
alongside the other two layers; `check` prints what the control plane said
under an `unanswered` heading, because a category is a bucket and the sentence
is the answer.

Two smaller lessons, same root:

* **Ask both lists at the same visibility.** Partitions were fetched with `--all` and nodes
  without it, so the two disagreed about what the cluster contained. Slurm applies the
  hidden-partition filter to both, and `scontrol show node <name>` hides the node too, so
  there is no way to recover it afterwards.
* **Read a sentence as its facts, not as a sentence.** One chained regex over
  `Job N to start at T using P processors on nodes L` lost `P` and `L` the moment 25.11 put
  a stray token after `T`. Three patterns against that one line cannot fail that way.

### 6a. The mark says which; the colour says how much

`◐` meant "somewhere between free and full", and that is all it meant, on a row
whose numbers ranged from 3 idle cores to 114. Asked to carry the ratio, the
first attempt graded the *shape* — `◔ ◑ ◕` between `○` and `●` — and it was put
back after one look at a real column:

* `●` and `○` are a plain yes/no in three other tables (a backend usable here, a
  queue on your allowlist, a dry-run accepted). A shape that means "62% free"
  in this table and "permitted" in the next is exactly the collision `·` was
  split out to stop.
* `◐` and `◑` are one scanned column away from being the same character, and
  they would have meant opposite things — "nothing free" against "half free".
* Five levels need a legend, and this table has none; the marks are currently
  inferable without one.
* ASCII has no quarter circles, and the four characters close enough to imply a
  ramp are already spoken for.

So the alphabet stays four glyphs and the ratio rides in the **colour**, on the
ramp step the free-core count beside it already uses — `st.tint(g.partial,
heat[node])`, handed the same step as the number, so the two cannot disagree
about one node. A column of marks can then be scanned by hue: bright green at
114 of 128 cores, cool blue where nothing is free.

The trade-off, stated: with `--no-color`, under `NO_COLOR`, or in a pipe, the
gradation is gone and the mark is exactly what it was before. That is acceptable
here and would not be for a *status* distinction — colour is the only channel
this view has left, and the number and the meter beside it still carry the
quantity in text.

### 1l. One snapshot is right for a printout and wrong for a session

Every command takes exactly one reading and every number in its output describes
that instant -- a guarantee with a test behind it, and the reason a report never
says "607 nodes, 549 up" above a table that adds to 606. For a printout that is
the whole answer: you ran it, you read it, it was true.

A browse is not a printout. It renders that one reading for as long as it is
open, and a cluster changes every second: jobs start and end, nodes drain and
come back. Sitting in the browser for ten minutes meant reading ten-minute-old
figures with nothing on screen saying so -- and the frame is deliberately built
once and never re-rendered in place, so there was no way for it to say so.

Three things, in the order they matter:

* **The frame carries its own age.** `read 12s ago  ·  r re-reads`, at the foot
  of every level, silent under five seconds and silent on a replay (where the
  header already dates the recording). Rebuilt on every repaint, so any keypress
  updates it.
* **`r` takes the reading again**, and `main` loops rather than the browse
  patching itself: the whole snapshot is replaced and the view rebuilt from it,
  which is the only way a refresh cannot leave two instants on one screen. The
  browse remembers its stack and cursor, so the level and the row survive and
  only the data changes.
* **A timer, but only where a re-read is cheap.** Paced by what the last full
  turnaround actually cost -- not by the load alone, because `status` re-runs a
  dry-run per partition and one 607-node cluster took ~20s to rebuild against a
  3.3s load. Under a second, refresh at twenty times the cost and no faster than
  every five; over a second, no timer at all and `r` is the only way. Measured:
  0.06s on two clusters (every 5s), 0.82s on a PBS site (every 16s), 3.3s and
  70s on the two big ones (manual only). A view that stalls for twenty seconds
  under the reader's hands would be a worse answer than a stale one.

### 1k. One window, whatever is in it

The frame sized itself to its content, which is right for a printout and wrong for a
screen you move around in. Stepping from the overview into `3 down` shrank the box to a
third of its width and four rows:

```
╭─────────────────────────────────────────────────╮
│ 3 down                                          │
│    partition      why   nodes  cores  models    │
│ ❯  eng        down     48   4608            │
╰─────────────────────────────────────────────────╯
```

*"whatever we choose in the ui, the window should stay the same and the text and
information getting displayed should dynamically get adjusted."* Every view now draws at
`term_width()` × `term_height()` and pads up to it, so the box is where the eye left it and
only the contents change. `term_height()` is the window less one line — the spare line the
repaint needs — capped at 30 for the same reason `MAX_WIDTH` exists, and floored where the
chrome no longer fits.

Escape got the same treatment as Left when Left had to stop leaving the program, and should
not have: at the root it did nothing, which is indistinguishable from a hang. It is now its
own key — out of a nested view, out of the program at the root — while Left stays a
movement with nowhere to go.

### 1g. The submit line, checked against the scheduler that will read it

The `submit` line exists to be copied, so the question is not whether it looks
right — it is whether `sbatch` accepts it. Fed back through `sbatch --test-only`,
every shape nodetop places is accepted verbatim: GPU counts, core counts,
walltimes in each accepted spelling, and `--mem` round-tripping as
`--mem=65536M`. A 2 TiB request is refused by nodetop (`NOWHERE NOW`,
`SHAPE_UNAVAILABLE` on every reachable partition) *and* by Slurm — the two agree
on the negative case as well as the positive ones.

Doing that found a papercut in the input rather than the output. `--mem` was a
bare float in GiB, so `--mem 64G` — which is exactly what `sbatch` takes, and
therefore the first thing a Slurm user types — was an argparse error reading
`invalid float value: '64G'`. A tool whose premise is scheduler fluency should
not refuse the scheduler's own notation. Both memory flags now accept `64`,
`64G`, `64GB`, `64Gi`, `64GiB`, `65536M`, `2T` and `0.5T`, case-insensitively;
bare numbers still mean GiB so nothing that worked before changes meaning, and
suffixes are binary multiples like `sbatch`'s, so there is no second convention
to guess between. `Gi` is in there because this tool speaks Kubernetes too, and
that is how Kubernetes writes it.

## Speed, where it turned out to be

A bare `nodetop` on a 607-node cluster took **10.26s** at its worst and ~3.0s
typically. Profiled rather than guessed at, and the answer was almost entirely
*waiting*: 3.53s of scheduler queries, 6.40s of dry-run probes, 0.09s of Python
imports and 0.05s of arithmetic. So the wins are in how the waiting is
organised, not in the code that runs between the waits.

**The snapshot's queries are independent reads, so they overlap now.** Six
commands -- nodes, partitions, config, associations, QOS, free times -- used to
be issued one after another. Measured five interleaved rounds: **0.57s
sequential against 0.23s together**, and on a congested controller the
sequential form drifted to 3.97s while the concurrent form stayed at 0.23s.
Lower variance is most of what "feels fast" means. Nothing races on the control
plane's side, these commands only look -- and the readings land *closer*
together in time, so the one-instant guarantee gets tighter rather than looser.
Each backend's caches are locked for it, so a shared answer is still fetched
exactly once: verified on PBS, where `load_queues` needs the nodes and a second
fetch would be 14 MB.

**The dry-runs overlap too, three at a time.** Ten probes against a live
controller, five interleaved rounds, medians: 0.91s sequential, 0.73s at three
concurrent, 0.73s at six. It saturates at three because the controller takes a
lock for the submit path, so all concurrency can overlap is the process spawn
and the RPC. Three is therefore the cap -- going wider buys nothing measurable
and asks more of somebody else's controller.

**Measuring a cell's width is now remembered.** A table sizes each column by
its widest cell and then pads every cell to it, so one render measures each cell
at least twice, and a listing repeats itself: `MIXED`, `64/64`, a meter, a dash.
`nodes --all` over 607 nodes called `width` **19,600 times per render**. An
8192-entry cache took the first render from 50.8 ms to 26.4 ms and a *repaint*
to 13.1 ms -- and the repaint is the number that matters, because the
interactive browse re-renders the whole frame on every keypress.

End to end on that cluster, same interpreter, medians of seven:

| | before | after |
|---|---|---|
| `nodetop` (status, with probes) | 3.02s | **2.00s** |
| `nodetop nodes --all` | 0.72s | **0.37s** |
| `nodetop queues` | 0.80s | **0.38s** |

### The one remaining stall was a wait, so it moved off the keypress

Opening a node's job list was the least smooth thing left, and the first time it
had been measured: per-node shares are one whole-cluster `scontrol show job -d`
-- 9 MB, ~92 ms of parsing, **1165 ms end to end** on a 2,116-job cluster -- and
all of it was paid on the keypress, with no frame until it finished.

It cannot be made cheaper (see below), so it is no longer paid at that moment.
Stepping *into* a partition's node listing starts the fetch in the background:
nobody opens a node listing to admire the borders, so by the time a row is
chosen the answer is usually in hand.

**And it is two queries, not one** -- which the first attempt at this missed,
fixing half the stall. The frame reads `jobs` (`squeue`, 175 ms cold) before it
reads the shares (1042 ms), so with only the shares warmed the keypress still
cost 126 ms. Both are warmed now, in that order and in one worker: the cheaper
one the frame needs first lands first, and at most one extra query is in flight
against somebody else's controller. Measured:

| the reader spends this long choosing a node | job list appears in | shares in |
|---|---|---|
| nothing (straight through) | 128 ms | 1042 ms |
| 0.5s | **0.3 ms** | 643 ms |
| 1.5s | **0.3 ms** | **0 ms** |

Deliberately *not* started when the browse opens. Most readers look at the
overview and leave, and firing a whole-cluster query at somebody else's
controller for a view nobody asked for is a worse trade than the hitch it would
hide. The lazy accessor is locked so the prefetch and the view cannot both
fetch -- one whole-cluster query is the most expensive duplicate available --
and a failure is still recorded rather than raised, because the view works
without it.

Two attempts to make the parse itself cheaper, one kept:

* **Kept.** `expand` walked every character of a nodelist in Python to find the
  commas outside brackets. With no bracket in the string -- which is every
  `Partitions=` field and every single-node job -- there are none to find, so it
  splits. 8.92 ms against 1.67 ms over a realistic mix, checked against the
  general path on 417 inputs including 400 generated ones and the degenerate
  `a],b`, `[`, `a[1-2` and empty-segment cases.
* **Rejected.** Replacing the per-line scan of `scontrol show job -d` with one
  `finditer` over the whole 9 MB: the interesting lines are 8,000 of 200,000, so
  skipping the rest in C sounds obviously right. It is 25% **slower** -- 115.5 ms
  against 92.4 ms -- because `splitlines` plus two `startswith` is already C, and
  a line-anchored regex over 9 MB is not free. Identical output, worse time,
  reverted.

### Parsing was a third one function, and the function was backtracking

The fourth pass profiled the one stage that had not been looked at: turning
`scontrol`'s text into the model. On 607 live nodes that took **42.4 ms**, and a
third of it was `_fields`.

The reason is in the pattern. Each value ends with a lazy match and a lookahead
-- `(?P<val>.*?)(?=\s+Ident=|$)` -- so for every character of every value the
engine evaluates "does an identifier followed by `=` start here?". 607 nodes at
~36 fields each came to 460,680 calls to `re.Match.group` and a great deal of
backtracking under them.

Splitting on spaces cannot backtrack: a token shaped `Ident=` starts a field,
anything else belongs to the value being collected. Same first-wins rule, same
`strip`, same output. Two things make that safe rather than merely fast:

* **It is checked against the regex, not assumed equal to it.** Field by field
  over every record of two live clusters -- 607 nodes and 87 partitions -- and
  over adversarial records: a key-shaped token mid-value (the `Reason=replacing
  NodeName=n2` case documented above), `=` inside a value, doubled spaces, an
  empty value, leading and trailing space.
* **It hands back where it cannot see.** The regex splits on any whitespace;
  splitting on spaces does not see a tab. A one-pass search for exotic
  whitespace sends such a record to the regex, because the failure otherwise
  would be the worst available -- the whole record read as one field, so a node
  with no state, reported as UNKNOWN and unschedulable.

Then the key test itself became the hot line, and it asks the same question over
and over: `scontrol`'s vocabulary is fixed and small. 607 nodes asked 217,383
times about **37 distinct** strings. A bounded cache took it to 37 regex calls.
Bounded rather than a plain dict -- which was faster still and unbounded -- since
a `Reason` full of `ticket=12345`-shaped words would add an entry per node, and
this process can outlive one parse by hours when somebody leaves the browse
open.

`parse_nodes`: **42.4 ms -> 32.9 ms** over those 607 nodes, 22%.

What that is worth end to end is worth stating plainly, because it is not much
where the numbers above were measured: a live run on that cluster is
I/O-dominated and 9.5 ms sits inside its run-to-run noise. It shows up where the
network does not: **4% on a replayed snapshot**, and it scales with the node
count -- a 10,000-node cluster spends about 150 ms less per run in the parser.

### Startup is a file-count problem, and on NFS it is the whole problem

The second pass went after the fixed cost, and the first measurement moved the
target. On the login node the numbers are unremarkable -- a 31 ms interpreter
floor, 95 ms to `import nodetop`, 160 ms for `nodetop --version`. On a cluster
whose home is NFS they are not:

| | login node | NFS home |
|---|---|---|
| `python -c pass` | 31 ms | 64 ms |
| `import nodetop` | 95 ms | **443 ms** |

That is not CPU. It is ~24 module files, each an open and a read across the
network -- roughly **19 ms per module**. Which makes the lever obvious and
unusual: *import fewer files*, and the wins scale with the filesystem rather
than with the code.

**The six backends are now imported one at a time, stopping at the first that
detects.** Asking for the names -- which `build_parser` does, before argparse
has looked at a single argument -- used to import all six, and `subprocess`,
`termios`, `getpass` and `grp` behind them.

| | eager | lazy |
|---|---|---|
| `--version`, login node | 95.0 ms | 87.2 ms |
| `--version`, NFS home | 456.3 ms | **376.5 ms** |
| `nodetop backends` (must ask all six) | 95.8 ms | 94.9 ms |

**And the keyboard layer is imported only when something is going to browse.**
`--version`, `--help`, `--json`, `nodes`, `queues`, `health`, `accelerators`,
`where` and `check` never touch it now; a non-`--json` `status` still does,
because it has to ask whether this is a terminal.

This pass, end to end on the NFS home, thirteen runs each, medians: `--version`
428 -> 361 ms, `--help` 450 -> 364 ms, `queues` 458 -> 417 ms, `nodes --all` 458
-> 431 ms.

A note for whoever reads this next: an earlier attempt at the lazy registry was
**reverted on a bad measurement** -- "3 ms, and one command slower". That
comparison was against a `git archive` of an older commit, which differed in a
dozen other files. Two trees, identical but for the one file, said 8 ms locally
and 80 ms on NFS. If a performance change looks like noise, check that the two
things being compared differ only in the thing being measured.

### The docstrings are 61% of the bytecode, and that is a packaging problem

The zipapp idea recorded here last time turned out to be the largest remaining
win by a distance, so it was measured properly and built. Two facts came out of
the measurement, and the second was a surprise:

* One zip on `sys.path` is **one** open where the installed layout is two dozen.
* This codebase's **docstrings are 61% of its compiled size** -- 1,303,123 bytes
  of `.pyc` against 501,758 with `-OO` -- and every start reads them off the
  network in order to discard them.

That second number is the direct cost of the thing that makes this repository
worth reading: every fix carries its measurement and its reasoning in the
docstring beside it. The answer is emphatically *not* to write fewer of them. It
is to leave them in the repository and out of the artifact.

`tools/build_pyz.py` builds that artifact. Measured on a cluster whose home is
NFS, thirteen runs each, medians:

| | installed files | pyz (sources) | pyz `--fast` |
|---|---|---|---|
| `--version` | 406.5 ms | 351.2 ms | **222.5 ms** |
| `queues` | 474.9 ms | 337.6 ms | **239.6 ms** |
| `nodes --all` | 603.7 ms | 402.1 ms | **240.6 ms** |

45 to 60 per cent, on the environment where starting the tool actually hurts.
`--fast` ships this interpreter's stripped bytecode and is therefore locked to
its Python version -- bytecode carries a magic number and `zipimport` refuses a
mismatch -- so it is built where it will run. The default build ships sources,
runs anywhere, and is still well ahead of two dozen files. The installed package
and its console scripts are untouched by either.

It is tested, because an artifact nobody exercises is the one that turns out to
be broken during an outage, which is exactly when somebody reaches for the
fast-starting copy. That was not a hypothetical: taking it to the other clusters
found **two defects in it**, both of the kind that report success.

* **The shebang did not name an interpreter that could run the archive.** A
  `--fast` build made under 3.12 carried `#!/usr/bin/env python3`, and on a
  cluster whose system `python3` is 3.9 *every* command failed with a `runpy`
  traceback -- unloadable bytecode fails before `main` can say "nodetop needs
  Python 3.10". The version-locked build now names the interpreter that built
  it. The portable build keeps `python3`, because its sources do reach `main`
  and the caller gets the floor in one line.
* **The archive swallowed every exit status.** `zipapp`'s generated shim is
  `import nodetop.cli; nodetop.cli.main()` -- the return value dropped -- so the
  pyz exited 0 whatever happened. `queues -q nope` exits 2 from the installed
  package and exited **0** from the archive, and `nodetop check && sbatch ...`
  would have waved a caller through a refusal. The archive now carries
  `raise SystemExit(main())`, which is what `nodetop/__main__.py` has always
  done for `python -m nodetop`.

Both were invisible to the builder's own check, because it ran the archive with
an explicit interpreter and ignored the status -- so it now runs it through the
shebang as well, and the tests compare the archive's exit codes against the
installed package's. Verified on three clusters afterwards: two Slurm (one with
a 3.9 system Python) and one PBS, every command matching the installed status,
including `check` returning 1 where nothing is accepted and 2 where the system
has no dry-run at all.

Two more ideas were measured and rejected in the same pass, which is worth as
much as the ones that worked:

* **A lazy package `__init__`.** Deferring the public API re-exports behind PEP
  562 sounds like it should shed files from a CLI start. It sheds none: `cli`
  imports those modules directly anyway, so the module count for a real run is
  15 either way, and on NFS it measured 29 ms *slower*.
* **Replacing the ANSI stripper's character loop with a regex.** 6.5x faster in
  isolation (3.84 us against 0.59 us per call) and identical output on every
  case tried -- but once `width` is memoised the stripper no longer appears in a
  repaint profile at all, and the regex and the loop disagree on *truncated*
  escape sequences. A faster function that is never hot, bought with a
  behavioural difference on malformed input, is not a bargain.

**A dry-run is not spent on a queue that accepts nothing from anybody.**
Disabled, never-starts, empty allowlist, every node down: `reachable` is already
False whatever the control plane would answer, so the round trip could not
change what is printed. Access blockers are deliberately *not* in that set -- a
declared ACL disagreeing with the control plane is the thing this tool exists to
catch -- nor are soft blockers, where the entitlement answer beside "your job is
too big" is still worth having. The honest figure is small: **3 of 89 probes on
`where --all`**, and none on the default views, where the entitlement filter has
already dropped those partitions before ranking sees them.

What is left is the control plane's own latency: a probe took 90 ms on an idle
controller and 3.3s on a busy one, and 21 of them is most of what remains. That
is not ours to optimise; it is why `r` and the age line exist instead.

### The 1.6s wait, spent differently rather than saved

Everything above shaved the fixed cost until the fixed cost stopped mattering:
`status` on the 607-node cluster is 1.93s and **1.60s of it is dry-runs**. Where
that goes, measured:

| | |
|---|---|
| `sbatch --help` (the client, no RPC) | 14.8 ms |
| `scontrol ping` (client + a round trip) | 15.8 ms |
| `sbatch --test-only` | **98 ms** |

So ~82 ms of each probe happens inside the controller, running the site's submit
plugin, serialised against every real submission. Concurrency does not help (per
probe latency rises linearly, wall time is flat past three workers) and the
partitions cannot be batched into one request (the plugin refuses a list if any
member is inadmissible). That time is not ours to make smaller.

**So a session spends it differently.** It opens on what the last run was told,
re-asks in the background, and reloads itself if the answer moved. Measured
through a pty against the live cluster, three consecutive starts: **329, 340 and
327 ms** to the first frame, against 1.9s -- and on the loaded controller that
produced the 7s cold start above, the warm start was still 329 ms.

What makes that honest rather than merely fast:

* **A printout never does it.** `nodetop status | grep` gets one shot at being
  right, so it probes and waits exactly as before; so does `--json`, and so does
  a `--replay`, which cannot dry-run at all. Only a session -- which can correct
  itself on screen and can say how old its answer is -- reads the file. This is
  the same split §1l already draws between a printout and a session.
* **The recheck always runs.** The screen is at most a second or two behind the
  control plane, not as old as the cache. If the fresh answer matches, nothing
  happens at all: no repaint, no flicker, and the reader never learns there was a
  check. If it differs, the browse reloads itself -- the same thing `r` does,
  landing the cursor on the same row.
* **It says so.** Once the remembered answer is more than five seconds old the
  frame carries `access checked 7m ago · re-checking`, on its own line under the
  reading's age, and the line disappears the moment the recheck confirms.
* **Only settled verdicts are written down.** "The control plane did not answer"
  is not a finding; the next run asks again.
* **One recheck at a time.** `r` held down would otherwise start a dry-run pass
  per keypress, and three at once is nine concurrent probes against a controller
  this tool is careful to ask three of.
* **The file is a convenience, never a dependency.** No HOME, a read-only cache
  directory, half a document, a schema from another version: every one of them
  ends in "ask the cluster". `NODETOP_ACCESS_TTL=0` switches it off entirely;
  the default bound is 15 minutes, which only limits how stale a *first frame*
  can be, since the recheck lands seconds later.

Three shape mistakes are worth recording, because each one worked well enough to
look finished and was found only by measuring the thing again.

**Keying on the question's candidates.** The first version keyed the remembered
answer on the set of partitions it had asked about. That set is "the ones with
room right now", so a partition filling up between two runs changed the key --
**2.3s instead of 0.33s**, a miss every time the cluster moved. Verdicts are
stored per partition now, merged across runs.

**All-or-nothing coverage.** With per-partition verdicts the key was stable, but
a partition that gained room since the last run had no verdict, and a missing
verdict threw the whole answer away: five consecutive starts measured 328, 2151,
328, ... The verdicts are independent questions, so the gap is probed on its own
-- **one dry-run instead of nineteen** -- and the frame reports the age of the
older half, because that is the part a reader might want to distrust. Five starts
after the fix: 327, 337, 326, 326, 325 ms.

**A refresh that never came back, and one that came 0.3s in.** Two bugs in the
polling itself, both mine, both found by watching a live session rather than by a
test -- which is why the tests that now cover them drive `select` with a real
callback instead of checking that one was passed.

The first was polarity. `select` reloads when its `on_idle` says True; the
callback was written as *"still waiting?"*, which returns True while it wants to
wait. So a browse re-read 0.3s after its first frame and then never again --
`continue` forever, frozen on one reading. Two funnels in twelve seconds is what
gave it away.

The second was the clock it paced off. `_TURNAROUND` was measured by `main`
around the whole dispatch, and for an interactive command that includes however
long the reader sits looking at the screen. So the first idle refresh, five
seconds in, concluded that a rebuild costs five seconds -- over the 1.0s
threshold -- and switched refreshing off for good: **one re-read in a hundred
idle seconds**, where the point was a paced series. The browse now stamps the
cost at the moment it has something on screen, which is also better information
than the previous command's timing, and a printout is still timed end to end
because there the whole command *is* the rebuild.

Instrumented afterwards, one line per pass, and this is the intended shape:

    cost=0.23 idle=5.0 step= 5.0 backoff=0 recheck=True
    cost=0.22 idle=5.0 step=10.0 backoff=1 recheck=False
    cost=0.22 idle=5.0 step=20.0 backoff=2 recheck=False
    cost=0.22 idle=5.0 step=40.0 backoff=3 recheck=False

**And the refresh backs off while nobody is there.** Remembering the access
answer made a re-read cheap on a cluster where it never used to be, so a browse
that used to sit still started re-reading every few seconds forever, six queries
at a time. Fine for one reader; fifty terminals left open is 45 queries a second
against a controller that has jobs to schedule. The interval doubles each time
the refresh finds nobody there, capped at five minutes, and any keypress puts it
back -- so a terminal in use is as current as it ever was and one nobody is
watching settles down. The dry-runs are paced separately, at twenty times their
own cost.

**And the remembered answer now lasts a day, not fifteen minutes.** Every session
re-asks in the background within seconds of opening, so the TTL bounds nothing
except how old a *first frame* may be -- the window in which anyone could act on
a stale answer is those couple of seconds either way. What fifteen minutes did
control was how often the 1.9s wait came back: coming back to the terminal after
lunch paid it again, which is most of what this mechanism exists to avoid.

**One document for every answer.** The remembered answers started life in a
single JSON file, read-merged-rewritten on each save. A login node runs several
of these at once -- a browse, a printout, a background recheck -- so the last
rename wins and the others' entries vanish. Twelve concurrent writers on twelve
different keys: **five of the twelve lost**, and losing a whole key costs a full
dry-run pass rather than the one probe a lost verdict costs. One small file per
question now: they cannot collide, they need no locking (which would be its own
adventure on an NFS home), and they prune by when each was last written, so the
answers a reader is actually using are the ones that survive.

**Making the frame cheap turned the auto-refresh on.** The browse re-reads by
itself when the last pass was cheap -- twenty times its cost -- and moving the
dry-runs into the background took them out of that cost. So an idle terminal
started re-reading every 7s and dragging nineteen dry-runs along each time,
which is a worse citizen than the slow version it replaced. Counted with `ps`
against a live session, which is the only reason it was noticed. A recheck now
waits twenty times *its own* cost before the next one -- ~32s here, ~6s on a
four-partition cluster -- so an idle session runs one dry-run pass per 40
seconds instead of six. The first recheck of a new process is never paced: a
fresh start always confirms what it opened on.

### So the wait says what it is waiting for

Put a number on "most of what remains", because it is worth knowing how lopsided
this is: `nodetop status` on the 607-node cluster takes **1.93s, of which 1.60s
is dry-runs** -- 83%. `--declared`, which skips them, is 0.33s. For that 1.60s
the terminal showed nothing, which is indistinguishable from a hang, and the
complaint that started this whole pass was "it takes a while to start".

The obvious fix is not available. Painting the partition list first and
correcting it when the verdicts land would show 19 partitions and then take
eleven of them away -- the declared list is 19 where the dry-run accepts 8 --
and over-reporting entitlement and then retracting it is the exact conflation
this view exists to prevent. `--declared` already carries a banner for a reason.

So the wait stays, and reports itself: one line, rewritten in place, counting
settled partitions. `rank` takes an `on_progress(done, total)` and knows nothing
about terminals; the CLI half writes to **stderr**, and only when stderr is a
terminal, so a pipe, a `--json` reader and a redirected log are all untouched.
It is called from the probe pool's workers, so the write is locked, and the line
is wiped to its own measured width before anything else prints.

Two details that were wrong first and are worth keeping written down: `pool.map`
yields in *submission* order, so the slowest partition was reported as the last
to *start* -- backwards for a progress count -- hence futures keyed by position
and reassembled after; and a `sys.stderr` of `None`, which some launchers hand
over, turned `isatty()` into an `AttributeError` in the middle of a command that
was otherwise working.

### A browse re-rendered every row of the partition on every keypress

The rows of a listing are a function of the snapshot, and the snapshot does not
change between reloads -- but the browse rebuilt them for every frame, which is
to say for every arrow key. About twenty-five rows are on screen; the work was
proportional to the *partition*.

| partition size | per keypress, before | after |
|---|---|---|
| 190 nodes | 6.6 ms | — |
| 607 nodes (this cluster's largest) | 17.6 ms | **0.75 ms** |
| 10,624 nodes (a real PBS partition's shape) | 344.5 ms | **0.85 ms** |

A third of a second of dead keyboard on the big one. The rows are now built once
per partition and width, four cached at a time -- enough for the one being read
and the ones stepped through to reach it -- and the first frame still pays the
build (365 ms at 10,624 nodes, which is a wait *once* rather than per key).

The correctness catch is worth naming, because a cache here is not free: the
cursor is written *into* a row, in place, so handing the cached list out twice
accumulates cursors -- two rows marked, and no way to tell which one `Enter`
would open. The frame copies the list before marking it, and the test that
matters asserts exactly one marked row per frame rather than asserting a timing.

### Three imports that no command needed before reading its arguments

With everything above done, `--version` is still ~80% imports. What was left
was not the tool's own modules but the standard library modules it pulled in for
paths it rarely takes: `pathlib` (8.5 ms) reached only by `snapshot` and
`--replay`, `difflib` (1.1 ms) only by the did-you-mean hint on a misspelled
queue, and `concurrent.futures` (4.6 ms, with `logging` behind it) only by
`Cluster.load`. Four call sites, in a tool where every other command paid for
all three.

Two trees identical but for those imports, strictly alternating runs -- because
the first attempt at this measured all of one tree and then all of the other,
and NFS caching handed back a **+90 ms** result with the sign reversed:

| | login node (warm bytecode) | NFS home |
|---|---|---|
| `--version` | 84.0 -> **77.2 ms** | 345.6 -> **327.8 ms** |
| `--help` | 84.9 -> **79.2 ms** | — |
| `import nodetop.cli` | — | 332.6 -> **292.8 ms** |

An import inside a function is invisible as a *property*: nothing stops a later
edit from hoisting it back, and nothing would fail. `tests/test_startup.py`
therefore names the three modules and asserts they are absent from
`sys.modules` after a real run -- checked against a bare interpreter's baseline,
not against a no-argument run of ours, because a hoisted import would appear in
that baseline and exempt itself.

### Measuring and colouring the same handful of strings, ten thousand times

Profiling a 10,624-row table -- the shape of a real PBS partition -- found the
same pattern three times over: an answer rebuilt for every cell of every row,
when the set of possible answers is tiny.

* **`_strip_ansi`, 53,537 calls**, walking a Python loop per character to remove
  escapes from strings that mostly have none. It now returns the string
  unchanged when there is no `ESC` in it, which is a C-level scan.
* **`width`, 42,907 calls and 385,333 `unicodedata` lookups each** for
  wide-character and combining-mark tests. The memo above it cannot help at this
  scale -- 10,624 nodes produce far more than 8,192 distinct strings, so a big
  listing is mostly misses. ASCII cannot be wide and cannot combine, so an ASCII
  string's width *is* its length, and `str.isascii()` is a flag on the string
  object rather than a scan. Meters and marks are not ASCII and still take the
  loop, which is exactly right.
* **`_sgr`, 94,294 calls**, formatting one of a few dozen escape strings.
  Memoised per `Style`, because the escape depends on that style's colour depth
  -- one shared cache would paint a 16-colour terminal with truecolour.
* **one `bar` per row**, when a meter is one of very few pictures: `size` cells
  at eighth resolution in one of nine ramp steps. Memoised on the *rounded*
  numbers the body actually uses -- both of them -- so a hit is the string the
  miss would have built rather than one that merely looks like it.

**And every cell was measured four times.** Sizing the columns, capping to
`limits`, shrinking to the window and padding each asked `width` for every cell
-- ~255,000 calls for 85,000 answers. The width is measured once now and carried
in a parallel array, updated wherever a cell is rewritten. Phase timings on the
10,624-row frame that motivated it: `_node_rows` 135 ms, pad+join 50 ms, measure
41 ms, the two truncate passes 22 ms each.

**And permission to paint was asked 74,526 times.** `paint` and `tint` rebuilt
the same escape prefix from the same role or ramp step for every cell, and each
first asked `Style.enabled` -- a property over a `depth` that cannot change
after construction. The prefix is memoised per style (clamped before it becomes
a key, so a wild ramp step shares the entry it shares the colour with) and
`enabled` is an attribute.

| first frame | before | after |
|---|---|---|
| 607 nodes | 19.0 ms | **12.7 ms** |
| 10,624 nodes | 365 ms | **216 ms** |

End to end against a recorded 607-node cluster, `nodes --all` is 5.5% faster and
the commands whose cost is elsewhere are unchanged -- which is the honest shape
of this: it is worth a third of the frame on the biggest listings and nothing at
all on a `queues` that renders ten rows.

The carried width is trusted, so a cell rewritten without its width rewritten
misaligns the row -- and that is nearly invisible, because `truncate` usually
lands exactly on the column and a stale width then costs nothing. It shows only
where the cut lands *short*: a wide character cannot straddle the last column,
so `ノードノードノード` cut to six columns yields five and the cell needs a space
a stale width would deny it. The test therefore sweeps terminal widths 15..45
rather than picking one -- at 22 columns the cut is exact and the bug is
invisible; at 15, 16, 19, 20 and 23 it is a column adrift. Each of the three
rewriting passes was broken on purpose to confirm a test notices.

The whole change is only allowed if the output does not move, so that was
checked rather than assumed: **22 command outputs byte-identical** -- Unicode and
ASCII, PBS and Slurm data, colour on. The tests compare against a reference
computed the long way rather than against a recorded string, and each shortcut
was broken on purpose to confirm a test notices.

### The same node, asked the same question, once per queue it is in

A `Node` is a reading; a `Queue` holds a list of them; and a node belongs to
several queues. So every aggregate -- the funnel, the heat ramp, `effective_free
_cpus`, "with room" -- walks the same nodes again, and each walk recomputed the
same arithmetic. Counted on a 607-node cluster with 84 partitions:
**`memory_exhausted` called 4,830 times for 607 nodes**.

Six answers on `Node` are memoised now (`schedulable`, `memory_free_mb`,
`memory_exhausted`, `effective_free_cpus`, `effective_free_gpus`, `has_room`).
Measured on the worst case -- every node in every queue -- and on the one path
that visits each node exactly once:

| | plain | 2 memoised | 6 memoised |
|---|---|---|---|
| aggregate capacity, 607 nodes x 84 queues | 53.9 ms | 25.7 ms | **13.1 ms** |
| build a 10,624-row table (one visit each) | 363.5 ms | 371.1 ms | 368.5 ms |

End to end on the live cluster, `status --declared`: our own compute went
**56 ms -> 46 ms**. The single-visit path pays 1.4%, on a frame whose cost is
string work either way, and it happens once per partition opened rather than
once per command.

This is only correct because a Node is written once, so two things enforce that.
Grid Engine used to *patch* accelerator counts in place after parsing `qhost`
-- exactly the shape that turns a memoised answer stale on one backend only --
and now rebuilds with `dataclasses.replace`, which yields a fresh object with an
empty cache. And a test walks the AST of every source file looking for an
assignment to any Node field; it fails with the file and line if one appears.
Queue-level aggregates are deliberately left uncached: `Queue.nodes` is wired up
after construction and four backends patch `node_names` later still, so a cache
there would be a cache of an unfinished object.

### Two more things measured and left alone

**Computing those six node answers eagerly instead of lazily.** `cached_property`
costs one Python-level descriptor call per node per answer on first read, and a
listing reads each exactly once -- so computing them in `__post_init__` and
reading plain attributes is faster on *both* paths: **-7.7%** on the 10,624-row
frame and -7.1% on the aggregate pass. Not taken. It means either six fields
whose long explanations move to private methods, or six `init=False` dataclass
fields, which changes what `dataclasses.fields` and `asdict` report about a
Node. In this file the docstrings are the design record, and 16 ms of a 216 ms
frame does not buy making them harder to find.

**The QOS query's 220 ms is one TRES resolution, and it is already minimal.**
`sacctmgr show qos format=Name,MaxWall` is 140 ms; adding *any* TRES column
takes it to 220 ms, and adding a second is free -- slurmdbd resolves the TRES
table once. So the ceilings cost 80 ms and there is no cheaper way to ask for
them. Narrowing with `where name=...` is *slower* (260 ms), which is worth
knowing before someone tries it: the filter costs more than the rows it saves.

### Reading associations from the controller instead of the database: refused

The load is 233 ms and it is exactly its slowest query -- the six run together,
and the two accounting queries are the slow ones. `scontrol show assoc_mgr`
serves the same data from the controller's memory instead of slurmdbd, and it is
much faster:

| | via slurmdbd | via the controller |
|---|---|---|
| associations | `sacctmgr show assoc` 203 ms | `scontrol show assoc_mgr flags=assoc` **66 ms** |
| QOS ceilings | `sacctmgr show qos` 221 ms | `scontrol show assoc_mgr flags=qos` **23 ms** |

Which would take the load from 233 ms to about 125 ms. It is still refused, for
three reasons found while measuring it. The `users=youzhi` filter **is ignored**:
the reply carried 1,296 association records of which 34 were this user's, just
under a megabyte. That megabyte is serialised by the *controller*, under the
assoc_mgr locks, on the path every real job submission uses -- so this trades
230 ms of a database's time, which is what a database is for, for tens of
milliseconds of the one process the whole cluster waits on. And the format is a
third dialect to parse (`GrpTRES=cpu=N(14454),...`, usage counters, limits
inline), version-fragile in exactly the way §7 is about. A tool that reads a
cluster should not make the cluster slower to be quicker itself.

### Ctrl-C during the wait printed a traceback

Found by watching the progress line: interrupt a `status` while the dry-runs are
running and the tool printed twenty lines ending in
`threading.py ... waiter.acquire() KeyboardInterrupt`. The browse had always handled Ctrl-C --
`select` catches it and quits -- but the 1.6s *before* the first frame had
nothing to catch it, and that is the window a reader is most likely to use it in.
It now exits **130** (128 + SIGINT, what a shell reports for Ctrl-C), after a
newline that closes whatever partial line the ticker had written so the prompt
does not land inside it. Measured against the real controller: 0.5-1.0s to exit,
which is one in-flight `sbatch` finishing -- the probe pool joins its workers,
and killing the children to shave that is more machinery than a quit is worth.

### Three more measurements, two of them refusals

**`json` came off the startup path and it was worth 1.9 ms, not the 40 ms
predicted.** Eleven `--json` sites called `json.dumps` directly, so every run
imported four modules to serialise nothing; they go through one helper now.
Local, 21 alternating pairs: `--version` **-1.9 ms**, `--help` -1.8 ms. On the
NFS home: nothing (-3.6 ms by minimum, +3.8 ms by median). Which corrects the
rule of thumb recorded above -- **~19 ms per module file applies to *this*
package's files on a cold cache, not to standard-library modules**, which every
Python process on the machine reads and which are therefore already cached. The
change earns its place on the other half of the deal anyway: `indent=2,
default=str` in one place instead of eleven, where two sites passed
`default=str` and nine did not.

**Wider probe concurrency, re-measured, still refused.** The earlier note said
the dry-runs saturate at three; that was ten probes on an idle controller, so it
was worth re-testing at scale on a busy one. Twenty probes, two rounds in both
orders:

| workers | wall | per-probe median |
|---|---|---|
| 1 | 2.10s | 91 ms |
| 3 | **1.54s** | 230 ms |
| 6 | 1.58s | 473 ms |
| 9 | 1.50s | 552 ms |

Per-probe latency rises *linearly* with the worker count while wall time is flat
from three onwards: the controller serialises the submit path, so extra workers
queue inside it rather than beside it. Three stays. It also means `where`'s 9.5s
over 86 partitions is the controller's throughput, not ours -- 26.8s of
subprocess time compressed into 9.5s, and no arrangement of ours makes the
serialised part smaller.

**Asking about many partitions in one dry-run: refused by the scheduler, not by
us.** If `sbatch --test-only --partition=a,b,c` named the partition it chose,
the accepting set could be peeled off in one call each and a `status` pretest
would cost as many probes as it has *acceptances* -- eight instead of nineteen
here. It does not. Measured on this cluster: `--partition=bigmem` is accepted,
`--partition=ssd` is refused, and `--partition=ssd,bigmem` is **refused as a
unit** -- the site's job_submit plugin validates the whole list, so one
inadmissible member sinks a request that would otherwise succeed. A per-list
answer would also lose the per-partition reason that the `blocked by` column
prints. Dead end, and cheap to establish.

### One wave of queries, and a straggler nobody had noticed

A sweep across every command looking for outliers found one: `nodetop health`,
median 320 ms, **worst 2171 ms**, with three runs in twenty over 600 ms. Nothing
else in the tool was doing that.

A per-query timeline showed why, and it was a scheduling accident rather than a
slow query. `Cluster.load` fires its queries together at t≈18 ms -- but
`scontrol show config` was not one of them. `load_nodes` reached for it partway
through its own parse, so it left at t≈90 ms, *after* the node query had already
come back. On the slow runs that straggler took 2.0s while the four first-wave
queries were entirely normal:

        17 ->     59 ms  (   42)  scontrol show node --all --oneliner
        18 ->     37 ms  (   19)  scontrol show partition --all
        18 ->    239 ms  (  220)  sacctmgr show qos
        19 ->    219 ms  (  200)  sacctmgr show assoc
        90 ->   2046 ms  ( 1956)  scontrol show config      <-- alone, and late

Asked on its own that query is 17 ms, twenty times out of twenty. So
`Backend.warm` now exists -- an optional hook for anything a loader would
otherwise fetch from inside its own work -- and the load submits it with the
rest. Slurm implements it by warming the config cache the loaders already share,
so it is one query either way, which the query-discipline tests enforce.

**What the measurement does and does not show.** Immediately after the change,
twelve consecutive runs were clean: 240-256 ms total, config 17-27 ms every
time. But an interleaved A/B of both versions a while later found **no stalls in
either** -- the controller had calmed down -- so those twelve clean runs are not
proof the change caused them. One later stall, with the change in place, had
*both* `scontrol` queries taking ~1s from the same start time, which is the
controller pausing rather than anything about staging.

So this is kept on the argument rather than the number: the same queries, at the
same instant, in one wave instead of one wave and a straggler. The median is
unchanged, and it is honest to say so -- the fast path always had the straggler
finishing inside the load's own window.

### The measurement trap: no `.pyc` cache means recompiling on every run

Worth writing down because it silently inflated every absolute number taken on
this machine. The development environment sets `PYTHONDONTWRITEBYTECODE=1`, so
no `__pycache__` is ever written, so **every run recompiled the package from
source**: 15 calls to `builtins.compile`, 50 ms of a 100 ms import, invisible in
any wall-clock reading and obvious the moment the import is profiled.

| `nodetop --version` | |
|---|---|
| bytecode cached (what an installed user gets) | **83 ms** |
| recompiling every run | 130 ms |

A/B comparisons taken in that regime are still sound -- both sides paid it --
but an absolute claim is not. `pip install` compiles at install time, so the
83 ms row is the honest one, and it is also why the `--fast` pyz (which ships
bytecode) wins as much as it does.


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
  default, which turns `--test-only` into `--test-` / `only` and `gpu-a-0001`
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

- **Slurm and PBS have been validated end-to-end against live clusters** — Slurm against
  two (a 607-node site and a Slurm 25.11 one), PBS against two more (PBS Pro 2022.1 at
  10,624 nodes and 2026.1 at 24) — plus the ssh pool against a real unscheduled GPU box.
  **LSF, Grid Engine and Kubernetes are not confirmed against a live control plane**: they
  are built from those systems' documented output formats and tested against
  format-faithful fixtures, which predicts that they will not crash, not that the numbers
  are right. Every live cluster so far has produced defects a green suite could not — see
  §7.
- **A RESERVED node is assumed to be someone else's.** Slurm says a node is in a
  reservation but not whether *you* may use it, so nodetop treats the state as
  blocking -- the conservative direction, and wrong on a site whose reservations
  are `MAGNETIC` with an open ACL. Measured on one such cluster: three nodes and
  up to 12 of 168 accelerators sat outside the reported total while being
  usable. Reading `scontrol show reservation` and evaluating its ACL (including
  the negated `Accounts=-a,-b` form) against the caller would settle it; until
  that is verified against a live reservation the caller can actually enter,
  `health` names the state so the reader can check with `scontrol show
  reservation` themselves.
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
pytest          # 4301 tests, no batch system required
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
