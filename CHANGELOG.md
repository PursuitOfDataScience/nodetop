# Changelog

All notable changes to nodetop are documented here, newest first.

The format is based on [Keep a Changelog](https://keepachangelog.com), and this
project adheres to [Semantic Versioning](https://semver.org).

## [0.7.0] — 2026-09-14

Two changes, neither of which alters what nodetop reports. The colour system was
rebuilt against the colour-science literature after a reader said the bars made no
sense, and the tool learned to serve itself to an AI agent. The first changes how a
report is drawn; the second changes who can ask for one.

### Added

- **`nodetop mcp` — the same reports, served to an AI agent over MCP.** Seven
  read-only tools on stdin/stdout, and no new dependency: a tools-only server
  needs four JSON-RPC methods, so it is standard library like everything else
  here. Depending on the official SDK would have pulled in pydantic, anyio,
  httpx and starlette and broken the rule pyproject.toml states with a reason.

  **It adds no data, and that is the point.** An agent with a shell could
  already run `nodetop where -g 4 --json`; what it could not do is know the
  flag. `--gpu` is an ambiguous prefix of `--gpus` and `--gpu-mem`, so the
  obvious spelling of the commonest question is a usage error — one a model
  reads as "no capacity" rather than as "wrong flag". A schema with
  `gpus: integer` deletes that class of failure. Clients with no shell at all
  are the second reason.

  **Nothing is re-derived.** There is one JSON view per command, built inside
  that command, so rather than assemble those dictionaries a second time for
  the protocol, `_print_json` grew a collector and a tool call runs the real
  command through `main()`. Every guard comes with it — backend detection, the
  broken-snapshot refusal, the unknown-queue check, the exit code. A report
  that disagrees with the CLI is something a reader can see; a tool result that
  disagrees is something a model repeats with confidence.

  Three things follow from a server outliving the state it describes. Each call
  takes its own reading, because serving a remembered answer would be this
  tool's cardinal sin committed by its newest surface. `sys.stdout` is pointed
  at stderr for the session and the real descriptor is kept for framing alone —
  one stray `print` out of the package's 111 would corrupt the stream. And
  dry-runs are throttled: a person types `where` a few times an hour, an agent
  in a loop will call it fifty times a minute at somebody else's controller, so
  a probing call within ten seconds of the last is answered from the declared
  allowlists and carries a second content block saying so. The payload itself
  stays byte-identical to `--json`; a caveat belongs beside the answer, not
  smuggled into it.

  `snapshot`, `exclude`, `backends` and `check` are deliberately not exposed —
  the first writes a file, the second emits shell input, the third answers about
  this host rather than the cluster, and the fourth exists only to spend probes
  that `where` already reports.

### Changed

- **The palette was rebuilt against the colour-science literature, because it
  was confusing to read.** Three defects produced that, and each is now a test.

  *Hue was carrying an ordered quantity.* The twelve-step heat ramp ran blue →
  cyan → green with its brightest step in the **middle** — L\* climbed to 91 at
  cyan and fell back to 76 at the green end, six lightness reversals — so a
  mid-range value drew the loudest row on the screen and the true extreme read
  as mid-range. That is the jet-colormap failure mode. The ramp is now a path
  through OKLCH from blue to turquoise resampled at twelve points of **equal
  ΔE2000 arc length**: monotonic in lightness, steps 4.3–5.1 apart instead of
  2.8–15.3, and every step clears WCAG AA 4.5:1 on a dark terminal.

  *Steps collapsed onto each other.* Four of the top five were one green (ΔE
  2.8, 6.4, 4.8, 4.7 apart), so two partitions an order of magnitude apart came
  out the same colour. On a 256-colour terminal steps 1 and 2 were literally
  the same index; at sixteen colours the twelve became six.

  *Verdicts and quantities shared hues.* `ok` green was ΔE 8 from the ramp's
  top step, so an idle node's green dot and its green core count were one
  colour by accident. The ramp now stops at turquoise and every verdict is at
  least ΔE 18 from all of it — and the verdicts are separated in **lightness**
  as well as hue, which is what survives red–green colour vision deficiency.
  `accent` moved from terracotta to magenta: it was ΔE 9.1 from `bad`, so the
  colour for "this is what you asked about" was a near-match for the colour for
  "this is broken". At sixteen colours `accent` no longer shares SGR 33 with
  `warn`, nor `text` 37 with `muted`.

- **Hue is no longer spent on identifiers.** Partition names, node names and
  job ids are plain. Painting a name on the heat ramp spends the one channel
  the eye reads as identity on a quantity the row already states twice — and on
  a real cluster fourteen of twenty-two partition names came out the same blue,
  so the column that says *which row this is* read as a rainbow with repeats.

- **Column headings are labels, not values.** They were painted in ramp steps 2
  and 9, so `cores free` wore the colour of a mid-sized core count and the two
  accelerator headings were indistinguishable from each other. All headings are
  now the label grey, bold on the column the table is sorted by.

- **A meter's length and its colour measure the same thing.** In `queues` the
  bar drew each queue's share of its own capacity while its tone was that
  queue's rank against the list, so `amd-hm` drew a *full* bar in the coldest
  tone (all of one node) under `amd` drawing a half bar in the warmest (half of
  eighty). The bar now draws the column the table is sorted by, as `where`
  already did. In `nodes`, core counts were ranked across the listing while
  memory and accelerators were coloured by each node's own share — two scales
  drawn from one set of twelve colours, with nothing on screen to say which was
  which. The whole table now uses the share.

- **The panel frame left the data's hues.** Its gradient swept light cyan
  through aqua, which is the ramp's own territory: a cyan border around a cyan
  column. It now runs periwinkle to light orchid, at least ΔE 21 from every
  ramp step and every verdict.

- **`--help` placeholders are no longer amber.** `NAME` in `--backend NAME` wore
  the colour this tool uses for a degraded node and a blocked queue. They take
  the identity magenta instead, and `info` — every flag and sub-command on the
  page — moved from a muddy slate to a legible periwinkle.

- **The `status` funnel counted one answer as two.** It printed
  "66 no access · 11 refused": two terms for two *filters* — what the queue
  declares (none of your accounts in its `AllowAccounts`/`AllowGroups`, read
  off the queue, free) versus what the scheduler did when a dry-run was
  actually submitted to a queue whose declared list named you. The distinction
  is real; naming it on the headline was wrong anyway. It answers a question
  nobody asked: a reader looking at that line wants to know why the table has
  eight rows, and both groups answer it the same way.

  Every wording tried for the second term failed differently — `refused` and
  `denied` read as synonyms of the first, `not listed` named a Slurm field the
  reader has never seen, `didn't work` invited the one question a four-word
  label cannot answer. So it is now **one term, `no access`**, and the note
  under it carries the only part a reader could not have guessed — that some of
  those look open on paper and a test job could not get in — plus where the
  per-queue answer lives.

  The split survives where someone asked for it. Opening the term lists each
  partition with its own reason — `no account` or `tried, no luck` — and the
  `--json` `excluded[].reason` codes are unchanged. Wire codes and screen
  wording are now separate things (`_EXCLUSION_LABELS`, `_REASON_LABELS`).

- **The output stopped saying `refused`.** `refused`, `denied` and `rejected`
  describe the cluster doing something *to* the reader; they carry nothing the
  plain words do not, and a screen that says them five times reads as an
  accusation rather than as a status. `where`'s header, its `access` column
  (`NOT_ENTITLED`) and three `fit` caveats were reworded. Two tests guard it,
  and a second pair rejects any label that needs inside knowledge to read.

- **The header line had four numbers and two denominators.** It read
  `330 of 608 nodes, 324 up  ·  230 of 358 GPUs, 58 free`, and nothing on it
  said which total each count belonged to: `324 up` is 324 of the **330**, not
  of the 608 printed beside it, and `58 free` is 58 of the **230**. The two
  cluster totals were therefore the denominators of nothing on the line, while
  the real denominators stayed implicit — so the natural reading was the wrong
  one. "why 330 of 608 nodes? what does it mean? why 324 up? are the rest of
  them down? why 58 free? what are the rest of them?"

  The cluster totals are gone: they answered "how much of this machine can I
  touch", which the funnel directly below already accounts for partition by
  partition, and which `nodetop gpus` states outright. What is left names its
  own denominator — `324 of 330 nodes up  ·  57 of 230 GPUs free`.

  No possessive either. An intermediate version said `324 of your 330 nodes
  up`, which claims something untrue: these are somebody else's nodes the
  reader is permitted to submit to. "i don't own these gpus or nodes. why are
  they mine?"

- **No prose annotating the funnel.** A note was tried under it — "11 look open
  on paper; a test job could not get in. check -q says why" — to carry the one
  thing the merged `no access` term cannot say, that some of its members passed
  the first filter. It was the longest line on the screen and a footnote in the
  middle of the answer. The fact stays reachable rather than displayed: opening
  the term names each partition's own reason, and `--json` carries both codes.

- **The header stopped naming the backend on every line.** `nodetop · slurm ·
  youzhi` spent a word on a per-machine constant: autodetection found exactly
  one batch system, and the reader could neither act on that nor have changed
  it. It is still named when it was *not* autodetected — `--backend` overrode
  the detection, or `--replay` is reading a file — because then it says which
  world the numbers came from. `nodetop backends` answers the question on
  purpose.

### Fixed

- **`status` shows memory free per partition, beside the wholly-idle nodes.**
  Memory is the gate the table was not naming: 45 of caslake's 183 usable
  nodes have no allocatable memory left while the partition still advertises
  ~800 free cores. `effective_free_cpus` already knew — it is why the figure
  beside it is not raw `cpus_free` — but nothing on the row said what had
  eaten the difference.

  A partition **total**, in the same `free/total` shape as the cores beside it,
  so the row does not switch to a per-node figure halfway across. It is
  therefore *not* more exact: it carries the same fragmentation caveat the core
  count carries, and answers "is there memory here" rather than "will one node
  take my job". `zoom` is per node and `where --mem` does the fit.

  It replaced `nodes idle` for one round — that column reads 0 in seven of this
  account's eight partitions, by its own definition, since a node counts only
  when WHOLLY free — and then sat down next to it. The question it answers has
  no other short answer: work wanting a whole machine needs that number and
  cannot get it from a core count. It is just not the column to read first,
  which it no longer has to be.

  GB on both sides, always. A switch to TB above four figures was tried and is
  wrong twice: it took the unit from one side and the number from the other
  (`5390/32.2G`), and even corrected it would make the column incomparable down
  its own length. GB is also the unit `--mem` is written in.

- **The header row is a tier again, and an even one.** `cores free` was bold
  white among grey headings to mark the sorted column: a heading that differs
  from the ones either side reads as a different kind of thing, and the sort
  was already stated twice by the data (the rows descend by it, and so does
  the meter). Levelling them all to the label grey then went too far the other
  way — that grey is `muted`, which is what the `/9033G` denominators wear, so
  the headings joined the table. Every heading is now bold `text`: one step up
  in weight and lightness from anything below it, the same step for every
  column, no hue spent.

- **`status` had no accent colour in it at all.** Counted: 71 of the roughly
  100 painted runs on the screen were one of the four greys, and the palette's
  identity hue appeared zero times — fallout from dropping `· slurm ·`, the
  possessive and the funnel note, each of which took a coloured element with
  it. Two places had it coming anyway: `gpu model` was `muted` here and
  `accent` in both `nodes --gpu` and `accelerators`, so one fact wore
  different colours in three views of one cluster; and `nodetop` names itself
  in the identity colour, which it did until the backend name carrying accent
  was removed.

- **The meter draws one composite figure and the list is ordered by it.** It
  drew each partition's free cores against the *largest* free-core count in the
  list — a denominator that appeared nowhere on the row it sat in, so the column
  needed a heading to explain a number the table did not contain. It briefly
  got one, `vs roomiest`, which is not a thing a reader should have to be told:
  "no vs or anything."

  The bar is now **`usable`**: the fraction of that partition which can
  actually be had, and the rows run most free to least. It is the *minimum* of
  the core and memory shares, not the average — a partition is only as free as
  its scarcest resource, and 800 idle cores on nodes with a gigabyte of memory
  left between them run nothing, which a mean would report as half free.

  The heading took three tries. `free share` needed a fourth word (share of
  what?); `% usable` answered that but put a symbol in a column of plain words
  and digits. One word does it: a bar drawn against a visible empty track
  already reads as a proportion, so what it was missing was never a number but
  a name for what the proportion is *of*. "usable" rather than "free" because
  the fraction is deliberately smaller than the free-core ratio beside it —
  `amd` has a quarter of its cores free and a sixth of it usable, memory being
  the binding constraint — and "free" against `1479/5120` would read as an
  arithmetic error.

  Two exclusions, both deliberate. The wholly-idle **node count** is out: a
  partition with no completely empty machine is not 0% free, and folding that
  in would zero almost every row. **Accelerators** are out too, which is the
  closer call — a GPU partition whose cards are all taken can still run CPU
  work, and including them ranked `gpu` below a partition that simply has none,
  comparing a scarcity against an absence. Both keep their own columns.

  A share ranking does put small partitions near the top: `build` leads at 72%
  of one node while `amd` sits sixth at 13% of forty. The three absolute
  columns beside the bar are what keep that honest — `42/48` against
  `1354/5120` says how much each fraction is worth. The sort key falls back to
  absolute cores, then accelerators, then name, so an all-idle cluster does not
  come out alphabetical.

- **`gpu free` counted devices and did not say so.** "when you say gpu free, is
  it gpu nodes free or gpu free?" — `beagle3` is 44 nodes carrying 176
  accelerators, so `36/176` is cards, and the only thing on the row indicating
  that was the denominator being too large to be a node count. It is `gpus
  free` in all three tables now, plural, which parallels the `nodes idle`
  column two along that really does count nodes.

- **`cores free` is no longer tinted.** Its colour came from `core_heat[i]`,
  which is exactly what the bar beside it draws — in its length *and* in its
  fill — so one variable was encoded three times in adjacent cells, the same
  waste the partition names were guilty of. Colour is the channel to drop:
  Cleveland & McGill rank position and length well above hue and saturation for
  judging quantity, so the cell that can only offer colour should not be the
  one carrying it. The number now states an exact count over its own capacity
  and the bar states how it compares to the list — two denominators, two
  questions. `mem free` and `gpu free` keep their tint; they have no meter, so
  there colour is the only proportional cue available.

- **The meter is capped at 20 cells, not 40.** Above twenty it buys nothing —
  a bar is read by its proportion of the track, and forty cells resolve 1/320
  with eighths for an eye that cannot see past about a twentieth — while
  taking width from the columns that carry digits. Below the width where the
  digits and a ten-cell bar both fit it is dropped entirely rather than
  squeezed: it draws the number printed immediately to its left, so losing it
  costs a picture of a figure still on the row. Handing `table` the real width
  with `fit=True` was tried instead and is worse; it shrinks every column, so
  a narrow terminal rendered `1423/903…` and a truncated bar as well.

- **The layout takes the terminal's full width.** It was capped at 100 columns,
  so the tool sat in a box off to the left of a wide window: "i think the app
  should take the entire hortizontal space. the current one looks so squeezed
  and unnatural." Frames, tables and the interactive browse now span whatever
  the window is, static and interactive alike, so a redirected report and a
  browsed one are the same shape.

  **The meter absorbs the extra room**, without which the width would only
  move the border and leave the same rows huddled at the left of a bigger box.
  Every other cell is sized by its text and cannot use spare columns; a bar is
  a proportion, so more cells is strictly more resolution. It runs 10 cells at
  80 columns to 40 at 112 and above — bounded at both ends, because below ten
  the sub-cell eighths stop separating a nearly-empty queue from an empty one,
  and past forty a bar reads as a rule across the screen rather than as a
  quantity.

  Two widths, not one. `MAX_WIDTH` is now a sanity bound (400) rather than a
  design measure, `FALLBACK_WIDTH` (100) is what a pipe with no window gets —
  inheriting the cap there would have set every redirected table to four
  hundred columns — and prose keeps its own `PROSE_WIDTH` (96), because tables
  taking the whole window is no reason for paragraphs to.

- **`table()` padded a final left-aligned cell for nothing.** A column sized to
  `V100, RTX6000` padded the heading `gpu model` out by four spaces. Invisible
  to a reader, and not invisible to anything that *measured* the table:
  `status` draws its rule at `max(width(...))`, so the rule came out 97 columns
  for content ending at 79. Nothing follows a last cell to align against, so
  the padding is no longer added. Not fixed with `rstrip`, which was tried and
  is wrong — a padded cell keeps its spaces *inside* the styled run, so the
  plain line loses them and the coloured twin does not, which breaks the one
  property `TestLayoutStability` exists to protect.

- **Two blocks inside the frame wrapped to the terminal, not to the frame.**
  A drained node's `Reason` and a job's detail pairs both sized themselves
  from `term_width()` while being drawn inside a narrower box, so the panel
  truncated the overflow — an ellipsis through the full reason text, which is
  the one thing that view exists to show whole. Only visible once the frame
  stopped being as wide as the window. Both wrap to the frame now, as the
  funnel's note already did.

- **Every run pushed a dozen lines of terminal history off the screen.** The
  interactive frame is padded to one height so the box does not jump as you
  move between levels — that part was asked for and is right. The height was
  wrong: it was the whole window, up to 30 rows, and this cluster's overview
  is 16. So `nodetop` drew fourteen blank rows inside its own border and
  scrolled fourteen lines of the reader's scrollback away to make room for
  them, on every single invocation. The height is now the overview's, clamped
  to what the window can hold — still one number for every level, so nothing
  jumps, and deeper levels page inside it exactly as they already paged inside
  the larger one. 31 rows → 19 on this cluster; 23 → 13 for a four-partition
  listing on a 24-row terminal.

- **Quitting destroyed the report.** `q` erased the block on the way out, so
  the tool printed an answer, waited, and took it away again: "when i exit it
  ... everything shown before is gone." The erase exists so each nested level
  replaces the last instead of appending a transcript of screens — right for
  stepping between levels, wrong for leaving, where nothing replaces those
  rows. The final frame now stays where it can be read, scrolled back to and
  copied out of, with the cursor already on the line below it.

- **The screen went blank for over a second just after `status` opened.**
  A reload — `r`, the idle timer, or the background access re-check landing —
  left `select` through a path that *erased* the block, and the new frame was
  only drawn once a full re-read of the cluster came back, 1.4s later on this
  cluster. So the report painted, sat there, vanished, and returned. It is the
  same erase-then-write defect `paint` already fixed for keypresses, except the
  gap is a scheduler query rather than a few microseconds. A reload now leaves
  its frame standing and hands the row count to the next one, which winds back
  and overwrites in place; nothing is ever blank. It showed up most on the
  *second* run, where a warm access cache paints instantly and the background
  re-check then triggers exactly this reload — which is why it read as a
  start-up glitch rather than as a refresh.

- **A note inside a panel lost its last words.** `_note` wrapped prose to the
  window while a panel is four columns narrower than that — two of border, two
  of padding — so the final line of a framed note overran by exactly four and
  the panel truncated it. Silently, and only at the narrow terminal sizes where
  the sentence had least room to spare. `_note` now takes the width it is
  wrapping into.

- **The first 1.68 seconds showed nothing at all.** Cold on a 607-node cluster
  that is how long `Cluster.load` spends in `sinfo`/`squeue`/`scontrol` before
  anything reaches the terminal, so typing `nodetop` left the shell history on
  screen for the better part of two seconds. The probe phase after it already
  announces itself, for the stated reason that a silent wait is
  indistinguishable from a hang; the phase a reader meets *first* did not. It
  now prints `reading <backend>` on stderr and clears it. First byte: 1.68s →
  0.08s.

- **`health` painted a clean result in alarm colours.** `0 degraded` was amber
  and `0 out` red, so a cluster with nothing wrong drew the two loudest colours
  in the palette on its headline. A count of a bad thing takes its verdict
  colour only when the count is not zero.

## [0.6.0] — 2026-09-10

An audit of 0.5.2, each finding reproduced by running it before it was fixed.
Two new public helpers, one new `--json` key, and input that used to be accepted
is now refused — so a minor bump rather than a patch. Every entry below shipped
with a regression test and a control verified in both states.

The five behavioural fixes share a shape: **bad input did not fail, it quietly
removed a check.**

### Fixed

- **`exclude` printed a sentence nothing could substitute.** With no matching
  nodes the command wrote `(no matching nodes)` — ANSI-wrapped on a tty — to
  **stdout**, while `--json` answered `{"count": 0, "nodelist": ""}`. The whole
  point of the command is to be substituted, so
  `sbatch --exclude=$(nodetop exclude --unschedulable)` handed that sentence to
  the scheduler as a node name, escape codes included. Nothing goes to stdout
  now; the note goes to stderr and the two surfaces agree. DESIGN.md 1c is about
  exactly this shape — returning something unusable is worse than returning
  nothing, because the caller believes it has exclusions.

- **`--needs bf16typo` matched EVERY node.** `hardware.supports` answers `None`
  for a capability it does not know, and `capability_gap` records only an
  explicit `False`, so an unknown requirement passed the filter instead of
  failing it. Measured on an A100 spec:
  `hardware_ok(node, JobShape(requires=("bf16typo",)))` returned `(True, ())` —
  the opposite of what a filter is for. The capability names are now a single
  mapping that `supports` reads and `CAPABILITIES` exposes, so the two cannot
  drift, and the CLI refuses an unknown name with exit 2. The tri-state `None`
  is deliberate and stays: it means "we do not know what this node is".

- **`--time garbage` silently meant "unlimited".** `-t/--time` had no `type=`,
  and `parse_duration` answers `None` both for the sentinels (`unlimited`,
  `n/a`, `0`) and for anything it cannot read, while every ceiling check
  downstream skips a `None`. So `1w` (weeks are not a unit here), `1h30`
  (missing the second unit) and `1.5h` (no floats) each disabled the
  `MAX_WALLTIME` comparison they were meant to tighten. `--mem` has always
  rejected bad input this way; `--time` now does too, and the sentinels are
  still accepted because "no limit" is a real thing to ask for.

- **PBS reported `10GiB` as a node with no memory.** `_mem_to_mb` had no slot
  for the `i` of the IEC spelling, so the match failed outright and the size
  read `0` — "not read" — and a memory ceiling of nothing. `10GiB` and `10Gib`
  both gave 0 while `10gb`, `10g`, `1.5gb` and `10 gb` were correct. Sites
  increasingly emit the IEC form, and these suffixes are already binary, so
  `GiB` and `GB` mean the same thing to this parser.

- **A reversed hostlist range excluded nothing.** `expand("n[10-1]")` returned
  `n01 … n10`: the endpoints were swapped but the zero-pad width was still taken
  from the endpoint *as written*. `n01` is a different node from `n1` — which is
  the whole reason `n[1-10]` was fixed to answer `n1 … n10` — so a range typed
  backwards by hand named ten nodes that do not exist.

### Added

- **`where --json` verdicts carry `durable`.** The text surface distinguishes
  `BLOCKED` from `NO ANSWER`; a `--json` consumer could only reproduce that by
  vendoring `TRANSIENT_CATEGORIES`. A wire vocabulary that cannot say "this
  refusal is real" forces every consumer to copy the table.
- **`duration.understood(text)`** — is this a walltime spelling the module
  recognises at all? Separates "no limit" from "could not read that" without
  changing what `parse_duration` returns, so existing callers are unaffected.
- **`hardware.CAPABILITIES`** — every capability name `supports` understands,
  for callers that need to validate one before asking.

### Changed

- `-t/--time` and `--needs` now exit **2** on input they previously accepted
  and ignored. That is the point of the change, but it is a behaviour change:
  a script passing a walltime this tool cannot parse used to get an unlimited
  ceiling, and now gets an error naming the spellings that work.

### Documentation

- `-p` was described as "accepted everywhere as an alias for `-q`" in both
  README and DESIGN.md. It is on six of eleven commands. The flag-to-command
  table is now measured from the parser and the prose is checked against it.
- DESIGN.md's `--all` list omitted `zoom`, `nodes` and `accelerators`; README
  listed `--all`/`--detail`/`--static` unqualified; each doc was missing the
  other's backend vocabulary (`pool`, `namespace`).
- `zoom --help` hardcoded "partition", so it said the wrong word on every
  non-Slurm backend. The parser is built before a backend is detected, so the
  neutral "queue/partition" is the honest spelling.
- **`status` answering exit 0 on an empty cluster is documented and pinned.**
  Through `main()` that input exits 3; the direct library call returns 0,
  because 3 is a code `main()` owns and every `cmd_*` returns 0/1/2. The
  difference was undocumented and untested, which is what made it look like a
  bug. Both halves are now asserted from both ends.

## [0.5.2] — 2026-09-09

Polish and bugfix work on top of 0.5.1 — no API changes, so a patch release.
Every entry below shipped with a regression test and a control verified in both
states.

### Fixed

- **`snapshot` was the one command that recorded a cluster-wide outage in
  silence.** Every other command names each failed query on stderr and, when the
  snapshot is empty as well, refuses to print numbers and exits 3. `snapshot` is
  deliberately exempt from that guard — a recording of a broken cluster is a
  legitimate artifact, `errors` is a field of it, and replaying it *is* rejected —
  but it said nothing at the time, so a recording made during a controller outage
  looked like any other and exited 0. It now names each failed query on stderr,
  like the rest, leaving stdout (and `-o -`) untouched.

- **A meter drawn completely full did not mean full.** Every caller prints an exact
  ratio beside `render.bar` — `cores free` shows `5115/5120`, the feature table
  `231/232`, `render.gauge` `88/176 gpu` — so unlike this family's percentage
  gauges there is no rounded label for the bar to agree with: the only fullness the
  number admits is numerator == denominator. Both draw paths round, and both
  reached a solid bar early. Measured across 90 → 100% at eight and eighteen cells,
  in Unicode and under `NODETOP_ASCII`: ASCII drew `########` for **96.6%** of 8
  cells and `##################` for **98%** of 18; Unicode drew a solid
  `████████` from **99.4%** of 8 and a solid eighteen from **99.9%**. So a
  partition with 5115 of its 5120 cores free rendered as *every core free*, which
  is the one distinction the `cores free` column exists to make, and the 9-cell
  accelerator gauge turned `175/176` into a solid `█████████`.

  The final unit is now withheld until the fraction reaches 1.0 — one eighth on the
  Unicode path, because sub-cell resolution is this meter's stated point and an
  eighth is all it takes for the tip to stop being `█`, and one whole cell on the
  ASCII path, which has no eighths to spend. `175/176` at nine cells now reads
  `████████▉ 175/176` and `176/176` still reads `█████████ 176/176`. Nothing below
  the turnover moved: the reserve can only bite where the uncapped fill would have
  reached `size` whole cells, above 98.4% at eight cells, so every share a reader
  normally looks at draws exactly what it drew before, and the low end is untouched
  (1/128 of a 16-cell bar is still a visible tip, a true zero still an empty track).

  **The meter memo had to learn the same boundary.** It is keyed on the rounded
  eighths and rounded cells, and at eight cells both are identical for 0.999 and
  1.0 — the two really were one picture and one key until this change made them
  two. Left alone, the solid bar drawn for a 5120/5120 row would have been handed
  straight back to the 5115/5120 row under it; the repo's own
  `test_a_memoised_meter_is_the_meter_it_would_have_drawn` catches it at size 1.
  The key now carries the predicate the body branches on.

  Third and last member of a family-wide sweep of this invariant: `slurmwatch`
  reserves through 99.5–99.99 because its labels carry no decimals, `slurmpast`
  through 99.95–99.99 because its labels carry one. Here the label is an integer
  ratio, so the reserve runs all the way to the whole.

- **`queues --json` did not carry the column the `queues` table prints first.** The
  table's own header names it — `nodes up`, the count of a partition's nodes the
  scheduler would still place work on — and the document published the denominator
  plus three *other* numerators, so a row the screen showed as `39/40` offered
  `idle_nodes_advertised: 26`, `nodes_with_room: 38` and `nodes: 40`, none of them
  39. It was not derivable either: `nodes --json` carries `schedulable` per node but
  no partition, so there is nothing to join on. Rows now carry
  `nodes_schedulable`, which is the name `status --json` already uses for the same
  measurement cluster-wide, and it is read from the `Queue.schedulable_nodes` the
  renderer uses rather than from a second copy of the rule. Verified against a
  frozen snapshot with both surfaces on identical input: `amd` 39/40, `beagle3`
  43/44, `caslake` 190/190, `test` 552/608, screen and document alike.

- **`where --json` erased the access distinction the table makes.** The table's
  ACCESS column marks a partition whose allowlist names one or two accounts
  `group-only` — somebody's own hardware, and the strongest thing sayable about
  access without a dry-run — while `entitlement_source` walked a *second* copy of
  the same ladder that had no rung for it. Measured on one frozen snapshot of this
  cluster, driving both surfaces off the same replay so the input was identical
  (1 node, 8 CPUs, 32 GiB, 4:00:00): 19 partitions, the table marking **11** of
  them `group-only`, and **all 19** rows of the document reporting
  `"entitlement_source": "declared"` — which means "there was no dry-run to run
  here", true of a PI's partition and of `caslake` alike. Diffed field by field, a
  `ssd` row and a `caslake` row differed only in capacity numbers and
  `submit_flags`, so a consumer ranking partitions had nothing to tell private
  hardware from a general queue. The ladder now lives in one function,
  `_entitlement_source`, called once per row by each surface, and the table keeps
  only the choice of word and colour; the same replay now reports `group-only` on
  exactly those 11. The row also gains a `dedicated` boolean, because that line has
  one slot and a verdict outranks the heuristic in it — measured live, `ssd` comes
  back `entitlement_source: "refused"`, and without the extra key the row would
  again stop saying whose hardware refused it. `status --json` has published
  `dedicated` per queue all along. Nothing on the text surface changed: byte-for-byte
  the same table before and after.

  Settled in the same pass and deliberately **not** changed: `"verdict": null` on
  every row of a replay is correct. That key publishes the `Verdict` dataclass —
  the control plane's own answer, `None` from a backend that has no dry-run, which
  a replay by construction is — while the table's verdict column is
  `_verdict_label`, a placement label computed from five inputs of which the
  verdict is one. All five are in the row, `blockers` carrying the `fatal` flag
  that separates the last two, so the document publishes the ingredients and
  leaves the label to the renderer, exactly as it does for `fits/need`.

- **`FORCE_COLOR=0` turned colour on at full strength.** The variable's value is
  read as a level — `1`/`true` mean 256 colours, anything else means truecolor —
  so the one value that names *zero* selected the top of the ladder: measured,
  `FORCE_COLOR=0` and `FORCE_COLOR=false` both gave depth 24 with colour enabled,
  which is also the reading that contradicts the line directly above the code
  ("0 = no colour"). Both now disable colour, like `NO_COLOR`. An empty value is
  still "unset, not off" — the distinction the same function already made for
  `NO_COLOR` — and every other value keeps its meaning, including
  `FORCE_COLOR=3` for truecolor.

- **`status` claimed "every query failed" when only one had.** The message chose
  between two sentences on "is there any error at all" while asserting that *all*
  of them failed. A cluster that answered the node query in full and lost only
  `scontrol show partition` printed, two lines apart, `query failed: queues: …`
  and then `no data: every query failed` — naming the single query that failed and
  immediately contradicting it. `Cluster.load` labels six independent reads and
  records only the failures, so "every" is not derivable from it: the universal
  sentence is now used only when both primary reads are gone, and the failed
  queries are named otherwise ("the limits and queues queries failed").
- **An unmeasured ROCm GPU was counted as free.** The probe runs
  `rocm-smi --showproductname`, which reports a name and nothing else, so no memory
  reading for the card exists — but it was recorded as `0`, and a GPU "holding no
  memory" is read as idle. On an AMD pool every card therefore reported free
  however much was running on it, which is the collision the occupancy rule exists
  to prevent. An unmeasured card now counts as occupied; a measured idle card is
  still free, so an idle NVIDIA card beside a ROCm one reports one busy, not two.
- **AMD and Intel cards were left unnamed.** `name_accelerator` reports what the
  scheduler called a card, vocabulary or not, but its shape test accepted `nvidia`
  bare and had no `amd` or `intel` form — so `AMD Instinct MI250X` and
  `Intel(R) Data Center GPU Max 1550` came back nameless even though the spec table
  recognised them. Both are now named. The prefixes are admitted only in front of a
  known GPU family, because both vendors ship CPUs and `AMD EPYC 7763` /
  `Intel Xeon Gold 6248` must not be reported as a node's accelerator.
- **Long NVIDIA product names were rejected.** The name cap counted 12 characters
  after the vendor word, which `NVIDIA A100 80GB PCIe` just fits and
  `NVIDIA GeForce RTX 4090` does not, so the latter was refused a name entirely.
- The PBS backend ran a duplicated component check, and `snapshot`'s guard carried
  a `# pragma: no cover` for a branch that is reached and returns exit 2.

### Documentation

- The exit-status paragraph credited `where` with codes 2 and 3 it cannot return.
