# Changelog

All notable changes to nodetop are documented here, newest first.

The format is based on [Keep a Changelog](https://keepachangelog.com), and this
project adheres to [Semantic Versioning](https://semver.org).

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
