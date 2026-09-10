# nodetop audit — verified findings

Audit of `/home/youzhi/nodetop` at v0.5.2. Every item below was verified
against the code and, where stated, by executing it. Items that did not
survive verification were dropped (see §8). No code was changed.

Test baseline: suite is green (`pip install -e ".[dev]"`, `python -m pytest -q`
passes). The findings are therefore edge cases, contract breaks, and
docs-vs-code drift that the suite does not cover — not regressions in the
happy path.

---

## 1. Critical

### 1.1 `exclude` prints a literal sentence to stdout when there is nothing to exclude

`src/nodetop/cli.py:4423-4425`:

```python
if not nodelist:
    print(st.dim("(no matching nodes)"))
    return 0
```

Trigger: healthy cluster, `nodetop exclude --unschedulable >out.txt` when
nothing is down. `out.txt` contains `(no matching nodes)` (with ANSI escapes
when stdout is a tty, via `st.dim`), exit 0. The `--json` path (`:4419-4422`)
correctly returns `{"count": 0, "nodelist": "", ...}` — text and JSON diverge.

Impact: `sbatch --exclude=$(nodetop exclude --unschedulable)` passes the
literal string as a node name. The DESIGN.md guard (§1c) was written
precisely to keep this command pipe-safe; the empty case reintroduces the
same shape. Fix: print nothing to stdout (note to stderr if anything).

Verify: `nodetop exclude --unschedulable >out.txt; cat -A out.txt` on a
cluster with zero matches vs `nodetop exclude --unschedulable --json | jq .nodelist`.

---

## 2. Major — silent acceptance of bad input

### 2.1 `--needs <typo>` is silently ignored

Chain: `src/nodetop/core/hardware.py:639-659` `supports()` returns `None`
for unknown requirements (`table.get(...)` → `None`); `src/nodetop/core/model.py:1351-1361`
`capability_gap()` only records `supports(...) is False`; `src/nodetop/core/capacity.py:77-78`
only appends gaps. `src/nodetop/cli.py:1028,1044` takes `--needs` with no
validation.

Verified:

```python
hardware_ok(node_with_A100, JobShape(nodes=1, gpus_per_node=1, requires=("bf16typo",)))
# -> (True, ())
```

`--needs bf16typo`, `--needs cuda2`, `--needs ""`-adjacent strings all match
every node. Case and surrounding whitespace are fine (`supports` strips and
lowers); only unknown tokens vanish. A misspelled capability filter that
matches everything is the wrong default — it should error (exit 2) or at
least warn. Same for `JobShape(requires=("",))` at the library level.

### 2.2 `-t/--time <garbage>` silently becomes "unlimited"

`src/nodetop/cli.py:1007` registers `-t/--time` with no `type=`; `src/nodetop/core/duration.py:41-101`
`parse_duration()` returns `None` for sentinels **and** for unparseable
input; `JobShape.walltime_seconds` (`src/nodetop/core/model.py:1159-1162`)
passes that through; `Limits.blockers()` (`model.py:937-942`) and the queue
check (`fit.py:241-254`) skip when `want is None`.

Verified:

```python
JobShape(walltime=t).walltime_seconds is None
for t in ["garbage", "1w", "1h30", "1.5h", "", "-5"]
```

So `--time garbage`, `--time 1w` (weeks unsupported), `--time 1h30`
(missing second unit), `--time 1.5h` (floats unsupported) all disable every
`MAX_WALLTIME` check instead of failing with exit 2. `--mem` correctly
rejects bad input via `memory_gb` (`cli.py:852-868`); `--time` has no
equivalent. A typo'd walltime that removes the ceiling check inverts the
tool's purpose on that axis.

### 2.3 `parse_duration` conflates "unlimited" with "unparseable"

Same function as §2.2: `None` means both "no limit" and "could not parse".
Callers cannot distinguish a deliberate `UNLIMITED` from a typo, so every
caller either treats typos as unlimited (§2.2) or must re-validate the raw
string. Root cause; fixing here (raise / sentinel) fixes §2.2 structurally.
Related minor: `format_duration(0)` (`duration.py:121-130`) renders
`"0:00:00"` but `parse_duration` never returns `0` (int path `text if text >
0 else None`, `"0"` in `_SENTINELS`), so that arm is dead; bare
`"9999999999"` (minutes) yields ~19M years with no cap.

### 2.4 `where` exits 0 when every answer is transient (unanswered)

`src/nodetop/cli.py:4073`:

```python
exit_ok = any(p.reachable and not p.hardware_incompatible for p in places)
```

`Placement.reachable` (`src/nodetop/core/fit.py:92-108`) is `True` for
transient verdicts by design (only durable refusals make it `False`), and
the default `where` filter (`cli.py:4051-4054`) keeps transient rows. So a
control-plane outage (`CONTROL_PLANE_DOWN` everywhere, `NO ANSWER` rows, best
placement `note="queues"`) yields exit `0`. `check` in the same state exits
`1` unless something was `allowed` — strict, per DESIGN.md §1a ("waving a
`check && sbatch` caller through on an unanswered probe is the one outcome
worth being strict about"). `where && sbatch` is waved through on the same
input. Either `where` should exit 1 when nothing is confirmed, or the
difference should be documented and tested; currently the two commands
disagree with no comment at the `exit_ok` site.

### 2.5 `status --static` opens on the remembered (stale) answer

`src/nodetop/cli.py:1630`:

```python
if not args.json and not cluster.replayed and _interactive().supported():
    remembered = access.load(cache_key)
```

No `args.static` check, against the comment directly above (`:1620-1621`):
"a session may open on the previous answer and correct itself; a printout
may not, because it gets one shot at being right." `--static` is the
printout spelling (`--static` only exists on `status`, `:1143-1144`), starts
a background `_Recheck` (`:1655`) it never waits for, and never prints the
`access checked Xs ago` line (browse-only, `:2251-2257`). Poison the cache,
run `--static` vs `--json` (JSON always probes fresh, `:1649-1653`) on a tty:
they disagree with no warning on the `--static` side.

### 2.6 `--json` on a fatal snapshot prints no JSON

`_reject_broken_snapshot` (`cli.py:249-341`) returns 3 before dispatch on
both paths (`:5109-5113` replay, `:5187-5193` live), printing only to stderr.
`status_json`'s docstring (`:1397-1403`) promises the opposite: "a `--json`
that prints nothing there is worse than one that prints zeros: the caller
cannot tell 'no nodes' from 'the command crashed'", and the degenerate
branch (`:1483-1490`) that would emit it is unreachable via `main()` (see
§2.7). `nodetop --json status | jq .` on a dead control plane yields empty
stdout, exit 3. Contrast `check --json` with `can_probe is False` (`:4235`),
which does emit a document and exit 2. Every `--json` consumer must special-
case empty stdout.

### 2.7 `cmd_status` empty-nodes branch is dead and returns the wrong code

`src/nodetop/cli.py:1483-1490`:

```python
if not cluster.nodes:
    if args.json:
        return status_json()
    ...
    print(panel(body, "", st))
    return 0
```

Unreachable via `main()`: the guard computes `fatal = not cluster.nodes or
...` (`:283`) and returns 3 first. Direct callers get exit `0` ("success")
for "no nodes — wrong backend, or the control plane is down", contradicting
the exit-3 contract in the guard's docstring (`:266-269`). Dead code with a
wrong code is a trap for the next refactor.

### 2.8 `status` reports partial failures on stdout; everything else uses stderr

`src/nodetop/cli.py:1479-1481` appends `FAILED <query>` lines to the `panel`
body (stdout). The guard deliberately skips stderr naming for non-fatal
`status` (`:297 `if fatal or command != "status"``), leaving those lines as
the only record — while every other command names failures via
`_name_failed_queries` on stderr (`:229-246`) to keep stdout pipeable
(`:231-234`). `status | cut` ingests `FAILED` rows; `status --json` carries
the failures only inside `summary.errors` (`cluster.py:538`, spread at
`cli.py:1404-1405`), truncated vs full. The streams disagree per command.

---

## 3. Medium — wire/text divergence and exit-code honesty

### 3.1 `where --json` verdict omits durability

`src/nodetop/cli.py:4163-4169` emits
`allowed/category/reason/filter_verdict/effective_qos` but not `durable`.
Text distinguishes `BLOCKED` from `NO ANSWER` (`:3741`, `:3978`); JSON
consumers must hardcode the category→durable map (`TRANSIENT_CATEGORIES`,
`model.py:1288-1296`) to reproduce it. A wire vocabulary that cannot answer
"was this asked?" forces every consumer to vendor the table.

### 3.2 Dynamic top-level JSON keys break consumers per backend

- `zoom --json`: `src/nodetop/cli.py:3195` `f"{cluster.queue_term}": ...`
  → `partition` on Slurm, `queue` on PBS/LSF/SGE, `namespace` on K8s.
- `accelerators --json`: `:4630,4634,4636`
  `f"{cluster.queue_term}s[_open|_group_only]"` → `partitions…` vs `queues…`.

`status --json` deliberately uses fixed `"listed"` with a comment (`:1878-1881`)
explaining the PBS collision; `zoom`/`accelerators` repeat the pattern. A
consumer must branch on the backend to find the queue list.

### 3.3 Ctrl-C in the browse exits 0; before the browse it exits 130

`interactive.select` maps `KeyboardInterrupt` → `Key.QUIT`
(`interactive.py:521-522`); `_browse` root maps non-`int` → `return 0`
(`cli.py:2673-2674`, also `:2697-2698,2716-2717,2745-2746` for deeper
levels). `main()` maps `KeyboardInterrupt` around load/dispatch → 130
(`:5198-5200`). Same key, two codes depending on whether the 1.6s probe
phase or the browse owned the terminal. README promises 130 for Ctrl-C.

### 3.4 One `Style` for two streams

`main()` builds a single `st = Style(...)` (`cli.py:5076-5079`); `_depth()`
(`render.py:497-521`) keys on `sys.stdout.isatty()`, `TERM`, `NO_COLOR`.
The same `st` paints stdout tables and stderr diagnostics
(`:5102,5125,5134,3175,4917`). Piped-stdout/tty-stderr loses stderr colour;
tty-stdout/file-stderr writes ANSI into the log. `_name_failed_queries`
correctly uses `Glyphs.detect(sys.stderr)` (`:244`); colour does not.

---

## 4. Backend parsing gaps (all fail toward "unknown", none crash)

### 4.1 PBS rejects `GiB`/`Gib` memory suffixes

`src/nodetop/backends/pbs.py:111`
`r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)([bw]?)\s*$"` has no `i` slot, so:

```python
_mem_to_mb("10GiB")  # -> 0 (want 10240)
_mem_to_mb("10Gib")  # -> 0
```

(`10gb`/`10g`/`1.5gb`/`10 gb` all correct.) `0` means "not read":
`Node.memory_mb = 0` skips the memory constraint (`capacity.py:116-120`,
safe direction) and `_pbs_mem_mb` records `unreadable` — so the cost is lost
information, not phantom capacity. Still, `GiB` is what sites increasingly
emit. Verify: `python -c "from nodetop.backends.pbs import _mem_to_mb; print(_mem_to_mb('10GiB'))"`.

### 4.2 Hostlist reversed ranges mis-pad

`src/nodetop/hostlist.py:109-117`: `width = len(lo_s)` is taken before the
`hi < lo` swap:

```python
expand("n[10-1]")  # -> ['n01', ..., 'n10']  (want ['n1'..'n10'] or descending)
```

`n01` is a different node from `n1` on most clusters (cf. the `n[1-10]`
comment at `:91-97`), so a hand-typed reversed range silently excludes
nothing. Rare input; wrong output. Related nits, same file, verified:
`host[1-3,3-5]` yields `host3` twice (no dedup; harmless downstream because
`Cluster.load` dedups via `by_name`/`dict.fromkeys`, `cluster.py:161-178`);
`n[]` expands to `[]` (node vanishes rather than staying literal);
`expand` preserves input duplicates while `collapse` dedups.

### 4.3 K8s fractional CPU truncates; odd-case quantities read as 0

`_quantity_to_cpu("1500m")` → `1` (1.5 cores, truncated — safe direction,
understates room). `_quantity_to_mb("256MI"/"256gi")` → `0` ("not read").
The latter matches the K8s spec (units are case-sensitive: `Mi`/`Gi`), so
this is arguably correct strictness; noted because a site emitting
lowercase gets silent "unknown memory" rather than a named `unreadable`.

### 4.4 LSF `Feb 29` unparseable in non-leap years

`src/nodetop/core/duration.py:236-258` retries LSF short-form timestamps as
`f"{t} {now.year}"`. In 2026 (non-leap) `parse_timestamp("Feb 29 12:00")`
→ `None` instead of trying an adjacent leap year. Effect: missing
`earliest_free` for those jobs only, one day in four years. Noted for
completeness; the New-Year rollback and 1900-sentinel handling around it are
correct.

---

## 5. Library validation gaps (CLI is guarded; direct users are not)

`JobShape` (`src/nodetop/core/model.py:1124-1198`) accepts `nodes=0`,
negative counts, and empty/unknown `requires`:

```python
assess_capacity([node], JobShape(nodes=0)).satisfies(JobShape(nodes=0))  # True
JobShape(nodes=-1).total_cpus  # -1
```

CLI floors via `_at_least` (`cli.py:821-849,996-1004`), so this is only
reachable programmatically (`Cluster.load()`, `rank()` per README §Library).
A 0-node shape is trivially "runnable everywhere". Either floor in the model
or document that the model trusts the CLI.

---

## 6. Docs vs code (all verified against `build_parser()`)

Parser truth table (this host): `-p`/`-q` on
queues/nodes/where/check/exclude/accelerators only; `--static` on status
only; `--detail` on queues only; `--all` on
status/queues/zoom/nodes/where/accelerators (not check/health/backends/snapshot).

- README.md:174-175 "`-p` is accepted everywhere as an alias for `-q`" —
  false for status/zoom/health/backends/snapshot. The helper's own docstring
  (`cli.py:975-976` "on every command") over-claims identically.
- README.md:159-163 lists `--all`/`--detail`/`--static` unqualified;
  README.md:140 "`--static` prints and exits" with no status-only scope.
- DESIGN.md:867-869 "`--all` widens status, queues and where" — omits
  nodes/zoom/accelerators that have it.
- README.md:174-175 vocabulary omits `sshpool:pool`
  (`backends/sshpool.py:63`); DESIGN.md:888-889 omits
  `kubernetes:namespace` (`backends/kubernetes.py:115`). Each doc is missing
  the other's backend.
- `accelerators` canonical vs alias: code (`cli.py:1231`)
  `accelerators, aliases=[accel, gpus]`; README.md:152 and DESIGN.md:844
  present `gpus` as the command, DESIGN.md:754 as `accelerators`.
- `zoom` help hardcodes `partition` (`cli.py:1158`
  `f"open one {'partition'} up..."`) instead of the backend's `queue_term`.
- CHANGELOG.md has only `0.5.2` though tags `v0.1.0…v0.5.2` and commits
  `0.5.1`/`0.5.0` exist — violates its own Keep-a-Changelog header.
- Stale DESIGN.md:857-861 ("dry-run, capped per queue") contradicts
  DESIGN.md:191-195 + `fit.py:505,515` (`MAX_PROBES_PER_QUEUE=12` backstop,
  `MAX_PROBES_TOTAL=150` global). Stale §156-159 (split free/total columns)
  contradicts `cli.py:1715-1742` + README.md:113-116 (reverted to
  `free/total` under a numerator header, with rationale in comments).
  DESIGN.md:832 ("unusable queues first") contradicts DESIGN.md:108-110 +
  `cli.py:1665-1675` (failures last).
- README `can_probe` note ("False on PBS and LSF") is host-dependent:
  `sge/kubernetes/slurm` are also `False` here without their clients, and any
  replay forces `False` (`cluster.py:256-264`). README omits the replay case
  that DESIGN.md:2364-2366 states.
- Counts: README `~4855 tests` / DESIGN `4855` vs 4903 collected — stale
  numbers, no behavioural impact.

---

## 7. Minor nits (verified, low impact)

- `_GUARD_EXEMPT` (`cli.py:226`) is defined and documented but never
  referenced (`rg` hits: definition + comment at `:4819` only). `backends`
  bypasses via early return (`:5081-5082`), `snapshot` via `:5155-5164`. A
  new early-return command silently bypasses the guard the set claims to
  enumerate. Enforcement or a `test_guard`-style assertion referencing the
  set would close it (the existing `test_guard_covers_every_command.py`
  classifies by spying; it does not read the set).
- Snapshot encoding asymmetry: written `encoding="utf-8"` (`cli.py:4895`),
  read via `Path.read_text()` with no encoding (`:4942`) — locale-dependent.
  No traceback (raised `UnicodeDecodeError` is a `ValueError` subclass and
  the handler at `:5087` catches `OSError,ValueError,KeyError` → clean
  "cannot replay" + exit 2), but round-trip under `LC_ALL=C` depends on
  UTF-8 mode rather than the file's actual encoding. Pass `encoding="utf-8"`
  on the read.
- `_Recheck` robustness (`cli.py:575-591`): `access.save(key, got)` sits
  outside the `try` wrapping `ask()`, with no `finally` for `done`. In
  practice `save()` catches `OSError/ValueError/UnicodeError` internally
  (`access.py:320`) and returns `False`, so disk-full/permission failures do
  not escape — the window is unexpected exceptions only (leaving
  `done` unset, `moved()` false, pacing state unstamped). Move into
  `try`/`finally` and honour the return.
- `-q/-p` repeated uses last-wins (single `dest="queue"`, `:978-983`):
  `queues -q a -p b` silently shows `b`. `--top 0/-5` silently clamps to 1
  (`:3281,3428` `max(1, args.top)`) while `where -N/-c/-g` error via
  `_at_least` (exit 2). Pick one policy per flag class.
- Interactive quit erases the report (`interactive.py:502-574` `finally:
  erase`): quitting the browse leaves a blank screen where `--static`
  leaves the report. Arguably correct terminal hygiene; inconsistent with
  the printout path either way.
- `tools/build_pyz.py:146-150`: `--portable` defaults `True`, so
  `--fast` yields `portable is True and fast is True`; harmless today (code
  branches only on `fast`) but explicit `--portable --fast` does not error
  and there is no negation. `default=None` + resolve, or a mutually
  exclusive group, would say what is meant.
- `render.py` `inverse()` (`:636-637`) returns text unchanged when disabled,
  while `supported()` docs call highlight "structure, not decoration".
  Latent only (`_browse.highlight` currently uses cursor glyphs), but a
  `NO_COLOR` browse degrades selection to no distinction.

---

## 8. Claims investigated and dropped (not bugs)

- *Partial cache refresh thrash* (`cli.py:1645` saving only `fresh`): invalid.
  `access.save()` merges with the file's existing verdicts internally
  (`access.py:268-303`), so saving the 1-entry gap yields 20 entries, not 1.
- *`ProbeBudget` race / missing global cap*: invalid. Budget is global
  (`fit.py:515,550-568`), shared per question (`:803`), and locked
  (`_lock` at `:555,573,580,584`). `MAX_PROBES_TOTAL` living in `fit.py`
  rather than `cli.py` is scoping, not absence.
- *`FORCE_COLOR` ladder / `TERM=dumb` ordering* (`render.py:497-521`):
  dropped as a bug. `1`/`true`→8, else→24 is the documented internal mapping
  (`0/4/8/24` are depths, not `FORCE_COLOR` levels); `FORCE_COLOR` levels
  beyond 0/1 are not standardised by force-color.org. `TERM=dumb`→0 before
  force is at worst a judgement call about undisplayable escapes.
- *Replay `UnicodeDecodeError` traceback* (`cli.py:4942`): overstated as a
  crash. `UnicodeDecodeError ⊂ UnicodeError ⊂ ValueError`, caught by the
  existing `except (OSError, ValueError, KeyError)`. Retained above only as
  the encoding asymmetry.
- *`snapshot` of a broken cluster exits 0*: intentional and documented
  (guard-exempt by design, `cli.py:210-226,4818-4827`; replay of the artifact
  exits 3). Not filed.
- *`backends` exits 0 with zero usable*: diagnostic command; arguably
  correct. Not filed.
