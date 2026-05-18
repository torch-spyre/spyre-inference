---
name: debug-spyre
description: Debug numerical, compilation, or correctness failures anywhere the Spyre backend is involved — custom ops, attention, model runner, end-to-end vLLM runs. Use whenever a user reports failing Spyre tests, accuracy/tolerance mismatches against a CPU reference, torch.compile errors on `spyre` device, silent CPU fallbacks, or generally "something on Spyre is broken." Most bugs here are not in our code — they are torch-spyre op gaps or dtype/layout limitations, so debugging requires tracing into the torch-spyre site-packages, not just reading `spyre_inference/`. For attention-specific hints (cache-seeding, KV alignment, MHA/MQA, broadcast diagnosis), see `attention-notes.md` next to this file.
---

# Debug Spyre Failures

`spyre_inference/` is a vLLM platform plugin that runs on IBM Spyre via the `torch-spyre` PyTorch backend. A lot of the code you'll see exists to *work around* gaps in `torch-spyre` — so when tests fail, the answer is rarely "our code is wrong": it's usually "torch-spyre behaves differently than we expected for some op, dtype, shape, or layout."

Use this skill whenever a Spyre-related test fails, produces large numerical errors, silently falls back to CPU, or triggers a compile error on `spyre`. Do not assume failures are local bugs — trace into `torch-spyre` first.

If the failing surface is the attention backend, the workflow below still applies; read `attention-notes.md` (sibling of this file) for attention-specific limitations and worked examples, but don't take "open the attention test file" as an instruction unless the user has actually pointed you at it.

## Before you start

### Bypass-permissions mode (strongly recommended)

This skill runs an autonomous hypothesis loop — many pytest invocations, many writes into `logs/<slug>/`, many `git checkout`s to revert experiments. Each new log filename, each new env-var combination, and each scratch-script execution is a separate permission prompt under path-specific rules; over a typical session that's dozens of interruptions, and the loop is the whole point.

The cleanest setup is to launch the session in bypass-permissions mode (a.k.a. "yolo mode") so the agent can iterate without prompting. Either:

- Start Claude Code with `claude --dangerously-skip-permissions`, or
- Toggle "Bypass permissions" in the harness's permission-mode picker before invoking this skill.

If you can't enable bypass mode (shared workstation, paranoid environment, untrusted branch), fall back to the explicit allowlist below. The allowlist gets you most of the way but will still prompt on commands you didn't anticipate — when that happens, either approve and continue, or pause the session, add the new pattern to `.claude/settings.json`, and resume.

### Permissions (fallback if bypass mode is off)

Debugging Spyre code runs a handful of commands repeatedly — pytest with env-var prefixes, `tee` into session log files, `grep` into those logs, `git checkout` to revert experiments, and `Write`/`Edit` on the HTML/artifacts directory. Pre-approve these in `.claude/settings.json` under `permissions.allow` so the loop isn't interrupted every time a new log file appears:

```json
"Bash(uv run pytest*)",
"Bash(uv run --no-sync pytest*)",
"Bash(uv run python .claude/skills/debug-spyre/logs/**)",
"Bash(uv run --no-sync python .claude/skills/debug-spyre/logs/**)",
"Bash(tee .claude/skills/debug-spyre/logs/**)",
"Bash(SPYRE_DEBUG_*=* uv run pytest*)",
"Bash(TORCH_SENDNN_LOG=* uv run pytest*)",
"Bash(DTLOG_LEVEL=* uv run pytest*)",
"Bash(DT_DEEPRT_VERBOSE=* uv run pytest*)",
"Bash(git checkout -- spyre_inference/**)",
"Bash(git checkout -- tests/**)",
"Bash(git diff --stat*)",
"Bash(git status)",
"Bash(grep * .claude/skills/debug-spyre/logs/**)",
"Bash(grep -A* * .claude/skills/debug-spyre/logs/**)",
"Bash(grep -B* * .claude/skills/debug-spyre/logs/**)",
"Write(.claude/skills/debug-spyre/logs/**)",
"Edit(.claude/skills/debug-spyre/logs/**)"
```

If you don't set these first, expect ~10–15 permission prompts during a typical investigation — each new log filename is a new approval under path-specific rules.

### What you're allowed to do without asking

Local experiments are fair game:

- Run any pytest selection under `tests/`.
- Write scratch scripts and artifacts under `.claude/skills/debug-spyre/logs/<slug>/`.
- Edit source files in this repo (`spyre_inference/`, `tests/`) to prototype and test a fix — but revert before marking the session `resolved` or `blocked` unless the user has explicitly approved the patch.

The following still require human confirmation:

- `git commit`, `git push`, creating/merging PRs.
- Editing anything under `.venv/` (torch-spyre site-packages) — those changes are invisible to version control and wiped on the next `uv sync`. If you need to instrument torch-spyre, write a standalone repro script instead.
- Installing, upgrading, or downgrading dependencies, or changing `pyproject.toml` / `uv.lock`.

### Hardware constraint: Spyre is a single accelerator

**Never run two Spyre-backed commands at once.** This includes:

- Launching multiple `uv run pytest` invocations in parallel tool calls.
- Using `run_in_background=true` on a Spyre test and then starting another before it finishes.
- `pytest -n` / `xdist` — always run serially.

The card is contested by a single process at a time; parallel invocations either hang, produce undefined device state, or corrupt the compile cache in ways that make subsequent runs non-reproducible. Any pytest command that touches Spyre is an exclusive critical section. If you need both a targeted probe and a full matrix run, do them sequentially.

### Iterating on a locally-vendored torch-spyre: always pass `--no-sync`

When you are editing the local `torch-spyre/` checkout (in this repo or any sibling path) and reinstalling it into the spyre-inference venv via `uv pip install /path/to/torch-spyre`, the next plain `uv run pytest …` will silently undo your work. `uv run` re-syncs deps from `pyproject.toml` on every invocation, and `pyproject.toml` pins torch-spyre to the upstream git rev — so each `uv run` reinstalls the upstream commit on top of your local-source install. Symptom: you `uv pip install` the local edit, confirm the installed file has your changes, run the test, and it still fails as if nothing changed. (Look for `Uninstalled 1 package … Installed 1 package …` in the front of pytest output — that's uv quietly reverting your install.)

**Always use `uv run --no-sync …`** for any iteration cycle that depends on a hand-installed local dependency:

```bash
# After: uv pip install --no-deps --force-reinstall ./torch-spyre
uv run --no-sync pytest -m "not upstream" path/to/test.py        # YES — keeps your local install
uv run pytest -m "not upstream" path/to/test.py                  # NO  — silently restores upstream
uv run --no-sync python .claude/skills/debug-spyre/logs/<slug>/repro.py
```

A few related gotchas worth knowing:

- `uv pip install` from a path also caches the built wheel. If a fresh edit doesn't seem to take effect, `uv cache clean torch-spyre` and reinstall with `--no-cache --no-build-isolation`.
- The torch-spyre wheel build is C++-heavy and takes ~50 s. Plan your iteration loop accordingly — batch up source changes before each rebuild rather than one-edit-one-rebuild.
- Verify the install actually landed before running the test: `grep -n <new-symbol-name> /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/torch_spyre/.../<file>.py`. If the symbol you just added isn't there, the install didn't take and `--no-sync` won't help yet — fix the install first.

## The debug loop: run it autonomously, log everything to HTML

This skill is meant to be run without stopping to ask the human which hypothesis to try next. The human watches by refreshing an HTML debug log that you maintain continuously. **Do not** ask "which should I try first?" — enumerate all candidate hypotheses or fixes into the log, then work them one by one and record the outcome of each before moving on.

### Scope of one invocation: a cluster of related failures

Each invocation of this skill investigates **one cluster of related failures**, not a single failing test. A cluster is a set of failing parametrizations that look like they share a root cause — same symptom shape, overlapping traceback/warnings, same stage of the pipeline. Full test runs surface multiple failures; you group the ones that smell alike and debug them together.

Before doing anything else, build the cluster:

1. Run the failing test surface once (or read a recent run the human points you at). Use the narrowest selector that still includes every failure you've been told about — a single test file, a marker, or a path. Tee to the session directory:

   ```bash
   uv run pytest <selector> -m "not upstream" --tb=short -q 2>&1 \
     | tee .claude/skills/debug-spyre/logs/<slug>/full-matrix.txt
   ```

2. Read every failure. Sort them by symptom:
   - near-100 % mismatch with large abs diff → likely one cluster (structural bug),
   - small-ratio mismatch with small abs diff → likely a different cluster (fp16 accumulation),
   - exceptions/compile errors → each error class is usually its own cluster.
   Affected parametrizations that differ only in shape-like axes (block counts, head sizes, sequence lengths) are almost always the same bug surfaced under different shapes — group them.
3. Name the cluster with a slug describing the shared symptom, not one representative test (e.g. `<feature>-<symptom>` like `rmsnorm-fp16-overflow`, `embedding-fallback-regression`).
4. In the HTML log header, list **every** failing test node id the cluster covers. The "failing tests" field is plural on purpose.
5. Pick the **narrowest failure in the cluster as the baseline probe** — the one with smallest shapes / shortest runtime / simplest parametrization. That's the single test you run in Step 0.5 and iterate on. You're not ignoring the others; you're using the cheapest member as a fast oracle for the whole cluster.

If two symptom classes are both failing, they get **two separate invocations of the skill, each with its own HTML log**. Don't try to debug unrelated clusters inside one log — the hypothesis queue and the narrative get tangled.

Before you mark the cluster `resolved`, you must validate the fix against **every** test in the cluster (see Step 4 below). A fix that clears the probe but leaves another cluster member failing means either (a) your cluster was too broad and one of the members belongs in a different cluster, or (b) the fix is incomplete. Log which it is and re-plan.

### Step 0: create the debug log (first action of every invocation)

Before any other investigation, create an HTML file so the human has something to refresh from the start:

- Directory: `.claude/skills/debug-spyre/logs/` (create if missing).
- Filename: `debug-YYYYMMDD-HHMMSS-<slug>.html` where `<slug>` names the **cluster of failures**, not a single test. Generate the timestamp from `date +%Y%m%d-%H%M%S`.
- Per-invocation sibling directory: `.claude/skills/debug-spyre/logs/<same-stem>/` for baseline output, standalone repros, and any other artifacts produced during the session. All session artifacts go here — never `/tmp/`.
- The filename is **unique per invocation** (do not reuse an earlier session's log). **Within the invocation, overwrite the same HTML file in place** as you add updates — that's the point, so the human can refresh.
- Print the absolute path of the log to the human on the first turn, then again whenever you add a significant update, so they can open/refresh it.

The cluster-building step above may populate `logs/<slug>/full-matrix.txt` before the HTML exists yet — that's fine. Create the HTML immediately after you've settled on a slug, and link `full-matrix.txt` from the log's "Key findings / artifacts" section.

Structure the HTML so it is scannable while you are still working:

1. **Header** — timestamp, cluster slug + one-line symptom summary, **list of every failing test node id in the cluster** (mark the one you're using as the baseline probe), current status (`investigating` / `blocked` / `resolved`).
2. **Session summary** — a few sentences, updated as understanding evolves. What is the bug in plain English? What is the current best theory? What has been ruled out?
3. **Hypothesis / fix queue** — a checklist table. Columns: `#`, `hypothesis or fix`, `how to test`, `status` (`queued` / `in-progress` / `confirmed` / `ruled out` / `inconclusive`), `outcome`, `evidence / artifacts`. Every new idea lands here as `queued` before you test it — no silent investigation.
4. **Experiment log** — reverse-chronological, most recent first. Each entry has: timestamp, hypothesis # being tested, exact command(s) run, tail of relevant output (keep it short — link or reference a file for the full log), conclusion.
5. **Cluster validation** — a small table with one row per test node id in the cluster, columns: `node id`, `pre-fix status`, `post-fix status`, `notes`. Populated at the end (Step 4) before marking `resolved`.
6. **Key findings / artifacts** — files created or modified, minimal repros, references into `torch-spyre` internals.
7. **Next actions** — explicit, ordered. Mirrors the `queued` rows from the hypothesis table.

Keep the HTML self-contained (inline CSS, no external requests) and safe for re-save — overwrite the same file as the investigation progresses. A good baseline: a `<style>` block that color-codes statuses (green = confirmed/resolved, red = ruled out, amber = in-progress, grey = queued), monospace `<pre>` blocks for command output, and anchor links between sections.

### Step 0.5: capture a baseline of the failure

You can't form good hypotheses without seeing the current output. Before brainstorming:

1. Read the failing test, the module under test, and the relevant torch-spyre surface (see "Orient yourself" below). If the module's docstring or top-of-file comments mention torch-spyre limitations being worked around, those are load-bearing — read them carefully.
2. Run the **baseline probe** (the narrowest failure in the cluster you picked above) and tee its output:

   ```bash
   uv run pytest -m "not upstream" \
     '<baseline-probe node id>' --tb=long 2>&1 \
     | tee .claude/skills/debug-spyre/logs/<slug>/baseline.txt
   ```

3. Paste the load-bearing bits into the HTML log under "Session summary":
   - the `Mismatched elements` line and `Greatest absolute/relative difference`,
   - every `FallbackWarning` (file:line + the op name),
   - the innermost traceback frame if there was an error.

4. Sanity-check that the probe's symptom actually matches the rest of the cluster — compare its error signature against the symptom summaries you wrote down while building the cluster. If the probe looks qualitatively different from the others, you either picked the wrong probe or the cluster is wrong: re-split and update the log header's node-id list before continuing.

That's the evidence you reason from. Don't skip it even if you have a strong prior — the baseline often contradicts the prior.

### Step 1: populate the hypothesis queue before testing anything

Now brainstorm **every plausible hypothesis you can name** and write them all into the queue. Include the obvious ones even if you are fairly confident in one — surfacing the alternatives is a core feature, not noise. Typical starter set when something on Spyre is wrong:

- A `FallbackWarning` op behaves differently on CPU than expected (dtype/layout mismatch on the round-trip, or accumulation order differs from the on-device kernel that *would* have run).
- An op the code thinks is on-device is silently dispatching to CPU — check `eager.py` vs `fallbacks.py` for the specific overload (e.g. `aten.mm.out` is different from `aten.mm`).
- Dtype gets coerced somewhere unexpected (Spyre is fp16-only; an op that produces int64 indices, fp32 intermediates, or bool masks may force a fallback or silently clobber).
- Layout / contiguous expectation broken — torch-spyre rejects in-device transpose+contiguous combinations in some places.
- A shape that escapes whatever bucket the module pre-aligns to (alignment constants exist precisely to avoid recompilation; an unbucketed shape may dispatch to an untested kernel path).
- fp16 accumulation order difference (Spyre matmul vs CPU) — only when mismatch is small and looks like noise.
- Test fixture bug (wrong inputs, wrong reference computation), not backend bug.
- Compile-cache or device-warmup carryover from a previous test in the same worker (results genuinely differ between isolated and full-matrix runs — see the gotcha below).

If the failing surface is attention specifically, also consult `attention-notes.md` for the standard attention-specific hypotheses (cache seeding, mask shape, scatter/gather selector order, broadcast/dispatch bugs in 4D matmul) and add the relevant ones to the queue.

**Ranking**: attack any high-confidence hypothesis whose experiment runs in ≤1 minute first. Defer anything that needs a full test run (~3 minutes due to vLLM startup) or a code change. If two hypotheses are equally cheap, pick the one most directly supported by the baseline output.

### Step 2: run hypotheses one at a time, record outcomes

For each `queued` row:

1. Move it to `in-progress`, write the command you are about to run into the experiment log.
2. Execute the experiment (usually: a targeted test invocation, a small standalone repro script, or a diff of intermediate tensor values).
3. Write the outcome back to the hypothesis row (`confirmed`, `ruled out`, or `inconclusive`) with a one-sentence conclusion and a short evidence excerpt (≤10 lines). If the full output is worth keeping, write it to a sibling file under `logs/` and link from the HTML.
4. If the experiment surfaces a new hypothesis, append it to the queue (do not derail into it immediately — finish the current one first unless the new one *invalidates* the current experiment).
5. Update the session summary to reflect new understanding.
6. Move to the next `queued` row.

Stop the hypothesis loop only when one of these holds:

- You have a fix that clears the baseline probe → proceed to Step 4 (cluster validation). Do not mark the cluster `resolved` yet.
- You have identified the bug but a fix needs human judgment (design choice, scope question) → document a concrete next action in the log and set status to `blocked`.
- Every hypothesis in the queue has been tested and none explain the failure → log this explicitly and propose the next tier of experiments (instrument torch-spyre itself, escalate) before pausing.

**Never ask the human "which of these should I try?"** If there are multiple viable hypotheses, they all go in the queue, and you work them in order. The human reads the log at their own pace and can redirect you if they want — but they should not be *required* to redirect for you to make progress.

(Autonomy boundaries are listed under "Before you start: What you're allowed to do without asking" at the top of this doc.)

### Step 3: keep the log current

- Update the HTML after each experiment concludes, not at the end of the session.
- Timestamp every update so the human can see freshness.

### Step 4: validate the fix across the whole cluster before marking resolved

A fix that clears the baseline probe is a strong signal, not a finish line. Before the log can move to `resolved`:

1. Run **every** failing test node id from the cluster (not just the probe) with the fix in place:

   ```bash
   uv run pytest -m "not upstream" \
     '<node id 1>' '<node id 2>' '<node id 3>' ... --tb=short 2>&1 \
     | tee .claude/skills/debug-spyre/logs/<slug>/post-fix.txt
   ```

2. Populate the "Cluster validation" table in the HTML: for each node id, record pre-fix status (from baseline), post-fix status, and a short note.

3. **If the fix touched shared code, re-run the broader matrix the cluster came from** — not just the cluster. A reshape/expand/dispatch tweak that clears the cluster probe frequently shifts fp16 accumulation noise on other shapes enough to push them over tolerance. Run the same selector you used to build the cluster:

   ```bash
   uv run pytest <same selector> -m "not upstream" --tb=short -q 2>&1 \
     | tee .claude/skills/debug-spyre/logs/<slug>/post-fix-matrix.txt
   ```

   Compare pass/fail counts against the baseline matrix. Any regression — even by tiny tolerance margins — is a hard blocker for `resolved`.

4. Interpret the outcome:

   - **All cluster members pass AND no regressions in the broader matrix** → set the header status to `resolved`. Write a short retrospective: what turned out to be the bug, which hypotheses were wasted effort, what signal would have pointed at the right answer faster.
   - **Probe passes but another cluster member still fails** → the cluster was too broad *or* the fix is incomplete. In the retrospective, state which: either (a) carve the still-failing test(s) out into a new cluster, start a fresh invocation of this skill for them, and link the new log from this one, or (b) return to Step 1 with a new hypothesis explaining the remaining failure, keeping this log `investigating`.
   - **Fix regresses a previously passing test** → status goes back to `investigating`, add a hypothesis for the regression, continue the loop. Do not ship a fix that trades cluster members for non-cluster failures, even when the net count is better.

5. Only after the validation table is populated and the outcome is logged is it acceptable to set header status to `resolved` or `blocked`.

## Orient yourself before touching code

1. **Identify the failing surface.** Which test file? Which module under `spyre_inference/`? Which custom op? Read the test, the module under test, and any CPU reference the test compares against.
2. **Read the module's docstring and top-of-file comments.** In this repo, modules that wrap torch-spyre often enumerate the limitations they route around (no advanced indexing, no in-device transpose+contiguous, no simultaneous dtype+device conversion, alignment constants, etc.). That list tells you which lines exist for *correctness* vs which exist for *performance* — and the former are landmines if you "simplify" them.
3. **Re-read `CLAUDE.md`** — Spyre-specific constraints (head_size%64, fp16-only, TP=1, eager mode in the platform layer) are documented there.
4. **Look at the relevant torch-spyre site-packages files** — see "Tracing into torch-spyre" below. The site-packages copy is the source of truth for what is and isn't on-device.

## Reproducing a single failure

Full test runs are slow (~3 minutes) because of vLLM startup. Use single-test invocations. **Always pass `-m "not upstream"`** so a broad selector doesn't accidentally match an upstream vLLM test pulled in by the testing plugin.

Parametrize IDs frequently contain `(`, `=`, and `,`, which `-k` cannot parse (pytest evaluates `-k` as a Python-ish expression, and those characters produce syntax errors or silent zero matches). Two options:

1. **Run by full node id (recommended for reproducibility).** List node ids first, then copy-paste:

   ```bash
   uv run pytest <test-file> -m "not upstream" --collect-only -q | grep <test-name>
   # then, single-quoting so the shell does not expand the brackets:
   uv run pytest -m "not upstream" \
     '<full node id>' \
     --tb=long
   ```

2. **Use a plain-letter substring with `-k` (quick but coarse).** Avoid `(`, `=`, `,`:

   ```bash
   uv run pytest <test-file> -m "not upstream" -k "<plain words>" -x --tb=long
   ```

Add `-s` if you need `print()` output. Warnings are suppressed after the first occurrence — to turn `FallbackWarning` into an exception use the specific filter (not `-W error::UserWarning`, which also catches unrelated warnings like flex_attention's and sends you chasing a red herring):

```bash
uv run pytest -m "not upstream" '<node id>' \
  -W "error::torch_spyre.ops.fallbacks.FallbackWarning" --tb=long
```

## The single most important signal: FallbackWarning

`torch-spyre` silently routes unsupported ops to CPU and emits a `FallbackWarning`. Example shape:

```shell
spyre_inference/.../<file>.py:<line>: FallbackWarning:
    torch.ops.spyre.<op> is falling back to cpu
    <the line that triggered it>
```

A fallback is not always wrong, but it:

- Changes the numerical path (CPU vs Spyre matmul kernels differ in precision & accumulation order for fp16).
- Incurs extra D2H/H2D transfers (sometimes these silently clobber dtype or layout).
- Can mask a real bug: the op that fell back may not match the op you thought you called.

Turn the warning into an error using the specific `FallbackWarning` filter shown in the "Reproducing" section above — it gives you a full traceback pointing at the exact line that triggered the fallback.

## Tracing into torch-spyre

`torch-spyre` is installed from GitHub (see `pyproject.toml` `tool.uv.sources`), so the site-packages copy is the source of truth. Key files:

- `.venv/lib/python3.12/site-packages/torch_spyre/ops/eager.py`
    - Lists every aten op registered as a Spyre kernel via `register_torch_compile_kernel([...])`. If an op is **not** in that list, it either has a fallback or is missing entirely. Currently covered: `mm`, `bmm`, `cat`, `add`, `mul`, `div`, `exp`, `log`, `_softmax`, `sum`, `sqrt`, `rsqrt`, `sigmoid`, `relu`, `tanh`, `sub`, `addmm`, `eq`, `ge`, `gt`, `lt`, `maximum`, `pow`, `linalg_vector_norm`, and a few others. Anything else goes through fallbacks.
- `.venv/lib/python3.12/site-packages/torch_spyre/ops/fallbacks.py`
    - Every op that moves to CPU and back. Notable: `arange`, `sin`, `cos`, `embedding`, `tril`, `triu`, `bitwise_or`/`xor`, `argmax`, `cumsum`, `repeat.out`, `isin`, `max_dim_int64_fallback`, `max_default_int64_fallback`. Read the top-of-file comment block — it documents the *process* for adding new fallbacks.
- `.venv/lib/python3.12/site-packages/torch_spyre/_monkey_patch.py`
    - Patches `torch.Tensor.to`, `torch.empty`, and `__repr__` for Spyre. Explains why `.to(device="spyre")` can behave oddly when combined with dtype changes (the patch only kicks in when `device_layout` is provided).
- `.venv/lib/python3.12/site-packages/torch_spyre/_inductor/` — decompositions, lowerings, and custom ops. If a compile error blames an inductor pass, this is the place.

Grep patterns that usually get you to the right file fast:

```bash
# Is <op> registered as a compiled Spyre kernel?
grep -n "aten\.<op>\b" .venv/lib/python3.12/site-packages/torch_spyre/ops/eager.py

# Is <op> registered as a CPU fallback?
grep -n "aten\.<op>\b" .venv/lib/python3.12/site-packages/torch_spyre/ops/fallbacks.py

# Any decomposition / lowering / constant involving it?
grep -rn "<op>" .venv/lib/python3.12/site-packages/torch_spyre/_inductor/ | head
```

## Known torch-spyre limitations that shape this codebase

When you see odd code in this repo, it usually exists because of one of these. Before "fixing" it, confirm the limitation still applies — torch-spyre is under active development, and a limitation may have been lifted.

| Limitation | How code in this repo works around it |
|---|---|
| No advanced/fancy indexing on device | Build one-hot row/col selectors on CPU, use `bmm` + mask blend instead of `cache[idx] = values` |
| No in-device transpose + contiguous | Compute permutations on CPU where possible, or interleave `transpose + contiguous` against ops that tolerate non-contiguous inputs |
| No simultaneous dtype + device conversion | `custom_ops/utils.py::convert` does the dtype change on CPU first, then moves to Spyre |
| `torch.compile` recompiles on shape change | Modules pre-align inputs to fixed bucket sizes (e.g. KV-length alignment, query-chunk size) so the same compiled kernel is reused |
| No tensor parallelism | All custom layers assume `TP=1` |
| Only `float16` is supported | `TorchSpyrePlatform` forces dtype; custom ops list `float16` only as `supported_dtypes` |

For attention-backend-specific limitations (head_size constraints, MHA/MQA compile gaps, KV alignment / query chunk numbers), see `attention-notes.md`.

> **Gotcha — Spyre results vary across test selections.** The same failing test can show very different numbers when run alone vs. as part of a larger matrix (seen in practice: abs diff 2.35 in matrix run, abs diff 63616 in isolated run of the same node id, same seed, same git rev). Spyre's compile cache and device warmup state carry across tests in a worker. Do not compare magnitudes across selection scopes — re-run with the same scope you were debugging against before calling something a regression.

## Debugging workflow: numerical mismatch

When a test fails with `Tensor-likes are not close`:

1. **Gauge the severity.** Look at `Mismatched elements` and `Greatest absolute difference`. ~100 % mismatch with abs diff > 1.0 almost always means a structural bug (wrong inputs, wrong reshape, wrong index, wrong dispatch). ~5–30 % mismatch with small diffs usually means fp16 accumulation order differences — loosen tolerances only after ruling out structural issues.

2. **Bisect the pipeline.** Identify the clean stages of the failing operation (most modules in this repo have an obvious 3–6 stage shape: prep inputs → device transfer → compute → reshape/slice → return). Run each stage's inputs and outputs on CPU against a reference and diff. The first stage where the diff goes nonzero is the bug site. A diff that only appears after the compute stage points at a kernel/dispatch problem in torch-spyre; a diff before that points at our own logic.

3. **Compare Spyre output to CPU output for the same kernel.** The cleanest way is a standalone repro script under `logs/<slug>/repro_cpu.py` that constructs the same module under test and runs it with `torch.device("cpu")` instead of `spyre`. If your module reads its target device from a field (e.g. `_target_device`), monkeypatch that field rather than the global default. Run the same inputs the failing parametrization uses. If the CPU variant passes, the bug is in torch-spyre's realization of the operation; if it still fails on CPU, the bug is in our own logic. (Note: pytest fixtures like `requires_spyre` will skip the real test on CPU-only hosts, so do this in a standalone script, not by editing the device and rerunning pytest.)

4. **Inspect any non-data tensors.** Masks, indices, one-hot selectors, padding, alignment buffers — anything where a single bit-flip silently changes meaning. These are the most common silent-disaster bugs in torch-spyre workarounds. Print them for a small case and verify by hand.

5. **When outputs blow up to near-fp16-max (~60000+), suspect a broadcast/dispatch bug.** Print magnitudes along each suspect axis and look for **periodic patterns**: if every Nth row/column is huge and the rest are sane, that's near-certain evidence of a broadcast bug along that axis in either a matmul or softmax kernel. A workaround is usually `.expand(-1, N, -1, -1).contiguous()` on the singleton dim of the suspect input — which is **diagnostic** (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix. See `attention-notes.md` for a worked code snippet.

## Debugging workflow: compile / runtime error on Spyre

1. Capture the full traceback. Errors from torch-spyre usually bubble up through inductor — the frame of interest is the one inside `.venv/lib/python3.12/site-packages/torch_spyre/_inductor/`.
2. Before calling the suspicious op, run it on a **tiny** input (e.g. shape `(1, 1, 1)`) on Spyre in isolation. If that also fails, you have a minimal repro you can hand to the torch-spyre team.
3. Enable verbose torch-spyre logging for one run:

   ```bash
   TORCH_SENDNN_LOG=INFO DT_DEEPRT_VERBOSE=1 DTLOG_LEVEL=info \
     uv run pytest -m "not upstream" '<node id>' -s 2>&1 \
     | tee .claude/skills/debug-spyre/logs/<slug>/verbose.log
   ```

   (See `torch_spyre/__init__.py::_autoload` for the env var defaults — these are set to quiet levels unless overridden.)

4. If the error is `falling back to cpu` on an op we thought was implemented, the dtype or shape may be off. `torch-spyre` registers kernels per-overload: `aten.mm`, `aten.mm.out`, `aten.mm.Scalar` are different ops. Check `op.overloads()` and confirm the overload you actually hit is the one registered.

## Debugging workflow: "compiles and runs, but is slow or falls back"

- Every fallback is a D2H + CPU-compute + H2D roundtrip. One fallback in a hot loop destroys performance.
- Grep the repo for `FallbackWarning` references — some are filtered as known-accepted. If you see a new one, decide: route around it (preferred), or request a torch-spyre kernel.
- Compile-mode gotcha: vLLM's `CompilationConfig` distinguishes between `mode = CompilationMode.NONE` (i.e. `0`) and `mode = None` (Python `None`). Modules that gate `torch.compile` on `cfg.mode == CompilationMode.NONE` will *still wrap* their kernels with `torch.compile` when `cfg.mode is None`, which is what happens under the pytest `default_vllm_config` fixture. So "compile is off in tests" is not always true — confirm before assuming eager when comparing pytest vs standalone-script behavior.

## When to stop debugging locally and escalate

Escalate to the torch-spyre team (open an issue / file it in the shared tracker) when:

- You have a minimal repro that runs an op on Spyre and produces either wrong output or an unhelpful error.
- The op is listed as registered in `torch_spyre/ops/eager.py` but still falls back.
- A compile error references internal inductor files (`torch_spyre/_inductor/*.py`) with no obvious misuse on our side.
- An overload that *should* be supported (per `op.overloads()`) is not.

Before escalating, always re-run the same scenario on CPU to confirm the bug is Spyre-specific, and capture:

1. The failing test node id.
2. The full traceback.
3. torch-spyre identification — `torch_spyre.__version__` is currently pinned at `"0.0.1"` and is **not** a useful identifier on its own. Capture the actual source revision:

   ```bash
   # Preferred: exact git revision recorded by uv
   grep -A2 '"torch-spyre"' uv.lock | head
   # Or: the wheel's recorded source
   uv run pip show torch-spyre | grep -iE "version|home-?page|location"
   ```

   Include the revision hash in the escalation.
4. The torch version (`uv run python -c "import torch; print(torch.__version__)"`).
5. A repro, in this order of preference — always prefer the minimal one that reliably triggers the bug:
   - **First choice — standalone script** (no pytest, no vLLM, no spyre-inference). A ≤20-line file at `logs/<slug>/repro.py` that only imports `torch` and `torch_spyre` and exercises the failing op. This is what the torch-spyre team can run fastest.
   - **Second choice — standalone script with vLLM config context** (`set_current_vllm_config` + `set_forward_context`). Use when the bug requires the vLLM compile/backend wiring but does not require the full module-under-test pipeline.
   - **Third choice — self-contained pytest file that uses the `default_vllm_config` fixture and drives the failing module directly.** Some bugs only fire under this full pipeline and vanish in a plain `torch.compile`'d script with identical shapes and inputs. When that's the case, a pytest repro is acceptable — note the failure in the escalation so the torch-spyre team knows minimization further is a task for them. If you try and confirm a smaller repro does *not* reproduce, capture that negative result in the HTML log as evidence.

When you escalate, set the debug log header status to `blocked`, link the repro and artifacts from the log, and summarize which hypotheses were ruled out — the log becomes the handoff document.

## Quick reference: files to open

- The failing module under `spyre_inference/` and the test that exercises it (start here, identified per-session).
- `spyre_inference/custom_ops/utils.py::convert` — the canonical dtype+device dance.
- `spyre_inference/platform.py` — platform constraints (dtype, compile mode).
- `tests/plugin/spyre_testing_plugin/pytest_plugin.py` — `default_vllm_config` fixture.
- `.venv/lib/python3.12/site-packages/torch_spyre/ops/eager.py` — compiled kernels.
- `.venv/lib/python3.12/site-packages/torch_spyre/ops/fallbacks.py` — CPU fallbacks.
- `.venv/lib/python3.12/site-packages/torch_spyre/_monkey_patch.py` — Tensor patches.
- `.venv/lib/python3.12/site-packages/torch_spyre/_inductor/` — compiler internals.
- `attention-notes.md` (sibling of this file) — attention-specific limitations and worked examples, when the failing surface is the attention backend.
