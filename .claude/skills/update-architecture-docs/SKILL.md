# Update Architecture Docs

Revise the architecture documentation and D2 diagrams in `docs/architecture/` to reflect
the current state of the codebase. This skill produces two D2 diagrams (system overview +
component view) and a markdown page that explains the plugin architecture.

## Source Files to Inspect

Discover the current source layout rather than relying on a hardcoded list — files get
added, renamed, and removed over time. Run:

```bash
find spyre_inference -type f -name "*.py" | sort
```

Then read every file under these areas to understand the current architecture:

- **Plugin registration & platform** — top-level `spyre_inference/` (entry point
  `__init__.py`, `platform.py`).
- **Worker & model runner** — everything under `spyre_inference/v1/worker/`.
- **Attention backend** — everything under `spyre_inference/v1/attention/`.
- **Custom ops (OOT layer replacements)** — every `.py` file in
  `spyre_inference/custom_ops/`. Start with `__init__.py` to see the registration list,
  then read each op module that it references.
- **Scheduler and other v1 core** — everything under `spyre_inference/v1/core/` (may be
  empty if no overrides are present).

When you read these, note the public class names — the diagrams label nodes with them, so
any rename needs to propagate to the `.d2` files.

## vLLM Base Classes to Cross-Reference

The system overview diagram shows vLLM's own class hierarchy alongside the Spyre overrides.
Read these vLLM source files (in `.venv/lib/python3.12/site-packages/vllm/`) to verify
base class names, inheritance relationships, and any new extension points:

**Platform:**

- `platforms/interface.py` — `Platform` base class
- `platforms/cpu.py` — `CpuPlatform` (parent of `TorchSpyrePlatform`)

**Worker & model runner:**

- `v1/worker/cpu_worker.py` — `CPUWorker` base class
- `v1/worker/gpu_model_runner.py` — `GPUModelRunner` (pattern reference for model runners)

**Engine & scheduling:**

- `v1/engine/core.py` — `EngineCore`
- `v1/engine/async_llm.py` — `AsyncLLM`
- `v1/core/kv_cache_manager.py` — `KVCacheManager`
- `v1/core/scheduler.py` — vLLM's `Scheduler`

**Attention:**

- `v1/attention/backend.py` — `AttentionBackend` base class

**API:**

- `entrypoints/openai/api_server.py` — `APIServer` entry point

Check these for:

- Renamed classes or moved modules (vLLM refactors frequently)
- New base class methods that spyre-inference should override
- New components in the engine/worker pipeline (e.g., new coordinator classes)

Note: The vLLM path may change if the Python version or venv location changes. Use
`python -c "import vllm; print(vllm.__path__[0])"` to find the current install path.

## Output Files

Update these three files in `docs/architecture/`:

1. **`system-overview.d2`** — Process-level view: API server → EngineCore → Worker → Platform.
   Shows class hierarchy with inheritance (dashed `▷` arrows) and composition (solid arrows).

2. **`plugin-architecture.d2`** — Component view of a Granite decoder layer showing each
   custom op replacement, the attention pipeline detail, and model bookends (embedding + LM head).

3. **`index.md`** — Explanatory page referencing both SVGs. Contains tables for custom op
   replacements, attention pipeline steps, and key constraints.

## D2 Diagram Style Guide

Follow these conventions exactly:

**Container (subgraph) styling — light background for both light/dark mode:**

```d2
container: "Label" {
  style.fill: "#e0e0e0"
  style.stroke: "#9e9e9e"
  style.font-color: "#212121"
}
```

**Node color scheme:**

- Spyre-native (runs on Spyre device): `fill: "#1565c0"`, `stroke: "#0d47a1"`, `font-color: "#ffffff"`
- CPU fallback: `fill: "#e65100"`, `stroke: "#bf360c"`, `font-color: "#ffffff"`
- Mixed CPU + Spyre: `fill: "#2e7d32"`, `stroke: "#1b5e20"`, `font-color: "#ffffff"`
- vLLM base class (not overridden): `fill: "#37474f"`, `stroke: "#263238"`, `font-color: "#ffffff"`
- Model (gold): `fill: "#827717"`, `stroke: "#558b2f"`, `font-color: "#ffffff"`
- Spyre plugin classes get `style.bold: true`

**Relationship arrows:**

- Inheritance: `A -> B: "▷" {style.stroke-dash: 3}`
- Composition/has-a: solid arrow with label
- Data flow: solid arrow with label describing what flows

**Plugin-architecture.d2 specific:**

- Include a legend container with `direction: right` showing the three device colors
- Attention detail panel connects from the ATTN node via `{style.stroke-dash: 3}`

## Rendering

After updating the `.d2` files, render them to SVG:

```bash
d2 --pad 50 docs/architecture/system-overview.d2 docs/architecture/system-overview.svg
d2 --pad 50 docs/architecture/plugin-architecture.d2 docs/architecture/plugin-architecture.svg
```

If `d2` is not installed, install it once to `~/.local/bin` without sudo:

```bash
curl -fsSL https://d2lang.com/install.sh | sh -s -- --prefix="$HOME/.local"
```

## Layout debugging

D2 uses ELK (or dagre) for auto-layout. The result is opinionated and not always what
you expect, especially with uneven container sizes and cross-container arrows. **Always
render after every change and inspect the result before declaring the diagram done.**

### Inspecting positions without opening the SVG

Each shape ends up as `<rect x="..." y="..." width="..." height="...">` in the rendered
SVG, prefixed with `class="<base64-of-path>"`. Grep the SVG to confirm where things
landed:

```bash
# Canvas size
grep -oE 'viewBox="[^"]+"' docs/architecture/system-overview.svg | head -1

# All rect positions inside the worker container
grep -oE 'class="d29[^"]*"><g class="shape" ><rect x="[0-9.]+" y="[0-9.]+"' \
  docs/architecture/system-overview.svg
```

(`d29ya2Vy` is base64 for `worker`. Decode any class name with
`echo "<class>" | base64 -d`.)

### Common problems and fixes

**Long diagonal arrows cutting across the page.** Cross-container arrows pull the target
node toward whatever side ELK thinks is closest to the source. If a target like
`worker.CPUWorker` lands far from where `engine.EngineCore` enters the worker container,
the arrow has to traverse the whole box. The fix is usually to let ELK's
connection-based placement do its job naturally: **remove forced directions** (e.g.
`direction: right` on a container) and let the default flow position incoming-arrow
targets near the edge closest to their source. The 2×2 grid with default direction in
each cell tends to put `worker.CPUWorker` at the top-right (closest to the engine cell
above it) and `worker.SpyreComm` at the bottom-left (closest to the platform cell to its
left), giving short clean arrows on both sides.

**Dead space above small containers in `direction: down`.** A vertical stack with one
tall container (worker) and several short ones (api, engine, platform) leaves the small
ones floating left with nothing on their right. Force a 2×2 grid by putting BOTH
`grid-rows: 2` and `grid-columns: 2` at the top level. Order children
`api, engine, platform, worker` so the layout is:

```text
[api ]  [engine]
[plat]  [worker]
```

`grid-columns: 2` alone is not enough — D2 will pack column-major when one cell is much
taller than the others, putting the tall worker in its own full column.

**Reserved keywords as identifiers.** `top` (and other layout keywords) cannot be used
as a shape name. The compiler error reads `"top" must be the last part of the key`. Pick
a descriptive name instead (e.g. `control_layer`).

**`near` keyword limits.** `near` is more restricted than it looks:

- Cannot reference shapes inside grid cells (descendants of grid containers).
- Only accepts an absolute shape path OR one of `top-left`, `top-center`, `top-right`,
  `center-left`, `center-right`, `bottom-left`, `bottom-center`, `bottom-right`.
- Compound forms like `near: shape-top-left` are NOT supported.

If you reach for `near`, the layout is probably already fighting you — try restructuring
connections or container direction first.

**Transparent sub-containers.** You can group nodes with a wrapper to force them to
cluster, using `style.fill: transparent` AND `style.stroke: transparent` to hide the
wrapper itself. This adds visual depth (extra nesting) and tends to grow the canvas — try
the layout without it first. When you do use it, all references from outside need the
new path (e.g. `worker.CPUWorker` → `worker.bootstrap.CPUWorker`).

### Iteration loop

1. Make a small structural change (reorder, add/remove direction, swap connection).
2. Re-render.
3. Grep the SVG for the canvas size and key node positions.
4. Decide whether the change helped, then repeat.

Don't fight the layout engine. If three attempts haven't fixed an arrow path, the right
move is usually to **remove** a constraint (a forced direction, a sub-container, an
explicit `near`), not add one.

## What to Look For When Updating

Compare the current code against the diagrams and docs:

1. **New or removed custom ops** — check `custom_ops/__init__.py` for the registration list.
   Each `@ClassName.register_oot()` decorator defines a replacement. Update both the
   component diagram and the "Custom Op Replacement" table.

2. **Class renames or new classes** — check worker, model runner, platform, and attention
   files for any new classes or renamed ones. Update the system overview diagram.

3. **Attention pipeline changes** — read `SpyreAttentionImpl.forward()` for the step-by-step
   pipeline. Update the attention detail panel and the "Attention Backend" table.

4. **Device placement changes** — check which ops run on CPU vs Spyre. The rotary embedding
   CPU fallback and attention scatter/gather are the main CPU ops. Update colors if anything
   moved to Spyre or if new CPU fallbacks were added.

5. **Architecture changes** — new processes, new inheritance relationships, new composition.
   E.g., if a new scheduler override is added, it should appear in the system overview.

6. **Key constraints** — check alignment constants (KV_LEN_ALIGNMENT, QUERY_CHUNK_SIZE,
   head size requirements) and update the docs if they changed.

## index.md Image Styling

Keep the figure tags with width overrides for legibility:

```markdown
![System Overview](system-overview.svg){: style="width: 140%; max-width: 1200px; margin-left: -20%;" }
![Plugin Architecture](plugin-architecture.svg){: style="width: 140%; max-width: 1000px; margin-left: -20%" }
```
