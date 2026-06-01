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
