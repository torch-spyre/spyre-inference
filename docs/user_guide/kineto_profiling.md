# Profiling Spyre with Kineto

This manual explains how to profile Spyre workloads with
`spyre-inference` using PyTorch's Kineto profiler.

---

## 1. Prerequisites

### 1.1 AIUPTI profiler backend

A standard CPU-only PyTorch build includes the Kineto profiling
framework, but its `PrivateUse1` backend — PyTorch's generic slot for
third-party accelerators — is a no-op. Run the profiler as-is and the
device timeline in Perfetto is empty.

The AIUPTI activity provider connects Kineto's `PrivateUse1` profiler
slot to `libaiupti.so`, the AIU hardware performance-counter library
that records kernel start/end timestamps directly on the card.
Historically this shipped as a separate `+aiu.kineto` torch wheel; as
of torch-spyre PR
[#1856](https://github.com/torch-spyre/torch-spyre/pull/1856)
(merged 2026-08-04) it lives inside `torch-spyre` itself and is
registered via PyTorch's `REGISTER_PRIVATEUSE1_PROFILER` macro. The
Python-facing API is unchanged — you still use
`torch.profiler.profile(activities=[CPU, PrivateUse1])`.

The AIUPTI backend is compiled in **only** when torch-spyre is built
with `USE_SPYRE_PROFILER=1`. The `spyre-inference` `pyproject.toml`
currently pins this flag to `"0"` (under
`[tool.uv.extra-build-variables.torch-spyre]`) to avoid a long
model-load stall on Z, so a plain `uv sync` produces a torch-spyre
wheel with **no AIU device events** in the trace — CPU-side rows are
present, the device row is empty, and no error is raised.

### 1.2 Requirements on the target system

1. **`libaiupti.so` and its headers are present:**

   ```bash
   ls /opt/ibm/spyre/runtime/lib/libaiupti.so
   ls /opt/ibm/spyre/runtime/include/libaiupti/*.h
   ```

   The runtime library and its headers ship with the Spyre runtime.
   Both are needed at build time; the `.so` is needed at run time.

2. **The Spyre device is accessible** — typically a `/dev/vfio/<N>`
   node exposed to the container.

3. **`uv` is available** (or an equivalent Python package installer).

### 1.3 Enabling the AIUPTI backend

**Option A — persistent (`pyproject.toml` flip):**

Edit `pyproject.toml` in `spyre-inference`:

```toml
[tool.uv.extra-build-variables.torch-spyre]
USE_SPYRE_PROFILER = "1"   # was "0"
```

Then `uv sync` — the resulting wheel has AIUPTI compiled in. This is
the right path for anyone using this venv for profiling long-term.
Because the flag was originally set to `"0"` to work around long Z
model-load times, flipping it in-tree is a project-level decision, not
a silent local edit.

**Option B — rebuild in place after each pod recreate:**

If you want to leave `pyproject.toml` alone, force-reinstall
torch-spyre with the flag set:

```bash
USE_SPYRE_PROFILER=1 \
  LIBAIUPTI_INSTALL_DIR=/opt/ibm/spyre/runtime \
  LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH" \
  /path/to/your-venv/bin/python -m pip install \
    --no-deps --force-reinstall --no-cache-dir \
    "torch-spyre @ git+https://github.com/torch-spyre/torch-spyre@<rev>"
```

Pin `<rev>` to whatever commit `[tool.uv.sources.torch-spyre]` in
`pyproject.toml` resolves to. Do **not** pass `--no-build-isolation`:
the build system's `torch~=2.13.0` requirement needs an isolated env,
and the venv's torch is not trusted for build. Build isolation
downloads ~1 GB of torch into `/tmp/pip-build-env-*/` and cleans up
after; watch `/tmp` space on tight pods.

`uv sync` / `uv run` will silently re-install the pinned non-profiler
wheel on top of this — invoke Python from the venv directly, or use
`uv run --no-sync …`, when iterating.

### 1.4 Verify installation

Stock torch is expected — the `+aiu.kineto` suffix is gone:

```bash
python -c "import torch; print(torch.__version__)"
# Expected: 2.13.0+cpu   (no +aiu.kineto suffix)
```

The real signal is that the AIUPTI backend is linked into
`torch_spyre/_C.so`. Four checks — all four must pass:

```bash
SO=$(python -c "import torch_spyre, os; print(os.path.join(os.path.dirname(torch_spyre.__file__), '_C.so'))")

# 1. Wheel size — profiler-enabled build is ~5× bigger (~77 MB vs ~15 MB).
ls -la "$SO"

# 2. libaiupti actually linked.
ldd "$SO" | grep libaiupti
# → libaiupti.so => /opt/ibm/spyre/runtime/lib/libaiupti.so

# 3. AIUPTI symbols present in the shared object.
nm -CD "$SO" | grep -i AiuptiActivityProfilerSession | head -3
# → several T (defined text) entries

# 4. A produced trace contains `kernel`-category events.
python -c "
import json, collections
t = json.load(open('<path>.pt.trace.json'))
cats = collections.Counter(e.get('cat','') for e in t['traceEvents'])
assert cats.get('kernel', 0) > 0, f'no kernel events: {cats}'
print('OK', cats.most_common(5))
"
```

Do **not** trust `torch_spyre.profiler.is_available()` as a health
signal — it is hardcoded to `return False` regardless of build flags
(comment: "more to be implemented later").

---

## 2. Enabling profiling

vLLM in `spyre-inference` runs the worker **in the same process** as
user code (via `distributed_executor_backend="external_launcher"`).
This lets `torch.profiler.profile(...)` wrap `llm.generate()`
directly.

### 2.1 Minimum example

```python
import os

# external_launcher reads these from env
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig

llm = LLM(
    model="ibm-granite/granite-3.3-8b-instruct",
    dtype="float16",
    max_model_len=32,
    max_num_seqs=1,
    num_gpu_blocks_override=64,
    attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    distributed_executor_backend="external_launcher",  # worker in-process
)

prompts = ["What do you know about Zurich?"]
samplings = [SamplingParams(max_tokens=4, temperature=0.0)]

# Warmup
for _ in range(2):
    llm.generate(prompts, samplings)

# Profiled generate
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/"),
    record_shapes=True,
    acc_events=True,
) as prof:
    outputs = llm.generate(prompts, samplings)

# Optional terminal summary
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
      .replace("CUDA", "AIU"))

os._exit(0)  # avoids TimestampCalibrator abort at teardown
```

### 2.2 `profile(...)` argument reference

| Argument | Purpose |
|---|---|
| `activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]` | Which event streams to record. `PrivateUse1` is PyTorch's generic slot for third-party accelerators; Spyre registers under it. Without a torch-spyre wheel built with `USE_SPYRE_PROFILER=1` (see §1.3), this slot is a no-op — the trace file is written but contains zero Spyre-device events. |
| `on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/")` | Callback that fires when the profiler context exits. Writes a Chrome/Perfetto-format JSON file (`.pt.trace.json`) into `logs/` with a timestamped filename. Despite the "tensorboard" name, output is Chrome trace JSON — the name is a historical artefact. |
| `record_shapes=True` | Capture input tensor shapes for each op, so the Perfetto trace shows e.g. `aten::mm [4096, 128] x [128, 4096]` rather than just `aten::mm`. Small overhead, useful for identifying which shape bucket each attention call landed in. |
| `with_stack=True` | Optional: capture Python call stacks per op. Event names then embed absolute source paths and line numbers, which makes the trace self-authenticating — you can tell after the fact which branch/tree it came from. Adds noticeable overhead; drop it for tight measurement runs. |
| `acc_events=True` | Retain events in memory after the trace is written, so `prof.key_averages().table(...)` works after the context exits. Drop it if you only care about the trace file. |

### 2.3 Load-bearing environment variables

The recommended way to set these is to source `setup_profile_env.sh`
before launching:

```bash
source ./setup_profile_env.sh
python -u profile_spyre_inference.py
```

The script must be *sourced*, not executed: `export` and `source
/opt/spyre-inference/bin/activate` only take effect in the current
shell, so running it as `./setup_profile_env.sh` in a child shell has
no effect on your interactive environment.

Contents of the script and why each entry matters:

| Variable / action | Value | Purpose |
|---|---|---|
| (venv activation) | `source /opt/spyre-inference/bin/activate` | Puts the venv's `python` and packages on `PATH`. This is a deployment-style path; for a local `uv sync` install use `source .venv/bin/activate`. |
| `VLLM_PLUGINS` | `spyre_inference` | Required for vLLM to load the Spyre platform plugin |
| (AIUPTI check) | — | Warns if `torch_spyre/_C.so` was not built with `USE_SPYRE_PROFILER=1` — verify via `ldd .../_C.so \| grep libaiupti` and presence of `AiuptiActivityProfilerSession` symbols. Do **not** inspect `torch.__version__` for a `+aiu.kineto` suffix; that flow was retired by torch-spyre PR #1856. See §1.3. |
| `OMP_NUM_THREADS` | `1` | Pin OpenMP thread pool so BLAS work does not compete with Spyre dispatch |
| `OPENBLAS_NUM_THREADS` | `1` | Pin OpenBLAS thread pool (same rationale) |
| `MKL_NUM_THREADS` | `1` | Pin MKL thread pool (same rationale) |
| `NUMEXPR_NUM_THREADS` | `1` | Pin NumExpr thread pool (same rationale) |
| `VECLIB_MAXIMUM_THREADS` | `1` | Pin Apple vecLib thread pool (harmless on Linux; kept for portability) |

Additionally required but **not** exported by the setup script — set
these where the run is launched:

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | Safety margin so cumulative D2H sync stalls do not kill the run mid-generate (see §4.2) |
| `RANK` | `0` | Torchrun-style distributed rank. `distributed_executor_backend="external_launcher"` reads it from env; set explicitly for single-process use |
| `LOCAL_RANK` | `0` | Torchrun-style local rank; same rationale as `RANK` |
| `WORLD_SIZE` | `1` | Total number of ranks in the job |
| `LOCAL_WORLD_SIZE` | `1` | Ranks on the local node. **Mandatory** — `libspyre_comms.so.1` hard-aborts if this is unset |
| `MASTER_ADDR` | `127.0.0.1` | Rendezvous address for the (single-process) distributed group |
| `MASTER_PORT` | `29500` | Rendezvous port; any free port works |

The six torchrun-style vars can also be set inside the script via
`os.environ.setdefault(...)` before importing vLLM, as shown in §2.1.
Setting them in the environment or in the script is equivalent.

`VLLM_ENABLE_V1_MULTIPROCESSING=0` is **auto-set by vLLM** when
`distributed_executor_backend="external_launcher"` is passed to
`LLM(...)`, so it does not need to appear in the setup script or the
run environment. Its effect matters — the worker must run in-process
so Kineto can capture Spyre events (`PrivateUse1` events only appear in
the process that owns the Spyre device handle) — but the constructor takes care
of it. If a different executor backend is used and Spyre events are
missing from the trace, set this to `0` manually.

### 2.4 Why warmup matters

`spyre-inference` uses `torch.compile` with an Inductor cache. Each unique
`(num_blocks, query_len)` combination compiles to its own kernel
bundle. If the profiled `llm.generate()` call hits a shape bucket
that was not compiled during warmup, an ~3-second
`PyCodeCache.load_by_key_path` or `entire_frame_compile` event fires
inside the measurement window and skews the numbers.

The warmup loop drains compilation before the profiled window. Two
warmup passes are typically sufficient for a single-prompt profile;
more if the workload exercises multiple shape buckets (e.g. a
long-prompt prefill followed by a decode).

---

## 3. Viewing traces

Traces are written as `.pt.trace.json` (or `.pt.trace.json.gz` if
compressed). Open them in the Perfetto UI:

<https://ui.perfetto.dev> → "Open trace file"

The trace shows CPU thread rows on top and one Spyre device row below.

For a terminal summary (`acc_events=True` required):

```python
prof.key_averages().table(sort_by="cpu_time_total", row_limit=20).replace("CUDA", "AIU")
prof.key_averages().table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU")
```

The `cuda_time_total` column contains Spyre device time despite the
name — a PyTorch legacy naming artefact. The `.replace("CUDA", "AIU")`
call is a display fix.

---

## 4. Known runtime issues (not profiler-specific)

These affect any Spyre run, but appear most frequently during profiling
because profile runs are longer and more numerous than smoke tests.

### 4.1 libflex lost-wakeup wedge

**Symptom:** `RuntimeStream::synchronize()` warning at 60s, escalating
to 120s, 180s, ... with `in_flight_=1` and no recovery. Process
consumes 100–200% CPU without forward progress.

**Cause:** lost-wakeup race in `libflex.so`'s `QueueCbs`, triggered by
the first H2D transfer.

**Fix:** flex PR #1165 ("Fix lost control blocks in PF mode: serialize
`QueueCbs` and make the batch-timestamp ring race-free") resolves it.
On systems that ship an older `libflex.so`, load a patched build via
`LD_PRELOAD` before launching:

```bash
LD_PRELOAD=/path/to/libflex_patched.so:/path/to/libflexhdma_patched.so \
    python -u your_profile_script.py
```

The system `libflex.so` typically lives at
`/opt/ibm/spyre/runtime/lib/libflex.so`; `LD_PRELOAD` is used because
that directory is often read-only.

### 4.2 60-second D2H stall

**Symptom:** `RuntimeStream::synchronize() still waiting after 60000ms:
in_flight_=0` followed by `completed after 60000ms`. Recovers on its
own after exactly 60s.

**Cause:** the same lost-wakeup mechanism, milder variant. Fires on
D2H copies (typically the LM-head D2H after each decode step). The
60s comes from `RuntimeStream::synchronize()`'s fixed poll interval
for warning messages.

**Fix:** the same PR #1165 eliminates most of these. Remaining stalls
come from cold-start flakes and compile-mode warmup drain.

**Safety margin** — prevents vLLM from killing the run if multiple
60s stalls accumulate:

```python
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
```

### 4.3 VFIO DMA mapping exhaustion

**Symptom:** `RAS::VFIO::MapDMAFailed / "No space left on device"`.

**Cause:** the Linux kernel caps VFIO DMA mappings at **65,535
entries per container**. Each Spyre tensor allocation that participates
in a transfer consumes one entry. The KV cache alone consumes
`num_blocks × num_layers × 2` mappings (K + V per layer). At
`num_gpu_blocks_override=64` and 40 layers, that's 5,120 mappings —
well within budget. But an unbounded `max_num_seqs` (default 256 on
some vLLM versions) can multiply the block count and blow the budget.

**Fix:** pass explicit values for `max_num_seqs` and
`num_gpu_blocks_override` to `LLM(...)`. Sizing them so
`num_blocks × num_layers × 2` stays well under 65,535 keeps the
mapping table healthy. The "consider restarting Linux" text in the
error message is misleading — pod restart does not help. This is a
configuration issue, not an IOMMU-state issue.

**Note:** the current KV cache layout — one K and one V tensor per
block per layer — is temporary. It exists because Spyre does not yet
support indirect addressing, so every page has to be an individually
mappable tensor. Once indirect addressing lands, the KV cache can
collapse to a small number of per-layer tensors indexed at runtime,
which will drastically reduce both the total tensor count and the
number of VFIO DMA mappings the cache consumes. The sizing guidance
above will remain correct until then.

### 4.4 `TimestampCalibrator` abort at exit

**Symptom:** abort message printed at process exit.

**Cause:** imperfect clock-alignment calibration between host TSC and
Spyre hardware counter. Residual offset causes the calibrator's C++
destructor to fail its own invariants at teardown.

**Fix:** call `os._exit(0)` at end of `main()` to bypass Python's
normal C++ destructor chain.

### 4.5 AIU trace buffer exhaustion — truncated device timeline

**Symptom:** stderr line
`Exceeded max AIU buffer count (5 > 5) - terminating tracing`
during a profiled run. The trace file is still written and still opens
in Perfetto, but the device row is truncated: kernels appear up to
some point mid-run and then abruptly stop, while the CPU rows keep
going.

**Cause:** the AIUPTI backend allocates a fixed pool of trace buffers
(hardcoded max 5) and terminates capture when the pool fills. Even a
4-token smoke test can trip this once compilation flushes are
included; anything longer will trip it deterministically.

**Fix:** no environment knob is exposed as of PR #1856. Mitigations:

- Keep the profiled window as short as possible — profile a single
  `generate()` call, not the warmup passes.
- Use `with_stack=False` and `record_shapes=False` when hunting a
  timeline scope issue; both add per-event overhead.
- For a permanent lift, patch `AiuptiActivityApi.cpp` (search for
  the buffer-count constant) and rebuild torch-spyre, or file an
  upstream request for an env knob.

Do **not** interpret an empty tail on the device row as "the workload
went idle" — check stderr for the exhaustion message first.

---

## 5. Monitoring device utilization with `aiu-smi`

Kineto captures per-op timings inside a bounded profiling window;
`aiu-smi` complements it by streaming per-second AIU hardware counters
(compute busy %, power, temperature, HBM bandwidth, host↔device DMA)
across the whole run. This surfaces long idle gaps, DMA saturation,
and thermal throttling that a short profiler window would miss.

### 5.1 Two-terminal recipe

`aiu-smi` reads counters from a metric file that the workload process
writes. `SENLIB_DEVEL_CONFIG_FILE` must be exported in **both**
terminals for them to agree on the file path — the most common source
of a row of `-` values.

**Terminal 1 — workload:**

```bash
source examples/offline_inference/setup_profile_env.sh
export DTCOMPILER_KEEP_EXPORT=true
export SENLIB_DEVEL_CONFIG_FILE=<venv-prefix>/etc/senlib_config_aiusmi.json
python examples/offline_inference/profile_spyre_inference.py
```

**Terminal 2 — monitor:**

```bash
SENLIB_DEVEL_CONFIG_FILE=<venv-prefix>/etc/senlib_config_aiusmi.json \
  aiu-smi
```

The monitor may be started before or after the workload; it will emit
`-` values until the workload begins writing counters.

Sample output:

```text
#ID Date      Time      hostcpu hostmem  pwr  gtemp busy  rdmem  wrmem  rxpci  txpci  rdrdma  wrrdma  rsvmem
#   YYYYMMDD  HH:MM:SS        %       %    W      C    %   GB/s   GB/s   GB/s   GB/s    GB/s    GB/s      MB
  0 20260715  11:45:28    828.3     5.2   75     42   87    4.2    3.1    0.8    0.2     1.2     0.9     512
```

### 5.2 Column reference

| Column | Meaning |
|---|---|
| `busy` | AIU compute utilization %; primary metric for kernel occupancy. |
| `pwr` / `gtemp` | Device power (W) and temperature (°C). |
| `rdmem` / `wrmem` | On-device HBM bandwidth. |
| `rdrdma` / `wrrdma` | Host↔device DMA bandwidth. KV-cache traffic surfaces here. |
| `rxpci` / `txpci` | PCIe bandwidth; typically 0 on a single-card setup. |
| `rsvmem` | Reserved HBM (MB). Under-reports in `aiu-monitor` 1.0.0. |
| `hostcpu` | Host CPU %, summed across cores (~800% on 8 cores is normal). |

### 5.3 Useful options

```bash
aiu-smi -d 2               # poll every 2 seconds (default 1s)
aiu-smi -s -f run.csv      # log to CSV file for offline analysis
aiu-smi -g A               # emit all metric groups (default: D M P)
aiu-smi --mem-details      # break down HBM reservation
```

CSV output pairs well with a Kineto trace: run `aiu-smi -s -f run.csv`
alongside a profiled `generate()` and the row timestamps align to the
Perfetto trace window.

### 5.4 Known limitations

- PF (physical-function) mode only; VF mode is unsupported by
  `aiu-monitor` 1.0.0.
- `rsvmem` and `pt_act` counters are not populated correctly upstream.
- If either terminal fails to export `SENLIB_DEVEL_CONFIG_FILE`, every
  numeric column reads as `-`. Always check this first when the output
  is empty.
