# Profiling Spyre with Kineto

This manual explains how to profile Spyre workloads using PyTorch's
Kineto profiler on both the **old stack** (`sendnn-inference`) and the
**new stack** (`spyre-inference`). Both stacks use the same class of
Kineto-patched torch build, but the vLLM-level API for enabling
profiling differs.

---

## 1. Prerequisites

### 1.1 Kineto-patched torch wheel (both stacks)

A standard CPU-only PyTorch build includes the Kineto profiling
framework, but its `PrivateUse1` backend — PyTorch's generic slot for
third-party accelerators — is a no-op. Run the profiler as-is and the
device timeline in Perfetto is empty.

The patched build connects Kineto's `PrivateUse1` profiler slot to
`libaiupti.so`, the AIU hardware performance-counter library that
records kernel start/end timestamps directly on the card.

Wheels are published at the `kineto-spyre` release page:

<https://github.com/IBM/kineto-spyre/releases>

**Pick the wheel that matches the torch build currently installed in
your venv.** The patched wheel replaces the stock torch install, so
its version must line up with what is already there. Check first:

```bash
python -c "import torch, sys; print(torch.__version__, sys.version_info[:2])"
```

Then pick the release asset whose:

- **torch major.minor version** matches the reported `torch.__version__`
  prefix
- **`cpXY` Python tag** matches the reported Python version (e.g.
  `cp312` for Python 3.12)
- **platform tag** matches your OS/architecture (typically
  `linux_x86_64`)

Filenames follow the pattern
`torch-<MAJOR.MINOR.PATCH>+aiu.kineto.<KINETO_VER>-<pyver>-<pyver>-<platform>.whl`.

### 1.2 Requirements on the target system

Before installing the wheel, verify:

1. **`libaiupti.so` is present** on the system:

   ```bash
   ls /opt/ibm/spyre/runtime/lib/libaiupti.so
   ```

   This is the AIU hardware performance-counter library that the
   patched Kineto build links against. It ships with the Spyre
   runtime. If it is missing, profiling will silently produce empty
   device rows.

2. **The Spyre device is accessible** — typically a `/dev/vfio/<N>`
   node exposed to the container.

3. **`uv` is available** (or an equivalent Python package installer).

### 1.3 Installing the wheel

Activate the venv that contains your vLLM install and force-reinstall
the patched wheel:

```bash
source /path/to/your-venv/bin/activate
uv pip install --no-deps --force-reinstall /path/to/downloaded-wheel.whl
```

`--no-deps` prevents unrelated dependency updates. `--force-reinstall`
is required because the base version prefix matches the stock CPU
wheel — without it, `uv` considers the install already satisfied and
skips.

### 1.4 Verify installation

After install, `torch.__version__` should include the `+aiu.kineto`
suffix:

```bash
python -c "import torch; print(torch.__version__)"
# Expected: <MAJOR.MINOR.PATCH>+aiu.kineto.<KINETO_VER>
```

If it still reports the plain `+cpu` string, `--force-reinstall` was
not applied or the wheel was installed into the wrong venv.

---

## 2. Enabling profiling — new stack

The new stack's vLLM runs the worker **in the same process** as user
code (via `distributed_executor_backend="external_launcher"`). This
lets `torch.profiler.profile(...)` wrap `llm.generate()` directly.

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
| `activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]` | Which event streams to record. `PrivateUse1` is PyTorch's generic slot for third-party accelerators; Spyre registers under it. Without the kineto-patched wheel this slot is a no-op. |
| `on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/")` | Callback that fires when the profiler context exits. Writes a Chrome/Perfetto-format JSON file (`.pt.trace.json`) into `logs/` with a timestamped filename. Despite the "tensorboard" name, output is Chrome trace JSON — the name is a historical artefact. |
| `record_shapes=True` | Capture input tensor shapes for each op, so the Perfetto trace shows e.g. `aten::mm [4096, 128] x [128, 4096]` rather than just `aten::mm`. Small overhead, useful for identifying which shape bucket each attention call landed in. |
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
| (venv activation) | `source /opt/spyre-inference/bin/activate` | Puts the spyre-inference venv's `python` and installed packages on `PATH` |
| `VLLM_PLUGINS` | `spyre_inference` | Required for vLLM to load the Spyre platform plugin |
| (kineto check) | — | Inspects `torch.__version__` and warns if the stock wheel is installed instead of the `+aiu.kineto` patched build; prints the install command to fix it |
| `OMP_NUM_THREADS` | `1` | Pin OpenMP thread pool so BLAS work does not compete with Spyre dispatch |
| `OPENBLAS_NUM_THREADS` | `1` | Pin OpenBLAS thread pool (same rationale) |
| `MKL_NUM_THREADS` | `1` | Pin MKL thread pool (same rationale) |
| `NUMEXPR_NUM_THREADS` | `1` | Pin NumExpr thread pool (same rationale) |
| `VECLIB_MAXIMUM_THREADS` | `1` | Pin Apple vecLib thread pool (harmless on Linux; kept for portability) |

Additionally required but **not** exported by the setup script — set
these where the run is launched:

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | Safety margin so cumulative D2H sync stalls do not kill the run mid-generate (see §5.2) |
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

The new stack uses `torch.compile` with an Inductor cache. Each unique
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

## 3. Enabling profiling — old stack

The old stack's vLLM launches the worker in a **separate process**.
`torch.profiler.profile(...)` wrapped around `llm.generate()` in user
code would only see the launcher process — the actual model execution
and Spyre kernels would be invisible.

Instead, the profiler configuration is passed to `LLM(...)` at
construction time via `ProfilerConfig`, and toggled with
`llm.start_profile()` / `llm.stop_profile()`. These are remote-procedure
calls that tell the already-running worker to start/stop its internally
constructed profiler context.

### 3.1 Minimum example

```python
import os
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig

llm = LLM(
    model="ibm-granite/granite-3.3-8b-instruct",
    dtype="float16",
    max_model_len=1024,
    max_num_seqs=1,
    num_gpu_blocks_override=64,
    profiler_config=ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="logs/",
        torch_profiler_record_shapes=True,
    ),
)

prompts = ["What do you know about Zurich?"]
samplings = [SamplingParams(max_tokens=4, temperature=0.0)]

# No warmup — old stack has no Inductor cache; compilation is part of the run

llm.start_profile()
try:
    outputs = llm.generate(prompts, samplings)
finally:
    llm.stop_profile()

os._exit(0)  # avoids TimestampCalibrator abort at teardown
```

### 3.2 `ProfilerConfig` field reference

| Field | Purpose |
|---|---|
| `profiler="torch"` | Selects the torch.profiler backend (the only supported option today). |
| `torch_profiler_dir="logs/"` | Directory where the worker will write the `.pt.trace.json` file. |
| `torch_profiler_record_shapes=True` | Same effect as `record_shapes=True` on the new stack — captures input tensor shapes per op. |
| `torch_profiler_with_stack=True` | Optional: captures Python call stacks for each op, so Perfetto can show the source-line origin. Adds noticeable overhead. |

The worker constructs the underlying `torch.profiler.profile()` with
`activities=[CPU, PrivateUse1]` automatically — user code cannot
override the activity list on this path.

### 3.3 Load-bearing environment variables

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_PLUGINS` | `sendnn_inference` | Required for vLLM to load the sendnn platform plugin. Without it, `current_platform.device_type` is empty and `LLM(...)` aborts with `RuntimeError: Device string must not be empty`. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | Safety margin for cumulative sync stalls |

`HOME` must point to a directory writable by the running user (for
Hugging Face weight caching).

---

## 4. Viewing traces

Traces are written as `.pt.trace.json` (or `.pt.trace.json.gz` if
compressed). Open them in the Perfetto UI:

<https://ui.perfetto.dev> → "Open trace file"

The trace shows CPU thread rows on top and one Spyre device row below.

For a terminal summary (new stack only, `acc_events=True` required):

```python
prof.key_averages().table(sort_by="cpu_time_total", row_limit=20).replace("CUDA", "AIU")
prof.key_averages().table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU")
```

The `cuda_time_total` column contains Spyre device time despite the
name — a PyTorch legacy naming artefact. The `.replace("CUDA", "AIU")`
call is a display fix.

---

## 5. Known runtime issues (not profiler-specific)

These affect any Spyre run, but appear most frequently during profiling
because profile runs are longer and more numerous than smoke tests.

### 5.1 libflex lost-wakeup wedge

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

### 5.2 60-second D2H stall

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

### 5.3 VFIO DMA mapping exhaustion

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

### 5.4 `TimestampCalibrator` abort at exit

**Symptom:** abort message printed at process exit.

**Cause:** imperfect clock-alignment calibration between host TSC and
Spyre hardware counter. Residual offset causes the calibrator's C++
destructor to fail its own invariants at teardown.

**Fix:** call `os._exit(0)` at end of `main()` to bypass Python's
normal C++ destructor chain.

---

## 6. Monitoring device utilization with `aiu-smi`

Kineto captures per-op timings inside a bounded profiling window;
`aiu-smi` complements it by streaming per-second AIU hardware counters
(compute busy %, power, temperature, HBM bandwidth, host↔device DMA)
across the whole run. This surfaces long idle gaps, DMA saturation,
and thermal throttling that a short profiler window would miss.

### 6.1 Two-terminal recipe

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

### 6.2 Column reference

| Column | Meaning |
|---|---|
| `busy` | AIU compute utilization %; primary metric for kernel occupancy. |
| `pwr` / `gtemp` | Device power (W) and temperature (°C). |
| `rdmem` / `wrmem` | On-device HBM bandwidth. |
| `rdrdma` / `wrrdma` | Host↔device DMA bandwidth. KV-cache traffic surfaces here. |
| `rxpci` / `txpci` | PCIe bandwidth; typically 0 on a single-card setup. |
| `rsvmem` | Reserved HBM (MB). Under-reports in `aiu-monitor` 1.0.0. |
| `hostcpu` | Host CPU %, summed across cores (~800% on 8 cores is normal). |

### 6.3 Useful options

```bash
aiu-smi -d 2               # poll every 2 seconds (default 1s)
aiu-smi -s -f run.csv      # log to CSV file for offline analysis
aiu-smi -g A               # emit all metric groups (default: D M P)
aiu-smi --mem-details      # break down HBM reservation
```

CSV output pairs well with a Kineto trace: run `aiu-smi -s -f run.csv`
alongside a profiled `generate()` and the row timestamps align to the
Perfetto trace window.

### 6.4 Known limitations

- PF (physical-function) mode only; VF mode is unsupported by
  `aiu-monitor` 1.0.0.
- `rsvmem` and `pt_act` counters are not populated correctly upstream.
- If either terminal fails to export `SENLIB_DEVEL_CONFIG_FILE`, every
  numeric column reads as `-`. Always check this first when the output
  is empty.
