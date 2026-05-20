# Helion-to-Spyre Transpilation Experiments

This directory contains experimental tools and scripts for transpiling Helion kernels to Spyre-compiled executables. The project demonstrates how to bridge Helion's high-level tiling abstractions with Spyre's hardware-accelerated execution model.

## Overview

The **pastamachine** framework provides a complete pipeline for converting Helion kernels into Spyre-optimized code:

1. **Helion Kernel** → FX Graph (aten ops)
2. **FX Graph** → Spyre-compatible transformations
3. **Spyre Compilation** → Hardware-executable code
4. **SDSC Analysis** → Performance insights and core utilization

## Dependencies

- **torch**: PyTorch framework
- **helion**: Helion kernel language
- **torch_spyre**: Spyre backend for PyTorch
- **torch.profiler**: Performance profiling (optional)

### Installation

```bash
# Install Helion
uv pip install helion debugpy

# Other dependencies should already be available in your torch-spyre environment
```

## Directory Structure

```text
spyre-inference/experimental/helion/
├── pastamachine/                    # Core transpilation framework
│   ├── __init__.py                 # Public API exports
│   ├── _logging.py                 # Logging utilities
│   ├── analyze.py                  # SDSC metadata analysis tools
│   ├── transpile.py                # Main transpilation pipeline (1457 lines)
│   ├── util.py                     # Metadata containers and helpers
│   └── verify.py                   # CPU verification utilities
├── basic_transpile_example.py      # Basic transpilation example
├── profiling_sweep_experiment.py   # Advanced profiling & sweep experiments
└── README.md                       # This file
```

## The `pastamachine` Framework

### Core Components

#### `transpile.py`

The heart of the transpilation pipeline. Key functions:

- **`compile_helion_to_spyre()`**: Main entry point for transpilation
    - Converts Helion kernels to Spyre-compiled callables
    - Supports CPU verification before hardware execution
    - Optional config-aware core division planning
    - Returns metadata for SDSC analysis when `return_meta=True`

- **`transpile_fx_graphs()`**: Helion IR → FX graph conversion
    - Extracts aten operations from Helion device IR
    - Handles block-size parameters and tiling dimensions
    - Tracks which operations are affected by Helion configs

#### `analyze.py`

SDSC metadata inspection and validation:

- **`extract_key_sdsc_info()`**: Parses SDSC JSON files
- **`summarize_meta()`**: Aggregates info from all kernels
- **`print_meta_summary()`**: Pretty-prints analysis with config validation

#### `verify.py`

CPU-side correctness checking:

- **`verify_on_cpu()`**: Runs FX graph on CPU before Spyre compilation
- Returns CPU results for comparison with hardware execution

## Example Scripts

### `basic_transpile_example.py`

Basic transpilation example demonstrating the core workflow:

```python
@helion.kernel()
def inplace_add(a, b, out):
    for tile in hl.tile(out.size()):
        out[tile] += a[tile] + b[tile]

# Transpile with config-aware core division
config = helion.Config(block_sizes=[32])
compiled_fn, meta = pastamachine.compile_helion_to_spyre(
    inplace_add,
    (a_spyre, b_spyre, out_spyre),
    do_verify_run_on_cpu=True,
    return_meta=True,
    config=config,
)

# Analyze SDSC metadata
pastamachine.print_meta_summary(meta)

# Execute on Spyre hardware
result = compiled_fn(a_spyre, b_spyre, out_spyre)
```

**Features:**

- Single vector size (configurable: 512 by default)
- Config-driven core division (block_size=32)
- CPU verification before hardware execution
- SDSC analysis and core count validation
- Correctness checking against expected results

### `profiling_sweep_experiment.py`

Advanced profiling and parameter sweep experiments:

```python
# Sweep over vector sizes: 32, 64, 128, ..., 8192
VECTOR_SIZES = [2**k for k in range(5, 14)]

# Sweep over Helion configs
HELION_CONFIGS = [
    None,                          # Default
    helion.Config(block_sizes=[16]),
    helion.Config(block_sizes=[32]),
    helion.Config(block_sizes=[64]),
    helion.Config(block_sizes=[128]),
    helion.Config(block_sizes=[256]),
]

# Profile each combination
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]):
    result = compiled_fn(a_spyre, b_spyre, out_spyre)
```

**Features:**

- Multi-dimensional parameter sweeps (size × config)
- PyTorch profiler integration (CPU + Spyre/AIU)
- JSON output for all profiling data
- Top-5 operations by Spyre time share
- Automatic core count verification
- Memory operation analysis (Memset, HtoD transfers)
- Timestamped result directories

**Output Structure:**

```text
path/to/results/YYYYMMDD_HHMMSS_profiling_sweep_experiment/
├── profiling_v6_vecsize32_cfgdefault_sencoresdefault.json
├── profiling_v6_vecsize64_cfgbs16_sencoresdefault.json
├── ...
└── sweep_summary.json  # Aggregated results
```

### SDSC Metadata

SDSC (Spyre Data Structure Compiler) files contain:

- **numCoresUsed**: Actual cores allocated
- **numWkSlicesPerDim**: Work division per dimension
- **coreIdToDsc**: Core-to-DSC mapping
- **computeOp**: Execution unit and operation details

The framework validates that `numCoresUsed == prod(numWkSlicesPerDim)`.

### CPU Verification

Setting `do_verify_run_on_cpu=True` runs the FX graph on CPU before Spyre compilation:

- Catches transpilation errors early
- Provides reference results for correctness checking
- Useful for debugging complex kernels
