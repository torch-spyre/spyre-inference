# Helion-to-Spyre Transpilation Experiments

This directory contains experimental tools and scripts for transpiling Helion kernels to Spyre-compiled executables. The project demonstrates how to bridge Helion's high-level tiling abstractions with Spyre's hardware-accelerated execution model.

## Overview

The **pastamachine** framework provides a complete pipeline for converting Helion kernels into Spyre-optimized code:

1. **Helion Kernel** → FX Graph (aten ops)
2. **FX Graph** → Spyre-compatible transformations
3. **Spyre Compilation** → Hardware-executable code
4. **SDSC Analysis** → Performance insights and core utilization

## Directory Structure

```
helion-experiments/spyre-inference/experimental/helion/
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

## The pastamachine Framework

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

- **`prevent_reduction_upcasts()`**: Prevents float32 promotion
  - Pins reduction operations to input dtype (e.g., float16)
  - Critical for maintaining precision on Spyre hardware

- **`fix_scalar_args_for_spyre()`**: Converts `.Tensor` ops to `.Scalar`
  - Handles raw Python scalars in binary operations
  - Ensures Spyre backend compatibility

- **`_make_config_aware_core_division()`**: Config-driven parallelization
  - Uses Helion `block_sizes` to determine core allocation
  - Supports pointwise and reduction operations
  - Handles matmul and batch matmul with custom tiling

#### `analyze.py`
SDSC metadata inspection and validation:

- **`extract_key_sdsc_info()`**: Parses SDSC JSON files
- **`summarize_meta()`**: Aggregates info from all kernels
- **`print_meta_summary()`**: Pretty-prints analysis with config validation

#### `verify.py`
CPU-side correctness checking:

- **`verify_on_cpu()`**: Runs FX graph on CPU before Spyre compilation
- Returns CPU results for comparison with hardware execution

#### `util.py`
Metadata and helper utilities:

- **`TranspileMeta`**: Container for compilation metadata
  - SDSC file paths
  - FX graph module
  - Block-size tracking
  - Tile dimension mappings

- **`_capture_sdsc_paths()`**: Context manager for SDSC file collection

### Public API

```python
from pastamachine import (
    compile_helion_to_spyre,  # Main transpilation function
    verify_on_cpu,            # CPU verification
    TranspileMeta,            # Metadata container
    extract_key_sdsc_info,    # SDSC analysis
    summarize_meta,           # Multi-kernel summary
    print_meta_summary,       # Pretty-print analysis
)
```

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
```
/home/ngl/results/YYYYMMDD_HHMMSS_profiling_sweep_experiment/
├── profiling_v6_vecsize32_cfgdefault_sencoresdefault.json
├── profiling_v6_vecsize64_cfgbs16_sencoresdefault.json
├── ...
└── sweep_summary.json  # Aggregated results
```

## Usage Examples

### Basic Transpilation

```python
import torch
import helion
import helion.language as hl
import pastamachine

@helion.kernel()
def my_kernel(x, y, out):
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]

# Prepare Spyre tensors
x = torch.randn(1024, device="spyre")
y = torch.randn(1024, device="spyre")
out = torch.zeros(1024, device="spyre")

# Compile
compiled = pastamachine.compile_helion_to_spyre(
    my_kernel,
    (x, y, out),
    do_verify_run_on_cpu=True,
)

# Execute
compiled(x, y, out)
```

### With Config and Metadata

```python
config = helion.Config(block_sizes=[64])

compiled, meta = pastamachine.compile_helion_to_spyre(
    my_kernel,
    (x, y, out),
    config=config,
    return_meta=True,
)

# Analyze SDSC
pastamachine.print_meta_summary(meta)

# Inspect specific SDSC file
sdsc_data = meta.load_sdsc(0)
info = pastamachine.extract_key_sdsc_info(sdsc_data)
print(f"Cores used: {info['numCoresUsed']}")
print(f"Work slices: {info['numWkSlicesPerDim']}")
```

### Profiling Integration

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    record_shapes=True,
) as prof:
    compiled(x, y, out)

# View results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Key Concepts

### Block Sizes and Core Division

Helion's `Config(block_sizes=[...])` controls how work is divided across Spyre cores:

- **block_size**: Determines the granularity of parallelization
- **num_cores** = `vector_size / block_size` (capped at `SENCORES`, default 32)
- Smaller block sizes → more cores → higher parallelism
- Block size must evenly divide the vector size

Example:
```python
# vector_size=512, block_size=32 → 16 cores
# vector_size=512, block_size=64 → 8 cores
# vector_size=512, block_size=16 → 32 cores (max)
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

## Environment Variables

- **`SENCORES`**: Maximum cores available (default: 32)
  - Used by core division planning
  - Affects parallelization strategies

## Dependencies

- **torch**: PyTorch framework
- **helion**: Helion kernel language
- **torch_spyre**: Spyre backend for PyTorch
- **torch.profiler**: Performance profiling (optional)

### Installation

```bash
# Install Helion
uv pip install helion

# Other dependencies should already be available in your torch-spyre environment
```

## Path Configuration

The scripts assume the following directory structure:
```
/home/ngl/gitrepos/spyre/
├── helion-experiments/
│   └── spyre-inference/experimental/helion/  # This directory
└── new_stack/
    └── torch-spyre/  # Spyre backend
```

Update `sys.path` in scripts if your layout differs:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/path/to/torch-spyre')
```

## Performance Tips

1. **Choose appropriate block sizes**: Start with powers of 2 (16, 32, 64)
2. **Profile before optimizing**: Use v6 script to identify bottlenecks
3. **Watch for memory ops**: Memset and HtoD can dominate small kernels
4. **Verify core utilization**: Check SDSC metadata for expected parallelism
5. **Test multiple configs**: Optimal block size varies by problem size

## Troubleshooting

### Import Errors
```python
# Ensure paths are correct
import sys
sys.path.insert(0, '/path/to/torch-spyre')
sys.path.insert(0, '/path/to/helion-experiments')
```

### Core Count Mismatches
- Check that `block_size` evenly divides `vector_size`
- Verify `SENCORES` environment variable
- Review SDSC metadata with `print_meta_summary()`

### Float32 Upcasts
- The framework automatically prevents these via `prevent_reduction_upcasts()`
- If issues persist, check that input tensors are float16

### Profiler Issues
- Ensure `PrivateUse1` activity is supported
- Check that Spyre backend is properly initialized
- Verify profiler output with `.table()` method

## Future Work

- Support for multi-dimensional tiling
- Automatic block size tuning
- Integration with torch.compile
- Extended reduction operation support
- Performance regression testing

## License

Copyright 2026 The Torch-Spyre Authors

Licensed under the Apache License, Version 2.0

## References

- Helion Documentation: [Internal]
- Spyre Backend: `/new_stack/torch-spyre`
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html