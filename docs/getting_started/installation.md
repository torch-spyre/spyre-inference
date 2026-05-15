# Installation

This guide covers the installation of `spyre-inference` using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python >= 3.11
- `uv` package manager installed ([installation guide](https://github.com/astral-sh/uv))
- Access to IBM Spyre hardware with the Spyre Runtime Stack (required for torch-spyre compilation)

## Installation with uv sync

The `spyre-inference` plugin uses `uv` for dependency management and installation. The project configuration in `spyre_inference/pyproject.toml` includes several important settings that ensure proper installation:

### Basic Installation

From the `spyre_inference` directory, run:

```bash
uv sync --frozen
```

This command will:

1. Install all project dependencies
2. Build vLLM from source with CPU backend support
3. Build torch-spyre from source
4. Install PyTorch 2.11.0 from the CPU-specific index

### Configuration Highlights

The `pyproject.toml` file includes several key configurations:

#### 1. Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.11.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "cpu" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.11.0)
- vLLM is built specifically for the CPU backend

#### 2. Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
```

This ensures that both torch-spyre and vllm are compiled from source, instead of pulling pre-compiled wheels from pypi.

#### 3. PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

We ensure that the cpu flavor of pytorch is installed, as we're not building cuda support.

## Verification

After installation, verify the plugin is correctly installed:

```bash
python -c "import spyre_inference; print(spyre_inference.__version__)"
```

## Development Installation

For development work, install with the dev dependency group:

```bash
uv sync --group dev
```

This includes additional tools like pytest, pytest-asyncio, and other testing utilities.

## Troubleshooting

### Build Failures

If you encounter build failures:

1. **torch-spyre compilation**: Ensure the Spyre Runtime Stack is available on your system. See internal development documentation for environment setup.
2. **vLLM build**: Check that you have sufficient memory and CPU resources for compilation
3. **Dependency conflicts**: Review the `override-dependencies` section in `pyproject.toml`

## Next Steps

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference
```

You can then use vLLM as usual:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_model_len=128,
    max_num_seqs=2,
)
```

See the [Examples](../examples/offline_inference/torch_spyre_inference.md) page for more usage patterns.
