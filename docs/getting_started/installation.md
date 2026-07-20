# Installation

This guide covers the installation of `spyre-inference` using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) package manager
- Access to IBM Spyre hardware with the Spyre Runtime Stack (required for torch-spyre compilation)

## Install

From the repository root, run:

```bash
uv sync --frozen
```

This command will:

1. Install all project dependencies
2. Build vLLM from source with the empty backend (`VLLM_TARGET_DEVICE=empty`, no device-specific C kernels)
3. Build torch-spyre from source
4. Install PyTorch 2.11.0 from the CPU-specific index

## Verification

After installation, verify the plugin is correctly installed:

```bash
python -c "import spyre_inference; print(spyre_inference.__version__)"
```

## Troubleshooting

### Build Failures

If you encounter build failures:

1. **torch-spyre compilation**: Ensure the Spyre Runtime Stack is available on your system. See internal development documentation for environment setup.
2. **vLLM build**: Check that you have sufficient memory and CPU resources for compilation
3. **Dependency conflicts**: Review the `override-dependencies` section in `pyproject.toml`
