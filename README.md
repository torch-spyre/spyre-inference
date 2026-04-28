# spyre-inference

A vLLM platform plugin for IBM Spyre AI accelerators.

## Overview

`spyre-inference` is the next evolution of [`sendnn-inference`](https://github.com/torch-spyre/sendnn-inference), providing seamless integration of IBM's Spyre hardware accelerators with vLLM for high-performance large language model inference.

This plugin leverages `torch-spyre` to utilize PyTorch's native Inductor compiler backend, enabling optimized model execution on Spyre devices through vLLM's plugin architecture.


## Requirements

- Python >= 3.11
- Access to IBM Spyre hardware with the Spyre Runtime stack
- PyTorch 2.10.0 (CPU backend)

## Installation

```bash
# Clone the repository
git clone https://github.com/torch-spyre/spyre-inference
cd spyre-inference

# Install with uv (recommended)
uv sync --frozen
```

**Note:** `torch-spyre` compilation requires access to IBM Spyre hardware with the Spyre Runtime stack. See internal development documentation for environment setup.

## Usage

The plugin automatically registers with vLLM when installed.
Use it by setting `VLLM_PLUGINS=spyre_inference"

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_model_len=128,
    max_num_seqs=2,
)
```

## Testing

The test suite includes:
- **Local tests** (`-m spyre`) - Spyre-specific functionality validation
- **Upstream tests** (`-m upstream`) - vLLM compatibility verification

Upstream tests are automatically synced from the vLLM repository at the commit specified in `pyproject.toml`.

## Contributing

See [Contributing Guide](docs/contributing/README.md) for:
- Issue reporting and feature requests
- Development setup
- Testing guidelines
- Pull request process

## Documentation

- [Installation Guide](docs/getting_started/installation.md)
- [Contributing Guide](docs/contributing/README.md)

## License

Apache 2.0

## Related Projects

- [torch-spyre](https://github.com/torch-spyre/torch-spyre) - PyTorch backend for Spyre accelerators
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference engine
- [sendnn-inference](https://github.com/torch-spyre/sendnn-inference) - Previous generation Spyre vLLM plugin
