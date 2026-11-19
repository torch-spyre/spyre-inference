# Installation

This guide covers the installation of `spyre-inference` using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) package manager
- Access to IBM Spyre hardware with the Spyre Runtime Stack (required for torch-spyre compilation)

CI provisions the RPM versions in `spyre-rpms.lock` before building the Python environment.
On a development pod whose image-baked runtime does not match that lock, install and
activate the pinned RPMs first (requires access to the configured RPM repository):

```bash
bash scripts/install-pinned-rpms.sh --rebuild
source ~/spyre-libs/env.sh
```

The `--rebuild` option rebuilds torch-spyre against those libraries, clearing uv's
revision-keyed wheel cache first. After an RPM-only update, `uv sync` by itself can
reuse a wheel built against the old libraries. The installer leaves the system
`/opt/ibm/spyre` tree unchanged and points the unpacked binaries' hardcoded paths at a
`/tmp/<tag>/spyre` symlink to itself. Source its `env.sh` in each new shell.
Unlike the Install command below, `--rebuild` runs
`uv sync --group dev --reinstall-package torch-spyre` without `--frozen` and may update
`uv.lock`.

## Install

From the repository root, run:

```bash
uv sync --frozen
```

This command will:

1. Install all project dependencies
2. Build vLLM from source with the empty backend (`VLLM_TARGET_DEVICE=empty`, no device-specific C kernels)
3. Install or reuse the torch-spyre wheel for the pinned source revision (RPMs are provisioned separately)
4. Install PyTorch 2.14.1 from the CPU-specific index

## MRV2 kernel parity tests (x86 only)

The Model Runner V2 host kernels are checked against upstream's Triton kernels on
triton-cpu. triton is never a runtime dependency; it lives in the opt-in `triton-cpu`
dependency group, which installs nothing off x86:

```bash
uv sync --frozen --group dev --group triton-cpu
uv run --group triton-cpu pytest -m triton_cpu
```

The `triton_cpu` tests are collected only when `-m` names the marker. A later plain
`uv sync` removes triton again (a plain `uv run` leaves it). Every other test expects a
triton-free environment, as on Power and Z: `test_triton_is_not_installed` fails while
the group is installed.

## Verification

After installation, verify the plugin is correctly installed:

```bash
uv run --no-sync python -c "import spyre_inference; print(spyre_inference.__version__)"
```

## Troubleshooting

### Build Failures

If you encounter build failures:

1. **torch-spyre compilation**: Ensure the Spyre Runtime Stack is available on your system. See internal development documentation for environment setup.
2. **vLLM build**: Check that you have sufficient memory and CPU resources for compilation
3. **Dependency conflicts**: Review the `override-dependencies` section in `pyproject.toml`
