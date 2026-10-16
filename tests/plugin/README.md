# spyre-testing-plugin

pytest plugin for spyre-inference upstream test integration.

This downloads and caches the upstream vllm unit tests, running a subset of them configured by upstream_tests.yaml.

## Installation

```bash
uv sync --group dev  # from parent project
```

## Usage

The plugin is activated by the `addopts = ["-p", "spyre_testing_plugin.pytest_plugin"]` entry in the
parent project's `pyproject.toml`, so any pytest run rooted in spyre-inference loads it. It is
deliberately *not* registered as a `pytest11` entry point: that would load it into every pytest
session sharing the venv, including a sibling `torch-spyre` checkout's own suite.

Upstream tests are opt-in — see the marker gate in `pytest_plugin.py`'s module docstring.

Running the plugin against a vLLM checkout is the one case that needs the flag by hand:

```bash
cd ~/vllm && pytest -p spyre_testing_plugin.pytest_plugin -m upstream
```

## Development

The plugin's own behaviour is exercised from the parent project (there are no tests under
`tests/plugin/`, and a pytest run rooted here resolves that config instead and collects nothing):

```bash
uv sync --group dev            # from the repo root
uv run pytest tests/test_upstream_gating.py -m "not upstream"
```
