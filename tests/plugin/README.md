# spyre-testing-plugin

pytest plugin for spyre-inference upstream test integration.

This downloads and caches the upstream vllm unit tests, running a subset of them configured by upstream_tests.yaml.

## Installation

```bash
uv sync --group dev  # from parent project
```

## Usage

The plugin is automatically discovered by pytest via the `pytest11` entry point.

## Development

```bash
cd tests/plugin
uv sync
uv run pytest
```
