# Contributing to Spyre Inference

Thank you for your interest in contributing to the Spyre plugin for vLLM! There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Developing

Follow the [Installation Guide](../getting_started/installation.md) to get the base package installed, then install the dev dependency group:

```bash
uv sync --group dev
```

This includes `pytest`, `pyyaml`, and the `spyre-testing-plugin` for running the test suite.

### Linting

When submitting a PR, please make sure your code passes all linting checks. We use prek with a .pre-commit-config.yaml file to run checks on every commit.

The `format.sh` script will run prek from an isolated virtual environment using [uvx](https://docs.astral.sh/uv/guides/tools/). The only requirement is that you have `uv` installed.

```sh
bash format.sh
```

Alternatively, you can [install prek](https://github.com/j178/prek?tab=readme-ov-file#installation) and set up a git hook to run it on every commit with:

```sh
prek install
```

### Testing

The project includes both local tests (located in `tests/`) for spyre-inference specific functionality, and upstream vLLM tests automatically cloned from the vLLM repository at the commit specified in `pyproject.toml`, for compatibility verification.

#### Test Markers

The test suite uses pytest markers to categorize tests:

```python
--8<-- "pyproject.toml:test-markers-definition"
```

Some useful overrides:

```bash
# Run only local tests
pytest

# Run all upstream tests
pytest -m upstream

# Run attention tests from upstream only (see tests/plugin/spyre_testing_plugin/upstream_tests.yaml for markers on upstream tests)
pytest -m "attention"
```

#### Upstream Test Integration

Upstream tests are cloned from the vLLM repository at the commit pinned in `pyproject.toml`, fetching only the `tests/` directory. Cloned tests are cached in `~/.cache/vllm-upstream-tests` (or `$XDG_CACHE_HOME/vllm-upstream-tests`) with separate worktrees per commit, allowing multiple vLLM versions to be tested simultaneously. All upstream tests run with `VLLM_PLUGINS=spyre_inference,spyre_inference_ops,spyre_inference_hf_adaptor` set automatically. See `tests/plugin/spyre_testing_plugin/pytest_plugin.py` for implementation details.

!!! tip
    To force a re-clone, remove `~/.cache/vllm-upstream-tests`.

    ```bash
    rm -rf ~/.cache/vllm-upstream-tests
    ```

#### Configuration

**SKIP_UPSTREAM_TESTS**: Skip upstream tests entirely. Accepts `1`, `true`, or `yes`.

**VLLM_COMMIT**: Override the vLLM commit SHA from `pyproject.toml`.

**VLLM_REPO_URL**: Override the vLLM repository URL. Defaults to `https://github.com/vllm-project/vllm`.

**UPSTREAM_TESTS_PATHS**: Not currently consumed by the plugin — the set of upstream test paths is auto-derived from the `rel_path` entries in `tests/plugin/spyre_testing_plugin/upstream_tests.yaml`.

!!! tip
    Environment variables can be passed directly to the `pytest` command, e.g. `VLLM_COMMIT=abc123def456 pytest`.

### Docs

Install MkDocs along with the [plugins](https://github.com/torch-spyre/spyre-inference/blob/main/mkdocs.yaml) used in the Spyre Inference documentation.

```bash
uv pip install -r docs/requirements-docs.txt
```

!!! note
    Ensure that your Python version is compatible with the plugins (e.g., `mkdocs-awesome-nav` requires Python 3.10+)

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. Make sure you're in the same directory as the `mkdocs.yaml` configuration file and run:

```bash
mkdocs serve
```

Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to see a live preview. For additional features and advanced configurations, refer to the official [MkDocs Documentation](https://www.mkdocs.org/).

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/torch-spyre/spyre-inference/issues?q=is%3Aissue) first to see if it has already been reported. If not, please create a new issue, by using our [issue templates](https://github.com/torch-spyre/spyre-inference/issues/new/choose):

- **🐛 Bug Report**: For reporting bugs and unexpected behavior
- **🚀 Feature Request**: For suggesting new features or improvements

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Pull Requests

### DCO and Signed-off-by

When contributing, you must agree to the <gh-file:DCO>. Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [torch-spyre Documentation](https://github.com/torch-spyre/torch-spyre)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [uv Documentation](https://docs.astral.sh/uv/)

## License

See <gh-file:LICENSE>.
