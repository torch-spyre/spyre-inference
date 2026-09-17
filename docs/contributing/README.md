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

If you already have a local `torch-spyre` checkout installed (e.g. editable, for `torch-spyre` development), a plain `uv sync` will rebuild and reinstall the pinned git rev from `pyproject.toml`, discarding it. To keep your local install instead:

```bash
uv sync --group dev --no-install-package torch-spyre --inexact
```

`--no-install-package torch-spyre` skips resolving and rebuilding the pinned rev; `--inexact` stops uv from uninstalling the now-unreferenced local package. Note this only affects `uv sync` itself — any subsequent plain `uv run` will still re-sync and revert it, so use `uv run --no-sync …` for those (see `CLAUDE.md`'s "Iterating on a Local `torch-spyre` Checkout" section).

### One venv for both repos

To work on a `torch-spyre` bug and validate against both test suites, build on this repo's venv — it already carries the pinned `torch` and `vllm` — then add your `torch-spyre` checkout and its dev dependencies:

```bash
cd spyre-inference
uv sync --group dev --no-install-package torch-spyre --inexact
uv pip install -e ~/torch-spyre
uv pip install --group ~/torch-spyre/pyproject.toml:dev
```

`spyre-testing-plugin` is scoped to this repo — it is activated by an `addopts` entry in our `pyproject.toml` rather than a global `pytest11` entry point — so `pytest` inside `torch-spyre` behaves exactly as it does without spyre-inference installed. Run each suite from its own checkout.

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

Upstream vLLM tests are opt-in: they are cloned and collected only when the `-m` expression names the `upstream` marker, or `--upstream` is passed. A negative mention doesn't count, so `-m "not upstream"` and `-m "attention and not upstream"` skip the clone entirely.

```bash
# Run only local tests (no vLLM clone)
pytest

# Run all upstream tests
pytest -m upstream

# Run upstream attention tests only (see tests/plugin/spyre_testing_plugin/upstream_tests.yaml for markers on upstream tests)
pytest -m "attention and upstream"

# Run local AND upstream attention tests: --upstream adds them to a marker
# expression that doesn't name `upstream` itself
pytest --upstream -m "attention"
```

#### Model Output Quality Gate

The quality gate holds the product models to their output, and runs as one suite of two
kinds of case:

- **`model_quality`** — each product model is loaded **compiled** (the platform default) and
  compared against a CPU HF reference: greedy token ids and per-token probabilities for the
  decoders (`tests/e2e/test_model_quality.py`), cosine similarity for the embedding models
  and sigmoid scores plus document ranking for the cross-encoder rerankers
  (`tests/e2e/test_encoder_models.py`).
- **`gsm8k`** — server-based GSM8K accuracy evals (one per config in
  `tests/plugin/spyre_testing_plugin/gsm8k_configs/models-spyre.txt`), each starting a vLLM
  server and running a batched eval. These live in the upstream vLLM tree, so the gate's
  marker expression pulls the cached checkout in automatically (no `--upstream` flag).

```bash
make test-quality              # the whole gate, one card
make test-quality-shard-0      # one CI shard (QUALITY_SHARDS=8)
```

CI runs the gate as `QUALITY_SHARDS` parallel 1-card jobs, weighted by recorded runtime like
the smoke and attention suites. The slowest single case bounds the useful shard count, so
resize with the `rebalance-test-shards` skill rather than by raising it on a hunch.

The models are too large to run through transformers in CI, so the references are checked
into `tests/data/` and regenerated by hand where the weights are cached:

```bash
python tests/data/generate_decoder_output_refs.py --models ibm-granite/granite-4.1-8b
python tests/data/generate_encoder_embed_refs.py
python tests/data/generate_rerank_score_refs.py
```

Regenerate only when the *expected* output changes (a new model or prompt), never to make a
failing test pass — that is the regression the gate exists to catch. Prompt sets are per
model: see `MODEL_PROMPTS` and `MODEL_DOCUMENTS` in the generators.

Every gated model is pinned to a revision, in the generator's `MODEL_REVISIONS` and in
`.github/cache_config/hf_models_and_datasets.yaml`. Each generator records the revision it
measured in its JSON and the test loads that one back, so bumping a pin means regenerating
that model's reference.

`SPYRE_TEST_MEAN_ABS_TOL` / `SPYRE_TEST_ABS_TOL` / `SPYRE_TEST_REL_TOL` (decoder
probabilities) and `SPYRE_TEST_SCORE_ABS_TOL` / `SPYRE_TEST_SCORE_REL_TOL` (reranker scores)
set the tolerances. For a low-confidence reference the stricter of the absolute and relative
bound applies, so it is held to a fraction rather than to the same absolute margin. Reranker
ranking is checked separately from the per-score bound.

The decoder gate is aggregate-first: `SPYRE_TEST_MEAN_ABS_TOL` bounds each prompt's *mean*
error and is what holds quality, while `SPYRE_TEST_ABS_TOL` only caps a single step against
gross breakage. A reference near p=0.5 is maximally ill-conditioned (`dp/dlogit` peaks at
`p(1-p)`), and one compiled graph has measured 0.115 apart on such a step between two CI pods
with every token still exact — a tight per-step bound buys flakiness, not coverage. Each case
prints `mean=`/`max=` per prompt, so a failure is readable without a rerun.
`SPYRE_TEST_TIE_ABS_TOL` holds token disagreements to a tighter bound, since picking a
different token is a stronger signal than drift. The FP8 decoder checkpoints are
load-and-decode cases with no reference of their own — their unquantized siblings gate the
numerics.

A greedy path that diverges from HF on a near-tie cannot be compared past the split, so how
much of the reference a case compares depends on the prompts. That is **reported, not
asserted**: the tolerances above are the gate, and coverage is a separate signal, because
truncation is all-or-nothing per prompt and a floor cannot tell an unlucky prompt from a
regression. Each decoder case prints `compared <n>/<total> reference steps`, records it as a
`refcoverage__<n>/<total>` JUnit tag, and warns (`LowReferenceCoverage`) below
`COVERAGE_WARN_FRACTION`. A warning means the case gates less than it looks like it does and
its prompts want replacing — not that the model regressed. The one coverage failure is zero:
a case where every prompt diverged on its first step asserted nothing at all.

#### Upstream Test Integration

Upstream tests are cloned from the vLLM repository at the commit pinned in `pyproject.toml`, fetching only the `tests/` directory. The clone happens on demand, the first time a run asks for upstream tests (see the marker gate above). Cloned tests are cached in `~/.cache/vllm-upstream-tests` (or `$XDG_CACHE_HOME/vllm-upstream-tests`) with separate worktrees per commit, allowing multiple vLLM versions to be tested simultaneously. All upstream tests run with `VLLM_PLUGINS=spyre_inference,spyre_inference_ops` set automatically. Pointing the plugin at a vLLM checkout instead of the cache is the one case that still needs the flag by hand: `pytest -p spyre_testing_plugin.pytest_plugin -m upstream` from the checkout root. See `tests/plugin/spyre_testing_plugin/pytest_plugin.py` for implementation details.

!!! tip
    To force a re-clone, remove `~/.cache/vllm-upstream-tests`.

    ```bash
    rm -rf ~/.cache/vllm-upstream-tests
    ```

#### Configuration

**--upstream**: Collect upstream tests even when the `-m` expression doesn't name the `upstream` marker.

**SKIP_UPSTREAM_TESTS**: Skip upstream tests entirely, overriding both the `-m` expression and `--upstream`. Accepts `1`, `true`, or `yes`.

**VLLM_COMMIT**: Override the vLLM commit SHA from `pyproject.toml`.

**VLLM_REPO_URL**: Override the vLLM repository URL. Defaults to `https://github.com/vllm-project/vllm`.

**UPSTREAM_TESTS_PATHS**: Not currently consumed by the plugin — the set of upstream test paths is auto-derived from the `rel_path` entries in `tests/plugin/spyre_testing_plugin/upstream_tests.yaml`.

!!! tip
    Environment variables can be passed directly to the `pytest` command, e.g. `VLLM_COMMIT=abc123def456 pytest -m upstream`.

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
