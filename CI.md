
# Continuous Integration

We use GitHub Actions workflows as our CI/CD framework. The workflows are located in the `.github/workflows` folder. The other folders in the `.github` folder are related to CI.

## Workflows

Located in `.github/workflows/`

- We run tests on each commit to pushed a PR branch using [test_each_commit.yaml](.github/workflows/test_each_commit.yaml)

- There is a workflow to add or remove models, datasets, etc. to/from the cache [add_or_remove_artifacts_from_cache.yaml](.github/workflows/add_or_remove_artifacts_from_cache.yaml)

- A weekday cron [pr_reminder_slack.yaml](.github/workflows/pr_reminder_slack.yaml) posts the stalest open PRs to Slack to keep reviews moving. Setup, configuration variables and the failure-escalation behaviour are documented in [scripts/pr_reminder/README.md](scripts/pr_reminder/README.md)

- Changes to `pyproject.toml`, `uv.lock`, or `spyre-rpms.lock` ask for confirmation from the PR author or a repository maintainer with `maintain` or `admin` permission. [environment-file-confirmation.yml](.github/workflows/environment-file-confirmation.yml) posts one commit-specific prompt and reports the `Environment file confirmation` check, which passes when no open PR sharing that head commit and base branch has unconfirmed environment-file changes. The check blocks merging only once it is added as a required GitHub Actions check to the `main` branch ruleset; until then it is advisory.

> NOTE: Unfortunately GitHub does not allow secrets in workflow runs triggered on pull requests from forks.
>
> <https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets#using-secrets-in-a-workflow>
>
> With the exception of `GITHUB_TOKEN`, secrets are not passed to the runner when a workflow is triggered from a forked repository.

So the workflow to manage the cache is required for running tests on models (e.g. `meta-llama/Meta-Llama-3-8B`) that requires a token to pull from HuggingFace Hub.

## Compile cache

Each test job seeds its `TORCHINDUCTOR_CACHE_DIR` from `/storage-1/spyre-inference/.cache/compile` and runs with `SPYRE_KERNEL_CACHE=1`, so it reuses the Inductor graphs and Spyre kernels earlier jobs compiled. A green job writes only what it compiled as its own tar; restore extracts them all. The cache is keyed on `uv.lock` and `spyre-rpms.lock`, and scoped like the uv cache: pushes to `main` and the merge queue write `main`, and a PR writes `pr-<N>` on top of it. The merge queue writes because its commit is exactly what lands on `main`: without it, a lockfile change compiles cold in the queue and again on the push. Runs that override torch-spyre or the RPMs skip it. See [nfs_compile_cache.py](.github/scripts/nfs_compile_cache.py).

Any change to either lockfile starts a cold key, so a PR that bumps a dependency runs fully cold, and PRs rebased onto a bump stay cold until the bump's merge-queue run fills the new key. Most lockfile changes are torch-spyre bumps, which have to recompile anyway. The cache never falls back to an older key: Inductor's graph key does not cover all of torch-spyre's lowering code, so an entry from before a torch-spyre bump could run stale generated code.

Because CI always sets `SPYRE_KERNEL_CACHE=1`, the default user path with the kernel cache off only runs in jobs that skip the cache (torch-spyre or RPM overrides, prebaked images) and in `tests/e2e/test_kernel_cache.py`.

The nightly [compile_cache_cleanup.yaml](.github/workflows/compile_cache_cleanup.yaml) keeps the two newest keys per scope and drops PR scopes untouched for 14 days. A delta that fails to extract is renamed to `.bad-delta-*.tar` and skipped.

The cache holds compiled code that test jobs load and run. To drop it, bump `KEY_SALT` in that script or delete the directory.

It also ships Inductor's CPU C++ kernels (about 150 `.so` files, under 10 MB, but several minutes per job to rebuild). Inductor builds them with `-march=native` and keys them on the vector ISA class (AVX2, AVX-512, ...) only, so a kernel built on a newer runner CPU could be loaded on an older one. If a test job dies with `SIGILL` (illegal instruction) in a `.so` under the Inductor cache dir, suspect this first. To confirm, rerun the job with the cache skipped. To fix, exclude `*.so` in `_shippable`, or pin `torch._inductor.config.cpp.march` in CI so the build and the key agree.

## Runners

We use self-hosted runners on an Openshift cluster for the workflows that require access to Spyre cards.

The runner sets are managed by the official GitHub ARC operator <https://github.com/actions/actions-runner-controller>

A runner set refers to a specific pod spec to use for running the workflow. It's a combination of hardware (CPU architecture, storage, Spyre cards, etc.) and software (OS, container image, etc.)

The runner set is selected using a set of labels in the `runs-on` field in each workflow

### Example

```yaml
    runs-on:
      - x86_64 # Intel CPU Architecture
      - spyre_pf_x1 # 1 Spyre card in PF mode
      - linux # Linux OS - Usually RHEL 9/10
      - image_torch_spyre # The torch-spyre container image
```

The workflow will only get picked up by a runner set that matches ALL of the labels specified in the `runs-on` field.

For workflows that require a different number of Spyre cards we also have:

```yaml
- spyre_pf_x0 # No Spyre cards at all
- spyre_pf_x1
- spyre_pf_x2
- spyre_pf_x4
- spyre_pf_x8
- spyre_pf_x12
```
