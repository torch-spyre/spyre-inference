
# Continuous Integration

We use GitHub Actions workflows as our CI/CD framework. The workflows are located in the `.github/workflows` folder. The other folders in the `.github` folder are related to CI.

## Workflows

Located in `.github/workflows/`

- We run tests on each commit to pushed a PR branch using [test_each_commit.yaml](.github/workflows/test_each_commit.yaml)

- There is a workflow to add or remove models, datasets, etc. to/from the cache [add_or_remove_artifacts_from_cache.yaml](.github/workflows/add_or_remove_artifacts_from_cache.yaml)

> NOTE: Unfortunately GitHub does not allow secrets in workflow runs triggered on pull requests from forks.
>
> <https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets#using-secrets-in-a-workflow>
>
> With the exception of `GITHUB_TOKEN`, secrets are not passed to the runner when a workflow is triggered from a forked repository.

So the workflow to manage the cache is required for running tests on models (e.g. `meta-llama/Meta-Llama-3-8B`) that requires a token to pull from HuggingFace Hub.

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
