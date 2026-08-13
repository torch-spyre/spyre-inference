SHELL := /bin/bash
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of tests to run (uniform knob across the
# product repos: torch-spyre, hf-adapters, spyre-inference). These tier names
# are literal, first-class values -- there is no alias-resolution layer:
#   unit        — all spyre-native tests (per-op + attention + distributed);
#                 excludes the heavy upstream-vLLM suites
#   integration — the smoke suite: fast per-op unit tests only (this is the
#                 ONLY valid top-level tier for that suite -- TEST_TYPE=smoke
#                 by itself is rejected below)
#   regression  — everything
#   trunk       — same coverage as regression; push-to-main CI label
#   perf        — vLLM benchmark suite (perf-tests target), not a pytest
#                 marker subset
# Empty / unset defaults to "regression". This Makefile is the single source
# of truth for the valid TEST_TYPE set and its validation -- _test_matrix.yaml
# resolves TEST_TYPE via `make -s print-test-type TEST_TYPE=...` instead of
# duplicating this logic in YAML.
TEST_TYPE ?= regression

empty :=
space := $(empty) $(empty)
VALID_TEST_TYPES := unit integration regression trunk perf
VALID_TEST_TYPES_DISPLAY := $(subst $(space), | ,$(VALID_TEST_TYPES))

# "smoke" is the internal marker-combo this Makefile maps "integration" to
# (see MARK_EXPR below) -- not itself a valid top-level tier, so a caller
# reaching for the old name gets pointed at the replacement instead of the
# generic "Invalid TEST_TYPE" error below.
ifeq ($(strip $(TEST_TYPE)),smoke)
$(info ::error::TEST_TYPE=smoke is not a valid tier -- use TEST_TYPE=integration to run that suite. Valid: $(VALID_TEST_TYPES_DISPLAY))
$(error TEST_TYPE=smoke rejected)
endif

ifeq ($(filter $(TEST_TYPE),$(VALID_TEST_TYPES)),)
$(info ::error::Invalid test_type '$(TEST_TYPE)'. Valid: $(VALID_TEST_TYPES_DISPLAY))
$(error Invalid TEST_TYPE '$(TEST_TYPE)')
endif

# Flags passed verbatim to pytest. Mirrors the CI invocation so `make test`
# reproduces CI verbosity; override e.g. `make test PYTEST_ARGS="-x -q"`.
PYTEST_ARGS ?= -s -vvv

# When set, write JUnit XML here (CI callers set this to collect results
# for artifact upload / result ingestion). Unset = no JUnit file.
JUNIT_XML ?=
ifneq ($(JUNIT_XML),)
JUNIT_ARGS := --junitxml=$(JUNIT_XML)
else
JUNIT_ARGS :=
endif

# Map TEST_TYPE to a pytest -m marker expression. regression -> no filter
# (all tests). MARK_OVERRIDE bypasses TEST_TYPE entirely for callers that
# need a marker expression finer than the 3 coarse tiers (e.g. CI splitting
# the "regression"-only upstream suites into separate parallel jobs) -- set
# MARK_OVERRIDE and the TEST_TYPE mapping below is skipped.
# perf is NOT a pytest marker subset: it is a benchmark mode of `make tests`
# that shells out to the vLLM benchmark suite (perf-tests target) instead of
# pytest, so it has no MARK_EXPR. It is accepted here (not rejected) and the
# `tests` target routes it to perf-tests. This keeps perf on the SAME
# `make tests TEST_TYPE=...` entry point as the sibling repos (torch-spyre,
# hf-adapters), so CI drives every suite through one knob.
ifneq ($(MARK_OVERRIDE),)
MARK_EXPR := -m "$(MARK_OVERRIDE)"
else ifeq ($(TEST_TYPE),regression)
MARK_EXPR :=
else ifeq ($(TEST_TYPE),trunk)
MARK_EXPR :=
else ifeq ($(TEST_TYPE),perf)
MARK_EXPR :=
else ifeq ($(TEST_TYPE),integration)
MARK_EXPR := -m "not (distributed or upstream or attention)"
else ifeq ($(TEST_TYPE),unit)
MARK_EXPR := -m "not upstream"
else
# The validation above already rejected any type outside VALID_TEST_TYPES, so
# a value that reaches here IS valid but has no marker mapping above -- i.e. a
# new type was added to VALID_TEST_TYPES without a case here. Point at the fix.
$(error TEST_TYPE '$(TEST_TYPE)' has no pytest marker mapping in this Makefile; add a case above)
endif

# Root all-suite JUnit output under one directory so a caller can glob it in
# one shot (ingest_xml.py globs `${RESULTS_DIR}/*.xml` non-recursively).
RESULTS_DIR ?= .

.PHONY: help test tests run-one aiu-setup perf-tests print-test-type \
        test-smoke test-attention test-distributed \
        test-upstream test-upstream-distributed test-upstream-model

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=unit|integration|regression|trunk|perf (default regression), MARK_OVERRIDE (raw -m expr, bypasses TEST_TYPE),"
	@echo "  PYTEST_ARGS (default '$(PYTEST_ARGS)'), JUNIT_XML (single-run path; unset = no JUnit file),"
	@echo "  RESULTS_DIR (aggregate JUnit output dir for TEST_TYPE=regression/trunk, default '$(RESULTS_DIR)')"

print-test-type: ## Internal: print the resolved/validated TEST_TYPE. Lets CI (_test_matrix.yaml) resolve TEST_TYPE via `make -s print-test-type TEST_TYPE=...` without duplicating this Makefile's validation logic.
	@echo "$(TEST_TYPE)"

# Marker set for GHA's _test_matrix.yaml is intentionally NOT duplicated in
# YAML: each matrix.cfg's "Run tests" step calls one of the test-<name>
# targets below by name, so this Makefile is the sole owner of the marker
# strings. Add/change a combo here only.

# The env sourcing itself must happen in every recipe shell (env vars can't
# persist across separate make/shell processes), but ibm-aiu-setup.sh's
# one-time host-level side effect (topo.json reset) only needs to happen once
# per `make test` invocation even though `test` fans out into 6 separate
# run-one sub-makes -- gated on a stamp file so repeat sub-makes skip it.
# define (not a target body) so run-one/perf-tests can inline the exact same
# setup commands via $(AIU_SETUP_CMD) without re-declaring them.
AIU_SETUP_STAMP := /tmp/.spyre-inference-aiu-setup-done
define AIU_SETUP_CMD
if [ ! -f "$(AIU_SETUP_STAMP)" ]; then rm -f /tmp/etc/ibm/spyre/topo.json; touch "$(AIU_SETUP_STAMP)"; fi; \
unset _IBM_AIU_SETUP; \
set +e; \
source "$$HOME/.bashrc"; \
source /etc/profile.d/ibm-aiu-setup.sh; \
set -e
endef

aiu-setup: ## Internal: source ibm-aiu-setup.sh and run its one-time side effects (memoized via a stamp file for this run).
	$(AIU_SETUP_CMD)

# uv invocations below pass --active --no-sync: they must use the prebaked image venv
# ($VIRTUAL_ENV) and skip re-resolution, since the lockfile pins wheels (torch +cpu,
# bitsandbytes) that have no ppc64le build even though the venv is already complete.
run-one: ## Internal: one pytest invocation for the resolved MARK_EXPR/JUNIT_ARGS.
	# ibm-aiu-setup.sh ends with a chmod of root-owned /tmp/etc that fails on
	# the Spyre image; env vars are already exported by then, so tolerate
	# that failure (handled by AIU_SETUP_CMD's set +e/-e wrap).
	$(AIU_SETUP_CMD); \
	echo "Running tests for TEST_TYPE=$(TEST_TYPE) MARK_OVERRIDE=$(MARK_OVERRIDE)..."; \
	uv run --active --no-sync pytest $(PYTEST_ARGS) $(MARK_EXPR) $(JUNIT_ARGS)

test-smoke: ## Run the smoke marker combo (non-distributed, non-upstream, non-attention).
	$(MAKE) run-one MARK_OVERRIDE='not (distributed or upstream or attention)' JUNIT_XML=$(JUNIT_XML)

test-attention: ## Run the attention-only marker combo.
	$(MAKE) run-one MARK_OVERRIDE='attention and not (distributed or upstream)' JUNIT_XML=$(JUNIT_XML)

test-distributed: ## Run the distributed marker combo.
	$(MAKE) run-one MARK_OVERRIDE='distributed and not upstream' JUNIT_XML=$(JUNIT_XML)

test-upstream: ## Run the upstream (non-distributed, non-model) marker combo.
	$(MAKE) run-one MARK_OVERRIDE='upstream and not distributed and not model' JUNIT_XML=$(JUNIT_XML)

test-upstream-distributed: ## Run the upstream+distributed marker combo.
	$(MAKE) run-one MARK_OVERRIDE='upstream and distributed' JUNIT_XML=$(JUNIT_XML)

test-upstream-model: ## Run the upstream+model (non-distributed) marker combo.
	$(MAKE) run-one MARK_OVERRIDE='upstream and model and not distributed' JUNIT_XML=$(JUNIT_XML)

# When MARK_OVERRIDE is unset and TEST_TYPE=regression (or trunk, same
# coverage), GHA's _test_matrix.yaml runs this as 6 separate marker-combo
# jobs, not one unfiltered run -- mirror that here so
# `make test TEST_TYPE=regression` is GHA-parity, one flat JUnit file per
# combo in RESULTS_DIR, same convention hf-adapters' Makefile uses.
tests: ## Run tests. TEST_TYPE=unit|integration|regression|trunk|perf (default regression) or set MARK_OVERRIDE directly.
	if [ "$(TEST_TYPE)" = "perf" ]; then \
	  $(MAKE) perf-tests RESULTS_DIR="$(RESULTS_DIR)"; \
	elif [ -n "$(MARK_OVERRIDE)" ] || { [ "$(TEST_TYPE)" != "regression" ] && [ "$(TEST_TYPE)" != "trunk" ]; }; then \
	  $(MAKE) run-one JUNIT_XML=$(JUNIT_XML); \
	else \
	  mkdir -p "$(RESULTS_DIR)"; \
	  rc=0; \
	  $(MAKE) test-smoke JUNIT_XML="$(RESULTS_DIR)/smoke.xml" || rc=1; \
	  $(MAKE) test-attention JUNIT_XML="$(RESULTS_DIR)/attention.xml" || rc=1; \
	  $(MAKE) test-distributed JUNIT_XML="$(RESULTS_DIR)/distributed.xml" || rc=1; \
	  $(MAKE) test-upstream JUNIT_XML="$(RESULTS_DIR)/upstream.xml" || rc=1; \
	  $(MAKE) test-upstream-distributed JUNIT_XML="$(RESULTS_DIR)/upstream-distributed.xml" || rc=1; \
	  $(MAKE) test-upstream-model JUNIT_XML="$(RESULTS_DIR)/upstream-model.xml" || rc=1; \
	  exit $$rc; \
	fi

test: tests  ## Alias for `tests`, matching torch-spyre's Makefile target name.

# On some arches (notably s390x) `uv run` refuses to reuse the prebaked image
# venv: it re-resolves the project, cannot find an s390x torch/vllm wheel
# (torch==2.11.0 publishes no s390x wheel), and builds a fresh workspace .venv
# WITHOUT torch, so every benchmark then dies with "No module named 'torch'".
# No combination of --active/--no-sync/--frozen/--inexact/--no-project avoids
# this. Set SKIP_UV_FOR_BENCHMARKING=1 to bypass uv entirely and invoke the
# already-activated venv's python3 directly (the setup sourced above exports
# $VIRTUAL_ENV, so plain python3 is the baked interpreter). Empty/unset keeps
# the uv path, correct on arches with a resolvable lockfile (amd64, ppc64le).
SKIP_UV_FOR_BENCHMARKING ?=
ifeq ($(strip $(SKIP_UV_FOR_BENCHMARKING)),)
BENCH_PY := uv run --active --no-sync python3
else
BENCH_PY := python3
endif

perf-tests: ## Run vLLM benchmark suite, writing JSON results under RESULTS_DIR. Set SKIP_UV_FOR_BENCHMARKING=1 to bypass uv and use the active venv's python3 directly (needed on s390x).
	mkdir -p "$(RESULTS_DIR)"
	$(AIU_SETUP_CMD); \
	$(BENCH_PY) .github/scripts/run_vllm_benchmarks.py \
		--configs-dir vllm-benchmarks/benchmarks/spyre \
		--results-dir "$(RESULTS_DIR)"
