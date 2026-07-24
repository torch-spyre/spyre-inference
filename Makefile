SHELL := /bin/bash
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of tests to run (uniform knob across the
# product repos: torch-spyre, hf-adapters, spyre-inference):
#   smoke — fast per-op unit tests only
#   core  — all spyre-native tests (per-op + attention + distributed);
#           excludes the heavy upstream-vLLM suites
#   full  — everything (default)
# Empty / unset defaults to "full".
TEST_TYPE ?= full

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

# Map TEST_TYPE to a pytest -m marker expression. full -> no filter (all tests).
# MARK_OVERRIDE bypasses TEST_TYPE entirely for callers that need a marker
# expression finer than the 3 coarse tiers (e.g. CI splitting the "full"-only
# upstream suites into separate parallel jobs) -- set MARK_OVERRIDE and the
# TEST_TYPE mapping below is skipped.
ifneq ($(MARK_OVERRIDE),)
MARK_EXPR := -m "$(MARK_OVERRIDE)"
else ifeq ($(TEST_TYPE),full)
MARK_EXPR :=
else ifeq ($(TEST_TYPE),smoke)
MARK_EXPR := -m "not (distributed or upstream or attention)"
else ifeq ($(TEST_TYPE),core)
MARK_EXPR := -m "not upstream"
else
$(error Invalid TEST_TYPE '$(TEST_TYPE)'. Valid values: smoke | core | full)
endif

# Root all-suite JUnit output under one directory so a caller can glob it in
# one shot (ingest_xml.py globs `${RESULTS_DIR}/*.xml` non-recursively).
RESULTS_DIR ?= .

.PHONY: help test tests run-one aiu-setup perf-tests \
        test-smoke test-attention test-distributed \
        test-upstream test-upstream-distributed test-upstream-model

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=smoke|core|full (default full), MARK_OVERRIDE (raw -m expr, bypasses TEST_TYPE),"
	@echo "  PYTEST_ARGS (default '$(PYTEST_ARGS)'), JUNIT_XML (single-run path; unset = no JUnit file),"
	@echo "  RESULTS_DIR (aggregate 'full' JUnit output dir, default '$(RESULTS_DIR)')"

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

run-one: ## Internal: one pytest invocation for the resolved MARK_EXPR/JUNIT_ARGS.
	# ibm-aiu-setup.sh ends with a chmod of root-owned /tmp/etc that fails on
	# the Spyre image; env vars are already exported by then, so tolerate
	# that failure (handled by AIU_SETUP_CMD's set +e/-e wrap).
	$(AIU_SETUP_CMD); \
	echo "Running tests for TEST_TYPE=$(TEST_TYPE) MARK_OVERRIDE=$(MARK_OVERRIDE)..."; \
	uv run pytest $(PYTEST_ARGS) $(MARK_EXPR) $(JUNIT_ARGS)

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

# When MARK_OVERRIDE is unset and TEST_TYPE=full, GHA's _test_matrix.yaml runs
# this as 6 separate marker-combo jobs, not one unfiltered run -- mirror that
# here so `make test TEST_TYPE=full` is GHA-parity, one flat JUnit file per
# combo in RESULTS_DIR, same convention hf-adapters' Makefile uses.
tests: ## Run tests. TEST_TYPE=smoke|core|full (default full) or set MARK_OVERRIDE directly.
	if [ -n "$(MARK_OVERRIDE)" ] || [ "$(TEST_TYPE)" != "full" ]; then \
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

perf-tests: ## Run vLLM benchmark suite, writing JSON results under RESULTS_DIR.
	mkdir -p "$(RESULTS_DIR)"
	$(AIU_SETUP_CMD); \
	uv run python3 .github/scripts/run_vllm_benchmarks.py \
		--configs-dir vllm-benchmarks/benchmarks/spyre \
		--results-dir "$(RESULTS_DIR)"
