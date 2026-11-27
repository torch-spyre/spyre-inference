# Thin wrappers over the spyre-bench CLI, adapted from torch-spyre's mk/bench.mk so both products
# are driven the same way. Settings are BENCH_* variables the CLI also reads from the environment.
# Advisory only: nothing here replaces `make perf-tests`. benchmarks/ is GENERATED from
# vllm-benchmarks/, which stays the single source of truth, and every case runs through
# `bench-case` below -- so the runner's path is written down once, here.
#   make bench-generate | bench-check-catalog | bench-validate | bench-list
#        bench-explain CASE=... | bench-check-ids | bench-plan | bench-run | bench-collect
#        bench-verdict | bench-smoke | bench-case CASE=...

BENCH ?= spyre-bench
BENCH_DIR ?= benchmarks
BENCH_CENTRAL ?=
BENCH_REQUIRED ?=
BENCH_OPTIONAL ?=
BENCH_PURPOSE ?= on-demand
BENCH_REUSE ?= never
BENCH_INGEST ?= 0
BENCH_SELECTION ?=
BENCH_CASES ?=
BENCH_TIER ?=
BENCH_SUITES ?=
BENCH_SUBJECT_ID ?=
BENCH_PLAN ?=
BENCH_SET ?=
BENCH_RUN_ID ?=
BENCH_PREFIX ?=
BENCH_QUIET ?= $(if $(CI),1,0)
BENCH_ENVIRONMENT ?=
BENCH_RESULTS_DIR ?= bench-results
BENCH_NATIVE_DIR ?= $(BENCH_RESULTS_DIR)/native
BENCH_PLAN_FILE ?= $(BENCH_RESULTS_DIR)/plan/plan.json
CASE ?=

export BENCH_PURPOSE BENCH_ENVIRONMENT BENCH_SELECTION BENCH_CASES BENCH_TIER BENCH_SUITES \
	BENCH_SUBJECT_ID BENCH_REUSE BENCH_PLAN BENCH_SET BENCH_INGEST BENCH_RUN_ID BENCH_QUIET

_bench_install = uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@main\#subdirectory=extensions/spyre-bench'

# The traces, models and card-count Environments the catalog refers to ship inside the engine, so a
# plain `make bench-validate` resolves with no extra arguments. `--central` is repeatable and
# BENCH_CENTRAL arrives on the command line from a CI leg, so the two COMPOSE: dropping the shared
# one when a leg supplies its own would leave every dataset reference dangling.
BENCH_SHARED_CATALOG ?= $(shell $(BENCH) catalogs --path llm 2>/dev/null)
_bench_central = $(if $(BENCH_SHARED_CATALOG),--central '$(BENCH_SHARED_CATALOG)') \
	$(if $(BENCH_CENTRAL),--central '$(BENCH_CENTRAL)')

# Without a shared catalog, every Model, Dataset and Environment this catalog names is
# unresolvable and the engine reports one error per reference -- twenty "no Model of that name"
# lines whose actual cause is that the installed engine has no `catalogs` verb. Say the cause once
# instead. An operator supplying BENCH_CENTRAL has provided the resources themselves, so the check
# only fires when both are empty.
_bench_need_central = \
	if [ -z "$(strip $(BENCH_SHARED_CATALOG))" ] && [ -z "$(strip $(BENCH_CENTRAL))" ]; then \
		echo "bench: no shared catalog, so every Model/Dataset/Environment reference would dangle." >&2; \
		echo "       '$(BENCH) catalogs --path llm' returned nothing: the installed engine predates" >&2; \
		echo "       spyre_bench/catalogs/. Install an engine that ships it:" >&2; \
		echo "         $(_bench_install)" >&2; \
		echo "       or pass BENCH_CENTRAL=<dir> holding the shared resources." >&2; \
		exit 2; \
	fi

_bench_soft = $(if $(BENCH_REQUIRED),,$(BENCH_OPTIONAL))
# $(call _bench_if,VERB,COMMAND): run COMMAND if the engine has VERB, else say how to install it and
# exit 2 -- or exit 0 in soft mode, for a caller that must not fail without the engine.
_bench_if = if command -v $(firstword $(BENCH)) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then \
		$(2); \
	else \
		echo "bench: '$(1)' skipped: '$(BENCH)' is not installed or has no such verb. Install spyre-bench: $(_bench_install)" >&2; \
		test -n "$(_bench_soft)" || { echo "bench: set BENCH_OPTIONAL=1 to skip silently when the engine is absent" >&2; exit 2; }; \
	fi
_bench_plan_cmd = $(BENCH) plan --catalog '$(BENCH_DIR)' $(_bench_central) --out '$(BENCH_PLAN_FILE)'

.PHONY: bench-validate bench-list bench-explain bench-check-ids bench-plan bench-run bench-collect bench-generate bench-check-catalog bench-case \
	bench-verdict bench-smoke

bench-validate: ## Validate benchmarks/ against the schemas, packs and cross-resource rules. Hardware-free.
	@$(_bench_need_central)
	@$(call _bench_if,validate,$(BENCH) validate '$(BENCH_DIR)' $(_bench_central))

bench-list: ## List the cases and Selections of benchmarks/. Hardware-free.
	@$(_bench_need_central)
	@$(call _bench_if,ids,$(BENCH) ids --catalog '$(BENCH_DIR)' $(_bench_central))

bench-explain: ## Show one case with every reference resolved: make bench-explain CASE=<name>.
	@test -n "$(CASE)" || { echo "usage: make bench-explain CASE=<case name>" >&2; exit 2; }
	@$(_bench_need_central)
	@$(call _bench_if,explain,$(BENCH) explain '$(CASE)' --catalog '$(BENCH_DIR)' $(_bench_central))

bench-check-ids: ## Check every case's identity block against the ingest's own derivation. Hardware-free.
	@python3 $(BENCH_DIR)/tools/check_identity_ids.py --bench '$(BENCH)' \
		--catalog '$(BENCH_DIR)' --configs-dir '$(BENCH_CONFIGS_DIR)'

bench-generate: ## Regenerate benchmarks/ from the configs in vllm-benchmarks/. Hardware-free.
	@python3 $(BENCH_DIR)/tools/generate_catalog.py --bench '$(BENCH)'

bench-check-catalog: ## Fail if benchmarks/ is stale against vllm-benchmarks/. Hardware-free.
	@python3 $(BENCH_DIR)/tools/generate_catalog.py --bench '$(BENCH)' --check

# The single place the benchmark runner's location is written down: every generated Benchmark's
# `run.command` calls this target, so moving the script is a one-line change here rather than an
# edit to twenty files.
bench-case: ## Run one benchmark by name through the benchmark runner (used by a case's run.command).
	@test -n "$(CASE)" || { echo "usage: make bench-case CASE=<test_name>" >&2; exit 2; }
	$(BENCH_PY) .github/scripts/run_vllm_benchmarks.py \
		--configs-dir "$(BENCH_CONFIGS_DIR)" \
		--bench-types "$(BENCH_TYPES)" \
		--tests "$(CASE)" \
		--results-dir "$(RESULTS_DIR)"

bench-plan: ## Write the plan for the BENCH_* request to BENCH_PLAN_FILE. Hardware-free.
	@$(_bench_need_central)
	@$(call _bench_if,plan,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd))

# The serve cases replay traces that are not in the repo. Resolving them is a step of its own, with
# .github/scripts/fetch_bench_datasets.py -- the same fetcher `perf-tests` uses -- because the
# engine deliberately withholds secret-looking variables from a benchmark's own command, so the
# store token exists here and not inside the run.
bench-run: ## Plan (unless BENCH_PLAN=plan.json), fetch the traces, then run on the devices the job holds.
	@$(_bench_need_central)
	@$(call _bench_if,run,\
		_bench_env="$$(mktemp)"; trap 'rm -f "$$_bench_env"' EXIT; \
		$(AIU_SETUP_CMD); \
		$(BENCH_DATA_CMD) && \
		$(if $(BENCH_PLAN),:,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd)) && \
		mkdir -p '$(BENCH_RESULTS_DIR)' && \
		$(BENCH_PREFIX) $(BENCH) run --catalog '$(BENCH_DIR)' $(_bench_central) \
			--plan '$(or $(BENCH_PLAN),$(BENCH_PLAN_FILE))' --out '$(BENCH_RESULTS_DIR)')

# The repeat-NNN level is kept: a repeated benchmark writes one `<test_name>.json` per repeat, and
# the ingest reads the repeat from that directory name.
bench-collect: ## Gather the run's per-case result files into BENCH_NATIVE_DIR for the ClickHouse ingest.
	@mkdir -p '$(BENCH_NATIVE_DIR)'
	@found=0; \
	for f in $(BENCH_RESULTS_DIR)/cases/*/raw/*.json $(BENCH_RESULTS_DIR)/cases/*/raw/repeat-*/*.json; do \
		[ -e "$$f" ] || continue; \
		rep=$$(basename "$$(dirname "$$f")"); \
		case "$$rep" in repeat-*) dest='$(BENCH_NATIVE_DIR)'/"$$rep";; *) dest='$(BENCH_NATIVE_DIR)';; esac; \
		mkdir -p "$$dest"; \
		cp "$$f" "$$dest/"; \
		found=$$((found+1)); \
	done; \
	echo "bench-collect: $$found result file(s) in $(BENCH_NATIVE_DIR)"; \
	test "$$found" -gt 0 || echo "bench-collect: nothing to ingest" >&2

bench-verdict: ## Print the stored verdict of the run (BENCH_RUN_ID, else the ids the run recorded).
	@$(call _bench_if,verdict,id='$(BENCH_RUN_ID)'; \
		[ -n "$$id" ] || id=$$(python3 -c "import json,sys;print((json.load(open('$(BENCH_RESULTS_DIR)/run.json')).get('stored_run_ids') or [''])[0])" 2>/dev/null); \
		test -n "$$id" || { echo "bench-verdict: no run id; pass BENCH_RUN_ID" >&2; exit 2; }; \
		$(BENCH) verdict --run-id "$$id")

bench-smoke: ## Selection 'smoke': one short case, one repeat, never ingested.
	@$(MAKE) --no-print-directory bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_TIER= \
		BENCH_PURPOSE=on-demand BENCH_REUSE=never BENCH_INGEST=0
