# Glue for the bench route: Jenkins runs it nightly and weekly, .github/workflows/bench-on-demand.yml on demand. Advisory only; `make tests TEST_TYPE=perf` is untouched.
# Included after mk/bench.mk, which supplies BENCH, _bench_exec and the BENCH_* request variables.

# native: perf-tests over the nightly suite files. engine: plan then `spyre-bench run`, whose per-case command is the same runner.
# stub: no cards, no benchmark. Both real runners leave the vLLM JSON in BENCH_NATIVE_DIR for the ingest step; the engine
# never writes ClickHouse rows.
BENCH_RUNNER       ?= native
# none until the vLLM path pins CPUs itself; the engine records it as run.exec.cpu_pinning.
BENCH_CPU_PINNING  ?= none
BENCH_NATIVE_DIR   ?= $(BENCH_RESULTS_DIR)/native
BENCH_LEG_MODEL    ?=
BENCH_LEG_TP       ?=
BENCH_VERDICT_RUN_ID ?=
# bench-ingest: what the GitHub composite action passes, for a caller without it (Jenkins). BENCH_STORED_RUN_ID is the
# orchestrator's uuid; the ingest uses it verbatim as the stored run_id.
BENCH_STORED_RUN_ID ?=
BENCH_WORKFLOW     ?= vLLM Benchmark
BENCH_BRANCH       ?=
BENCH_SHA          ?=
BENCH_STATE        ?= passed
BENCH_LEG_KEY      ?=
BENCH_REPOSITORY   ?=
BENCH_RPM_LOCK     ?= spyre-rpms.lock
# Interpreter for the case-list helper; the engine venv has PyYAML and is not the product venv.
BENCH_HELPER_PY    ?= python3

# The engine gives a command an allowlisted environment; these are what the vLLM runner needs on top of it.
BENCH_PASS_ENV     ?= HF_HOME,HF_HUB_OFFLINE,STORAGE_1_DIR,TOKENIZERS_PARALLELISM,SAVE_TO_PYTORCH_BENCHMARK_FORMAT,DTLOG_LEVEL,FLEX_COMPUTE,FLEX_DEVICE,AIU_SETUP_MULTI_AIU,VLLM_PLUGINS,DEVICE_NAME,DEVICE_TYPE,UV_NO_SYNC

export BENCH_CPU_PINNING BENCH_INVOKER BENCH_PASS_ENV

# The toolchain names the AIU setup exports are only known once it ran, so the list is completed inside the prefix.
BENCH_PREFIX += export BENCH_PASS_ENV="$$BENCH_PASS_ENV,$$(env | cut -d= -f1 | grep -E '^(AIU|FLEX|DT|SEN|COLL|TORCH|VLLM|OMP|UV|DEEPTOOLS|RUNTIME|LIB_VERSION|_IBM_AIU)' | grep -Ev 'TOKEN|PASS|SECRET|KEY|CRED' | paste -sd, -)";

_bench_leg_cases = $(BENCH_HELPER_PY) .github/scripts/bench_nightly_cases.py --model '$(BENCH_LEG_MODEL)' --tp '$(BENCH_LEG_TP)'

.PHONY: bench-datasets bench-legs bench-leg bench-native bench-collect bench-ingest bench-dry-run bench-verdict bench-conformance

bench-datasets: ## List each dataset's state (cached, absent, corrupt, preset) and the cases that read it; fetches nothing.
	@$(BENCH_HELPER_PY) .github/scripts/fetch_bench_datasets.py status

bench-legs: ## Print the nightly bench legs (model, TP, cards) as JSON. MODELS=<csv> narrows.
	@$(MAKE) --no-print-directory -s perf-legs BENCH_SUITE=nightly BENCH_TYPES=latency,throughput

bench-native: ## Run the leg's nightly entries with the vLLM bench runner; native JSON goes to BENCH_NATIVE_DIR.
	@test -n "$(BENCH_LEG_MODEL)" && test -n "$(BENCH_LEG_TP)" || { echo "set BENCH_LEG_MODEL and BENCH_LEG_TP" >&2; exit 2; }
	@test "$(BENCH_CPU_PINNING)" = none || { echo "the native runner cannot pin CPUs; use BENCH_RUNNER=engine" >&2; exit 2; }
	$(MAKE) --no-print-directory perf-tests BENCH_SUITE=nightly BENCH_TYPES=latency,throughput \
		MODELS='$(BENCH_LEG_MODEL)' TPS='$(BENCH_LEG_TP)' RESULTS_DIR='$(BENCH_NATIVE_DIR)'

bench-leg: ## One nightly leg with BENCH_RUNNER (native|engine|stub) for BENCH_LEG_MODEL and BENCH_LEG_TP.
	@case '$(BENCH_RUNNER)' in \
	native) $(MAKE) --no-print-directory bench-native ;; \
	engine) c="$$($(_bench_leg_cases))" || exit 2; $(MAKE) --no-print-directory bench-run BENCH_CASES="$$c"; rc=$$?; \
		$(MAKE) --no-print-directory bench-collect || [ $$rc -ne 0 ] || rc=1; exit $$rc ;; \
	stub) mkdir -p '$(BENCH_NATIVE_DIR)' && $(_bench_leg_cases) > '$(BENCH_NATIVE_DIR)/stub-cases.txt' && cat '$(BENCH_NATIVE_DIR)/stub-cases.txt' ;; \
	*) echo "BENCH_RUNNER must be native, engine or stub" >&2; exit 2 ;; \
	esac

bench-collect: ## Gather the engine's per-case vLLM JSON into BENCH_NATIVE_DIR, where the ingest step reads it.
	$(BENCH_HELPER_PY) .github/scripts/bench_nightly_collect.py --results-dir '$(BENCH_RESULTS_DIR)' --native-dir '$(BENCH_NATIVE_DIR)'

bench-ingest: ## Write BENCH_NATIVE_DIR with ingest_vllm_benchmarks.py, as the GitHub action does; CLICKHOUSE_* in the environment.
	@test -n "$(BENCH_STORED_RUN_ID)" -a -n "$(BENCH_BRANCH)" -a -n "$(BENCH_SHA)" || { echo "set BENCH_STORED_RUN_ID, BENCH_BRANCH and BENCH_SHA" >&2; exit 2; }
	$(BENCH_HELPER_PY) .github/scripts/ingest_vllm_benchmarks.py --results-dir '$(BENCH_NATIVE_DIR)' \
		--workflow '$(BENCH_WORKFLOW)' --branch '$(BENCH_BRANCH)' --sha '$(BENCH_SHA)' --gha-run-id '' --v2-run-id '$(BENCH_STORED_RUN_ID)' \
		--job-id 0 --pr-number 0 --arch '$(or $(BENCH_ARCH),x86_64)' --test-type perf --rpm-lock '$(BENCH_RPM_LOCK)' \
		--state '$(BENCH_STATE)' --leg-key '$(BENCH_LEG_KEY)' --repository '$(BENCH_REPOSITORY)' \
		--purpose '$(filter nightly weekly on-demand,$(BENCH_PURPOSE))'

bench-dry-run: ## Render the plan's commands without running them (needs bench-plan first).
	$(call _bench_exec,run,--plan '$(BENCH_RESULTS_DIR)/plan.json' --out '$(BENCH_RESULTS_DIR)' --dry-run)

bench-verdict: ## Rolling-baseline verdict (words and counts only) of BENCH_VERDICT_RUN_ID; a notice when the engine lacks the verb.
	@test -n "$(BENCH_VERDICT_RUN_ID)" || { echo "set BENCH_VERDICT_RUN_ID" >&2; exit 2; }
	$(call _bench_exec,verdict,--run-id '$(BENCH_VERDICT_RUN_ID)' --store clickhouse --quiet)

bench-conformance: ## Check the definitions against the real ingest producer (needs spyre-bench and the ingest package).
	python3 -m pytest -q -p no:cacheprovider -o addopts="" .github/scripts/test_bench_nightly.py
