# Thin wrappers over the spyre-bench CLI. Advisory only: nothing here gates a check.
# Never branches on architecture, card count or queue: those are variables the launcher and engine read.
# Without the engine (or a verb it does not have yet) a target prints a notice and exits 0, unless BENCH_REQUIRED is set.

BENCH             ?= spyre-bench
BENCH_DIR         ?= benchmarks
BENCH_CENTRAL     ?=
BENCH_BASE_DIR    ?=
BENCH_PREFIX      ?=
BENCH_REQUIRED    ?=
BENCH_PURPOSE     ?= on-demand
BENCH_ENVIRONMENT ?=
BENCH_SELECTION   ?=
BENCH_CASES       ?=
BENCH_TIER        ?=
BENCH_SUBJECT     ?=
BENCH_REUSE       ?= never
BENCH_PLAN        ?=
BENCH_SET         ?=
BENCH_INGEST      ?= 0
BENCH_ARCH        ?=
BENCH_CARDS       ?=
BENCH_QUEUE       ?=
BENCH_RESULTS_DIR ?= results/bench

export BENCH_PURPOSE BENCH_ENVIRONMENT BENCH_SELECTION BENCH_CASES BENCH_TIER BENCH_SUBJECT BENCH_REUSE \
	BENCH_PLAN BENCH_SET BENCH_INGEST BENCH_ARCH BENCH_CARDS BENCH_QUEUE

_bench_catalog = --catalog '$(BENCH_DIR)'$(if $(BENCH_CENTRAL), --central '$(BENCH_CENTRAL)')

# $(call _bench_exec,<verb>,<args>[,<shell prefix>[,<setup command ending in &&>]]): runs the verb if the engine has it.
define _bench_exec
@if command -v $(firstword $(BENCH)) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then \
	$(4) $(3) $(BENCH) $(1) $(2); \
else \
	echo "bench: skipped '$(1)': $(firstword $(BENCH)) is not installed or does not provide this verb yet. Set BENCH_REQUIRED=1 to fail instead." >&2; \
	[ -z "$(BENCH_REQUIRED)" ] || exit 2; \
fi
endef

# An unset selection must never fall back to every suite.
_bench_need_selection = @[ -n "$(BENCH_SELECTION)$(BENCH_CASES)$(CASE)" ] || { echo "set BENCH_SELECTION=<name> or BENCH_CASES/CASE=<case>" >&2; exit 2; }
_bench_select = $(if $(BENCH_SELECTION), --selection '$(BENCH_SELECTION)')$(if $(or $(BENCH_CASES),$(CASE)), --cases '$(or $(BENCH_CASES),$(CASE))')

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-smoke bench-report

bench-validate: ## Validate benchmarks/ (no hardware, no network); BENCH_BASE_DIR=<tree at the merge base> adds the identity diff.
	$(call _bench_exec,validate,'$(BENCH_DIR)'$(if $(BENCH_CENTRAL), --central '$(BENCH_CENTRAL)')$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-list: ## List benchmark cases and selections.
	$(call _bench_exec,list,$(_bench_catalog)$(if $(BENCH_SELECTION), --selection '$(BENCH_SELECTION)'))

bench-explain: ## Resolve one case: ids, environment request, policy. Needs CASE=<case name>.
	@test -n "$(CASE)" || { echo "CASE=<case name> is required" >&2; exit 2; }
	$(call _bench_exec,explain,'$(CASE)' $(_bench_catalog)$(if $(BENCH_ENVIRONMENT), --environment '$(BENCH_ENVIRONMENT)') --purpose '$(BENCH_PURPOSE)')

bench-plan: ## Plan only (no hardware): writes BENCH_RESULTS_DIR/plan.json. Needs BENCH_SELECTION or CASE.
	$(_bench_need_selection)
	$(call _bench_exec,plan,$(_bench_catalog)$(_bench_select) --out '$(BENCH_RESULTS_DIR)/plan.json',,mkdir -p '$(BENCH_RESULTS_DIR)' &&)

bench-run: ## Plan (or BENCH_PLAN=plan.json), run on the local devices, write the Result. Needs BENCH_SELECTION or CASE.
	$(_bench_need_selection)
	$(call _bench_exec,run,$(_bench_catalog)$(_bench_select) --out '$(BENCH_RESULTS_DIR)',$(BENCH_PREFIX),mkdir -p '$(BENCH_RESULTS_DIR)' &&)

bench-smoke: ## One case from the smoke selection, one repeat, never ingested.
	$(MAKE) bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_PURPOSE=on-demand BENCH_REUSE=never \
		BENCH_SET=run.repeats=1 BENCH_INGEST=0

bench-report: ## Summarise BENCH_RESULTS_DIR.
	$(call _bench_exec,report,'$(BENCH_RESULTS_DIR)')
