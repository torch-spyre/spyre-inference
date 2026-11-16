# Bench quality: record the scores the accuracy gate already measures. Advisory only; `make test-quality` (the pytest
# gate) is untouched and keeps asserting its own threshold. Included after mk/bench.mk and mk/bench-nightly.mk.
#
# Needs the gate's `pytest -s` output of the run being recorded in BENCH_QUALITY_LOG (CI: PYTEST_ARGS='-s -vvv' piped to
# tee); a stale log is not detected. Starts no server and uses no card. The architecture only selects a profile file.

BENCH_QUALITY_LOG             ?= $(abspath $(BENCH_RESULTS_DIR)/quality-gate.log)
BENCH_QUALITY_ARCH            ?= $(shell uname -m)
BENCH_QUALITY_PROFILES        ?= benchmarks/quality/profiles
BENCH_QUALITY_HARNESS_VERSION ?= unknown

export BENCH_QUALITY_LOG BENCH_QUALITY_HARNESS_VERSION

_bench_quality_cases = $(BENCH_HELPER_PY) .github/scripts/bench_quality_gate.py list --arch '$(BENCH_QUALITY_ARCH)' --profiles '$(BENCH_QUALITY_PROFILES)'

.PHONY: bench-quality bench-quality-cases

bench-quality-cases: ## The quality cases the BENCH_QUALITY_ARCH profile enables, comma separated.
	@$(_bench_quality_cases)

bench-quality: ## Record the quality gate's scores (BENCH_QUALITY_LOG) as Bench quality results. BENCH_INGEST=1 ingests them (CLICKHOUSE_* set).
	@test -f '$(BENCH_QUALITY_LOG)' || { echo "no gate log at $(BENCH_QUALITY_LOG): run the quality gate with pytest -s and tee its output there" >&2; exit 2; }
	@c="$$($(_bench_quality_cases) --log '$(BENCH_QUALITY_LOG)')" || exit 2; \
	$(MAKE) --no-print-directory bench-run BENCH_CASES="$$c" BENCH_TYPE=quality BENCH_PREFIX= BENCH_REUSE=never \
		BENCH_PASS_ENV='$(BENCH_PASS_ENV),BENCH_QUALITY_LOG,BENCH_QUALITY_HARNESS_VERSION,TMPDIR'
