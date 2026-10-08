#!/bin/bash
# Several workloads and tensor-parallel sizes in one go: the TTFT workload (run_1102.sh) at each
# of --ttft-tps and the ITL workload (run_1102.sh --itl) at each of --itl-tps, every one with
# the arms alternating. Checks every combination first and starts nothing if one would be
# refused. Ends with one shareable report.md over all of them, TP scaling included; each
# workload's own report.md sits in its subdirectory.
#
# Usage: run_suite.sh [--ttft-tps "1 2 4"] [--itl-tps "1 2 4"] [--out DIR] [run_1102.sh options]
#   --ttft-tps LIST  TP sizes of the TTFT workload (default "1"; "" skips the workload)
#   --itl-tps LIST   TP sizes of the ITL workload (default "1"; "" skips the workload)
#   --out DIR        suite dir (default $HOME/gemma4-benchmark/suite_<timestamp>)
#   Every other option (--env-script, --python, --hf-python, --arms, --main-ref, --repeat,
#   --check, --accept-warnings, --allow-profiler, --recompiles, --model) goes to each run.
# Exit: 0 ok; 1/2/3 as run_1102.sh (nothing started, or the suite stopped there); 4 some arm
#       failed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TTFT_TPS="1"
ITL_TPS="1"
OUT=""
CHECK_ONLY=0
PASS=()
die() { echo "run_suite.sh: $1" >&2; exit "${2:-2}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --ttft-tps) TTFT_TPS=$2; shift 2 ;;
    --itl-tps) ITL_TPS=$2; shift 2 ;;
    --tps) die "--tps is now --itl-tps, next to --ttft-tps" ;;
    --out) OUT=$2; shift 2 ;;
    --itl|--tp) die "$1 is set per workload: use --ttft-tps and --itl-tps" ;;
    --check) CHECK_ONLY=1; shift ;;
    --accept-warnings|--allow-profiler|--recompiles) PASS+=("$1"); shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) [ $# -ge 2 ] || die "$1 needs a value"
       PASS+=("$1" "$2"); shift 2 ;;
  esac
done
[ -n "$OUT" ] || OUT=${GEMMA4_BENCH_OUT:-$HOME/gemma4-benchmark}/suite_$(date +%Y-%m-%d_%H%M%S)

WORKLOADS=()
add() {  # add <name> <tps> <run_1102.sh flags>
  local tp seen=" "
  for tp in $2; do
    [[ "$tp" =~ ^[1-9][0-9]*$ ]] || die "--$1-tps takes positive integers, got '$tp'"
    [[ "$seen" != *" $tp "* ]] || die "--$1-tps lists $tp twice"
    seen+="$tp "
    WORKLOADS+=("$1_tp$tp|$3 --tp $tp")
  done
}
add ttft "$TTFT_TPS" ""
add itl "$ITL_TPS" "--itl"
[ ${#WORKLOADS[@]} -gt 0 ] || die "nothing to run: --ttft-tps and --itl-tps are both empty"

# An ERROR (1) outranks a usage problem (2), which outranks unaccepted WARNs (3).
severity() { case "$1" in 0) echo 0 ;; 3) echo 1 ;; 1) echo 3 ;; *) echo 2 ;; esac; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
rc=0
for w in "${WORKLOADS[@]}"; do
  name=${w%%|*}
  read -r -a extra <<< "${w#*|}"
  if [ "$CHECK_ONLY" = 1 ]; then
    echo "=== suite check: $name"
    bash "$HERE/run_1102.sh" "${PASS[@]+"${PASS[@]}"}" "${extra[@]}" --check
    r=$?
  else
    bash "$HERE/run_1102.sh" "${PASS[@]+"${PASS[@]}"}" "${extra[@]}" --check > "$STAGE/$name.log" 2>&1
    r=$?
    echo "=== suite check: $name exited $r"
    [ "$r" = 0 ] || sed 's/^/    /' "$STAGE/$name.log"
  fi
  [ "$(severity "$r")" -le "$(severity "$rc")" ] || rc=$r
done
[ "$CHECK_ONLY" = 1 ] && exit "$rc"
[ "$rc" = 0 ] || die "not starting: a workload's check exited $rc (see above)" "$rc"

for w in "${WORKLOADS[@]}"; do
  name=${w%%|*}
  read -r -a extra <<< "${w#*|}"
  echo "=== suite: $name  $(date -Is)"
  bash "$HERE/run_1102.sh" "${PASS[@]+"${PASS[@]}"}" "${extra[@]}" --out "$OUT/$name"
  r=$?
  echo "=== suite: $name exited $r  $(date -Is)"
  case "$r" in
    0) ;;
    4) rc=4 ;;
    *) echo "=== suite: stopping after $name"; rc=$r; break ;;
  esac
done

PY=$(sed -n 's/^python \([^ ]*\) .*/\1/p' "$OUT"/*/provenance.txt 2>/dev/null | head -1)
if [ -n "$PY" ]; then
  "$PY" -I -B "$HERE/report.py" --suite "$OUT" > "$OUT/report.md" || { [ "$rc" != 0 ] || rc=4; }
  echo "=== suite report: $OUT/report.md"
fi
exit "$rc"
