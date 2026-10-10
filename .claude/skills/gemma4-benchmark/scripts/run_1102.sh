#!/bin/bash
# Gemma-4 26B-A4B on Spyre, one workload at one tensor-parallel size: hf-adapters (#620) vs
# `vllm bench latency` on spyre-inference main and on the branch under test. Runs on an
# environment that already has everything installed. It never installs, syncs, rebuilds or
# checks anything out: it checks the environment, reports what deviates from the recipe, and
# stops for anything the user has not accepted. Ends with a shareable report.md.
#   TTFT (default): the maintainer's commands from spyre-inference#1102 (TP1 there)
#     https://github.com/torch-spyre/spyre-inference/issues/1102#issuecomment-6001115830
#   ITL (--itl): decode latency on the same prompt, ITL = (lat64 - lat1) / 63, the method of
#     the granite-3.3-8b TP4 ITL report (2026-10-02)
# Several workloads or TP sizes in one go: run_suite.sh.
#
# Usage: run_1102.sh [options]
#   --python PATH       interpreter of the environment to benchmark (default:
#                       $VIRTUAL_ENV/bin/python, else $UV_PROJECT_ENVIRONMENT/bin/python)
#   --hf-python PATH    interpreter for the hf arm only (default: --python)
#   --env-script FILE   source FILE first: the host's Spyre runtime environment, e.g. the
#                       venv's bin/activate (sourcing only sets shell variables)
#   --model PATH        model dir or HF id (default /models/google/gemma-4-26B-A4B)
#   --arms "hf main branch"  arms and their order (default). branch: the installed
#                       spyre-inference checkout; main: --main-ref, exported with git archive
#                       into a cache dir and put first on PYTHONPATH (vllm = branch)
#   --main-ref REF      default: <the torch-spyre/spyre-inference remote>/main, never fetched
#   --repeat N          run the arm list N times, alternating (default 1)
#   --itl               ITL: --output-len 1 and 64, 10 timed iterations, the vLLM arms under
#                       OMP_WAIT_POLICY=PASSIVE SPYRE_NUM_CPUS=8; hf runs hf_latency.py alike
#   --tp N              tensor-parallel size of every arm (default 1): vLLM
#                       --tensor-parallel-size N; hf runs hf_latency.py under torchrun with N
#                       ranks and tp_plan="auto". More than the host's cards is an ERROR
#   --check             check the environment, print findings and the plan, run nothing
#   --accept-warnings   benchmark despite WARN findings (the user has reviewed them)
#   --allow-profiler    downgrade a profiler build from ERROR to WARN
#   --recompiles        TORCH_LOGS=recompiles on the vLLM arms; count post-warmup recompiles
#   --out DIR           output dir (default $HOME/gemma4-benchmark/<timestamp>)
# Exit: 0 ok; 1 ERROR findings; 2 usage/runtime problem; 3 WARN findings not accepted;
#       4 an arm failed (its row in report.md says FAILED).
set -uo pipefail

SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

INPUT_LEN=1984
MODEL=/models/google/gemma-4-26B-A4B
PY=""
HF_PY=""
ENV_SCRIPT=""
ARMS="hf main branch"
MAIN_REF=auto
REPEAT=1
ITL=0
TP=1
CHECK_ONLY=0
ACCEPT_WARNINGS=0
ALLOW_PROFILER=0
RECOMPILES=0
OUT=""
ARGS=("$@")

die() {
  local msg=$1 code=${2:-2}
  echo "run_1102.sh: $msg" >&2
  exit "$code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY=$2; shift 2 ;;
    --hf-python) HF_PY=$2; shift 2 ;;
    --env-script) ENV_SCRIPT=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --arms) ARMS=$2; shift 2 ;;
    --main-ref) MAIN_REF=$2; shift 2 ;;
    --repeat) REPEAT=$2; shift 2 ;;
    --itl) ITL=1; shift ;;
    --tp) TP=$2; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --accept-warnings) ACCEPT_WARNINGS=1; shift ;;
    --allow-profiler) ALLOW_PROFILER=1; shift ;;
    --recompiles) RECOMPILES=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || die "--repeat takes a positive integer, got '$REPEAT'"
[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "--tp takes one positive integer, got '$TP' (several: run_suite.sh)"
ARMS=$(for arm in $ARMS; do [[ "$arm" = vllm ]] && echo branch || echo "$arm"; done | xargs)
for arm in $ARMS; do
  case "$arm" in hf|main|branch) ;; *) die "unknown arm '$arm' (hf, main, branch)" ;; esac
done
VLLM_ARMS=$(for arm in $ARMS; do [[ "$arm" = hf ]] || echo "$arm"; done | paste -sd, -)

if [[ -n "$ENV_SCRIPT" ]]; then
  [[ -r "$ENV_SCRIPT" ]] || die "cannot read --env-script $ENV_SCRIPT"
  set +u
  # shellcheck disable=SC1090
  source "$ENV_SCRIPT"
  set -u
fi
# After --env-script, which may be what sets VIRTUAL_ENV.
if [[ -z "$PY" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then PY=$VIRTUAL_ENV/bin/python
  elif [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then PY=$UV_PROJECT_ENVIRONMENT/bin/python
  else die "no environment: pass --python, or --env-script <venv>/bin/activate"; fi
fi
[[ -n "$HF_PY" ]] || HF_PY=$PY
for p in "$PY" "$HF_PY"; do [[ -x "$p" ]] || die "no interpreter at $p"; done

# The issue's workload, or the ITL report's on the same prompt.
if [[ "$ITL" = 1 ]]; then
  OUTPUT_LENS=(1 64); ITERS=10; HF_ITERS=10
  VLLM_ENV=(OMP_WAIT_POLICY=PASSIVE SPYRE_NUM_CPUS=8)
else
  OUTPUT_LENS=(1); ITERS=3; HF_ITERS=5
  VLLM_ENV=()
fi
# ttft.py is the issue's single-card script; hf_latency.py times the same way and shards.
HF_SCRIPT=hf_latency.py
[[ "$ITL" = 1 ]] || [[ "$TP" != 1 ]] || HF_SCRIPT=ttft.py

# Spyre cards: $SPYRE_DEVICES if pinned, else the numbered IOMMU-group nodes under /dev/vfio
# (the `vfio` container node is not one). hf ranks take the first TP of them, as hf-adapters'
# multicard script does.
if [[ -n "${SPYRE_DEVICES:-}" ]]; then
  CARDS=$(echo "${SPYRE_DEVICES//,/ }" | wc -w)
  HF_DEVICES=$(echo "${SPYRE_DEVICES//,/ }" | xargs -n1 | head -n "$TP" | paste -sd, -)
else
  CARDS=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$')
  HF_DEVICES=$(seq -s, 0 $((TP - 1)))
fi
HF_LAUNCH=("$HF_PY")
[[ "$TP" = 1 ]] || HF_LAUNCH=(env "SPYRE_DEVICES=$HF_DEVICES" "$HF_PY" -m torch.distributed.run
                            --nproc-per-node "$TP" --master-port 29500)
hf_args() {  # hf_args <output-len> <latency.json>
  local output_len=$1 latency_json=$2
  HF_ARGS=(--model "$MODEL" --input-len "$INPUT_LEN")
  [[ "$HF_SCRIPT" = ttft.py ]] || HF_ARGS+=(--output-len "$output_len" --iters "$HF_ITERS" --output-json "$latency_json")
  return 0
}
vllm_args() {  # vllm_args <output-len>
  local output_len=$1
  VLLM_ARGS=(--input-len "$INPUT_LEN" --output-len "$output_len" --batch-size 1 --num-iters-warmup 2
             --num-iters "$ITERS" --max-model-len 2048 --max-num-seqs 1)
  [[ "$TP" = 1 ]] || VLLM_ARGS+=(--tensor-parallel-size "$TP")
  return 0
}
[[ -n "$OUT" ]] || OUT=${GEMMA4_BENCH_OUT:-$HOME/gemma4-benchmark}/$(date +%Y-%m-%d_%H%M%S)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# ---------------------------------------------------------------- environment check
# One check per interpreter, from a neutral cwd so a checkout in the cwd cannot shadow
# what the benchmark itself imports.
check() {  # check <label> <python> <arms,comma-separated>
  local label=$1 python=$2 arms=$3
  local extra=()
  [[ "$ALLOW_PROFILER" = 0 ]] || extra=(--allow-profiler)
  (cd "$STAGE" && "$python" -I -B "$SKILL_DIR/scripts/check_env.py" --out "$STAGE/env_$label.json" \
    --arms "$arms" --model "$MODEL" --main-ref "$MAIN_REF" "${extra[@]+"${extra[@]}"}") > "$STAGE/check_$label.log" 2>&1
  [[ -s "$STAGE/env_$label.json" ]] || { echo "!! $label: check_env.py failed:"; tail -5 "$STAGE/check_$label.log"; return 1; }
}
jq_py() {  # jq_py <code> [args...]
  local code=$1
  "$PY" -I -B -c "$code" "${@:2}"
  return $?
}

if [[ "$HF_PY" = "$PY" ]]; then
  check env "$PY" "$(echo "$ARMS" | tr ' ' ',')" || die "environment check could not run"
  for arm in $ARMS; do cp "$STAGE/env_env.json" "$STAGE/env_$arm.json"; done
else
  if [[ " $ARMS " == *" hf "* ]]; then check hf "$HF_PY" hf || die "environment check could not run"; fi
  if [[ -n "$VLLM_ARMS" ]]; then
    check vllm "$PY" "$VLLM_ARMS" || die "environment check could not run"
    for arm in ${VLLM_ARMS//,/ }; do cp "$STAGE/env_vllm.json" "$STAGE/env_$arm.json"; done
  fi
fi

echo "=== environment findings (nothing was changed)"
counts=$(jq_py '
import json, sys
seen = {}
for path in sys.argv[1:]:
    for f in json.load(open(path))["findings"]:
        seen[(f["level"], f["message"])] = f
for (level, msg), f in sorted(seen.items(), key=lambda kv: ["ERROR", "WARN", "INFO"].index(kv[0][0])):
    print("  %-5s [%s] %s" % (level, f["arm"], msg))
print("COUNTS", sum(1 for k in seen if k[0] == "ERROR"), sum(1 for k in seen if k[0] == "WARN"))
' $(for arm in $ARMS; do echo "$STAGE/env_$arm.json"; done))
echo "$counts" | grep -v '^COUNTS '
read -r _ N_ERR N_WARN <<< "$(echo "$counts" | grep '^COUNTS ')"
[[ "${N_ERR:-0}" = 0 ]] && [[ "${N_WARN:-0}" = 0 ]] && echo "  (none)"

SI_DIR=""
MAIN_SHA=""
MAIN_SRC=""
if [[ -n "$VLLM_ARMS" ]]; then
  VLLM_ENV_JSON=$STAGE/env_${VLLM_ARMS%%,*}.json
  SI_DIR=$(jq_py 'import json, sys; print((json.load(open(sys.argv[1])).get("spyre-inference") or {}).get("checkout") or "")' "$VLLM_ENV_JSON")
  MAIN_SHA=$(jq_py 'import json, sys; print((json.load(open(sys.argv[1])).get("spyre-inference-main") or {}).get("commit") or "")' "$VLLM_ENV_JSON")
  [[ -z "$MAIN_SHA" ]] || MAIN_SRC=${GEMMA4_BENCH_CACHE:-$HOME/.cache/gemma4-benchmark}/spyre-inference-${MAIN_SHA:0:12}
fi
VLLM_CMD=("$(dirname "$PY")/vllm")
[[ -x "${VLLM_CMD[0]}" ]] || VLLM_CMD=("$PY" -m vllm.entrypoints.cli.main)
LENS=$(IFS=,; echo "${OUTPUT_LENS[*]}")
[[ "${#OUTPUT_LENS[@]}" = 1 ]] || LENS="{$LENS}"
vllm_args "$LENS"
hf_args "$LENS" "<run dir>/latency.json"
VLLM_LINE="${VLLM_ENV[*]+${VLLM_ENV[*]} }${VLLM_CMD[*]} bench latency --model $MODEL ${VLLM_ARGS[*]}"
PLAN=$(
  echo "  host   ${HOSTNAME:-$(uname -n)}  spyre cards=$CARDS"
  echo "  model  $MODEL  input_len=$INPUT_LEN output_len=$LENS tp=$TP"
  echo "  arms   $ARMS  x$REPEAT"
  [[ " $ARMS " != *" hf "* ]] || echo "  hf     OMP_NUM_THREADS=8 ${HF_LAUNCH[*]} $HF_SCRIPT ${HF_ARGS[*]}"
  [[ " $ARMS " != *" main "* ]] || echo "  main   (cd ${MAIN_SRC:-<export of --main-ref>}) PYTHONPATH=${MAIN_SRC:-<export>} $VLLM_LINE"
  [[ " $ARMS " != *" branch "* ]] || echo "  branch (cd ${SI_DIR:-<checkout>}) $VLLM_LINE"
  echo "  out    $OUT"
)
echo "=== plan"
echo "$PLAN"

if [[ "${N_ERR:-0}" != 0 ]]; then
  die "$N_ERR ERROR finding(s): this environment cannot be benchmarked as is; fix it yourself and re-run" 1
fi
if [[ "$TP" -gt 1 ]] && [[ "$CARDS" -lt "$TP" ]]; then
  die "ERROR: --tp $TP needs $TP Spyre cards, this host has $CARDS (\$SPYRE_DEVICES, else /dev/vfio); adjust --tp" 1
fi
if [[ "${N_WARN:-0}" != 0 ]] && [[ "$ACCEPT_WARNINGS" = 0 ]]; then
  [[ "$CHECK_ONLY" = 1 ]] && exit 3
  die "$N_WARN WARN finding(s) deviate from the issue's recipe; review them, then re-run with --accept-warnings" 3
fi
[[ "$CHECK_ONLY" = 1 ]] && exit 0
card_busy() {
  [[ -e /dev/vfio/vfio ]] && fuser /dev/vfio/vfio >/dev/null 2>&1
  return $?
}
card_busy && die "/dev/vfio/vfio is held by another process; the cards serve one process at a time"

# The main arm's source: an export of --main-ref (the checkout and its refs stay untouched),
# shadowing the editable install through PYTHONPATH. Gate on what that actually imports.
if [[ -n "$MAIN_SRC" ]]; then
  if [[ ! -d "$MAIN_SRC" ]]; then
    mkdir -p "$MAIN_SRC.tmp.$$" && git -C "$SI_DIR" archive --format=tar "$MAIN_SHA" | tar -x -C "$MAIN_SRC.tmp.$$" \
      && mv "$MAIN_SRC.tmp.$$" "$MAIN_SRC" || die "could not export $MAIN_SHA from $SI_DIR into $MAIN_SRC"
  fi
  MAIN_PYTHONPATH=$MAIN_SRC${PYTHONPATH:+:$PYTHONPATH}
  got=$(cd "$MAIN_SRC" && PYTHONPATH=$MAIN_PYTHONPATH "$PY" -B -c \
    'import os, spyre_inference; print(os.path.realpath(spyre_inference.__file__))' 2>/dev/null | tail -1)
  [[ "$got" == "$(realpath "$MAIN_SRC")"/* ]] \
    || die "main arm: spyre_inference imports from '${got:-nothing}', not the export $MAIN_SRC"
fi

# ---------------------------------------------------------------- benchmark
mkdir -p "$OUT"
for arm in $ARMS; do cp "$STAGE/env_$arm.json" "$OUT/"; done
printf '%s\n' "${ARGS[@]+"${ARGS[@]}"}" > "$OUT/args.txt"
echo "$PLAN" > "$OUT/plan.txt"
echo "{\"input_len\": $INPUT_LEN, \"output_lens\": [$(IFS=,; echo "${OUTPUT_LENS[*]}")], \"tp\": $TP," \
  "\"iters\": $ITERS, \"hf_iters\": $HF_ITERS, \"hf_script\": \"$HF_SCRIPT\"," \
  "\"hf_devices\": \"$([[ "$TP" = 1 ]] || echo "$HF_DEVICES")\", \"itl\": $([[ "$ITL" = 1 ]] && echo true || echo false)," \
  "\"vllm_env\": \"${VLLM_ENV[*]+${VLLM_ENV[*]}}\", \"arms\": \"$ARMS\", \"repeat\": $REPEAT}" > "$OUT/workload.json"
{
  echo "host ${HOSTNAME:-$(uname -n)}  date $(date -Is)"
  echo "python $PY  hf-python $HF_PY"
  echo "cpu $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1), $(nproc) online"
  echo "cards $CARDS (SPYRE_DEVICES=${SPYRE_DEVICES:-unset}; $(env | sed -n 's/^AIU_WORLD_RANK_[0-9]*=//p' | sort | paste -sd' ' -))"
  rpm -qa 2>/dev/null | grep -E '^ibm-' | sort
} > "$OUT/provenance.txt"

run_arm() {  # run_arm <arm> <rep> <output-len>
  local arm=$1 rep=$2 len=$3 dir log rc
  dir=$OUT/${arm}_out${len}_$rep
  mkdir -p "$dir"; log=$dir/run.log
  if card_busy; then
    echo "!! /dev/vfio/vfio is held by another process; not starting $arm #$rep" | tee "$log"
    echo "{\"arm\": \"$arm\", \"rep\": $rep, \"output_len\": $len, \"tp\": $TP, \"rc\": 3}" > "$dir/meta.json"
    return
  fi
  echo "=== $arm #$rep output_len=$len tp=$TP  $(date -Is)" | tee "$log"
  if [[ "$arm" = hf ]]; then
    hf_args "$len" "$dir/latency.json"
    (cd "$dir" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 "${HF_LAUNCH[@]}" "$SKILL_DIR/scripts/$HF_SCRIPT" \
      "${HF_ARGS[@]}") >> "$log" 2>&1
  else
    local extra=("${VLLM_ENV[@]+"${VLLM_ENV[@]}"}") cwd=$SI_DIR
    [[ "$RECOMPILES" = 0 ]] || extra+=(TORCH_LOGS=recompiles)
    [[ "$arm" = branch ]] || { cwd=$MAIN_SRC; extra+=("PYTHONPATH=$MAIN_PYTHONPATH"); }
    vllm_args "$len"
    (cd "${cwd:-$dir}" && env PYTHONDONTWRITEBYTECODE=1 "${extra[@]+"${extra[@]}"}" "${VLLM_CMD[@]}" bench latency \
      --model "$MODEL" "${VLLM_ARGS[@]}" --output-json "$dir/latency.json") >> "$log" 2>&1
  fi
  rc=$?
  echo "=== $arm #$rep output_len=$len tp=$TP finished rc=$rc $(date -Is)" | tee -a "$log"
  [[ "$rc" = 0 ]] || tail -20 "$log" | sed 's/^/    /'
  echo "{\"arm\": \"$arm\", \"rep\": $rep, \"output_len\": $len, \"tp\": $TP, \"rc\": $rc}" > "$dir/meta.json"
}

for rep in $(seq 1 "$REPEAT"); do
  for arm in $ARMS; do
    for len in "${OUTPUT_LENS[@]}"; do
      run_arm "$arm" "$rep" "$len"
    done
  done
done

"$PY" -I -B "$SKILL_DIR/scripts/report.py" "$OUT" | tee "$OUT/report.md"
[[ "${PIPESTATUS[0]}" = 0 ]] || exit 4
