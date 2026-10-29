#!/usr/bin/env bash
set -uo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
summary_tool=/home/ccyang/dt-inductor/spyre-perf-suite/.claude/skills/summarize-sdsc/batch_summarize_sdsc.py
output_dir="$root/explicit_kv_major_paged_attention_sweep"
table="$output_dir/summary.md"

rm -rf "$output_dir"
mkdir -p "$output_dir"
printf '| Q length | KV length | Device kernel time (us/run) | Status | Log | SDSC summary |\n|---:|---:|---|---|---|---|\n' > "$table"

total_runs=24
run=0
for qlen in 1 2 4 8 16 32; do
    for kvlen in 128 1024 2048 4096; do
        run=$((run + 1))
        name="q${qlen}_kv${kvlen}"
        cache_dir="$output_dir/cache_$name"
        log="$output_dir/$name.log"
        sdsc="$output_dir/$name.sdsc.md"
        rm -rf "$cache_dir"
        printf '[%02d/%02d] qlen=%-2s kvlen=%-4s profiling...\n' \
            "$run" "$total_runs" "$qlen" "$kvlen"

        if (
            cd "$root"
            TORCHINDUCTOR_CACHE_DIR="$cache_dir" \
                uv run --no-sync python explicit_kv_major_paged_attention.py \
                --qlen "$qlen" --kvlen "$kvlen" --profile-reps 5
        ) >"$log" 2>&1; then
            status=ok
            time_us=$(awk -F'kernel_us_per_run=' '/^RESULT / { print $2 }' "$log" | tail -1)
        else
            status=failed
            time_us='-'
        fi
        printf '[%02d/%02d] qlen=%-2s kvlen=%-4s %s; summarizing SDSC...\n' \
            "$run" "$total_runs" "$qlen" "$kvlen" "$status"

        (
            cd /home/ccyang/dt-inductor/spyre-perf-suite
            python3 "$summary_tool" "$cache_dir"
        ) > "$root/summarize_sdsc.md" 2>&1
        cp "$root/summarize_sdsc.md" "$sdsc"
        printf '| %s | %s | %s | %s | [%s](%s) | [%s](%s) |\n' \
            "$qlen" "$kvlen" "$time_us" "$status" "$name" "$(basename "$log")" \
            "$name" "$(basename "$sdsc")" >> "$table"
        if [[ "$status" == ok ]]; then
            printf '[%02d/%02d] qlen=%-2s kvlen=%-4s done: %s us/run\n' \
                "$run" "$total_runs" "$qlen" "$kvlen" "$time_us"
        else
            printf '[%02d/%02d] qlen=%-2s kvlen=%-4s failed; see %s\n' \
                "$run" "$total_runs" "$qlen" "$kvlen" "$log"
        fi
    done
done

printf 'Wrote %s\n' "$table"
