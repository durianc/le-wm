#!/usr/bin/env bash
set -euo pipefail

# models=(dinowm dinowm_noprop iql gcbc ivl)
models=(pldm lejepa)
tasks=(tworoom)

# 同时最多跑几个进程
MAX_JOBS=5

LOG_DIR="logs_eval"
mkdir -p "$LOG_DIR"

pids=()

run_one() {
    local model="$1"
    local task="$2"
    local log_file="$LOG_DIR/${task}_${model}.log"

    echo "================================="
    echo "Launching task=${task}, model=${model}"
    echo "Log: $log_file"
    echo "================================="

    bash run_eval.sh "$model" "$task" >"$log_file" 2>&1 &
    pids+=($!)
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 2
    done
}

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        wait_for_slot
        run_one "$model" "$task"
    done
done

echo "All jobs launched. Waiting..."
wait

echo "All jobs finished."