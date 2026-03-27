#!/usr/bin/env bash

# models=(dinowm dinowm_noprop pldm lejepa iql gcbc ivl)
models=(pldm lejepa iql gcbc ivl)
for m in "${models[@]}"; do
    echo "================================="
    echo "Running model: $m"
    echo "================================="
    bash run_eval.sh "$m" || echo "❌ Failed: $m"
done