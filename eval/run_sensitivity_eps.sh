#!/bin/bash
# ε (numerical floor) sensitivity experiments for Table 6
# Tests different values of eps used in MAD standardization

set -e

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
ENGINE=/home/luwa/Documents/DSCLR-remote/eval/engine_trace.py
DUAL_DIR=/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v6
OUTPUT_BASE=/home/luwa/Documents/DSCLR/evaluation_remote/sensitivity_eps_trace
MODEL=samaya-ai/RepLLaMA-reproduced

DATASETS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
EPS_VALUES=("1e-8" "1e-6" "1e-4" "1e-2" "1e-1")

for EPS in "${EPS_VALUES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        OUTPUT_DIR="${OUTPUT_BASE}/eps${EPS}/${DS}"
        DUAL_PATH="${DUAL_DIR}/dual_queries_v6_${DS}.jsonl"

        if [ -f "${OUTPUT_DIR}/trace_all_results.json" ]; then
            echo "SKIP: eps=${EPS} ${DS} (already exists)"
            continue
        fi

        echo "=== Running eps=${EPS} on ${DS} ==="
        mkdir -p "${OUTPUT_DIR}"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON} ${ENGINE} \
            --task_name "${DS}" \
            --model_name "${MODEL}" \
            --output_dir "${OUTPUT_DIR}" \
            --dual_queries_path "${DUAL_PATH}" \
            --eps "${EPS}" \
            --device cuda \
            --batch_size 8
    done
done

echo "All ε sensitivity experiments complete."
