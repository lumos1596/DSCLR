#!/usr/bin/env bash
set -euo pipefail

# Table 6: Sensitivity and efficiency
# - λ and τ sensitivity can be extracted from existing grid search results
# - K (candidate depth) needs new runs with different --top_k values
# - ε (numerical floor) needs new runs
# - Latency measurements are done separately

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/sensitivity_trace"
PY="/home/luwa/.conda/envs/dsclr/bin/python"
ENGINE="eval.engine_trace"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

TASKS=(
  "Core17InstructionRetrieval"
  "Robust04InstructionRetrieval"
  "News21InstructionRetrieval"
)

DUAL_VERSION="v6"

cd "$ROOT"

# Sensitivity to K (candidate depth)
# Note: current engine_trace.py uses all candidates from FollowIR, not a configurable top-k
# We skip this for now as it requires code changes

# Sensitivity to eps (numerical floor)
for eps_val in "1e-4" "1e-6" "1e-8"; do
  for task in "${TASKS[@]}"; do
    dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
    out_dir="${OUT_ROOT}/eps_${eps_val}/${task}"

    echo "============================================================"
    echo "Sensitivity: eps=${eps_val}  Task: ${task}"
    echo "============================================================"

    "$PY" -m "$ENGINE" \
      --task_name "$task" \
      --model_name "samaya-ai/RepLLaMA-reproduced" \
      --dual_queries_path "$dual_path" \
      --output_dir "$out_dir" \
      --huber_delta 1.345 \
      --lambda_boundary 1.0 \
      --tau_decay 0.2 \
      --regression_mode "huber" \
      --ablation "full" \
      --device "$DEVICE" \
      --batch_size 64 \
      --use_cache "true"
  done
done

echo "============================================================"
echo "All sensitivity runs complete."
echo "============================================================"
