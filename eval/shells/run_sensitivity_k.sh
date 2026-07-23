#!/usr/bin/env bash
set -euo pipefail

# Table 6: Candidate depth K sensitivity on FollowIR (RepLLaMA, TRACE full)
# K ∈ {10, 50, 100, 200}

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/sensitivity_k_trace"
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
MODEL="samaya-ai/RepLLaMA-reproduced"

K_VALUES=(10 50 100 200)

cd "$ROOT"

for K in "${K_VALUES[@]}"; do
  echo "============================================================"
  echo "K sensitivity: K=${K}"
  echo "============================================================"

  for task in "${TASKS[@]}"; do
    dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
    out_dir="${OUT_ROOT}/K${K}/${task}"
    echo "  K=${K}, ${task} -> ${out_dir}"
    "$PY" -m "$ENGINE" \
      --task_name "$task" \
      --model_name "$MODEL" \
      --dual_queries_path "$dual_path" \
      --output_dir "$out_dir" \
      --huber_delta 1.345 \
      --lambda_boundary 1.0 \
      --tau_decay 0.2 \
      --regression_mode "huber" \
      --ablation "full" \
      --candidate_depth "$K" \
      --device "$DEVICE" \
      --batch_size 8 \
      --use_cache "true"
  done
done

echo "============================================================"
echo "K sensitivity runs complete."
echo "============================================================"
