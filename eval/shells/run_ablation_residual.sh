#!/usr/bin/env bash
set -euo pipefail

# First ablation experiment (Table 3: Component and residual-estimation ablation)
# Four variants:
#   1. z_full_only:    S_final = z_full                         (Full)
#   2. pos_only:       S_final = z_full + p                     (Full + Positive)
#   3. raw_neg_subtract: S_final = z_full + p - z_neg           (+ Raw Negative)
#   4. linear:         S_final = z_full + p - r                 (+ Residual)

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/ablation_residual_trace"
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

ABLATIONS=(
  "z_full_only"
  "pos_only"
  "raw_neg_subtract"
  "linear"
)

DUAL_VERSION="v6"

cd "$ROOT"

for ablation in "${ABLATIONS[@]}"; do
  for task in "${TASKS[@]}"; do
    dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
    out_dir="${OUT_ROOT}/${ablation}/${task}"

    echo "============================================================"
    echo "Ablation: ${ablation}  Task: ${task}"
    echo "Output: ${out_dir}"
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
      --ablation "$ablation" \
      --device "$DEVICE" \
      --batch_size 64 \
      --use_cache "true"
  done
done

echo "============================================================"
echo "All ablation runs complete."
echo "Results under: ${OUT_ROOT}"
echo "============================================================"
