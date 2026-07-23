#!/usr/bin/env bash
set -euo pipefail

# Second ablation experiment (Table 4: Thresholded residual-based scoring ablation)
# Four variants:
#   1. linear:     S = z_full + p - r         (Direct subtraction)     [already run]
#   2. no_gate:    S = z_full + p - h         (Thresholded penalty)    [need to run]
#   3. gate_only:  S = z_full + p*g           (Attenuation only)       [need to run]
#   4. full:       S = z_full + p*g - h       (TRACE)                  [already run]

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/ablation_scoring_trace"
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
  "no_gate"
  "gate_only"
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
echo "All scoring ablation runs complete."
echo "Results under: ${OUT_ROOT}"
echo "============================================================"
