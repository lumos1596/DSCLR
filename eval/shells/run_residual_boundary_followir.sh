#!/usr/bin/env bash
set -euo pipefail

# Controlled FollowIR comparison for DeIR-Dual V2:
#   1) semantic: original tau = cos(Q_base, Q_neg) + delta
#   2) residual_bg: background-calibrated residual boundary
#
# Alpha and beta are fallback values only. With --per_query_ab true, the
# effective alpha_q and beta_q are derived at inference time from each query's
# candidate-score distribution.

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/residual_boundary_v8"
PY="/home/luwa/.conda/envs/dsclr/bin/python"
ENGINE="eval.engine_deir_dual_v2"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESIDUAL_MARGIN_SCALE="${RESIDUAL_MARGIN_SCALE:-3.0}"
export CUDA_VISIBLE_DEVICES

TASKS=(
  "Core17InstructionRetrieval"
  "Robust04InstructionRetrieval"
  "News21InstructionRetrieval"
)

BOUNDARY_MODES=(
  "semantic"
  "residual_bg"
)

cd "$ROOT"

for boundary_mode in "${BOUNDARY_MODES[@]}"; do
  for task in "${TASKS[@]}"; do
    dual_path="dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${task}.jsonl"
    out_dir="${OUT_ROOT}/${boundary_mode}/${task}"

    echo "============================================================"
    echo "Task: ${task}"
    echo "Boundary: ${boundary_mode}"
    echo "Output: ${out_dir}"
    echo "============================================================"

    "$PY" -m "$ENGINE" \
      --task_name "$task" \
      --model_name "samaya-ai/RepLLaMA-reproduced" \
      --dual_queries_path "$dual_path" \
      --output_dir "$out_dir" \
      --alphas "1.0" \
      --betas "1.0" \
      --deltas "0.0" \
      --boundary_mode "$boundary_mode" \
      --residual_margin_scale "$RESIDUAL_MARGIN_SCALE" \
      --per_query_ab "true" \
      --beta_derive_mode "max_mean" \
      --t_safety "20.0" \
      --device "$DEVICE" \
      --batch_size "1" \
      --use_cache "true"
  done
done

echo "Done. Compare all_results.json and metrics_summary.json under:"
echo "  ${OUT_ROOT}/semantic"
echo "  ${OUT_ROOT}/residual_bg"
