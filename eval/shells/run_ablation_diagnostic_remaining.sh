#!/usr/bin/env bash
set -euo pipefail

# Table 5: Remaining diagnostic ablation variants
# (trace_baseline and ols_fit already completed)
# Variants to run:
#   3. Mean/std scaling:     normalization_mode=mean_std
#   4. Uncentered residual:  uncentered_residual=true
#   5. Constrained slope:    constrained_slope=true
#   6. Raw-score fit:        raw_score_fit=true

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/ablation_diagnostic_trace"
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

# Variant 3: Mean/std scaling (Core17 already done)
for task in "Robust04InstructionRetrieval" "News21InstructionRetrieval"; do
  dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
  out_dir="${OUT_ROOT}/mean_std_scaling/${task}"

  echo "============================================================"
  echo "Variant: Mean/std scaling  Task: ${task}"
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
    --normalization_mode "mean_std" \
    --ablation "full" \
    --device "$DEVICE" \
    --batch_size 64 \
    --use_cache "true"
done

# Variant 4: Uncentered residual
for task in "${TASKS[@]}"; do
  dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
  out_dir="${OUT_ROOT}/uncentered_residual/${task}"

  echo "============================================================"
  echo "Variant: Uncentered residual  Task: ${task}"
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
    --uncentered_residual "true" \
    --ablation "full" \
    --device "$DEVICE" \
    --batch_size 64 \
    --use_cache "true"
done

# Variant 5: Constrained slope
for task in "${TASKS[@]}"; do
  dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
  out_dir="${OUT_ROOT}/constrained_slope/${task}"

  echo "============================================================"
  echo "Variant: Constrained slope  Task: ${task}"
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
    --constrained_slope "true" \
    --ablation "full" \
    --device "$DEVICE" \
    --batch_size 64 \
    --use_cache "true"
done

# Variant 6: Raw-score fit
for task in "${TASKS[@]}"; do
  dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"
  out_dir="${OUT_ROOT}/raw_score_fit/${task}"

  echo "============================================================"
  echo "Variant: Raw-score fit  Task: ${task}"
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
    --raw_score_fit "true" \
    --ablation "full" \
    --device "$DEVICE" \
    --batch_size 64 \
    --use_cache "true"
done

echo "============================================================"
echo "All remaining diagnostic ablation runs complete."
echo "============================================================"
