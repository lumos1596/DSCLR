#!/usr/bin/env bash
set -euo pipefail

# Table 8: NegConstraint transfer experiment
# 4 variants:
#   1. z_full_only:  Base retriever (S_final = z_full)
#   2. pos_only:     + Positive (S_final = z_full + p)
#   3. raw_neg_subtract: + Raw Negative (S_final = z_full + p - z_neg)
#   4. full:         TRACE (S_final = z_full + p*g - h)

ROOT="/home/luwa/Documents/DSCLR-remote"
PY="/home/luwa/.conda/envs/dsclr/bin/python"
ENGINE="eval.negconstraint_trace"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

ABLATIONS=(
  "z_full_only"
  "pos_only"
  "raw_neg_subtract"
  "full"
)

cd "$ROOT"

for ablation in "${ABLATIONS[@]}"; do
  echo "============================================================"
  echo "NegConstraint TRACE: ablation=${ablation}  encoder=repllama"
  echo "============================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m "$ENGINE" \
    --encoder "repllama" \
    --ablation "$ablation" \
    --lambda_boundary 1.0 \
    --tau_decay 0.2 \
    --huber_delta 1.345 \
    --candidate_depth 100 \
    --output_dir "results/negconstraint_trace/repllama/${ablation}"
done

echo "============================================================"
echo "All NegConstraint TRACE runs complete."
echo "============================================================"
