#!/usr/bin/env bash
set -euo pipefail

# Table 7: Cross-retriever portability on FollowIR
# 4 retrievers × 3 datasets (both base and TRACE)

ROOT="/home/luwa/Documents/DSCLR-remote"
OUT_ROOT="/home/luwa/Documents/DSCLR/evaluation_remote/cross_retriever_trace"
PY="/home/luwa/.conda/envs/dsclr/bin/python"

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

# ---------- Dense retrievers (neural encoders) ----------

DENSE_MODELS=(
  "intfloat/e5-mistral-7b-instruct"
  "BAAI/bge-large-en-v1.5"
  "samaya-ai/RepLLaMA-reproduced"
)

for model in "${DENSE_MODELS[@]}"; do
  model_short=$(echo "$model" | sed 's/\//_/g')
  echo "============================================================"
  echo "Cross-retriever: model=${model}"
  echo "============================================================"

  for task in "${TASKS[@]}"; do
    dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"

    # Base model (z_full_only)
    out_dir="${OUT_ROOT}/${model_short}/base/${task}"
    echo "  Base: ${task} -> ${out_dir}"
    "$PY" -m eval.engine_trace \
      --task_name "$task" \
      --model_name "$model" \
      --dual_queries_path "$dual_path" \
      --output_dir "$out_dir" \
      --huber_delta 1.345 \
      --lambda_boundary 1.0 \
      --tau_decay 0.2 \
      --regression_mode "huber" \
      --ablation "z_full_only" \
      --device "$DEVICE" \
      --batch_size 64 \
      --use_cache "true"

    # TRACE (full)
    out_dir="${OUT_ROOT}/${model_short}/trace/${task}"
    echo "  TRACE: ${task} -> ${out_dir}"
    "$PY" -m eval.engine_trace \
      --task_name "$task" \
      --model_name "$model" \
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

# ---------- BM25 (sparse retriever) ----------

echo "============================================================"
echo "Cross-retriever: model=BM25"
echo "============================================================"

for task in "${TASKS[@]}"; do
  dual_path="dataset/FollowIR_test/dual_queries_${DUAL_VERSION}/dual_queries_${DUAL_VERSION}_${task}.jsonl"

  # Base model (z_full_only)
  out_dir="${OUT_ROOT}/bm25/base/${task}"
  echo "  Base: ${task} -> ${out_dir}"
  "$PY" -m eval.engine_trace_bm25 \
    --task_name "$task" \
    --dual_queries_path "$dual_path" \
    --output_dir "$out_dir" \
    --ablation "z_full_only"

  # TRACE (full)
  out_dir="${OUT_ROOT}/bm25/trace/${task}"
  echo "  TRACE: ${task} -> ${out_dir}"
  "$PY" -m eval.engine_trace_bm25 \
    --task_name "$task" \
    --dual_queries_path "$dual_path" \
    --output_dir "$out_dir" \
    --ablation "full" \
    --huber_delta 1.345 \
    --lambda_boundary 1.0 \
    --tau_decay 0.2
done

echo "============================================================"
echo "All cross-retriever runs complete."
echo "============================================================"
