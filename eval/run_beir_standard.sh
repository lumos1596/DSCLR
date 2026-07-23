#!/bin/bash
# BEIR Standard Retrieval Preservation experiments for Table 9
# 6 BEIR datasets × 3 backbones (RepLLaMA, E5-Mistral, BGE)

set -e

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
DUAL_DIR=/home/luwa/Documents/DSCLR-remote/dataset/BEIR/dual_queries
OUTPUT_BASE=/home/luwa/Documents/DSCLR/evaluation_remote/beir_standard_trace

DATASETS=("trec-covid" "nfcorpus" "fiqa" "arguana" "scifact" "quora")
MODELS=("BAAI/bge-large-en-v1.5" "samaya-ai/RepLLaMA-reproduced" "intfloat/e5-mistral-7b-instruct")
MODEL_NAMES=("bge" "repllama" "e5-mistral")

for m_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$m_idx]}"
    MNAME="${MODEL_NAMES[$m_idx]}"

    for DS in "${DATASETS[@]}"; do
        OUTPUT_DIR="${OUTPUT_BASE}/${MNAME}/${DS}"
        DUAL_PATH="${DUAL_DIR}/${DS}_CONSERVATIVE_t01.jsonl"

        if [ -f "${OUTPUT_DIR}/metrics_summary.json" ]; then
            echo "SKIP: ${MNAME}/${DS} (already exists)"
            continue
        fi

        echo "=== Running ${MNAME} on ${DS} ==="
        mkdir -p "${OUTPUT_DIR}"

        cd /home/luwa/Documents/DSCLR-remote && \
        CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON} -m eval.beir_standard_retrieval \
            --dataset "${DS}" \
            --model_name "${MODEL}" \
            --dual_queries_path "${DUAL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --device cuda \
            --batch_size 8
    done
done

echo "All BEIR standard retrieval experiments complete."
