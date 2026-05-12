#!/bin/bash
# V2 engine grid search on test sets for both Repllama and Mistral
# This gives us the test-set optimal parameters as reference

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
cd /home/luwa/Documents/DSCLR

ALPHAS="0.1,0.3,0.5,0.7,1.0,1.5,2.0"
BETAS="0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.5"
DELTAS="-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15"

# ========== Repllama ==========
echo "========== Repllama V2 Grid Search =========="
for ds in Core17 Robust04 News21; do
  echo "=== ${ds} ==="
  $PYTHON -m eval.engine_deir_dual_v2 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/repllama-v2-grid/${ds} \
    --alphas=${ALPHAS} \
    --betas=${BETAS} \
    --deltas=${DELTAS} \
    --use_cache=true \
    --device=cuda
done

# ========== Mistral ==========
echo "========== Mistral V2 Grid Search =========="
for ds in Core17 Robust04 News21; do
  echo "=== ${ds} ==="
  $PYTHON -m eval.engine_deir_dual_v2 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=intfloat/e5-mistral-7b-instruct \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/mistral-v2-grid/${ds} \
    --alphas=${ALPHAS} \
    --betas=${BETAS} \
    --deltas=${DELTAS} \
    --use_cache=true \
    --device=cuda
done

echo "========== All Grid Searches Complete =========="
