#!/bin/bash
# Run DeIR-Dual V2.1 baseline (alpha=0, beta=0) and wider grid search with E5-Mistral-7B

cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python

echo "=========================================="
echo "Step 1: E5-Mistral-7B Baseline (alpha=0, beta=0)"
echo "=========================================="

for ds in Core17 Robust04 News21; do
  echo ""
  echo "=== $ds Baseline ==="
  $PYTHON -m eval.engine_deir_dual_v21 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=intfloat/e5-mistral-7b-instruct \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/e5-mistral-7b/v21_baseline/${ds} \
    --alphas=0.0 \
    --betas=0.0 \
    --deltas=0.0 \
    --gammas=1.0 \
    --use_cache=true \
    --device=cuda
done

echo ""
echo "=========================================="
echo "Step 2: E5-Mistral-7B Wider Grid Search"
echo "=========================================="

for ds in Core17 Robust04 News21; do
  echo ""
  echo "=== $ds Grid Search ==="
  $PYTHON -m eval.engine_deir_dual_v21 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=intfloat/e5-mistral-7b-instruct \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/e5-mistral-7b/v21_grid/${ds} \
    --alphas=0.3,0.5,0.7,1.0,1.5,2.0 \
    --betas=1.0,2.0,3.0,5.0,8.0,11.0,15.0 \
    --deltas=-0.10,-0.05,0.00,0.05,0.10,0.15 \
    --gammas=0.5,0.6,0.7,0.8,1.0 \
    --use_cache=true \
    --device=cuda
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
