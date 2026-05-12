#!/bin/bash
# Run DeIR-Dual V2.1 evaluation with E5-Mistral-7B on all three test sets
# Training-set optimal parameters: alpha=1.5, beta=4.0, delta=0.10, gamma=0.7

cd /home/luwa/Documents/DSCLR
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python

echo "=========================================="
echo "E5-Mistral-7B DeIR-Dual V2.1 Evaluation"
echo "Parameters: alpha=1.5, beta=4.0, delta=0.10, gamma=0.7"
echo "=========================================="

echo ""
echo "=== Core17 ==="
$PYTHON -m eval.engine_deir_dual_v21 \
  --task_name=Core17InstructionRetrieval \
  --model_name=intfloat/e5-mistral-7b-instruct \
  --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Core17InstructionRetrieval.jsonl \
  --output_dir=results/e5-mistral-7b/v21/Core17 \
  --alphas=1.5 \
  --betas=4.0 \
  --deltas=0.10 \
  --gammas=0.7 \
  --use_cache=true \
  --device=cuda

echo ""
echo "=== Robust04 ==="
$PYTHON -m eval.engine_deir_dual_v21 \
  --task_name=Robust04InstructionRetrieval \
  --model_name=intfloat/e5-mistral-7b-instruct \
  --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Robust04InstructionRetrieval.jsonl \
  --output_dir=results/e5-mistral-7b/v21/Robust04 \
  --alphas=1.5 \
  --betas=4.0 \
  --deltas=0.10 \
  --gammas=0.7 \
  --use_cache=true \
  --device=cuda

echo ""
echo "=== News21 ==="
$PYTHON -m eval.engine_deir_dual_v21 \
  --task_name=News21InstructionRetrieval \
  --model_name=intfloat/e5-mistral-7b-instruct \
  --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_News21InstructionRetrieval.jsonl \
  --output_dir=results/e5-mistral-7b/v21/News21 \
  --alphas=1.5 \
  --betas=4.0 \
  --deltas=0.10 \
  --gammas=0.7 \
  --use_cache=true \
  --device=cuda

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
