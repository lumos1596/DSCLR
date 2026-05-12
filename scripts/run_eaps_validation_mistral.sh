#!/bin/bash
# EAPS Strategy Validation: Test Mistral parameters derived from training set
# Training-set derived: alpha=0.3, beta=3.0, delta=0.05, gamma=1.0
# Test-set optimal: alpha=0.3, beta=3.0, delta=0.15, gamma=1.0

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
cd /home/luwa/Documents/DSCLR

# EAPS-derived parameters
for ds in Core17 Robust04 News21; do
  echo "=== Evaluating ${ds} with EAPS parameters (a=0.3, b=3.0, d=0.05, g=1.0) ==="
  $PYTHON -m eval.engine_deir_dual_v21 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=intfloat/e5-mistral-7b-instruct \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/e5-mistral-7b/eaps_validation/${ds} \
    --alphas=0.3 \
    --betas=3.0 \
    --deltas=0.05 \
    --gammas=1.0 \
    --use_cache=true \
    --device=cuda
done

echo "=== All evaluations complete ==="
