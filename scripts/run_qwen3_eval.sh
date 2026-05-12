#!/bin/bash
# 使用 Qwen3-0.6B 改写结果 + Repllama 编码器 + α=0.5, β=1.1, δ=0.0 进行 FollowIR 评估
# 必须直接在终端运行，绕过沙箱限制

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
cd /home/luwa/Documents/DSCLR

echo "========== FollowIR Evaluation: Qwen3-0.6B Reformulator + Repllama =========="
echo "Parameters: alpha=0.5, beta=1.1, delta=0.0"

for ds in Core17 Robust04 News21; do
  echo "=== Evaluating ${ds} ==="
  $PYTHON -m eval.engine_deir_dual_v2 \
    --task_name=${ds}InstructionRetrieval \
    --model_name=samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path=dataset/FollowIR_test/dual_queries_qwen3/dual_queries_qwen3_${ds}InstructionRetrieval.jsonl \
    --output_dir=results/qwen3-repllama-v2/${ds} \
    --alphas=0.5 \
    --betas=1.1 \
    --deltas=0.0 \
    --use_cache=true \
    --device=cuda
  echo "=== ${ds} Evaluation Done ==="
done

echo "========== All Evaluations Complete =========="
