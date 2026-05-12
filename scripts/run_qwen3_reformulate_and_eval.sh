#!/bin/bash
# 使用 Qwen3-0.6B 本地模型进行指令改写
# 必须直接在终端运行，绕过沙箱限制

PYTHON=/home/luwa/.conda/envs/dsclr/bin/python
cd /home/luwa/Documents/DSCLR

echo "========== Qwen3-0.6B Local Reformulation =========="

for ds in Core17 Robust04 News21; do
  echo "=== ${ds} ==="
  $PYTHON -m model.reformulator_qwen3_local \
    --task_name=${ds}InstructionRetrieval \
    --model_path=/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B \
    --output_dir=dataset/FollowIR_test/dual_queries_qwen3 \
    --device=cuda
  echo "=== ${ds} Done ==="
done

echo "========== All Reformulations Complete =========="
echo ""
echo "Now running FollowIR evaluation with Repllama + alpha=0.5, beta=1.1, delta=0.0"

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
