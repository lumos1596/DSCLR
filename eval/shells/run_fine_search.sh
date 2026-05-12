#!/bin/bash

cd /home/luwa/Documents/DSCLR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsclr

if [ "$CONDA_DEFAULT_ENV" != "dsclr" ]; then
    echo "❌ 错误: 无法激活 dsclr 环境"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"

TASK="Core17InstructionRetrieval"
GPU_ID=1
MODEL_NAME="samaya-ai/RepLLaMA-reproduced"
BATCH_SIZE=8
TOP_K=0
CONFIDENCE_BETA=0
GAP_TEMPERATURE=0
MAX_PENALTY_RATIO=0

echo "============================================================"
echo "🚀 细粒度搜索 + sbase_mode 对比实验"
echo "============================================================"

start_time=$(date +%s)

# 实验1: q_plus 模式，细粒度搜索
echo ""
echo ">>> 实验1: sbase_mode=q_plus, 细粒度 α/Δ 搜索"
DIR1="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_fine_qplus"
mkdir -p "${DIR1}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \
    --model_name ${MODEL_NAME} \
    --task_name ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${DIR1} \
    --use_dynamic_tau \
    --alphas_dynamic="1.5,1.8,2.0,2.5,3.0" \
    --deltas="-0.05,-0.04,-0.03" \
    --sbase_mode "q_plus" \
    --top_k "${TOP_K}" \
    --confidence_beta "${CONFIDENCE_BETA}" \
    --gap_temperature "${GAP_TEMPERATURE}" \
    --max_penalty_ratio "${MAX_PENALTY_RATIO}"

echo ">>> 实验1完成"

# 实验2: original 模式，相同参数范围
echo ""
echo ">>> 实验2: sbase_mode=original, 细粒度 α/Δ 搜索"
DIR2="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_fine_original"
mkdir -p "${DIR2}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \
    --model_name "${MODEL_NAME}" \
    --task_name "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${DIR2}" \
    --use_dynamic_tau \
    --alphas_dynamic="1.5,1.8,2.0,2.5,3.0" \
    --deltas="-0.05,-0.04,-0.03" \
    --sbase_mode "original" \
    --top_k ${TOP_K} \
    --confidence_beta ${CONFIDENCE_BETA} \
    --gap_temperature ${GAP_TEMPERATURE} \
    --max_penalty_ratio ${MAX_PENALTY_RATIO}

echo ">>> 实验2完成"

end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
duration_sec=$((duration % 60))

echo ""
echo "============================================================"
echo "✅ 全部实验完成"
echo "总耗时: ${duration_min}分${duration_sec}秒"
echo "============================================================"

echo ""
echo "========== 结果汇总 =========="
echo "目标: p-MRR > 0.156 AND MAP1000 > 0.216"
echo ""

for DIR in "$DIR1" "$DIR2"; do
    CSV="${DIR}/all_params_summary.csv"
    MODE=$(basename "$DIR" | sed 's/Core17_fine_//')
    if [ -f "$CSV" ]; then
        echo ""
        echo "--- sbase_mode=${MODE} ---"
        awk -F',' 'NR>1 {printf "α=%s,Δ=%s: p-MRR=%.4f, MAP1000=%.4f %s\n", $1,$2,$4,$21, ($4>0.156 && $21>0.216)?"✅双超":($4>0.156?"⚠️pMRR✅":($21>0.216?"⚠️MAP✅":"❌"))}' "$CSV"
    fi
done
