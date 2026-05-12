#!/bin/bash

cd /home/luwa/Documents/DSCLR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsclr

if [ "$CONDA_DEFAULT_ENV" != "dsclr" ]; then
    echo "❌ 错误: 无法激活 dsclr 环境，当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"

TASK="Core17InstructionRetrieval"
GPU_ID=1
MODEL_NAME="samaya-ai/RepLLaMA-reproduced"
BATCH_SIZE=8
SBASE_MODE="q_plus"
TOP_K=0
LAP_MODEL_PATH=""

OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_confidence"

mkdir -p "${OUTPUT_DIR}"

ALPHAS="1.5,2.0,3.0,5.0"
DELTAS="-0.05,-0.03,0.0"
CONFIDENCE_BETAS="0.5,1.0,2.0"

echo "============================================================"
echo "🚀 置信度加权惩罚实验"
echo "Alpha: ${ALPHAS}"
echo "Delta: ${DELTAS}"
echo "Confidence Beta: ${CONFIDENCE_BETAS}"
echo "============================================================"

start_time=$(date +%s)

LAP_ARG=""
if [ -n "$LAP_MODEL_PATH" ]; then
    LAP_ARG="--lap_model_path ${LAP_MODEL_PATH}"
fi

for BETA in $(echo $CONFIDENCE_BETAS | tr ',' ' '); do
    echo ""
    echo ">>> Running with confidence_beta=${BETA}"
    SUB_DIR="${OUTPUT_DIR}/beta_${BETA}"
    mkdir -p "${SUB_DIR}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${SUB_DIR} \
        --use_dynamic_tau \
        --alphas_dynamic="${ALPHAS}" \
        --deltas="${DELTAS}" \
        --sbase_mode ${SBASE_MODE} \
        --top_k ${TOP_K} \
        --confidence_beta ${BETA} \
        ${LAP_ARG}
    
    echo ">>> Completed beta=${BETA}"
done

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
for BETA in $(echo $CONFIDENCE_BETAS | tr ',' ' '); do
    SUB_DIR="${OUTPUT_DIR}/beta_${BETA}"
    CSV="${SUB_DIR}/all_params_summary.csv"
    if [ -f "$CSV" ]; then
        echo ""
        echo "--- beta=${BETA} ---"
        head -1 "$CSV"
        grep -v "^alpha" "$CSV" | sort -t',' -k4 -rn | head -3
    fi
done
