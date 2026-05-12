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
SBASE_MODE="q_plus"
TOP_K=0
CONFIDENCE_BETA=0

OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_gap"

mkdir -p "${OUTPUT_DIR}"

ALPHAS="2.0,3.0,5.0"
DELTAS="-0.05,-0.03,0.0"
GAP_TEMPS="10,20,50"

echo "============================================================"
echo "🚀 差距加权惩罚实验 (Gap-weighted Penalty)"
echo "Alpha: ${ALPHAS}"
echo "Delta: ${DELTAS}"
echo "Gap Temperature: ${GAP_TEMPS}"
echo "============================================================"

start_time=$(date +%s)

for TEMP in $(echo $GAP_TEMPS | tr ',' ' '); do
    echo ""
    echo ">>> Running with gap_temperature=${TEMP}"
    SUB_DIR="${OUTPUT_DIR}/gap_${TEMP}"
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
        --confidence_beta ${CONFIDENCE_BETA} \
        --gap_temperature ${TEMP}
    
    echo ">>> Completed gap_temperature=${TEMP}"
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
echo "目标: p-MRR > 0.156 AND MAP1000 > 0.216"
echo ""
for TEMP in $(echo $GAP_TEMPS | tr ',' ' '); do
    SUB_DIR="${OUTPUT_DIR}/gap_${TEMP}"
    CSV="${SUB_DIR}/all_params_summary.csv"
    if [ -f "$CSV" ]; then
        echo ""
        echo "--- gap_temperature=${TEMP} ---"
        echo "alpha,delta,pMRR,changed_MAP1000"
        head -1 "$CSV" | tr ',' '\n' | cat -n | grep -E "alpha|delta|pMRR|changed_MAP1000"
        awk -F',' 'NR>1 {printf "α=%s,Δ=%s: p-MRR=%.4f, MAP1000=%.4f %s\n", $1,$2,$4,$21, ($4>0.156 && $21>0.216)?"✅双超":($4>0.156?"⚠️pMRR✅":($21>0.216?"⚠️MAP✅":"❌"))}' "$CSV"
    fi
done
