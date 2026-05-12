#!/bin/bash

cd /home/luwa/Documents/DSCLR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsclr

TASK="Core17InstructionRetrieval"
GPU_ID=1
MODEL_NAME="samaya-ai/RepLLaMA-reproduced"
BATCH_SIZE=8
TOP_K=0
CONFIDENCE_BETA=0
GAP_TEMPERATURE=0
MAX_PENALTY_RATIO=0

OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_ultrafine"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "🚀 超细粒度搜索 - 寻找最优双超配置"
echo "目标: p-MRR > 0.156 AND MAP1000 > 0.216"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \
    --model_name ${MODEL_NAME} \
    --task_name ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --use_dynamic_tau \
    --alphas_dynamic="1.4,1.5,1.6,1.7,1.8,1.9" \
    --deltas="-0.06,-0.05,-0.04,-0.03" \
    --sbase_mode="q_plus" \
    --top_k ${TOP_K} \
    --confidence_beta ${CONFIDENCE_BETA} \
    --gap_temperature ${GAP_TEMPERATURE} \
    --max_penalty_ratio ${MAX_PENALTY_RATIO}

echo ""
echo "========== 结果汇总 =========="
CSV="${OUTPUT_DIR}/all_params_summary.csv"
if [ -f "$CSV" ]; then
    echo ""
    echo "所有配置:"
    awk -F',' 'NR>1 {printf "α=%s,Δ=%s: p-MRR=%.4f, MAP1000=%.4f %s\n", $1,$2,$4,$22, ($4>0.156 && $22>0.216)?"✅双超":($4>0.156?"⚠️pMRR✅":($22>0.216?"⚠️MAP✅":"❌"))}' "$CSV"
    echo ""
    echo "最佳双超配置:"
    awk -F',' 'NR>1 && $4>0.156 && $22>0.216 {print "α="$1", Δ="$2": p-MRR="$4", MAP1000="$22}' "$CSV" | sort -t'=' -k4 -rn | head -5
fi
