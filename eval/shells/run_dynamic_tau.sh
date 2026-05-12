#!/bin/bash

# ============================================================
# 动态 τ 实验脚本
# τ_q = Noise_q + Delta，其中 Noise_q = Cosine(Q_rich, Q^-_pure)
# ============================================================

# 切换到项目根目录
cd /home/luwa/Documents/DSCLR

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsclr

if [ "$CONDA_DEFAULT_ENV" != "dsclr" ]; then
    echo "❌ 错误: 无法激活 dsclr 环境，当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"
echo "✅ Python 路径: $(which python)"

# ============================================================
# 实验配置
# ============================================================
# 任务名称
TASK="${TASK:-Core17InstructionRetrieval}"

# GPU 设置
GPU_ID="${GPU_ID:-1}"

# 编码器配置
ENCODER_TYPE="${ENCODER_TYPE:-repllama}"
MODEL_NAME="${MODEL_NAME:-samaya-ai/RepLLaMA-reproduced}"
EMBED_DIM="${EMBED_DIM:-4096}"

# 批次大小
BATCH_SIZE="${BATCH_SIZE:-8}"

# 随机种子
SEED="${SEED:-42}"

# ============================================================
# LAP 模块配置
# ============================================================
# LAP 模型路径（可选，不使用则留空）
LAP_MODEL_PATH="${LAP_MODEL_PATH:-}"

# Top-K 重排配置（只对 Top-K 文档应用惩罚，保护长尾好文）
# 注意：设置为 0 表示不限制候选集大小，使用完整候选集（与原始评估一致）
TOP_K="${TOP_K:-0}"

# OG 锚点配置（统一方法: 用 OG 排序稳定 changed 排序）
ANCHOR_LAMBDA="${ANCHOR_LAMBDA:-0.0}"
ANCHOR_TOP_K="${ANCHOR_TOP_K:-0}"

# MAP 保留增益（对 OG Top-K 文档做温和加分，抑制深排 MAP 崩塌）
PRESERVE_LAMBDA="${PRESERVE_LAMBDA:-0.0}"
PRESERVE_TOP_K="${PRESERVE_TOP_K:-0}"

# ============================================================
# 自定义输出路径配置（直接修改此变量即可覆盖自动生成的路径）
# ============================================================
CUSTOM_OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/repllama-reproduced-max2600/Core17_gentle}"
# 示例: CUSTOM_OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/dynamic_tau/my_experiment"

# 输出目录
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
else
    OUTPUT_DIR="eval/output/dsclr_dynamic_tau/${TASK}_$(date +%Y%m%d_%H%M%S)"
fi

# ============================================================
# 动态 τ 参数配置 - Delta 一维网格搜索
# ============================================================
# Alpha 值 (固定为 1.5 和 2.0)
ALPHAS="${ALPHAS:-0.5,1.0,1.5,2.0}"

# Delta 值 (τ 的偏移量) - 包含正值和零
DELTAS="${DELTAS:-0.0,0.03,0.05,0.10}"

# S_base 计算模式: q_plus 或 original
SBASE_MODE="${SBASE_MODE:-q_plus}"  # q_plus: 使用 Q+ 计算; original: 使用原始查询计算

# 实验备注
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-Delta一维网格搜索: alpha=1.5/2.0, delta包含0基准点}"

echo "============================================================"
echo "🚀 动态 τ 实验"
echo "============================================================"
echo "任务: ${TASK}"
echo "编码器: ${ENCODER_TYPE} (${MODEL_NAME})"
echo "Alpha: ${ALPHAS}"
echo "Delta: ${DELTAS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 保存实验配置
CONFIG_FILE="${OUTPUT_DIR}/experiment_config.txt"
cat > "${CONFIG_FILE}" << EOF
============================================================
动态 τ 实验配置
============================================================

实验时间: $(date '+%Y-%m-%d %H:%M:%S')
输出目录: ${OUTPUT_DIR}

动态 τ 公式: τ_q = Noise_q + Delta
Noise_q = Cosine(Q_rich, Q^-_pure)

参数配置:
  Alpha (惩罚力度): ${ALPHAS}
  Delta (τ 偏移量): ${DELTAS}
  S_base 计算模式: ${SBASE_MODE}
  参数组数: ${ALPHAS} × ${DELTAS}

编码器配置:
  ENCODER_TYPE: ${ENCODER_TYPE}
  MODEL_NAME: ${MODEL_NAME}
  EMBED_DIM: ${EMBED_DIM}

运行配置:
  TASK: ${TASK}
  GPU_ID: ${GPU_ID}
  BATCH_SIZE: ${BATCH_SIZE}
  SEED: ${SEED}

实验备注:
  ${EXPERIMENT_NOTE}
EOF

echo "📝 实验配置已保存到: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

# ============================================================
# 运行实验
# ============================================================
echo ""
echo "============================================================"
echo "开始运行动态 τ 实验"
echo "============================================================"

# 记录开始时间
start_time=$(date +%s)

# 运行命令
echo "运行命令:"
echo "  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \\"
echo "    --model_name ${MODEL_NAME} \\"
echo "    --task_name ${TASK} \\"
echo "    --batch_size ${BATCH_SIZE} \\"
echo "    --output_dir ${OUTPUT_DIR} \\"
echo "    --use_dynamic_tau \\"
echo "    --alphas_dynamic ${ALPHAS} \\"
echo "    --deltas ${DELTAS} \\"
echo "    --sbase_mode ${SBASE_MODE} \\"
echo "    --top_k ${TOP_K} \\"
echo "    --anchor_lambda ${ANCHOR_LAMBDA} \\"
echo "    --anchor_top_k ${ANCHOR_TOP_K} \\"
echo "    --preserve_lambda ${PRESERVE_LAMBDA} \\"
echo "    --preserve_top_k ${PRESERVE_TOP_K} \\"
if [ -n "$LAP_MODEL_PATH" ]; then
    echo "    --lap_model_path ${LAP_MODEL_PATH}"
fi

# 切换到项目根目录运行
cd /home/luwa/Documents/DSCLR
LAP_ARG=""
if [ -n "$LAP_MODEL_PATH" ]; then
    LAP_ARG="--lap_model_path ${LAP_MODEL_PATH}"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -m eval.engine_dscrl \
    --model_name ${MODEL_NAME} \
    --task_name ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --use_dynamic_tau \
    --alphas_dynamic="${ALPHAS}" \
    --deltas="${DELTAS}" \
    --sbase_mode ${SBASE_MODE} \
    --top_k ${TOP_K} \
    --anchor_lambda ${ANCHOR_LAMBDA} \
    --anchor_top_k ${ANCHOR_TOP_K} \
    --preserve_lambda ${PRESERVE_LAMBDA} \
    --preserve_top_k ${PRESERVE_TOP_K} \
    ${LAP_ARG}

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
duration_sec=$((duration % 60))

echo ""
echo "============================================================"
echo "✅ 实验完成"
echo "============================================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "总耗时: ${duration_min}分${duration_sec}秒"
echo "============================================================"
