#!/bin/bash
# ============================================================
# DSCLR 静态网格搜索评测脚本
# 模式A：不使用 MLP，仅通过网格搜索寻找最佳 alpha 和 tau
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dsclr

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 设置 Python 路径
export PYTHONPATH="/home/luwa/Documents/DSCLR:$PYTHONPATH"

# 验证环境
if [ "$CONDA_DEFAULT_ENV" != "dsclr" ]; then
    echo "❌ 错误: 无法激活 dsclr 环境，当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"
echo "✅ Python 路径: $(which python)"

# ============================================================
# 模块开关配置（独立控制，自由组合）
# ============================================================
# USE_LAP: 是否使用 LAP 模块（true/false）
#   true  - 使用 LAP 投影负向查询
#   false - 不使用 LAP
USE_LAP=false

# USE_MLP: 是否使用 MLP 模块（true/false）
#   true  - 使用 MLP 动态预测 alpha/tau
#   false - 使用网格搜索（静态 alpha/tau）
USE_MLP=false

# ============================================================
# 模型路径配置（根据上面的开关设置）
# ============================================================
# LAP 模型路径（USE_LAP=true 时需要）
LAP_MODEL_PATH="train/output/Hybrid/4.3lap新损失测试/lap_only/4.4-2200-fully-adaptive/deir_best_lap_only.pt"
# MLP 模型路径（USE_MLP=true 时需要）
MLP_MODEL_PATH=""
# MLP 隐藏层维度
MLP_HIDDEN_DIM=256

# ============================================================
# 编码器类型配置: "bge", "mistral" 或 "repllama"
# ============================================================
ENCODER_TYPE="repllama"  # 修改这里切换模型

# ============================================================
# 根据编码器类型自动设置参数
# ============================================================
if [ "$ENCODER_TYPE" = "mistral" ]; then
    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    BATCH_SIZE=28
    EMBED_DIM=4096
    MODEL_SHORT="e5-mistral-7b"
elif [ "$ENCODER_TYPE" = "bge" ]; then
    MODEL_NAME="BAAI/bge-large-en-v1.5"
    BATCH_SIZE=256
    EMBED_DIM=1024
    MODEL_SHORT="bge-large-en"
elif [ "$ENCODER_TYPE" = "repllama" ]; then
    MODEL_NAME="samaya-ai/RepLLaMA-reproduced"
    BATCH_SIZE=8
    EMBED_DIM=4096
    MODEL_SHORT="repllama-reproduced"
else
    echo "错误: 未知的编码器类型 '$ENCODER_TYPE'"
    echo "支持的类型: bge, mistral, repllama"
    exit 1
fi

GPU_ID=1
SEED=42
VERBOSE=false
TASK="Core17InstructionRetrieval"
# TASK="Robust04InstructionRetrieval"
# TASK="News21InstructionRetrieval"

# 网格搜索参数（可配置）
ALPHAS="0.0,0.5,1.0,2.0,3.0,5.0"
TAUS="0.5,0.6,0.7,0.8,0.9,0.95"
NUM_SAMPLES=15  # 随机抽样的参数组合数量

# 实验备注
EXPERIMENT_NOTE="LAP模块评估: 完全自适应版本, sim_good=0.537, sim_bad=0.614, 动态目标=0.455"

# ============================================================
# 自定义输出路径配置（直接修改此变量即可覆盖自动生成的路径）
# ============================================================
CUSTOM_OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/grid_search/repllama-reproduced-max2600/Core17"
# 示例: CUSTOM_OUTPUT_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/grid_search/my_experiment"

# 输出路径
OUTPUT_BASE_DIR="/home/luwa/Documents/DSCLR/evaluation/dsclr/grid_search"
TIMESTAMP=$(date +%m%d-%H%M)
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_SHORT}_${TASK}_${TIMESTAMP}"

# ============================================================
# 显示帮助信息
# ============================================================
show_help() {
    cat << EOF
DSCLR 静态网格搜索评测脚本

用法: $0 [选项]

选项:
    -e, --encoder <type>      编码器类型: bge, mistral, repllama (默认: repllama)
    -t, --task <name>         评测任务 (必选)
    -O, --output <dir>        输出目录 (默认自动生成)
    -g, --gpu <id>            GPU编号: 0/1/2/3 (默认: 1)
    -b, --batch_size <size>   批处理大小 (默认根据encoder自动设置)
    -s, --seed <num>          随机种子 (默认: 42)
    -a, --alphas <values>     alpha 值列表，逗号分隔 (默认: 0.0,0.5,1.0,2.0,3.0,5.0)
    -T, --taus <values>       tau 值列表，逗号分隔 (默认: 0.5,0.6,0.7,0.8,0.9,0.95)
    -n, --num_samples <n>     随机抽样数量 (默认: 15)
    -N, --note <text>         实验备注 (可选，记录到配置文件中)
    -v, --verbose             显示详细日志
    -h, --help                显示帮助信息

示例:
    # 使用 Repllama 在 Core17 上进行网格搜索
    $0 --task Core17InstructionRetrieval

    # 使用 BGE 在 Robust04 上搜索
    $0 --encoder bge --task Robust04InstructionRetrieval

    # 指定 GPU 和输出目录
    $0 --task Core17InstructionRetrieval --gpu 0 --output /tmp/grid_search

    # 添加实验备注
    $0 --task Core17InstructionRetrieval --note "测试原始query+instruction作为S_base"
EOF
}

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--encoder)
            ENCODER_TYPE="$2"
            if [ "$ENCODER_TYPE" = "mistral" ]; then
                MODEL_NAME="intfloat/e5-mistral-7b-instruct"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MODEL_SHORT="e5-mistral-7b"
            elif [ "$ENCODER_TYPE" = "bge" ]; then
                MODEL_NAME="BAAI/bge-large-en-v1.5"
                BATCH_SIZE=256
                EMBED_DIM=1024
                MODEL_SHORT="bge-large-en"
            elif [ "$ENCODER_TYPE" = "repllama" ]; then
                MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
                BATCH_SIZE=28
                EMBED_DIM=4096
                MODEL_SHORT="repllama-v1-7b"
            fi
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -O|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -a|--alphas)
            ALPHAS="$2"
            shift 2
            ;;
        -T|--taus)
            TAUS="$2"
            shift 2
            ;;
        -n|--num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -N|--note)
            EXPERIMENT_NOTE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# 参数校验
# ============================================================
VALID_TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK} " ]]; then
    echo "❌ 错误: 无效任务 '$TASK'"
    echo "可用任务: ${VALID_TASKS[@]}"
    exit 1
fi

# 根据开关检查模型路径
if [ "$USE_LAP" = "true" ]; then
    if [ -z "$LAP_MODEL_PATH" ] || [ ! -f "$LAP_MODEL_PATH" ]; then
        echo "❌ 错误: USE_LAP=true 需要提供有效的 LAP_MODEL_PATH"
        echo "当前路径: $LAP_MODEL_PATH"
        exit 1
    fi
fi

if [ "$USE_MLP" = "true" ]; then
    if [ -z "$MLP_MODEL_PATH" ] || [ ! -f "$MLP_MODEL_PATH" ]; then
        echo "❌ 错误: USE_MLP=true 需要提供有效的 MLP_MODEL_PATH"
        echo "当前路径: $MLP_MODEL_PATH"
        exit 1
    fi
fi

# 如果设置了自定义输出路径，则使用它
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# 保存实验配置到文件（便于复现）
# ============================================================
CONFIG_FILE="${OUTPUT_DIR}/experiment_config.txt"
cat > "${CONFIG_FILE}" << EOF
============================================================
DSCLR 实验配置
============================================================

实验时间: $(date '+%Y-%m-%d %H:%M:%S')
输出目录: ${OUTPUT_DIR}

模块配置:
  USE_LAP: ${USE_LAP}
  USE_MLP: ${USE_MLP}

编码器配置:
  ENCODER_TYPE: ${ENCODER_TYPE}
  MODEL_NAME: ${MODEL_NAME}
  EMBED_DIM: ${EMBED_DIM}

模型路径:
  LAP_MODEL_PATH: ${LAP_MODEL_PATH:-未使用}
  MLP_MODEL_PATH: ${MLP_MODEL_PATH:-未使用}
  MLP_HIDDEN_DIM: ${MLP_HIDDEN_DIM}

网格搜索参数:
  ALPHAS: ${ALPHAS}
  TAUS: ${TAUS}
  NUM_SAMPLES: ${NUM_SAMPLES}

运行配置:
  TASK: ${TASK}
  GPU_ID: ${GPU_ID}
  BATCH_SIZE: ${BATCH_SIZE}
  SEED: ${SEED}

实验备注:
  ${EXPERIMENT_NOTE:-无}
EOF

echo "📝 实验配置已保存到: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

# ============================================================
# 主实验循环
# ============================================================
echo ""
echo "============================================================"
echo "开始运行 DSCLR 实验 [USE_LAP=${USE_LAP}, USE_MLP=${USE_MLP}]"
echo "============================================================"

# 记录开始时间
start_time=$(date +%s)

# 根据开关构建参数
EXTRA_ARGS=""
if [ "$USE_LAP" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --lap_model_path ${LAP_MODEL_PATH}"
fi
if [ "$USE_MLP" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --mlp_model_path ${MLP_MODEL_PATH} --mlp_hidden_dim ${MLP_HIDDEN_DIM}"
fi

# 运行实验
echo "运行命令:"
echo "  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u eval/engine_dscrl.py \\"
echo "    --model_name ${MODEL_NAME} \\"
echo "    --task_name ${TASK} \\"
echo "    --batch_size ${BATCH_SIZE} \\"
echo "    --alphas ${ALPHAS} \\"
echo "    --taus ${TAUS} \\"
echo "    --num_samples ${NUM_SAMPLES} \\"
echo "    --output_dir ${OUTPUT_DIR} \\"
if [ -n "$EXTRA_ARGS" ]; then
    echo "    ${EXTRA_ARGS}"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u eval/engine_dscrl.py \
    --model_name ${MODEL_NAME} \
    --task_name ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --alphas ${ALPHAS} \
    --taus ${TAUS} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    ${EXTRA_ARGS}

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
duration_sec=$((duration % 60))

echo ""
echo "============================================================"
echo "✅ 所有实验完成!"
echo "============================================================"
echo "总耗时: ${duration_min}分${duration_sec}秒"
echo "输出目录: ${OUTPUT_DIR}"
echo "实验配置: ${CONFIG_FILE}"
echo ""
echo "📊 结果汇总:"
python3 -c "
import json
from pathlib import Path
output_dir = Path('${OUTPUT_DIR}')
results = []
for exp_dir in sorted(output_dir.glob('trial_*')):
    metrics_file = exp_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        results.append({
            'dir': exp_dir.name,
            'map': metrics.get('map', {}).get('base', 0),
            'mrr': metrics.get('mrr', {}).get('base', 0),
            'p5': metrics.get('p5', {}).get('base', 0)
        })
if results:
    print(f'  共完成 {len(results)} 组实验')
    best = max(results, key=lambda x: x['map'])
    print(f'  最佳结果: {best[\"dir\"]}')
    print(f'    MAP: {best[\"map\"]:.4f}')
    print(f'    MRR: {best[\"mrr\"]:.4f}')
    print(f'    P@5: {best[\"p5\"]:.4f}')
"
