"""
Promptriever + TRACE FollowIR 评测脚本

实验目标：
  以 Promptriever (LLaMA 3.1 8B Instruct) 为初筛编码器，
  对比加入 TRACE 框架后是否能进一步增强 Promptriever 性能。

对比三种模式：
  1. baseline     —— TRACE 引擎但 ablation="z_full_only"
                     （z_full 是 S_full 的鲁棒标准化，单调变换，排名等价于原始 S_full，
                      即 Promptriever 原生相似度排名；作为基线）
  2. trace_full   —— TRACE 完整公式  S_final = z_full + p*g - h
  3. trace_pos_only —— 仅正通道      S_final = z_full + p
                     （项目记忆：在 RepLLaMA 上 pos_only 比 full 的 p-MRR 高 16pp）

每个 (mode, task) 进行 lambda × tau 网格搜索，取 p-MRR + changed_MAP + changed_nDCG 最大组合。

输出：
  results/promptriever_trace_followir/
    ├── baseline/{task}/trace_metrics_summary.json
    ├── trace_full/{task}/trace_metrics_summary.json
    ├── trace_pos_only/{task}/trace_metrics_summary.json
    └── comparison_summary.json   # 三种模式 × 3 任务 + target_avg

用法：
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_promptriever_trace_followir --device cuda
"""

import sys
import os

# IMPORTANT: 设置 HF 环境变量必须在 import torch/transformers 之前
# 因为 huggingface_hub.constants 在 import 时读取这些 env var
os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 允许在线下载（Promptriever adapter 和 LLaMA 3.1 基础模型需要从 HF 镜像下载）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gc
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List

import torch

# 显式覆盖 huggingface_hub 的 offline 常量（engine_dscrl.py 可能在 import 时设置了 env var）
try:
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HUB_OFFLINE = False
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "samaya-ai/promptriever-llama3.1-8b-instruct-v1"
TASKS = [
    "Core17InstructionRetrieval",
    "Robust04InstructionRetrieval",
    "News21InstructionRetrieval",
]
DUAL_QUERIES_DIR = "dataset/FollowIR_test/dual_queries_v7_mixed"
OUTPUT_BASE = "results/promptriever_trace_followir_v7_mean"

# 网格搜索参数空间
LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0]
TAU_LIST = [0.1, 0.2, 0.5, 1.0]


def get_dual_queries_path(task_name: str) -> str:
    return os.path.join(DUAL_QUERIES_DIR, f"dual_queries_v7_mixed_{task_name}.jsonl")


def run_trace_for_task(
    task_name: str,
    mode: str,
    output_dir: str,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    """运行单个 (mode, task) 的 TRACE 评测。

    mode 取值：
      - "baseline"      -> ablation="z_full_only"，单点参数（无网格搜索）
      - "trace_full"    -> ablation="full"，网格搜索
      - "trace_pos_only"-> ablation="pos_only"，网格搜索
    """
    from eval.engine_trace import TRACEEvaluator

    if mode == "baseline":
        ablation = "z_full_only"
        lambda_list = [1.0]
        tau_list = [0.2]
    elif mode == "trace_full":
        ablation = "full"
        lambda_list = LAMBDA_LIST
        tau_list = TAU_LIST
    elif mode == "trace_pos_only":
        ablation = "pos_only"
        lambda_list = LAMBDA_LIST
        tau_list = TAU_LIST
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("=" * 60)
    logger.info(f"Run TRACE | task={task_name} | mode={mode} | ablation={ablation}")
    logger.info("=" * 60)

    engine = TRACEEvaluator(
        model_name=MODEL_NAME,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=get_dual_queries_path(task_name),
        ablation=ablation,
        residual_pooling="mean",
        device=device,
        batch_size=batch_size,
        use_cache=True,
    )
    result = engine.run(lambda_list=lambda_list, tau_list=tau_list)

    # 释放显存
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return result


def extract_metrics(result: Dict[str, Any], task_name: str) -> Dict[str, float]:
    """从 TRACE 返回结果中提取关键指标。"""
    best_metrics = result.get("best_metrics", {}) or {}
    best_params = result.get("best_params", {}) or {}
    return {
        "task": task_name,
        "p-MRR": float(best_metrics.get("p-MRR", 0.0)),
        "og_MAP@1000": float(best_metrics.get("original", {}).get("map_at_1000", 0.0)),
        "og_nDCG@5": float(best_metrics.get("original", {}).get("ndcg_at_5", 0.0)),
        "changed_MAP@1000": float(best_metrics.get("changed", {}).get("map_at_1000", 0.0)),
        "changed_nDCG@5": float(best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)),
        "best_lambda": float(best_params.get("lambda", 0.0)),
        "best_tau": float(best_params.get("tau_decay", 0.0)),
    }


def compute_target_avg(per_task_metrics: Dict[str, Dict[str, float]]) -> float:
    """target_avg = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3"""
    c17 = per_task_metrics.get("Core17InstructionRetrieval", {}).get("changed_MAP@1000", 0.0)
    r04 = per_task_metrics.get("Robust04InstructionRetrieval", {}).get("changed_MAP@1000", 0.0)
    n21 = per_task_metrics.get("News21InstructionRetrieval", {}).get("changed_nDCG@5", 0.0)
    return (c17 + r04 + n21) / 3.0


def main():
    parser = argparse.ArgumentParser(description="Promptriever + TRACE FollowIR Evaluation")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task names or 'all'")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated modes: baseline,trace_full,trace_pos_only or 'all'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). Sets CUDA_VISIBLE_DEVICES.")
    args = parser.parse_args()

    # 限制可见 GPU（避免占用其他进程正在使用的 GPU）
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Restricting CUDA_VISIBLE_DEVICES to: {args.gpus}")

    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    all_modes = ["baseline", "trace_full", "trace_pos_only"]
    modes = all_modes if args.modes == "all" else args.modes.split(",")

    # 收集所有结果: {mode: {task: metrics}}
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in modes}

    # 按任务迭代（同一任务先编码一次文档，三种模式共享缓存）
    for task_name in tasks:
        for mode in modes:
            out_dir = os.path.join(args.output_base, mode, task_name)
            try:
                result = run_trace_for_task(
                    task_name=task_name,
                    mode=mode,
                    output_dir=out_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                all_results[mode][task_name] = extract_metrics(result, task_name)
            except Exception as e:
                logger.exception(f"Failed: mode={mode}, task={task_name}: {e}")
                all_results[mode][task_name] = {
                    "task": task_name,
                    "error": str(e),
                }

    # 生成对比汇总
    comparison = {
        "experiment": "Promptriever + TRACE FollowIR",
        "model": MODEL_NAME,
        "dual_queries_source": DUAL_QUERIES_DIR,
        "grid_search": {
            "lambda_list": LAMBDA_LIST,
            "tau_list": TAU_LIST,
        },
        "timestamp": datetime.now().isoformat(),
        "modes": {},
    }

    for mode in modes:
        per_task = all_results[mode]
        # 计算该模式的 target_avg（仅当 3 个任务都有结果时）
        valid_tasks = {t: m for t, m in per_task.items() if "error" not in m}
        target_avg = compute_target_avg(valid_tasks) if len(valid_tasks) == 3 else None
        avg_pmrr = (
            sum(m.get("p-MRR", 0.0) for m in valid_tasks.values()) / len(valid_tasks)
            if valid_tasks else None
        )
        comparison["modes"][mode] = {
            "per_task": per_task,
            "target_avg": target_avg,
            "avg_p-MRR": avg_pmrr,
        }

    os.makedirs(args.output_base, exist_ok=True)
    summary_path = os.path.join(args.output_base, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 打印汇总
    logger.info("\n" + "=" * 80)
    logger.info("Promptriever + TRACE FollowIR — Comparison Summary")
    logger.info("=" * 80)
    header = f"{'Mode':<18}{'Core17 pMRR':<14}{'Core17 cMAP':<14}{'R04 pMRR':<12}{'R04 cMAP':<12}{'N21 pMRR':<12}{'N21 cnDCG5':<14}{'target_avg':<12}{'avg_pMRR':<10}"
    logger.info(header)
    for mode in modes:
        m_info = comparison["modes"][mode]
        per_task = m_info["per_task"]
        c17 = per_task.get("Core17InstructionRetrieval", {})
        r04 = per_task.get("Robust04InstructionRetrieval", {})
        n21 = per_task.get("News21InstructionRetrieval", {})
        ta = m_info["target_avg"]
        ap = m_info["avg_p-MRR"]
        logger.info(
            f"{mode:<18}"
            f"{c17.get('p-MRR', 0):<14.4f}"
            f"{c17.get('changed_MAP@1000', 0):<14.4f}"
            f"{r04.get('p-MRR', 0):<12.4f}"
            f"{r04.get('changed_MAP@1000', 0):<12.4f}"
            f"{n21.get('p-MRR', 0):<12.4f}"
            f"{n21.get('changed_nDCG@5', 0):<14.4f}"
            f"{(ta if ta is not None else 0):<12.4f}"
            f"{(ap if ap is not None else 0):<10.4f}"
        )
    logger.info(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
