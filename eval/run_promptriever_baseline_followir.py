"""
Promptriever-LLaMA3 Baseline FollowIR 评测脚本

不加 DSCLR，直接用 Promptriever 编码查询+文档计算相似度。
用于对比 DSCLR 模块是否带来提升。

用法:
    python eval/run_promptriever_baseline_followir.py --tasks Core17InstructionRetrieval
    python eval/run_promptriever_baseline_followir.py --tasks all
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
import gc
from datetime import datetime

import torch

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TASKS = [
    "Core17InstructionRetrieval",
    "Robust04InstructionRetrieval",
    "News21InstructionRetrieval",
]


def run_promptriever_baseline(task_name: str, output_dir: str, device: str = "cuda", batch_size: int = 64):
    """Promptriever-LLaMA3 baseline 评测（不加 DSCLR）
    
    复用 DSCLREvaluatorEngine 的文档索引和结果提取逻辑，
    只是不应用 DSCLR 的 reward-penalty 公式。
    """
    from eval.engine_dscrl import DSCLREvaluatorEngine
    from eval.metrics import FollowIREvaluator

    logger.info(f"=" * 60)
    logger.info(f"Promptriever-LLaMA3 Baseline (no DSCLR) - {task_name}")
    logger.info(f"=" * 60)

    # 使用 DSCLREvaluatorEngine 加载编码器和文档索引
    model_name = "samaya-ai/promptriever-llama3.1-8b-instruct-v1"
    engine = DSCLREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        use_cache=True,
    )

    # 加载缓存的文档嵌入
    cache_dir = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/promptriever_llama31_8b_instruct"
    cache_emb_file = os.path.join(cache_dir, f"{task_name}_corpus_embeddings.npy")
    cache_ids_file = os.path.join(cache_dir, f"{task_name}_corpus_ids.json")
    
    if os.path.exists(cache_emb_file) and os.path.exists(cache_ids_file):
        import numpy as np
        embeddings = np.load(cache_emb_file)
        with open(cache_ids_file, 'r') as f:
            doc_ids = json.load(f)
        engine.retriever.set_embeddings(torch.tensor(embeddings), doc_ids)
        logger.info(f"Loaded cached document embeddings: {embeddings.shape}")
    else:
        logger.error(f"No cached document embeddings found at {cache_dir}! Run DeIR-Dual V2 first to cache them.")
        return None

    # 加载数据
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    og_ids = list(q_og.keys())
    changed_ids = list(q_changed.keys())

    # 编码 OG 查询
    og_queries = [q_og[qid] for qid in og_ids]
    logger.info(f"Encoding {len(og_queries)} OG queries...")
    q_emb_og = engine._encode_queries(og_queries)

    # 编码 Changed 查询
    changed_queries = [q_changed[qid] for qid in changed_ids]
    logger.info(f"Encoding {len(changed_queries)} Changed queries...")
    q_emb_changed = engine._encode_queries(changed_queries)

    # 计算相似度（不加 DSCLR，直接用原始相似度）
    dev = engine.retriever.doc_embeddings.device
    q_emb_og = q_emb_og.to(dev)
    q_emb_changed = q_emb_changed.to(dev)

    S_og = torch.matmul(q_emb_og, engine.retriever.doc_embeddings.T)
    S_changed = torch.matmul(q_emb_changed, engine.retriever.doc_embeddings.T)

    # 提取结果（复用引擎的 _extract_results 方法）
    results_og = engine._extract_results(S_og, og_ids, candidates)
    results_changed = engine._extract_results(S_changed, changed_ids, candidates)

    # 计算指标
    evaluator = FollowIREvaluator(task_name)
    metrics = evaluator.evaluate(results_og, results_changed)

    p_mrr = metrics.get('p-MRR', 0.0)
    og_map = metrics.get('original', {}).get('map_at_1000', 0.0)
    ch_map = metrics.get('changed', {}).get('map_at_1000', 0.0)
    og_ndcg5 = metrics.get('original', {}).get('ndcg_at_5', 0.0)
    ch_ndcg5 = metrics.get('changed', {}).get('ndcg_at_5', 0.0)

    logger.info("=" * 60)
    logger.info(f"Promptriever-LLaMA3 Baseline Results - {task_name}")
    logger.info(f"  p-MRR: {p_mrr:.4f}")
    logger.info(f"  OG MAP@1000: {og_map:.4f}, Changed MAP@1000: {ch_map:.4f}")
    logger.info(f"  OG nDCG@5: {og_ndcg5:.4f}, Changed nDCG@5: {ch_ndcg5:.4f}")
    logger.info("=" * 60)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'task': task_name,
        'encoder': model_name,
        'mode': 'Promptriever-LLaMA3 Baseline (no DSCLR)',
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 释放显存
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Promptriever-LLaMA3 Baseline FollowIR Evaluation")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task names or 'all'")
    parser.add_argument("--output_dir", type=str, default="results/promptriever_llama3_baseline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")

    all_results = {}

    for task_name in tasks:
        out_dir = os.path.join(args.output_dir, task_name)
        metrics = run_promptriever_baseline(task_name, out_dir, args.device, args.batch_size)
        all_results[task_name] = {
            'p-MRR': metrics.get('p-MRR', 0.0),
            'OG_MAP@1000': metrics.get('original', {}).get('map_at_1000', 0.0),
            'Changed_MAP@1000': metrics.get('changed', {}).get('map_at_1000', 0.0),
            'OG_nDCG@5': metrics.get('original', {}).get('ndcg_at_5', 0.0),
            'Changed_nDCG@5': metrics.get('changed', {}).get('ndcg_at_5', 0.0),
        }

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("Promptriever-LLaMA3 Baseline FollowIR Results Summary")
    logger.info("=" * 60)
    for task, res in all_results.items():
        logger.info(f"  {task}: p-MRR={res['p-MRR']:.4f}, OG_MAP={res['OG_MAP@1000']:.4f}, "
                     f"Changed_MAP={res['Changed_MAP@1000']:.4f}, "
                     f"OG_nDCG@5={res['OG_nDCG@5']:.4f}, Changed_nDCG@5={res['Changed_nDCG@5']:.4f}")


if __name__ == "__main__":
    main()
