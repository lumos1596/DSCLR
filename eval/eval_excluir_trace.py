"""
TRACE 引擎在 ExcluIR 基准上的评测脚本

ExcluIR: Exclusionary Neural Information Retrieval (https://arxiv.org/abs/2404.17288)
数据格式:
  - corpus.json: 文档列表 ["doc1", "doc2", ...]
  - test_manual_final.json: 查询列表 [{"ExcluQ": "...", "index": [neg_idx, pos_idx]}, ...]

评估指标:
  - R@1, R@5, R@10: 正例文档的召回率
  - MRR@10: 正例文档的倒数排名
  - ΔR@1: R@1(pos) - R@1(neg)
  - ΔMRR@10: MRR@10(pos) - MRR@10(neg)
  - RR: 正例排名高于负例的查询比例

Usage:
  python -m eval.eval_excluir_trace \
    --model_name BAAI/bge-large-en-v1.5 \
    --data_dir dataset/ExcluIR \
    --dual_queries_path dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl \
    --device cuda
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ExcluIR Metrics
# ============================================================

def compute_recall(result_list: List[List[int]], right_list: List[List[int]],
                   recall_k: List[int] = [1, 5, 10]) -> List[float]:
    """计算 Recall@K (ExcluIR 原始实现)"""
    recall_values = [0.0] * len(recall_k)
    for i in range(len(result_list)):
        retrieved = set(result_list[i])
        relevant = set(right_list[i])
        if not relevant:
            continue
        for j, k in enumerate(recall_k):
            hit = len(retrieved & set(result_list[i][:k]) & relevant)
            recall_values[j] += hit / len(relevant)
    return [r / len(result_list) for r in recall_values]


def compute_mrr(result_list: List[List[int]], right_list: List[List[int]]) -> float:
    """计算 MRR@10 (ExcluIR 原始实现)"""
    mrr = 0.0
    for i in range(len(result_list)):
        relevant = set(right_list[i])
        for rank, doc_idx in enumerate(result_list[i][:10]):
            if doc_idx in relevant:
                mrr += 1.0 / (rank + 1)
                break
    return mrr / len(result_list)


def compute_right_rank(result_list: List[List[int]],
                       neg_indices: List[int],
                       pos_indices: List[int]) -> float:
    """计算 RR (Right Rank): 正例排名高于负例的查询比例

    - 如果正例和负例都在结果中: 正例排名更高则正确
    - 如果只有正例在结果中(负例被排除): 正确
    - 其他情况: 不正确
    """
    right_count = 0
    for i in range(len(result_list)):
        neg_idx = neg_indices[i]
        pos_idx = pos_indices[i]
        result = result_list[i]

        pos_rank = None
        neg_rank = None
        for rank, doc_idx in enumerate(result):
            if doc_idx == pos_idx:
                pos_rank = rank
            if doc_idx == neg_idx:
                neg_rank = rank

        if pos_rank is not None and neg_rank is not None:
            # 两者都在结果中，正例排名更高则正确
            if pos_rank < neg_rank:
                right_count += 1
        elif pos_rank is not None and neg_rank is None:
            # 只有正例在结果中，负例被排除 → 正确
            right_count += 1
        # 其他情况: 只有负例/两者都不在 → 不正确

    return right_count / len(result_list)


def evaluate_excluir(result_list: List[List[int]],
                     neg_indices: List[int],
                     pos_indices: List[int]) -> Dict[str, float]:
    """计算所有 ExcluIR 指标"""
    right_list_pos = [[pos_indices[i]] for i in range(len(pos_indices))]
    right_list_neg = [[neg_indices[i]] for i in range(len(neg_indices))]

    recall_pos = compute_recall(result_list, right_list_pos)
    recall_neg = compute_recall(result_list, right_list_neg)
    mrr_pos = compute_mrr(result_list, right_list_pos)
    mrr_neg = compute_mrr(result_list, right_list_neg)
    rr = compute_right_rank(result_list, neg_indices, pos_indices)

    metrics = {
        "R@1": round(recall_pos[0] * 100, 2),
        "R@5": round(recall_pos[1] * 100, 2),
        "R@10": round(recall_pos[2] * 100, 2),
        "MRR@10": round(mrr_pos * 100, 2),
        "R@1_neg": round(recall_neg[0] * 100, 2),
        "MRR@10_neg": round(mrr_neg * 100, 2),
        "delta_R@1": round(recall_pos[0] * 100, 2) - round(recall_neg[0] * 100, 2),
        "delta_MRR@10": round(mrr_pos * 100, 2) - round(mrr_neg * 100, 2),
        "RR": round(rr * 100, 2),
    }
    return metrics


# ============================================================
# ExcluIR Data Loading
# ============================================================

def load_excluir_data(data_dir: str) -> Tuple[List[str], List[Dict]]:
    """加载 ExcluIR 数据

    Returns:
        corpus: 文档文本列表
        queries: 查询列表，每个包含 ExcluQ 和 index
    """
    corpus_path = os.path.join(data_dir, "corpus.json")
    queries_path = os.path.join(data_dir, "test_manual_final.json")

    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"Loaded {len(corpus)} documents")

    logger.info(f"Loading queries from {queries_path}...")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries")

    return corpus, queries


def load_dual_queries(dual_queries_path: str) -> Dict[int, Dict[str, Any]]:
    """加载 dual queries (q_plus, q_minus)

    Returns:
        {query_index: {"q_plus": ..., "q_minus": ...}}
    """
    dual_data = {}
    with open(dual_queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            qid = item["qid"]
            dual_data[qid] = item
    logger.info(f"Loaded {len(dual_data)} dual queries")
    return dual_data


# ============================================================
# TRACE Scoring (reused from engine_trace.py)
# ============================================================

from eval.engine_trace import (
    robust_standardize, fit_huber_regression, _mad,
    TRACEQueryResult, HuberFitResult,
)


def trace_score_query(
    s_full: torch.Tensor,
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    has_neg: bool,
    lambda_boundary: float = 1.0,
    tau_decay: float = 0.2,
    huber_delta: float = 1.345,
    eps: float = 1e-6,
) -> torch.Tensor:
    """对单个查询应用 TRACE 打分

    Args:
        s_full: 全查询与所有文档的相似度 [n_docs]
        s_pos: Q_plus 与所有文档的相似度 [n_docs]
        s_neg: Q_minus 与所有文档的相似度 [n_docs]
        has_neg: 是否有排除指令

    Returns:
        s_final: TRACE 修正后的得分 [n_docs]
    """
    n_docs = s_full.numel()
    if n_docs <= 1:
        return s_full.clone()

    if not has_neg:
        return s_full.clone()

    # Step 1: Robust standardization
    z_full = robust_standardize(s_full, eps)
    z_pos = robust_standardize(s_pos, eps)
    z_neg = robust_standardize(s_neg, eps)

    # Step 2: Huber regression z_neg = a + b * z_pos
    a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=huber_delta)

    # Step 3: Residual normalization
    e = z_neg - a_hat - b_hat * z_pos
    e_median = e.median()
    e_mad = _mad(e, eps)
    r = (e - e_median) / e_mad

    # Step 4: Monotone composition
    p = F.relu(z_pos)
    h = F.relu(r - lambda_boundary)
    g = torch.exp(-h / tau_decay)
    s_final = z_full + p * g - h

    return s_final


# ============================================================
# Main Evaluation
# ============================================================

def run_excluir_trace(
    model_name: str = "BAAI/bge-large-en-v1.5",
    data_dir: str = "dataset/ExcluIR",
    dual_queries_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    huber_delta: float = 1.345,
    lambda_boundary: float = 1.0,
    tau_decay: float = 0.2,
    eps: float = 1e-6,
    device: str = "auto",
    batch_size: int = 64,
    top_k: int = 10,
    use_cache: bool = True,
    lambda_list: Optional[List[float]] = None,
    tau_list: Optional[List[float]] = None,
):
    """运行 TRACE 在 ExcluIR 上的评测"""
    if output_dir is None:
        output_dir = f"evaluation/excluir_trace/{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    if device == "auto":
        try:
            torch.cuda._lazy_init()
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Load data
    corpus, queries = load_excluir_data(data_dir)
    n_docs = len(corpus)

    # Load dual queries if provided
    dual_data = {}
    has_dual = False
    if dual_queries_path and os.path.exists(dual_queries_path):
        dual_data = load_dual_queries(dual_queries_path)
        has_dual = len(dual_data) > 0

    # Load encoder
    from eval.models import ModelFactory
    logger.info(f"Loading encoder: {model_name}")
    encoder = ModelFactory.create(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    # Encode corpus
    cache_dir = os.path.join(data_dir, "embeddings")
    model_short_name = model_name.split("/")[-1].replace("-", "_")
    cache_file = os.path.join(cache_dir, f"{model_short_name}_corpus_embeddings.npy")

    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached corpus embeddings from {cache_file}")
        doc_embeddings = np.load(cache_file)
        doc_embeddings = torch.tensor(doc_embeddings, device=device, dtype=torch.float16)
    else:
        logger.info(f"Encoding {n_docs} documents...")
        doc_embeddings = encoder.encode_documents(corpus, batch_size=batch_size)
        doc_embeddings = doc_embeddings.to(device=device, dtype=torch.float16)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, doc_embeddings.cpu().float().numpy())
            logger.info(f"Cached corpus embeddings to {cache_file}")

    # Normalize
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=-1)
    logger.info(f"Corpus embeddings shape: {doc_embeddings.shape}")

    # Prepare query texts
    query_texts = []
    q_plus_texts = []
    q_minus_texts = []
    neg_indices = []
    pos_indices = []
    has_neg_flags = []

    for i, q in enumerate(queries):
        # Support both ExcluQ and RQ_rewrite field names
        query_text = q.get("ExcluQ", q.get("RQ_rewrite", ""))
        index = q.get("index", q.get("corpus_sub_index", []))

        if not query_text or len(index) < 2:
            continue

        query_texts.append(query_text)
        neg_indices.append(index[0])
        pos_indices.append(index[1] if len(index) > 1 else index[0])

        if has_dual and i in dual_data:
            d = dual_data[i]
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")
            if not q_minus or q_minus.strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", ""):
                q_minus = ""
                has_neg_flags.append(False)
            else:
                has_neg_flags.append(True)
            q_plus_texts.append(q_plus if q_plus else query_text)
            q_minus_texts.append(q_minus)
        else:
            q_plus_texts.append(query_text)
            q_minus_texts.append("")
            has_neg_flags.append(False)

    n_queries = len(query_texts)
    logger.info(f"Processing {n_queries} queries (dual queries: {has_dual})")

    # Encode queries
    logger.info("Encoding Q_full queries...")
    q_full_emb = encoder.encode_queries(query_texts, batch_size=batch_size)
    q_full_emb = F.normalize(q_full_emb.to(device).float(), p=2, dim=-1)

    logger.info("Encoding Q_plus queries...")
    q_pos_emb = encoder.encode_queries(q_plus_texts, batch_size=batch_size)
    q_pos_emb = F.normalize(q_pos_emb.to(device).float(), p=2, dim=-1)

    # Encode Q_minus (only non-empty ones)
    q_minus_emb = None
    if any(has_neg_flags):
        non_empty_minus = [(i, q_minus_texts[i]) for i in range(n_queries) if has_neg_flags[i]]
        minus_indices = [x[0] for x in non_empty_minus]
        minus_texts = [x[1] for x in non_empty_minus]

        logger.info(f"Encoding {len(minus_texts)} Q_minus queries...")
        q_neg_emb_all = encoder.encode_queries(minus_texts, batch_size=batch_size)
        q_neg_emb_all = F.normalize(q_neg_emb_all.to(device).float(), p=2, dim=-1)

        # Build full Q_minus embedding matrix (zeros for queries without Q_minus)
        q_minus_emb = torch.zeros_like(q_full_emb)
        for j, orig_idx in enumerate(minus_indices):
            q_minus_emb[orig_idx] = q_neg_emb_all[j]

    # Compute similarity scores
    logger.info("Computing similarity scores...")
    S_full = torch.matmul(q_full_emb, doc_embeddings.T)  # [n_queries, n_docs]
    S_pos = torch.matmul(q_pos_emb, doc_embeddings.T)    # [n_queries, n_docs]
    S_neg = torch.zeros_like(S_full)
    if q_minus_emb is not None:
        S_neg = torch.matmul(q_minus_emb, doc_embeddings.T)

    # ---- Baseline (no TRACE) ----
    logger.info("Computing baseline results (no TRACE)...")
    baseline_results = []
    for i in range(n_queries):
        scores = S_full[i]
        top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
        baseline_results.append(top_k_indices)

    baseline_metrics = evaluate_excluir(baseline_results, neg_indices, pos_indices)
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # ---- TRACE with grid search ----
    if lambda_list is None:
        lambda_list = [0.5, 1.0, 1.5, 2.0]
    if tau_list is None:
        tau_list = [0.1, 0.2, 0.5, 1.0]

    total_trials = len(lambda_list) * len(tau_list)
    best_metrics = None
    best_params = None
    all_results = []

    trial_idx = 0
    for lam in lambda_list:
        for tau_d in tau_list:
            trial_idx += 1

            trace_results = []
            for i in range(n_queries):
                s_full_i = S_full[i]
                s_pos_i = S_pos[i]
                s_neg_i = S_neg[i]
                has_neg_i = has_neg_flags[i]

                if has_neg_i:
                    s_final = trace_score_query(
                        s_full_i, s_pos_i, s_neg_i,
                        has_neg=True,
                        lambda_boundary=lam,
                        tau_decay=tau_d,
                        huber_delta=huber_delta,
                        eps=eps,
                    )
                else:
                    s_final = s_full_i

                top_k_indices = torch.topk(s_final, min(top_k, n_docs)).indices.cpu().tolist()
                trace_results.append(top_k_indices)

            metrics = evaluate_excluir(trace_results, neg_indices, pos_indices)

            logger.info(
                "[%d/%d] lambda=%.1f, tau=%.2f: R@1=%.2f, delta_R@1=%.2f, "
                "MRR@10=%.2f, delta_MRR@10=%.2f, RR=%.2f",
                trial_idx, total_trials, lam, tau_d,
                metrics["R@1"], metrics["delta_R@1"],
                metrics["MRR@10"], metrics["delta_MRR@10"],
                metrics["RR"],
            )

            all_results.append({
                "lambda": lam,
                "tau_decay": tau_d,
                **metrics,
            })

            composite = metrics["delta_R@1"] + metrics["delta_MRR@10"] + metrics["RR"]
            if best_metrics is None:
                best_metrics = metrics
                best_params = {"lambda": lam, "tau_decay": tau_d}
            else:
                best_composite = (
                    best_metrics["delta_R@1"]
                    + best_metrics["delta_MRR@10"]
                    + best_metrics["RR"]
                )
                if composite > best_composite:
                    best_metrics = metrics
                    best_params = {"lambda": lam, "tau_decay": tau_d}

    # Summary
    logger.info("=" * 60)
    logger.info("ExcluIR TRACE Evaluation Complete")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Best params: lambda={best_params['lambda']}, tau={best_params['tau_decay']}")
    logger.info(f"  Baseline: {baseline_metrics}")
    logger.info(f"  Best TRACE: {best_metrics}")
    logger.info("=" * 60)

    # Save results
    result_data = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_queries": n_queries,
        "n_docs": n_docs,
        "has_dual_queries": has_dual,
        "baseline_metrics": baseline_metrics,
        "best_trace_metrics": best_metrics,
        "best_params": best_params,
        "all_grid_results": all_results,
    }

    result_path = os.path.join(output_dir, "excluir_trace_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {result_path}")

    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE evaluation on ExcluIR")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--data_dir", type=str, default="dataset/ExcluIR")
    parser.add_argument("--dual_queries_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--huber_delta", type=float, default=1.345)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--tau_decay", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_cache", type=str, default="true")

    args = parser.parse_args()
    use_cache = args.use_cache.lower() == "true"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_excluir_trace(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        huber_delta=args.huber_delta,
        lambda_boundary=args.lambda_boundary,
        tau_decay=args.tau_decay,
        eps=args.eps,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        use_cache=use_cache,
    )

    print(f"\nBaseline: {result['baseline_metrics']}")
    print(f"Best TRACE: {result['best_trace_metrics']}")
    print(f"Best params: {result['best_params']}")
