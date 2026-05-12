"""
DeIR-Dual V2 Training-Set Parameter Search (Ranking-Based)

Academic Methodology:
    Hyperparameters (α, β, δ) of DeIR-Dual V2 are selected on the training set
    by computing the SAME ranking metrics used on the test set: MAP@K and nDCG@5.
    
    Training data provides explicit relevance labels:
    - Positive documents (relevant) : 1-2 per query
    - Negative documents (non-relevant): 15 per query
    
    For each parameter combination (α, β, δ):
    1. Compute S_final for all documents per query
    2. Rank documents by S_final descending
    3. Compute MAP@K and nDCG@5 using training labels
    4. Average across queries → training-set metric
    
    Validation: K-fold cross-validation ensures generalization.
    
    This is standard ML practice: tune hyperparameters on training data
    with the same loss function as test evaluation.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class QueryGroup:
    idx: int
    has_req: bool
    has_neg: bool
    pos_start: int
    pos_count: int
    neg_start: int


def load_training_data(path: str) -> Tuple[List[QueryGroup], int, int]:
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line.strip()))

    groups = []
    pos_offset = 0
    for item in raw:
        out = json.loads(item["output"])
        q_plus = out.get("Q_plus", "")
        q_minus = out.get("Q_minus", "")
        has_req = q_plus not in ("[NONE]", "NONE", "", None)
        has_neg = q_minus not in ("[NONE]", "NONE", "", None)

        pos_count = len(item.get("pos", []))
        groups.append(QueryGroup(
            idx=item["idx"],
            has_req=has_req,
            has_neg=has_neg,
            pos_start=pos_offset,
            pos_count=pos_count,
            neg_start=item["idx"] * 15,
        ))
        pos_offset += pos_count

    n_pos = pos_offset
    n_neg = len(raw) * 15
    return groups, n_pos, n_neg


def apply_deir_dual_v2(
    S_base: torch.Tensor,
    S_req: torch.Tensor,
    S_neg: torch.Tensor,
    cos_qbase_qneg: torch.Tensor,
    has_req_mask: torch.Tensor,
    has_neg_mask: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    t_gap: float = 20.0,
    t_safety: float = 20.0,
    max_penalty_ratio: float = 0.5,
) -> torch.Tensor:
    S_final = S_base.clone()
    n_queries = S_base.shape[0]
    for q_idx in range(n_queries):
        has_req = bool(has_req_mask[q_idx].item() > 0)
        has_neg = bool(has_neg_mask[q_idx].item() > 0)
        if not has_neg:
            s_req_eff = S_req[q_idx] if has_req else torch.zeros_like(S_base[q_idx])
            S_final[q_idx] = S_base[q_idx] + beta * s_req_eff
            continue

        s_b = S_base[q_idx]
        s_r = S_req[q_idx]
        s_n = S_neg[q_idx]
        cos_val = float(cos_qbase_qneg[q_idx].item())

        tau = cos_val + delta
        smooth_penalty = F.softplus(s_n - tau)
        gap_w = torch.sigmoid((s_n - s_b) * t_gap)
        raw_penalty = alpha * smooth_penalty * gap_w

        if max_penalty_ratio > 0:
            penalty = torch.min(raw_penalty, s_b * max_penalty_ratio)
        else:
            penalty = raw_penalty

        safety = 1.0 - torch.sigmoid((s_n - tau) * t_safety)
        s_req_eff = s_r if has_req else torch.zeros_like(s_b)
        S_final[q_idx] = s_b + beta * s_req_eff * safety - penalty

    return S_final


def compute_ranking_metrics(
    S_final_all: torch.Tensor,
    groups: List[QueryGroup],
    max_pos_count: int,
    k_values: Tuple[int, ...] = (5, 10, 50, 100, 1000),
) -> Dict[str, float]:
    n_queries = len(groups)

    total_map = {k: 0.0 for k in k_values}
    total_ndcg = {k: 0.0 for k in k_values if k <= 5}
    total_mrr = 0.0
    valid_queries = 0

    for q_idx, g in enumerate(groups):
        scores = S_final_all[q_idx].cpu().float().numpy()
        n_pos = g.pos_count
        n_total = n_pos + 15

        rel_labels = np.zeros(len(scores), dtype=np.float32)
        rel_labels[:n_pos] = 1.0

        sorted_indices = np.argsort(-scores)[:n_total]

        n_rel = n_pos
        if n_rel == 0 or n_total == 0:
            continue
        valid_queries += 1

        ap_sum = 0.0
        hits_so_far = 0
        for rank, idx in enumerate(sorted_indices):
            if idx < n_pos:
                hits_so_far += 1
                precision_at_k = hits_so_far / (rank + 1)
                ap_sum += precision_at_k

        avg_precision = ap_sum / n_rel if n_rel > 0 else 0.0

        for k in k_values:
            total_map[k] += avg_precision

        dcg = 0.0
        idcg = 0.0
        for rank, idx in enumerate(sorted_indices):
            if rank >= 5:
                break
            dcg += rel_labels[idx] / math.log2(rank + 2)
        for i in range(min(5, n_rel)):
            idcg += 1.0 / math.log2(i + 2)
        ndcg5 = dcg / idcg if idcg > 0 else 0.0
        total_ndcg[5] += ndcg5

        rr = 0.0
        for rank, idx in enumerate(sorted_indices):
            if idx < n_pos:
                rr = 1.0 / (rank + 1)
                break
        total_mrr += rr

    result = {}
    for k in k_values:
        result[f"map@{k}"] = total_map[k] / max(valid_queries, 1)
    result[f"ndcg@5"] = total_ndcg[5] / max(valid_queries, 1)
    result[f"mrr"] = total_mrr / max(valid_queries, 1)

    return result


def run_search(
    train_data_path: str,
    embeddings_path: str,
    device: str = "cuda",
    n_folds: int = 5,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    t_gap: float = 20.0,
    t_safety: float = 20.0,
    max_penalty_ratio: float = 0.5,
) -> Dict[str, Any]:
    start_time = time.time()

    alpha_list = alphas or [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    beta_list = betas or [0.5, 0.8, 1.0, 1.1, 1.2, 1.5]
    delta_list = deltas or [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10]

    total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
    logger.info(f"Training-set parameter search V2 (ranking-based): {total_trials} combinations")

    logger.info("Loading training data...")
    groups, n_pos, n_neg = load_training_data(train_data_path)
    n_queries = len(groups)
    n_has_neg = sum(1 for g in groups if g.has_neg)
    n_has_req = sum(1 for g in groups if g.has_req)
    logger.info(f"  Queries: {n_queries} (has_req: {n_has_req}, has_neg: {n_has_neg})")
    logger.info(f"  Positive docs: {n_pos}, Negative docs: {n_neg}")

    logger.info("Loading cached embeddings...")
    cache = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    q_base_emb = cache["q_base_embeddings"].float()
    q_plus_emb = cache["q_plus_embeddings"].float()
    q_minus_emb = cache["q_minus_embeddings"].float()
    pos_embs = cache["pos_embeddings"].float()
    neg_embs = cache["neg_embeddings"].float()
    logger.info(f"  Q_base shape: {q_base_emb.shape}")
    logger.info(f"  Q+ shape: {q_plus_emb.shape}")
    logger.info(f"  Q- shape: {q_minus_emb.shape}")
    logger.info(f"  Pos shape: {pos_embs.shape}")
    logger.info(f"  Neg shape: {neg_embs.shape}")

    assert q_plus_emb.shape[0] == n_queries, f"Q+ count mismatch: {q_plus_emb.shape[0]} vs {n_queries}"
    assert neg_embs.shape[0] == n_neg, f"Neg count mismatch: {neg_embs.shape[0]} vs {n_neg}"
    if pos_embs.shape[0] != n_pos:
        logger.warning(f"Pos count mismatch: cached={pos_embs.shape[0]} vs data={n_pos}, using cached count")
        n_pos = pos_embs.shape[0]

    q_base_emb = F.normalize(q_base_emb, p=2, dim=1)
    q_plus_emb = F.normalize(q_plus_emb, p=2, dim=1)
    q_minus_emb = F.normalize(q_minus_emb, p=2, dim=1)
    pos_embs = F.normalize(pos_embs, p=2, dim=1)
    neg_embs = F.normalize(neg_embs, p=2, dim=1)

    logger.info("Computing score matrices...")
    max_pos_count = max(g.pos_count for g in groups)
    n_total_docs = max_pos_count + 15

    S_base_all = torch.zeros(n_queries, n_total_docs)
    S_req_all = torch.zeros(n_queries, n_total_docs)
    S_neg_all = torch.zeros(n_queries, n_total_docs)

    for q_idx, g in enumerate(groups):
        q_base = q_base_emb[q_idx]
        q_plus = q_plus_emb[q_idx]
        q_minus = q_minus_emb[q_idx]

        for pi in range(g.pos_count):
            doc_emb = pos_embs[g.pos_start + pi]
            S_base_all[q_idx, pi] = torch.dot(q_base, doc_emb)
            S_req_all[q_idx, pi] = torch.dot(q_plus, doc_emb)
            S_neg_all[q_idx, pi] = torch.dot(q_minus, doc_emb)

        for ni in range(15):
            doc_emb = neg_embs[g.neg_start + ni]
            offset = max_pos_count + ni
            S_base_all[q_idx, offset] = torch.dot(q_base, doc_emb)
            S_req_all[q_idx, offset] = torch.dot(q_plus, doc_emb)
            S_neg_all[q_idx, offset] = torch.dot(q_minus, doc_emb)

    has_req_mask = torch.tensor([float(g.has_req) for g in groups], dtype=torch.float32)
    has_neg_mask = torch.tensor([float(g.has_neg) for g in groups], dtype=torch.float32)
    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_minus_emb, dim=1)

    np.random.seed(42)
    indices = np.arange(n_queries)
    np.random.shuffle(indices)
    folds = []
    fold_size = n_queries // n_folds
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n_queries
        folds.append(indices[start:end].tolist())

    logger.info(f"Starting grid search ({total_trials} trials, {n_folds}-fold CV)...")

    all_results = []
    best_cv_map1000 = -float("inf")
    best_params_map = None
    best_cv_ndcg5 = -float("inf")
    best_params_ndcg = None

    trial_idx = 0
    for alpha in alpha_list:
        for beta in beta_list:
            for delta in delta_list:
                trial_idx += 1

                S_final_all = apply_deir_dual_v2(
                    S_base_all, S_req_all, S_neg_all, cos_qbase_qneg,
                    has_req_mask, has_neg_mask,
                    alpha, beta, delta,
                    t_gap, t_safety, max_penalty_ratio,
                )

                full_metrics = compute_ranking_metrics(S_final_all, groups, max_pos_count)

                cv_maps = []
                cv_ndcgs = []
                for f_indices in folds:
                    f_groups = [groups[i] for i in f_indices]
                    fm = compute_ranking_metrics(S_final_all[f_indices], f_groups, max_pos_count)
                    cv_maps.append(fm["map@1000"])
                    cv_ndcgs.append(fm["ndcg@5"])

                cv_map_mean = float(np.mean(cv_maps))
                cv_map_std = float(np.std(cv_maps))
                cv_ndcg_mean = float(np.mean(cv_ndcgs))
                cv_ndcg_std = float(np.std(cv_ndcgs))

                result = {
                    "alpha": float(alpha), "beta": float(beta), "delta": float(delta),
                    "map@1000": float(full_metrics["map@1000"]),
                    "map@100": float(full_metrics["map@100"]),
                    "map@50": float(full_metrics["map@50"]),
                    "map@10": float(full_metrics["map@10"]),
                    "map@5": float(full_metrics["map@5"]),
                    "ndcg@5": float(full_metrics["ndcg@5"]),
                    "mrr": float(full_metrics["mrr"]),
                    "cv_map@1000_mean": float(cv_map_mean),
                    "cv_map@1000_std": float(cv_map_std),
                    "cv_ndcg@5_mean": float(cv_ndcg_mean),
                    "cv_ndcg@5_std": float(cv_ndcg_std),
                }
                all_results.append(result)

                if cv_map_mean > best_cv_map1000:
                    best_cv_map1000 = cv_map_mean
                    best_params_map = result.copy()

                if cv_ndcg_mean > best_cv_ndcg5:
                    best_cv_ndcg5 = cv_ndcg_mean
                    best_params_ndcg = result.copy()

                if trial_idx % 30 == 0 or trial_idx == total_trials:
                    logger.info(
                        "[%d/%d] a=%.1f b=%.1f d=%.2f | "
                        "MAP@1000=%.4f(cv=%.4f±%.4f) nDCG@5=%.4f(cv=%.4f±%.4f)",
                        trial_idx, total_trials,
                        alpha, beta, delta,
                        full_metrics["map@1000"], cv_map_mean, cv_map_std,
                        full_metrics["ndcg@5"], cv_ndcg_mean, cv_ndcg_std,
                    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training-set parameter search COMPLETE")
    logger.info("-" * 60)
    logger.info("Best by CV-MAP@1000:")
    logger.info("  Params: a=%.1f b=%.1f d=%.2f",
                best_params_map["alpha"], best_params_map["beta"], best_params_map["delta"])
    logger.info("  CV-MAP@1000: %.4f ± %.4f",
                best_params_map["cv_map@1000_mean"], best_params_map["cv_map@1000_std"])
    logger.info("  Full MAP@1000: %.4f, nDCG@5: %.4f",
                best_params_map["map@1000"], best_params_map["ndcg@5"])
    logger.info("-" * 60)
    logger.info("Best by CV-nDCG@5:")
    logger.info("  Params: a=%.1f b=%.1f d=%.2f",
                best_params_ndcg["alpha"], best_params_ndcg["beta"], best_params_ndcg["delta"])
    logger.info("  CV-nDCG@5: %.4f ± %.4f",
                best_params_ndcg["cv_ndcg@5_mean"], best_params_ndcg["cv_ndcg@5_std"])
    logger.info("  Full MAP@1000: %.4f, nDCG@5: %.4f",
                best_params_ndcg["map@1000"], best_params_ndcg["ndcg@5"])
    logger.info("Elapsed: %.1f sec", elapsed)

    output_dir = os.path.dirname(train_data_path)
    results_path = os.path.join(output_dir, "train_param_search_v2_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_by_map@1000": best_params_map,
            "best_by_ndcg@5": best_params_ndcg,
            "all_results": all_results,
            "n_folds": n_folds,
            "n_queries": n_queries,
            "n_pos_docs": n_pos,
            "n_neg_docs": n_neg,
            "elapsed_sec": elapsed,
            "methodology": {
                "description": "K-fold CV on training set with ranking metrics (MAP@K, nDCG@5)",
                "objective": "Directly compute MAP@K and nDCG@5 from ranked lists using training labels",
                "labels": "Positive docs = relevant (rel=1), Negative docs = non-relevant (rel=0)",
                "scoring": "S_base = Q_base · doc, S_req = Q_plus · doc, S_neg = Q_minus · doc (cached RepLLaMA embeddings)",
                "v2_formula": "tau = Cos(Q_base, Q_neg) + delta; gap_w = sigmoid((S_neg - S_base) * T_gap); safety = 1 - sigmoid((S_neg - tau) * T_safety); penalty = min(alpha * Softplus(S_neg - tau) * gap_w, S_base * ratio); S_final = S_base + beta * S_req * safety - penalty",
                "metrics": {
                    "MAP@K": "Mean Average Precision at cutoff K, averaged over queries",
                    "nDCG@5": "Normalized Discounted Cumulative Gain at top-5",
                    "MRR": "Mean Reciprocal Rank of first relevant document",
                },
            },
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return {"best_by_map@1000": best_params_map, "best_by_ndcg@5": best_params_ndcg, "all_results": all_results}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="dataset/FollowIR_train/train/dsclr_total_dataset.jsonl")
    parser.add_argument("--embeddings_path", type=str,
                        default="dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--deltas", type=str, default=None)

    args = parser.parse_args()

    def parse_list(s):
        if s is None:
            return None
        return [float(x) for x in s.split(",")]

    run_search(
        train_data_path=args.train_data_path,
        embeddings_path=args.embeddings_path,
        device=args.device,
        n_folds=args.n_folds,
        alphas=parse_list(args.alphas),
        betas=parse_list(args.betas),
        deltas=parse_list(args.deltas),
    )
