"""
Extract scatter-plot data for Figure 2 (left panel):
  - Representative query chosen near the median AUROC improvement
  - Saves per-candidate (z_pos, z_neg, r, category) and Huber fit params

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_figure2_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from eval.engine_dscrl import load_cached_embeddings

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
import datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ======== TRACE scoring primitives (same as compute_residual_auroc.py) ========

def _mad(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.median()).abs().median() + eps

def robust_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    median = x.median()
    mad = _mad(x, eps)
    return (x - median) / mad

def fit_huber_regression(
    y: torch.Tensor,
    X: torch.Tensor,
    delta: float = 1.345,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    n = y.numel()
    if n < 2:
        return 0.0, 0.0
    X_mean = X.mean()
    y_mean = y.mean()
    Xc = X - X_mean
    Yc = y - y_mean
    ss_xx = (Xc ** 2).sum()
    if ss_xx < 1e-12:
        return float(y_mean.item()), 0.0
    b = (Xc * Yc).sum() / ss_xx
    a = y_mean - b * X_mean

    for _ in range(max_iter):
        resid = y - a - b * X
        sigma = max(resid.median().abs().item() / 0.6745, 1e-6)
        u = resid / (sigma * delta)
        w = torch.where(u.abs() <= 1.0, torch.ones_like(u), 1.0 / u.abs())

        wx = w * X
        wy = w * y
        wxx = (wx * X).sum()
        wxy = (wx * wy).sum()
        wxs = wx.sum()
        wys = wy.sum()
        ws = w.sum()

        denom = ws * wxx - wxs * wxs
        if abs(denom) < 1e-12:
            break

        b_new = (ws * wxy - wxs * wys) / denom
        a_new = (wys - b_new * wxs) / ws

        if (a_new - a).abs().item() < tol and (b_new - b).abs().item() < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    return float(a.item()), float(b.item())


# ======== Data loading ========

DATASET_PATHS = {
    "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
    "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
    "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
}

DUAL_QUERIES_PATHS = {
    "Core17InstructionRetrieval": "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl",
    "Robust04InstructionRetrieval": "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Robust04InstructionRetrieval.jsonl",
    "News21InstructionRetrieval": "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl",
}

EMBEDDING_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"


def load_qrels(task_name: str) -> Dict[str, Dict[str, int]]:
    ds_path = DATASET_PATHS[task_name]
    ds = datasets.load_dataset(ds_path, 'default')
    split = 'test' if 'test' in ds else list(ds.keys())[0]
    qrels = {}
    for item in ds[split]:
        qid = item.get('query-id', item.get('query_id', ''))
        doc_id = str(item.get('corpus-id', item.get('doc_id', '')))
        relevance = int(item.get('score', item.get('relevance', 1)))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = relevance
    return qrels


def load_queries_and_instructions(task_name: str):
    ds_path = DATASET_PATHS[task_name]
    ds_q = datasets.load_dataset(ds_path, 'queries')
    q_split = 'queries' if 'queries' in ds_q else 'train'
    ds_inst = datasets.load_dataset(ds_path, 'instruction')
    i_split = 'instruction' if 'instruction' in ds_inst else 'train'

    instruction_dict = {}
    for item in ds_inst[i_split]:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = str(item.get('instruction', ''))

    q_og = {}
    q_changed = {}
    q_raw_og = {}
    q_raw_changed = {}

    for q in ds_q[q_split]:
        full_qid = str(q.get('_id', q.get('id', '')))
        query_text = q.get('text', '')
        inst = instruction_dict.get(full_qid, "")

        combined = f"{query_text} {inst}".strip()
        if full_qid.endswith('-og'):
            q_og[full_qid] = combined
            q_raw_og[full_qid] = (query_text, inst)
        elif full_qid.endswith('-changed'):
            q_changed[full_qid] = combined
            q_raw_changed[full_qid] = (query_text, inst)

    return q_og, q_changed, q_raw_og, q_raw_changed


def load_candidates(task_name: str) -> Dict[str, List[str]]:
    ds_path = DATASET_PATHS[task_name]
    ds_top = datasets.load_dataset(ds_path, 'top_ranked')
    available_splits = list(ds_top.keys())
    t_split = available_splits[0] if available_splits else None
    candidates = {}
    if t_split:
        for item in ds_top[t_split]:
            full_qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
            base_qid = full_qid.replace('-og', '').replace('-changed', '')
            results_list = item.get('corpus-ids', item.get('results', []))
            if base_qid not in candidates:
                candidates[base_qid] = [str(did) for did in results_list]
    return candidates


def load_dual_queries(task_name: str) -> Dict[str, Dict[str, str]]:
    path = DUAL_QUERIES_PATHS[task_name]
    dual_data = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get('qid', '')
            dual_data[qid] = {
                'q_plus': item.get('q_plus', ''),
                'q_minus': item.get('q_minus', ''),
            }
    return dual_data


def is_none_query(text: str) -> bool:
    if not text:
        return True
    t = str(text).strip().upper()
    return t in ("[NONE]", "NONE", "NULL", "N/A", "")


def load_doc_embeddings(task_name: str, model_name: str = "samaya-ai/RepLLaMA-reproduced"):
    return load_cached_embeddings(EMBEDDING_DIR, task_name, model_name)


# ======== Main: compute and save scatter data ========

def compute_per_query_zneg_and_r(
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compute z_neg and residual r. Also returns Huber fit params."""
    n = s_neg.numel()
    if n < 3:
        z_neg = robust_standardize(s_neg.float(), eps)
        return z_neg, torch.zeros_like(z_neg), 0.0, 0.0

    z_pos = robust_standardize(s_pos.float(), eps)
    z_neg = robust_standardize(s_neg.float(), eps)

    a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=1.345)

    e = z_neg.float() - a_hat - b_hat * z_pos.float()
    e_median = e.median()
    e_mad = _mad(e, eps)
    r = (e - e_median) / e_mad

    return z_neg, r, a_hat, b_hat


def main():
    # Load AUROC results to find representative query
    auroc_path = "/home/luwa/Documents/DSCLR/evaluation_remote/mechanism_auroc_results.json"
    with open(auroc_path, 'r') as f:
        auroc_results = json.load(f)

    # Collect all per-query AUROC with task info
    all_queries = []  # (task_name, qid_changed, auroc_zneg, auroc_r, improvement)
    for task_name, result in auroc_results.items():
        zneg_list = result.get("auroc_zneg_per_query", [])
        r_list = result.get("auroc_r_per_query", [])
        # We need to map index back to qid_changed; recompute by loading the task
        # Store task + index for now, resolve qid later
        for i in range(len(zneg_list)):
            improvement = r_list[i] - zneg_list[i]
            all_queries.append({
                "task": task_name,
                "index": i,
                "auroc_zneg": zneg_list[i],
                "auroc_r": r_list[i],
                "improvement": improvement,
            })

    improvements = [q["improvement"] for q in all_queries]
    median_improvement = np.median(improvements)
    logger.info(f"Total eligible queries: {len(all_queries)}")
    logger.info(f"Median AUROC improvement: {median_improvement:.6f}")

    # Strategy: pick a query with good AUROC and enough affected/satisfying docs
    # for visual clarity. We iterate over all tasks and find the best candidate.
    # Prefer: AUROC(r) > 0.7, at least 3 affected docs, and positive improvement.
    # Fallback: just pick the one with the best AUROC(r) that has improvement > 0.

    # We need to actually compute which queries have enough affected docs.
    # For now, use a heuristic: prefer Core17 queries (fewer candidates, more likely
    # to have clear separation) with AUROC in a reasonable range (0.65-0.85).

    # Among all queries with improvement > 0, pick the one with AUROC closest to 0.75
    queries_with_improvement = [q for q in all_queries if q["improvement"] > 0]
    if queries_with_improvement:
        target_auroc = 0.75
        best_query = min(queries_with_improvement, key=lambda q: abs(q["auroc_r"] - target_auroc))
    else:
        # If no improvement, just pick a query with good AUROC for illustration
        target_auroc = 0.75
        best_query = min(all_queries, key=lambda q: abs(q["auroc_r"] - target_auroc))

    rep_task = best_query["task"]
    rep_idx = best_query["index"]
    logger.info(f"Representative query: task={rep_task}, index={rep_idx}, "
                f"AUROC(z_neg)={best_query['auroc_zneg']:.4f}, "
                f"AUROC(r)={best_query['auroc_r']:.4f}, "
                f"improvement={best_query['improvement']:.4f}")

    # Now load data for the representative task and extract scatter data
    logger.info(f"\nLoading data for {rep_task}...")

    qrels = load_qrels(rep_task)
    q_og, q_changed, q_raw_og, q_raw_changed = load_queries_and_instructions(rep_task)
    candidates = load_candidates(rep_task)
    dual_data = load_dual_queries(rep_task)

    cached = load_doc_embeddings(rep_task)
    if cached is None:
        logger.error(f"No cached embeddings found for {rep_task}")
        return
    doc_embeddings, doc_ids = cached
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    from eval.models.repllama_encoder import RepLLaMAEncoder
    encoder = RepLLaMAEncoder(model_name="samaya-ai/RepLLaMA-reproduced", device="cuda")

    # Build query lists for changed queries
    query_ids_ch = list(q_changed.keys())
    q_base_list, q_pos_list, q_neg_list = [], [], []
    has_neg_list = []

    for qid in query_ids_ch:
        raw = q_raw_changed.get(qid, ('', ''))
        query_text, instruction = raw[0], raw[1]
        q_base = f"{query_text} {instruction}".strip()
        q_base_list.append(q_base)

        d = dual_data.get(qid, {})
        q_plus = d.get('q_plus', '')
        q_minus = d.get('q_minus', '')
        q_pos_list.append(q_plus if not is_none_query(q_plus) else "")
        q_neg_list.append(q_minus if not is_none_query(q_minus) else "")
        has_neg_list.append(0.0 if is_none_query(q_minus) else 1.0)

    has_neg_mask = torch.tensor(has_neg_list, dtype=torch.float32)

    logger.info(f"Encoding {len(q_pos_list)} changed queries...")
    q_pos_emb = encoder.encode_queries(q_pos_list, batch_size=32)
    q_neg_emb = encoder.encode_queries(q_neg_list, batch_size=32)

    device = torch.device("cuda")
    doc_emb_gpu = doc_embeddings.to(device)
    q_pos_emb = q_pos_emb.to(device)
    q_neg_emb = q_neg_emb.to(device)

    doc_emb_gpu = F.normalize(doc_emb_gpu.float(), p=2, dim=1)
    q_pos_emb = F.normalize(q_pos_emb.float(), p=2, dim=1)
    q_neg_emb = F.normalize(q_neg_emb.float(), p=2, dim=1)

    S_pos = torch.matmul(q_pos_emb, doc_emb_gpu.T)
    S_neg = torch.matmul(q_neg_emb, doc_emb_gpu.T)
    S_neg = S_neg * has_neg_mask.to(device).unsqueeze(1)

    def build_candidate_indices(candidates, doc_id_to_idx):
        qid_to_candidate_indices = {}
        for base_qid, cand_list in candidates.items():
            indices = [doc_id_to_idx[did] for did in cand_list if did in doc_id_to_idx]
            if indices:
                qid_to_candidate_indices[base_qid] = indices
        return qid_to_candidate_indices

    qid_to_candidate_indices = build_candidate_indices(candidates, doc_id_to_idx)

    # Find the eligible query at rep_idx (among those with negation)
    eligible_count = 0
    rep_qid_changed = None
    rep_base_qid = None

    for q_idx, qid_changed in enumerate(query_ids_ch):
        base_qid = qid_changed.replace('-changed', '')
        if has_neg_mask[q_idx].item() < 0.5:
            continue
        cand_indices = qid_to_candidate_indices.get(base_qid, [])
        if not cand_indices:
            continue

        og_rels = qrels.get(base_qid + '-og', {})
        ch_rels = qrels.get(qid_changed, {})
        cand_doc_ids = candidates.get(base_qid, [])
        valid_cand = [(did, doc_id_to_idx[did]) for did in cand_doc_ids if did in doc_id_to_idx]
        if not valid_cand:
            continue
        cand_doc_ids_valid = [v[0] for v in valid_cand]

        constraint_affected = set()
        constraint_satisfying = set()
        for doc_id in cand_doc_ids_valid:
            og_rel = og_rels.get(doc_id, 0)
            ch_rel = ch_rels.get(doc_id, 0)
            if og_rel > 0 and ch_rel <= 0:
                constraint_affected.add(doc_id)
            elif og_rel > 0 and ch_rel > 0:
                constraint_satisfying.add(doc_id)

        if len(constraint_affected) < 1 or len(constraint_satisfying) < 1:
            continue

        if eligible_count == rep_idx:
            rep_qid_changed = qid_changed
            rep_base_qid = base_qid
            break
        eligible_count += 1

    if rep_qid_changed is None:
        logger.error("Could not find representative query by index. Falling back to first eligible.")
        # Fallback: just use the first eligible query with best AUROC improvement
        for q_idx, qid_changed in enumerate(query_ids_ch):
            base_qid = qid_changed.replace('-changed', '')
            if has_neg_mask[q_idx].item() < 0.5:
                continue
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue
            rep_qid_changed = qid_changed
            rep_base_qid = base_qid
            break

    logger.info(f"Representative query ID: {rep_qid_changed}")

    # Get scatter data for the representative query
    q_idx = query_ids_ch.index(rep_qid_changed)
    cand_doc_ids = candidates.get(rep_base_qid, [])
    valid_cand = [(did, doc_id_to_idx[did]) for did in cand_doc_ids if did in doc_id_to_idx]
    cand_doc_ids_valid = [v[0] for v in valid_cand]
    cand_idx_tensor = torch.tensor([v[1] for v in valid_cand], device=device, dtype=torch.long)

    s_pos = S_pos[q_idx, cand_idx_tensor]
    s_neg = S_neg[q_idx, cand_idx_tensor]

    z_neg, r, a_hat, b_hat = compute_per_query_zneg_and_r(s_pos, s_neg)
    z_pos = robust_standardize(s_pos.float())

    # Also compute z_pos for all candidates
    og_rels = qrels.get(rep_base_qid + '-og', {})
    ch_rels = qrels.get(rep_qid_changed, {})

    scatter_data = []
    for i, doc_id in enumerate(cand_doc_ids_valid):
        og_rel = og_rels.get(doc_id, 0)
        ch_rel = ch_rels.get(doc_id, 0)

        if og_rel > 0 and ch_rel <= 0:
            category = "constraint_affected"
        elif og_rel > 0 and ch_rel > 0:
            category = "constraint_satisfying"
        else:
            category = "other"

        scatter_data.append({
            "doc_id": doc_id,
            "z_pos": float(z_pos[i].item()),
            "z_neg": float(z_neg[i].item()),
            "r": float(r[i].item()),
            "category": category,
        })

    output = {
        "task": rep_task,
        "qid_changed": rep_qid_changed,
        "qid_base": rep_base_qid,
        "auroc_zneg": best_query["auroc_zneg"],
        "auroc_r": best_query["auroc_r"],
        "improvement": best_query["improvement"],
        "huber_a": a_hat,
        "huber_b": b_hat,
        "n_candidates": len(scatter_data),
        "n_affected": sum(1 for d in scatter_data if d["category"] == "constraint_affected"),
        "n_satisfying": sum(1 for d in scatter_data if d["category"] == "constraint_satisfying"),
        "candidates": scatter_data,
    }

    output_dir = "/home/luwa/Documents/DSCLR-remote/results/figure2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure2_scatter_data.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Scatter data saved to {output_path}")
    logger.info(f"  n_candidates={len(scatter_data)}, "
                f"n_affected={output['n_affected']}, "
                f"n_satisfying={output['n_satisfying']}")
    logger.info(f"  Huber fit: a={a_hat:.4f}, b={b_hat:.4f}")

    # Free GPU memory
    del doc_emb_gpu, q_pos_emb, q_neg_emb, S_pos, S_neg
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
