"""
Compute AUROC for mechanism analysis (Section 4.3):
Measures whether residual scores (r) separate "constraint-affected" candidates
better than raw negative scores (z_neg).

Constraint-affected = documents whose relevance decreases under changed instruction
Constraint-satisfying = documents that remain relevant after the change

Usage:
  cd /home/luwa/Documents/DSCLR-remote && /home/luwa/.conda/envs/dsclr/bin/python -m eval.compute_residual_auroc
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a free GPU (1, 2, 3, 5 are available)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

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

# ======== TRACE scoring primitives (from engine_trace.py) ========

def _mad(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median Absolute Deviation."""
    return (x - x.median()).abs().median() + eps

def robust_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust standardization: (x - median) / MAD."""
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
    """Fit Huber regression: y = a + b*X using iterative reweighted least squares."""
    n = y.numel()
    if n < 2:
        return 0.0, 0.0

    # Initialize with OLS
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
    """Load qrels from HuggingFace, returns {qid: {doc_id: relevance}}."""
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
    """Load queries and instructions from HuggingFace."""
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
    """Load candidate document lists."""
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
    """Load dual queries (q_plus, q_minus)."""
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
    """Load cached document embeddings."""
    return load_cached_embeddings(EMBEDDING_DIR, task_name, model_name)


# ======== AUROC Computation ========

def compute_per_query_zneg_and_r(
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute z_neg and normalized residual r for a single query's candidates.
    
    Uses the same robust standardization + Huber regression as engine_trace.py.
    """
    n = s_neg.numel()
    if n < 3:
        z_neg = robust_standardize(s_neg.float(), eps)
        return z_neg, torch.zeros_like(z_neg)
    
    # Step 1: Robust standardization
    z_pos = robust_standardize(s_pos.float(), eps)
    z_neg = robust_standardize(s_neg.float(), eps)
    
    # Step 2: Huber regression: z_neg = a_hat + b_hat * z_pos
    a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=1.345)
    
    # Step 3: Compute residual and re-standardize
    e = z_neg.float() - a_hat - b_hat * z_pos.float()
    e_median = e.median()
    e_mad = _mad(e, eps)
    r = (e - e_median) / e_mad
    
    return z_neg, r


def compute_auroc_for_task(task_name: str) -> Dict[str, Any]:
    """Compute AUROC for a single task (dataset)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {task_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    logger.info("Loading qrels...")
    qrels = load_qrels(task_name)
    
    logger.info("Loading queries and instructions...")
    q_og, q_changed, q_raw_og, q_raw_changed = load_queries_and_instructions(task_name)
    
    logger.info("Loading candidates...")
    candidates = load_candidates(task_name)
    
    logger.info("Loading dual queries...")
    dual_data = load_dual_queries(task_name)
    
    logger.info("Loading document embeddings...")
    cached = load_doc_embeddings(task_name)
    if cached is None:
        logger.error(f"No cached embeddings found for {task_name}")
        return {}
    doc_embeddings, doc_ids = cached
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
    
    # Initialize model for encoding queries
    logger.info("Loading RepLLaMA model for query encoding...")
    from eval.models.repllama_encoder import RepLLaMAEncoder
    encoder = RepLLaMAEncoder(model_name="samaya-ai/RepLLaMA-reproduced", device="cuda")
    
    # ====== Batch-encode all changed queries ======
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
    
    # Encode queries in batch
    logger.info(f"Encoding {len(q_base_list)} changed queries (Q_pos, Q_neg)...")
    q_pos_emb = encoder.encode_queries(q_pos_list, batch_size=32)
    q_neg_emb = encoder.encode_queries(q_neg_list, batch_size=32)
    
    # Move to GPU
    device = torch.device("cuda")
    doc_emb_gpu = doc_embeddings.to(device)
    q_pos_emb = q_pos_emb.to(device)
    q_neg_emb = q_neg_emb.to(device)
    
    # Normalize embeddings for cosine similarity
    doc_emb_gpu = F.normalize(doc_emb_gpu.float(), p=2, dim=1)
    q_pos_emb = F.normalize(q_pos_emb.float(), p=2, dim=1)
    q_neg_emb = F.normalize(q_neg_emb.float(), p=2, dim=1)
    
    # Compute S_pos and S_neg for ALL queries at once
    logger.info("Computing S_pos, S_neg...")
    S_pos = torch.matmul(q_pos_emb, doc_emb_gpu.T)  # (n_q, n_docs)
    S_neg = torch.matmul(q_neg_emb, doc_emb_gpu.T)  # (n_q, n_docs)
    # Zero out S_neg for queries without negation
    S_neg = S_neg * has_neg_mask.to(device).unsqueeze(1)
    
    # Build candidate index mapping
    def build_candidate_indices(candidates, doc_id_to_idx):
        qid_to_candidate_indices = {}
        for base_qid, cand_list in candidates.items():
            indices = [doc_id_to_idx[did] for did in cand_list if did in doc_id_to_idx]
            if indices:
                qid_to_candidate_indices[base_qid] = indices
        return qid_to_candidate_indices
    
    qid_to_candidate_indices = build_candidate_indices(candidates, doc_id_to_idx)
    
    # ====== Compute AUROC per query ======
    all_zneg_aurocs = []
    all_r_aurocs = []
    eligible_queries = 0
    total_queries = len(q_changed)
    
    # Also collect pooled data for aggregate AUROC
    all_labels_pooled = []
    all_zneg_pooled = []
    all_r_pooled = []
    
    for q_idx, qid_changed in enumerate(query_ids_ch):
        base_qid = qid_changed.replace('-changed', '')
        qid_og = base_qid + '-og'
        
        # Check if this query has negation
        if has_neg_mask[q_idx].item() < 0.5:
            continue  # No exclusion query → not eligible
        
        # Get candidate documents
        cand_indices = qid_to_candidate_indices.get(base_qid, [])
        if not cand_indices:
            continue
        
        # Get candidate doc IDs
        cand_doc_ids = candidates.get(base_qid, [])
        # Filter to only those with valid indices
        valid_cand = [(did, doc_id_to_idx[did]) for did in cand_doc_ids if did in doc_id_to_idx]
        if not valid_cand:
            continue
        cand_doc_ids_valid = [v[0] for v in valid_cand]
        cand_idx_tensor = torch.tensor([v[1] for v in valid_cand], device=device, dtype=torch.long)
        
        # Get relevance labels for OG and changed
        og_rels = qrels.get(qid_og, {})
        ch_rels = qrels.get(qid_changed, {})
        
        # Identify constraint-affected and constraint-satisfying documents
        constraint_affected = set()  # doc_ids that lost relevance
        constraint_satisfying = set()  # doc_ids that kept relevance
        
        for doc_id in cand_doc_ids_valid:
            og_rel = og_rels.get(doc_id, 0)
            ch_rel = ch_rels.get(doc_id, 0)
            if og_rel > 0 and ch_rel <= 0:
                constraint_affected.add(doc_id)
            elif og_rel > 0 and ch_rel > 0:
                constraint_satisfying.add(doc_id)
        
        # Need at least 1 in each group for AUROC
        if len(constraint_affected) < 1 or len(constraint_satisfying) < 1:
            continue
        
        eligible_queries += 1
        
        # Get S_pos and S_neg for candidates
        s_pos = S_pos[q_idx, cand_idx_tensor]  # (n_cand,)
        s_neg = S_neg[q_idx, cand_idx_tensor]  # (n_cand,)
        
        # Compute z_neg and r
        z_neg, r = compute_per_query_zneg_and_r(s_pos, s_neg)
        
        # Build labels and scores for AUROC
        labels = []
        zneg_scores = []
        r_scores = []
        
        for i, doc_id in enumerate(cand_doc_ids_valid):
            if doc_id in constraint_affected:
                labels.append(1)  # positive class = constraint-affected
                zneg_scores.append(z_neg[i].item())
                r_scores.append(r[i].item())
            elif doc_id in constraint_satisfying:
                labels.append(0)  # negative class = constraint-satisfying
                zneg_scores.append(z_neg[i].item())
                r_scores.append(r[i].item())
        
        if len(set(labels)) < 2:
            continue
        
        labels = np.array(labels)
        zneg_scores = np.array(zneg_scores)
        r_scores = np.array(r_scores)
        
        # Collect for pooled AUROC
        all_labels_pooled.extend(labels.tolist())
        all_zneg_pooled.extend(zneg_scores.tolist())
        all_r_pooled.extend(r_scores.tolist())
        
        # AUROC: higher z_neg / r should indicate constraint-affected (label=1)
        try:
            auroc_zneg = roc_auc_score(labels, zneg_scores)
            auroc_r = roc_auc_score(labels, r_scores)
        except ValueError:
            continue
        
        all_zneg_aurocs.append(auroc_zneg)
        all_r_aurocs.append(auroc_r)
        
        logger.info(f"  {qid_changed}: affected={len(constraint_affected)}, satisfying={len(constraint_satisfying)}, "
                    f"AUROC(z_neg)={auroc_zneg:.4f}, AUROC(r)={auroc_r:.4f}")
    
    # Compute pooled AUROC
    pooled_auroc_zneg = 0.0
    pooled_auroc_r = 0.0
    if len(set(all_labels_pooled)) >= 2:
        pooled_auroc_zneg = float(roc_auc_score(all_labels_pooled, all_zneg_pooled))
        pooled_auroc_r = float(roc_auc_score(all_labels_pooled, all_r_pooled))
    
    result = {
        "task": task_name,
        "total_changed_queries": total_queries,
        "eligible_queries": eligible_queries,
        "eligible_pct": eligible_queries / total_queries * 100 if total_queries > 0 else 0,
        "n_queries_with_auroc": len(all_zneg_aurocs),
        "auroc_zneg_mean": float(np.mean(all_zneg_aurocs)) if all_zneg_aurocs else 0,
        "auroc_r_mean": float(np.mean(all_r_aurocs)) if all_r_aurocs else 0,
        "pooled_auroc_zneg": pooled_auroc_zneg,
        "pooled_auroc_r": pooled_auroc_r,
        "n_affected_total": sum(1 for l in all_labels_pooled if l == 1),
        "n_satisfying_total": sum(1 for l in all_labels_pooled if l == 0),
        "auroc_zneg_per_query": [float(x) for x in all_zneg_aurocs],
        "auroc_r_per_query": [float(x) for x in all_r_aurocs],
    }
    
    logger.info(f"\n  Summary for {task_name}:")
    logger.info(f"    Total changed queries: {total_queries}")
    logger.info(f"    Eligible (has negation): {eligible_queries} ({result['eligible_pct']:.1f}%)")
    logger.info(f"    With AUROC computed: {len(all_zneg_aurocs)}")
    logger.info(f"    Per-query Mean AUROC(z_neg): {result['auroc_zneg_mean']:.4f}")
    logger.info(f"    Per-query Mean AUROC(r):     {result['auroc_r_mean']:.4f}")
    logger.info(f"    Pooled AUROC(z_neg): {pooled_auroc_zneg:.4f}")
    logger.info(f"    Pooled AUROC(r):     {pooled_auroc_r:.4f}")
    logger.info(f"    Total affected: {result['n_affected_total']}, satisfying: {result['n_satisfying_total']}")
    
    # Free GPU memory
    del doc_emb_gpu, q_pos_emb, q_neg_emb, S_pos, S_neg
    torch.cuda.empty_cache()
    
    return result


def main():
    tasks = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval",
    ]
    
    all_results = {}
    total_eligible = 0
    total_changed = 0
    all_zneg = []
    all_r = []
    
    for task in tasks:
        result = compute_auroc_for_task(task)
        all_results[task] = result
        if result:
            total_eligible += result["eligible_queries"]
            total_changed += result["total_changed_queries"]
            all_zneg.extend(result.get("auroc_zneg_per_query", []))
            all_r.extend(result.get("auroc_r_per_query", []))
    
    # Aggregate
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total changed queries: {total_changed}")
    pct = total_eligible / total_changed * 100 if total_changed > 0 else 0
    logger.info(f"Total eligible (has negation): {total_eligible} ({pct:.1f}%)")
    logger.info(f"Total with AUROC: {len(all_zneg)}")
    if all_zneg:
        logger.info(f"Mean AUROC(z_neg): {np.mean(all_zneg):.4f}")
        logger.info(f"Mean AUROC(r):     {np.mean(all_r):.4f}")
        logger.info(f"Improvement (r - z_neg): {np.mean(all_r) - np.mean(all_zneg):.4f}")
    else:
        logger.info("No AUROC values computed")
    
    # Save results
    output_path = "/home/luwa/Documents/DSCLR/evaluation_remote/mechanism_auroc_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
