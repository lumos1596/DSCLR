"""
Extract per-document reward-penalty decomposition data for Figure 3.

Uses pre-cached query embeddings (q_base, q_req, q_neg) to avoid GPU requirement.
Computes on CPU with pre-cached doc + query embeddings.

For every FollowIR query with an exclusion, computes:
  - z_full, z_pos, z_neg, r, p, g, h, s_final
  - Base rank (by z_full) and TRACE rank (by s_final)
  - Document category: constraint_satisfying / constraint_affected / other
  - Score adjustment delta = p*g - h

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_figure3_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F

from eval.engine_dscrl import load_cached_embeddings
from eval.engine_trace import robust_standardize, _mad, fit_huber_regression

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
import datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ======== Config ========

DATASET_PATHS = {
    "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
    "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
    "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
}

DUAL_QUERIES_PATHS = {
    "Core17InstructionRetrieval": "/home/luwa/Documents/DSCLR-remote/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl",
    "Robust04InstructionRetrieval": "/home/luwa/Documents/DSCLR-remote/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Robust04InstructionRetrieval.jsonl",
    "News21InstructionRetrieval": "/home/luwa/Documents/DSCLR-remote/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl",
}

DOC_EMBEDDING_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"

# Pre-cached query embedding paths (q_base, q_req≈q_pos, q_neg≈q_minus for RepLLaMA)
# Format: {task_name: {suffix: path_pattern}}
QUERY_EMBEDDING_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/queries"

# Hash suffixes for each task's cached query embeddings
QUERY_CACHE_HASHES = {
    "Core17InstructionRetrieval": "15337234",
    "Robust04InstructionRetrieval": "ba41c9af",
    "News21InstructionRetrieval": "e0f9a455",
}

# TRACE hyperparameters
LAMBDA = 1.0
TAU = 0.2

OUTPUT_DIR = "/home/luwa/Documents/DSCLR-remote/results/figure3"


# ======== Data loading ========

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


def load_cached_query_embeddings(task_name: str):
    """Load pre-cached query embeddings (q_base, q_req, q_neg) for changed queries.

    q_req maps to q_pos (affirmative view), q_neg maps to q_minus (exclusion view).
    Returns (Q_base_emb, Q_pos_emb, Q_neg_emb) as torch tensors, shape (n_queries, dim).
    """
    h = QUERY_CACHE_HASHES.get(task_name)
    if not h:
        logger.error(f"No cache hash for {task_name}")
        return None, None, None

    q_base_path = os.path.join(QUERY_EMBEDDING_DIR,
                               f"{task_name}_RepLLaMA_reproduced_{h}_q_base_changed.npy")
    q_pos_path = os.path.join(QUERY_EMBEDDING_DIR,
                              f"{task_name}_RepLLaMA_reproduced_{h}_q_req_changed.npy")
    q_neg_path = os.path.join(QUERY_EMBEDDING_DIR,
                              f"{task_name}_RepLLaMA_reproduced_{h}_q_neg_changed.npy")

    for p in [q_base_path, q_pos_path, q_neg_path]:
        if not os.path.exists(p):
            logger.error(f"Cached query embedding not found: {p}")
            return None, None, None

    q_base = torch.tensor(np.load(q_base_path), dtype=torch.float32)
    q_pos = torch.tensor(np.load(q_pos_path), dtype=torch.float32)
    q_neg = torch.tensor(np.load(q_neg_path), dtype=torch.float32)

    logger.info(f"Loaded cached query embeddings: base={q_base.shape}, "
                f"pos={q_pos.shape}, neg={q_neg.shape}")
    return q_base, q_pos, q_neg


# ======== TRACE scoring (standalone, matching engine_trace.py) ========

def trace_score_single_query(
    s_full: torch.Tensor,
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    has_neg: bool,
    lambda_boundary: float = LAMBDA,
    tau_decay: float = TAU,
    eps: float = 1e-6,
):
    """Compute TRACE scores for one query, return all per-document components."""
    n = s_full.numel()

    z_full = robust_standardize(s_full.float(), eps)
    z_pos = robust_standardize(s_pos.float(), eps)
    z_neg = robust_standardize(s_neg.float(), eps)

    if not has_neg or n < 3:
        p = torch.clamp(z_pos, min=0)
        s_final = z_full + p
        return {
            'z_full': z_full, 'z_pos': z_pos, 'z_neg': z_neg,
            'r': torch.zeros_like(z_neg), 'p': p,
            'h': torch.zeros_like(z_neg), 'g': torch.ones_like(z_neg),
            's_final': s_final,
        }

    # Huber regression
    a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=1.345)

    # Residual
    e = z_neg.float() - a_hat - b_hat * z_pos.float()
    e_median = e.median()
    e_mad = _mad(e, eps)
    r = (e - e_median) / e_mad

    # Score composition
    p = torch.clamp(z_pos, min=0)
    h = torch.clamp(r - lambda_boundary, min=0)
    g = torch.exp(-h / tau_decay)
    s_final = z_full + p * g - h

    return {
        'z_full': z_full, 'z_pos': z_pos, 'z_neg': z_neg,
        'r': r, 'p': p, 'h': h, 'g': g, 's_final': s_final,
    }


# ======== Main extraction ========

def process_task(task_name: str) -> Dict[str, Any]:
    """Process one FollowIR task and return per-document decomposition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {task_name}")
    logger.info(f"{'='*60}")

    qrels = load_qrels(task_name)
    q_og, q_changed, q_raw_og, q_raw_changed = load_queries_and_instructions(task_name)
    candidates = load_candidates(task_name)
    dual_data = load_dual_queries(task_name)

    # Load cached document embeddings
    cached = load_cached_embeddings(DOC_EMBEDDING_DIR, task_name, "samaya-ai/RepLLaMA-reproduced")
    if cached is None:
        logger.error(f"No cached doc embeddings for {task_name}")
        return {}
    doc_embeddings, doc_ids = cached
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    # Load cached query embeddings (CPU, no GPU needed)
    q_base_emb, q_pos_emb, q_neg_emb = load_cached_query_embeddings(task_name)
    if q_base_emb is None:
        logger.error(f"No cached query embeddings for {task_name}")
        return {}

    # Build query lists for changed queries
    query_ids_ch = list(q_changed.keys())
    has_neg_list = []

    for qid in query_ids_ch:
        d = dual_data.get(qid, {})
        q_minus = d.get('q_minus', '')
        has_neg_list.append(0.0 if is_none_query(q_minus) else 1.0)

    has_neg_mask = torch.tensor(has_neg_list, dtype=torch.float32)

    # Normalize and compute similarity matrices on CPU
    logger.info(f"Computing similarity matrices on CPU ({len(query_ids_ch)} queries x {len(doc_ids)} docs)...")
    doc_emb_f = F.normalize(doc_embeddings.float(), p=2, dim=1)
    q_full_emb_f = F.normalize(q_base_emb.float(), p=2, dim=1)
    q_pos_emb_f = F.normalize(q_pos_emb.float(), p=2, dim=1)
    q_neg_emb_f = F.normalize(q_neg_emb.float(), p=2, dim=1)

    # Compute in chunks to manage memory
    S_full = torch.matmul(q_full_emb_f, doc_emb_f.T)
    S_pos = torch.matmul(q_pos_emb_f, doc_emb_f.T)
    S_neg = torch.matmul(q_neg_emb_f, doc_emb_f.T)
    S_neg = S_neg * has_neg_mask.unsqueeze(1)

    logger.info(f"Similarity matrices computed: S_full={S_full.shape}")

    def build_candidate_indices(candidates, doc_id_to_idx):
        qid_to_candidate_indices = {}
        for base_qid, cand_list in candidates.items():
            indices = [doc_id_to_idx[did] for did in cand_list if did in doc_id_to_idx]
            if indices:
                qid_to_candidate_indices[base_qid] = indices
        return qid_to_candidate_indices

    qid_to_candidate_indices = build_candidate_indices(candidates, doc_id_to_idx)

    # Process each query
    all_docs = []

    for q_idx, qid_changed in enumerate(query_ids_ch):
        base_qid = qid_changed.replace('-changed', '')
        has_neg = bool(has_neg_mask[q_idx].item() > 0)

        if not has_neg:
            continue

        cand_doc_ids = candidates.get(base_qid, [])
        valid_cand = [(did, doc_id_to_idx[did]) for did in cand_doc_ids if did in doc_id_to_idx]
        if not valid_cand:
            continue

        cand_doc_ids_valid = [v[0] for v in valid_cand]
        cand_idx_tensor = torch.tensor([v[1] for v in valid_cand], dtype=torch.long)

        s_full = S_full[q_idx, cand_idx_tensor]
        s_pos = S_pos[q_idx, cand_idx_tensor]
        s_neg = S_neg[q_idx, cand_idx_tensor]

        result = trace_score_single_query(s_full, s_pos, s_neg, has_neg=True)

        # Compute ranks
        base_ranks = torch.argsort(torch.argsort(s_full, descending=True)) + 1
        trace_ranks = torch.argsort(torch.argsort(result['s_final'], descending=True)) + 1
        rank_changes = base_ranks.float() - trace_ranks.float()  # positive = promoted

        # Score components
        z_full = result['z_full']
        p_g = result['p'] * result['g']  # effective reward
        neg_h = result['h']              # effective penalty
        delta = p_g - neg_h              # total adjustment

        # Document categories
        og_rels = qrels.get(base_qid + '-og', {})
        ch_rels = qrels.get(qid_changed, {})

        for i, doc_id in enumerate(cand_doc_ids_valid):
            og_rel = og_rels.get(doc_id, 0)
            ch_rel = ch_rels.get(doc_id, 0)

            if og_rel > 0 and ch_rel <= 0:
                category = "constraint_affected"
            elif og_rel > 0 and ch_rel > 0:
                category = "constraint_satisfying"
            else:
                category = "other"

            all_docs.append({
                "task": task_name,
                "qid": base_qid,
                "doc_id": doc_id,
                "category": category,
                "og_rel": og_rel,
                "ch_rel": ch_rel,
                "s_full": float(s_full[i].item()),
                "s_pos": float(s_pos[i].item()),
                "s_neg": float(s_neg[i].item()),
                "z_full": float(z_full[i].item()),
                "z_pos": float(result['z_pos'][i].item()),
                "z_neg": float(result['z_neg'][i].item()),
                "r": float(result['r'][i].item()),
                "p": float(result['p'][i].item()),
                "g": float(result['g'][i].item()),
                "h": float(result['h'][i].item()),
                "p_g": float(p_g[i].item()),
                "neg_h": float(neg_h[i].item()),
                "delta": float(delta[i].item()),
                "s_final": float(result['s_final'][i].item()),
                "base_rank": int(base_ranks[i].item()),
                "trace_rank": int(trace_ranks[i].item()),
                "rank_change": float(rank_changes[i].item()),
            })

    # Summary stats
    n_satisfying = sum(1 for d in all_docs if d["category"] == "constraint_satisfying")
    n_affected = sum(1 for d in all_docs if d["category"] == "constraint_affected")
    n_other = sum(1 for d in all_docs if d["category"] == "other")
    logger.info(f"  Total docs: {len(all_docs)}, satisfying: {n_satisfying}, "
                f"affected: {n_affected}, other: {n_other}")

    return {
        "task": task_name,
        "n_queries": sum(1 for h in has_neg_list if h > 0),
        "n_docs": len(all_docs),
        "n_satisfying": n_satisfying,
        "n_affected": n_affected,
        "n_other": n_other,
        "docs": all_docs,
    }


def main():
    all_tasks = []
    for task_name in ["Core17InstructionRetrieval", "Robust04InstructionRetrieval",
                       "News21InstructionRetrieval"]:
        result = process_task(task_name)
        if result:
            all_tasks.append(result)

    # Combine all docs
    all_docs = []
    for task_result in all_tasks:
        all_docs.extend(task_result["docs"])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "figure3_reward_penalty_data.json")

    output = {
        "lambda": LAMBDA,
        "tau": TAU,
        "note": "Using cached q_req/q_neg embeddings (q_req≈q_pos, q_neg≈q_minus)",
        "n_total_docs": len(all_docs),
        "per_task_summary": [{
            "task": t["task"],
            "n_queries": t["n_queries"],
            "n_docs": t["n_docs"],
            "n_satisfying": t["n_satisfying"],
            "n_affected": t["n_affected"],
            "n_other": t["n_other"],
        } for t in all_tasks],
        "docs": all_docs,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nData saved to {output_path}")
    logger.info(f"Total docs: {len(all_docs)}")

    # Print summary
    for cat in ["constraint_satisfying", "constraint_affected", "other"]:
        cat_docs = [d for d in all_docs if d["category"] == cat]
        if cat_docs:
            avg_delta = np.mean([d["delta"] for d in cat_docs])
            avg_p_g = np.mean([d["p_g"] for d in cat_docs])
            avg_neg_h = np.mean([d["neg_h"] for d in cat_docs])
            avg_rank_change = np.mean([d["rank_change"] for d in cat_docs])
            logger.info(f"  {cat}: n={len(cat_docs)}, avg_delta={avg_delta:.4f}, "
                        f"avg_p_g={avg_p_g:.4f}, avg_neg_h={avg_neg_h:.4f}, "
                        f"avg_rank_change={avg_rank_change:.2f}")


if __name__ == "__main__":
    main()
