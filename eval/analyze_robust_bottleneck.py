import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator


def analyze_dataset(task_name, dual_queries_path, cache_dir, device="cuda"):
    print(f"\n{'='*70}")
    print(f"Analyzing: {task_name}")
    print(f"{'='*70}")

    engine = DSCLREvaluatorEngine(
        model_name="repllama-reproduced",
        task_name=task_name,
        output_dir="/tmp/analysis",
        device=device,
    )

    dual_data = {}
    with open(dual_queries_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            dual_data[item["qid"]] = item

    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    all_doc_ids = sorted(set(d for docs in candidates.values() for d in docs))
    doc_id_to_idx = {did: i for i, did in enumerate(all_doc_ids)}
    n_docs = len(all_doc_ids)

    cached_data = load_cached_embeddings(cache_dir, task_name, "repllama-reproduced")
    if cached_data is None:
        print("No cached embeddings found!")
        return

    doc_emb = F.normalize(cached_data["doc_embeddings"].float(), p=2, dim=1).to(device)

    q_base_embs = F.normalize(cached_data["q_og_embeddings"].float(), p=2, dim=1).to(device)
    q_plus_changed_embs = F.normalize(cached_data["q_plus_changed_embeddings"].float(), p=2, dim=1).to(device)
    q_minus_changed_embs = F.normalize(cached_data["q_minus_changed_embeddings"].float(), p=2, dim=1).to(device)

    neg_mask = cached_data.get("neg_mask", torch.ones(len(q_raw_changed)))
    if isinstance(neg_mask, torch.Tensor):
        neg_mask = neg_mask.float()
    else:
        neg_mask = torch.tensor(neg_mask, dtype=torch.float32)

    req_mask = torch.ones(len(q_raw_changed), dtype=torch.float32)

    query_ids_changed = [f"{qid}-changed" for qid in q_raw_changed.keys()]
    query_ids_og = [f"{qid}-og" for qid in q_raw_og.keys()]

    n_queries = len(query_ids_changed)

    cos_qbase_qneg = F.cosine_similarity(q_base_embs, q_minus_changed_embs, dim=1)
    cos_qbase_qplus = F.cosine_similarity(q_base_embs, q_plus_changed_embs, dim=1)

    print(f"\nCosine similarity statistics:")
    print(f"  Cos(Q_base, Q_neg): mean={cos_qbase_qneg.mean():.4f}, std={cos_qbase_qneg.std():.4f}, min={cos_qbase_qneg.min():.4f}, max={cos_qbase_qneg.max():.4f}")
    print(f"  Cos(Q_base, Q_plus): mean={cos_qbase_qplus.mean():.4f}, std={cos_qbase_qplus.std():.4f}")

    S_base_all = torch.matmul(q_plus_changed_embs, doc_emb.T)
    S_req_all = torch.matmul(q_plus_changed_embs, doc_emb.T)
    S_neg_all = torch.matmul(q_minus_changed_embs, doc_emb.T)
    S_base_og = torch.matmul(q_base_embs, doc_emb.T)

    evaluator = FollowIREvaluator(task_name)

    # Baseline: OG mode (S_base_og only)
    results_og = {}
    for i, qid in enumerate(query_ids_og):
        base_qid = qid.replace("-og", "")
        cand = candidates.get(base_qid, [])
        if not cand:
            continue
        scores = {}
        for doc_id in cand:
            if doc_id in doc_id_to_idx:
                scores[doc_id] = float(S_base_og[i, doc_id_to_idx[doc_id]].item())
        results_og[qid] = scores

    # Baseline: Changed mode without DeIR (S_base only, no penalty/reward)
    results_changed_baseline = {}
    for i, qid in enumerate(query_ids_changed):
        base_qid = qid.replace("-changed", "")
        cand = candidates.get(base_qid, [])
        if not cand:
            continue
        scores = {}
        for doc_id in cand:
            if doc_id in doc_id_to_idx:
                scores[doc_id] = float(S_base_all[i, doc_id_to_idx[doc_id]].item())
        results_changed_baseline[qid] = scores

    baseline_metrics = evaluator.evaluate(results_og, results_changed_baseline)
    bl_og = baseline_metrics.get("original", {})
    bl_ch = baseline_metrics.get("changed", {})

    print(f"\nBaseline (no DeIR, S_base only for changed):")
    print(f"  OG: MAP@1000={bl_og.get('map_at_1000',0):.5f}, nDCG@5={bl_og.get('ndcg_at_5',0):.5f}")
    print(f"  Changed: MAP@1000={bl_ch.get('map_at_1000',0):.5f}, nDCG@5={bl_ch.get('ndcg_at_5',0):.5f}")
    print(f"  p-MRR={baseline_metrics.get('p-MRR',0):.4f}")

    # V2 with best params
    alpha, beta, delta = 0.5, 1.0, 0.0
    results_changed_v2 = {}
    for i, qid in enumerate(query_ids_changed):
        base_qid = qid.replace("-changed", "")
        cand = candidates.get(base_qid, [])
        if not cand:
            continue

        has_neg = bool(neg_mask[i].item() > 0)
        has_req = bool(req_mask[i].item() > 0)
        cos_val = cos_qbase_qneg[i].item()

        cand_indices = [doc_id_to_idx[d] for d in cand if d in doc_id_to_idx]
        s_b = S_base_all[i, cand_indices]
        s_r = S_req_all[i, cand_indices]
        s_n = S_neg_all[i, cand_indices]

        if not has_neg:
            s_req_eff = s_r if has_req else torch.zeros_like(s_b)
            s_final = s_b + beta * s_req_eff
        else:
            tau = cos_val + delta
            overflow = s_n - tau
            smooth_penalty = F.softplus(overflow)
            gap_w = torch.sigmoid((s_n - s_b) * 20.0)
            raw_penalty = alpha * smooth_penalty * gap_w
            penalty = torch.min(raw_penalty, s_b * 0.5)
            safety = 1.0 - torch.sigmoid((s_n - tau) * 20.0)
            s_req_eff = s_r if has_req else torch.zeros_like(s_b)
            s_final = s_b + beta * s_req_eff * safety - penalty

        scores = {}
        for j, doc_id in enumerate(cand):
            if doc_id in doc_id_to_idx:
                scores[doc_id] = float(s_final[j].item())
        results_changed_v2[qid] = scores

    v2_metrics = evaluator.evaluate(results_og, results_changed_v2)
    v2_og = v2_metrics.get("original", {})
    v2_ch = v2_metrics.get("changed", {})

    print(f"\nV2 (a=0.5, b=1.0, d=0.0):")
    print(f"  OG: MAP@1000={v2_og.get('map_at_1000',0):.5f}, nDCG@5={v2_og.get('ndcg_at_5',0):.5f}")
    print(f"  Changed: MAP@1000={v2_ch.get('map_at_1000',0):.5f}, nDCG@5={v2_ch.get('ndcg_at_5',0):.5f}")
    print(f"  p-MRR={v2_metrics.get('p-MRR',0):.4f}")

    # Per-query analysis: which queries lose the most MAP?
    print(f"\nPer-query penalty analysis (top 20 heaviest penalized):")

    qid_to_candidate_indices = {}
    for qid in candidates:
        qid_to_candidate_indices[qid] = [doc_id_to_idx[d] for d in candidates[qid] if d in doc_id_to_idx]

    query_penalty_info = []
    for i, qid in enumerate(query_ids_changed):
        base_qid = qid.replace("-changed", "")
        cand = candidates.get(base_qid, [])
        if not cand:
            continue

        has_neg = bool(neg_mask[i].item() > 0)
        cos_val = cos_qbase_qneg[i].item()

        cand_indices = [doc_id_to_idx[d] for d in cand if d in doc_id_to_idx]
        s_b = S_base_all[i, cand_indices]
        s_n = S_neg_all[i, cand_indices]

        if has_neg:
            tau = cos_val + delta
            over_threshold = (s_n > tau).sum().item()
            avg_s_neg = s_n.mean().item()
            max_s_neg = s_n.max().item()
            avg_s_base = s_b.mean().item()
            gap = avg_s_neg - avg_s_base
        else:
            over_threshold = 0
            avg_s_neg = 0
            max_s_neg = 0
            avg_s_base = s_b.mean().item()
            gap = 0

        query_penalty_info.append({
            "qid": base_qid,
            "cos_qbase_qneg": cos_val,
            "has_neg": has_neg,
            "over_threshold": over_threshold,
            "n_candidates": len(cand_indices),
            "avg_s_neg": avg_s_neg,
            "max_s_neg": max_s_neg,
            "avg_s_base": avg_s_base,
            "gap_neg_base": gap,
        })

    query_penalty_info.sort(key=lambda x: x["gap_neg_base"], reverse=True)

    print(f"{'QID':>8} {'Cos':>6} {'HasNeg':>6} {'OverThr':>7} {'AvgS_neg':>9} {'AvgS_base':>10} {'Gap':>7}")
    for q in query_penalty_info[:20]:
        print(f"{q['qid']:>8} {q['cos_qbase_qneg']:>6.3f} {q['has_neg']:>6} {q['over_threshold']:>7} {q['avg_s_neg']:>9.4f} {q['avg_s_base']:>10.4f} {q['gap_neg_base']:>7.4f}")

    # Distribution of Cos(Q_base, Q_neg)
    cos_vals = cos_qbase_qneg.cpu().numpy()
    neg_mask_np = neg_mask.cpu().numpy()
    cos_with_neg = cos_vals[neg_mask_np > 0]

    print(f"\nCos(Q_base, Q_neg) distribution (queries with Q_neg):")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pct = (cos_with_neg > threshold).mean() * 100
        print(f"  Cos > {threshold:.2f}: {pct:.1f}% of queries")

    # Score distribution analysis
    print(f"\nScore distribution analysis:")
    with_neg_idx = (neg_mask > 0).nonzero(as_tuple=True)[0]
    if len(with_neg_idx) > 0:
        s_neg_vals = S_neg_all[with_neg_idx].flatten().cpu().numpy()
        s_base_vals = S_base_all[with_neg_idx].flatten().cpu().numpy()
        s_neg_above_base = (s_neg_vals > s_base_vals).mean() * 100
        print(f"  S_neg > S_base: {s_neg_above_base:.1f}% of (query, doc) pairs")
        print(f"  Avg S_neg: {s_neg_vals.mean():.4f}")
        print(f"  Avg S_base: {s_base_vals.mean():.4f}")
        print(f"  Avg gap (S_neg - S_base): {(s_neg_vals - s_base_vals).mean():.4f}")

    return {
        "task_name": task_name,
        "baseline_changed_map": bl_ch.get("map_at_1000", 0),
        "baseline_changed_ndcg5": bl_ch.get("ndcg_at_5", 0),
        "v2_changed_map": v2_ch.get("map_at_1000", 0),
        "v2_changed_ndcg5": v2_ch.get("ndcg_at_5", 0),
        "cos_qbase_qneg_mean": float(cos_qbase_qneg.mean()),
        "cos_qbase_qneg_std": float(cos_qbase_qneg.std()),
    }


def main():
    device = "cuda"

    datasets = [
        {
            "task_name": "Core17InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_Core17InstructionRetrieval.jsonl",
            "cache_dir": "dataset/FollowIR_test/embeddings",
        },
        {
            "task_name": "Robust04InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_Robust04InstructionRetrieval.jsonl",
            "cache_dir": "dataset/FollowIR_test/embeddings",
        },
        {
            "task_name": "News21InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_News21InstructionRetrieval.jsonl",
            "cache_dir": "dataset/FollowIR_test/embeddings",
        },
    ]

    results = []
    for ds in datasets:
        r = analyze_dataset(ds["task_name"], ds["dual_queries_path"], ds["cache_dir"], device)
        results.append(r)

    print(f"\n{'='*70}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*70}")
    print(f"{'Dataset':>30} {'Base_MAP':>10} {'V2_MAP':>10} {'Δ_MAP':>10} {'Base_nDCG':>10} {'V2_nDCG':>10} {'Cos_mean':>10}")
    for r in results:
        delta_map = r["v2_changed_map"] - r["baseline_changed_map"]
        print(f"{r['task_name']:>30} {r['baseline_changed_map']:>10.5f} {r['v2_changed_map']:>10.5f} {delta_map:>10.5f} {r['baseline_changed_ndcg5']:>10.5f} {r['v2_changed_ndcg5']:>10.5f} {r['cos_qbase_qneg_mean']:>10.4f}")


if __name__ == "__main__":
    main()
