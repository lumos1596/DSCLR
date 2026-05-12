import json
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.engine_dscrl import DSCLREvaluatorEngine


def analyze_reward_signal(task_name, dual_queries_path, device="cuda"):
    print(f"\n{'='*70}")
    print(f"Analyzing reward signal: {task_name}")
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

    def is_none_query(text):
        if not text:
            return True
        t = str(text).strip().upper()
        return t in ("[NONE]", "NONE", "NULL", "N/A", "")

    query_ids_changed = list(q_changed.keys())

    q_base_list = []
    q_req_list = []
    q_neg_list = []
    has_neg_list = []
    has_req_list = []

    for qid in query_ids_changed:
        raw = q_raw_changed.get(qid, ("", ""))
        query_text, instruction = raw[0], raw[1]
        q_base = f"{query_text} {instruction}".strip() if query_text else q_changed.get(qid, "")
        q_base_list.append(q_base)

        d = dual_data.get(qid, {})
        q_plus = d.get("q_plus", "")
        q_minus = d.get("q_minus", "")

        q_req_list.append(q_plus if not is_none_query(q_plus) else "")
        q_neg_list.append(q_minus if not is_none_query(q_minus) else "")
        has_req_list.append(0.0 if is_none_query(q_plus) else 1.0)
        has_neg_list.append(0.0 if is_none_query(q_minus) else 1.0)

    print("Encoding queries...")
    q_base_emb = engine._encode_queries(q_base_list)
    q_req_emb = engine._encode_queries(q_req_list)
    q_neg_emb = engine._encode_queries(q_neg_list)

    q_base_emb = F.normalize(q_base_emb, p=2, dim=1)
    q_req_emb = F.normalize(q_req_emb, p=2, dim=1)
    q_neg_emb = F.normalize(q_neg_emb, p=2, dim=1)

    cos_base_req = F.cosine_similarity(q_base_emb, q_req_emb, dim=1)
    cos_base_neg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)
    cos_req_neg = F.cosine_similarity(q_req_emb, q_neg_emb, dim=1)

    has_neg = torch.tensor(has_neg_list, dtype=torch.float32)
    has_req = torch.tensor(has_req_list, dtype=torch.float32)

    with_neg = has_neg > 0
    with_req = has_req > 0

    print(f"\nCosine similarity statistics:")
    print(f"  Cos(Q_base, Q+) [all queries]: mean={cos_base_req.mean():.4f}, std={cos_base_req.std():.4f}, min={cos_base_req.min():.4f}, max={cos_base_req.max():.4f}")
    if with_req.any():
        print(f"  Cos(Q_base, Q+) [with Q+]: mean={cos_base_req[with_req].mean():.4f}, std={cos_base_req[with_req].std():.4f}")
    print(f"  Cos(Q_base, Q-) [all queries]: mean={cos_base_neg.mean():.4f}, std={cos_base_neg.std():.4f}")
    if with_neg.any():
        print(f"  Cos(Q_base, Q-) [with Q-]: mean={cos_base_neg[with_neg].mean():.4f}, std={cos_base_neg[with_neg].std():.4f}")
    print(f"  Cos(Q+, Q-) [all queries]: mean={cos_req_neg.mean():.4f}, std={cos_req_neg.std():.4f}")

    # Key analysis: S_req vs S_base correlation
    # If Cos(Q_base, Q+) is very high, then S_req ≈ S_base for most documents
    # This means the reward β × S_req is just amplifying S_base
    print(f"\nReward redundancy analysis:")
    print(f"  Cos(Q_base, Q+) > 0.95: {(cos_base_req > 0.95).sum().item()}/{len(cos_base_req)} queries ({(cos_base_req > 0.95).float().mean()*100:.1f}%)")
    print(f"  Cos(Q_base, Q+) > 0.90: {(cos_base_req > 0.90).sum().item()}/{len(cos_base_req)} queries ({(cos_base_req > 0.90).float().mean()*100:.1f}%)")
    print(f"  Cos(Q_base, Q+) > 0.85: {(cos_base_req > 0.85).sum().item()}/{len(cos_base_req)} queries ({(cos_base_req > 0.85).float().mean()*100:.1f}%)")
    print(f"  Cos(Q_base, Q+) > 0.80: {(cos_base_req > 0.80).sum().item()}/{len(cos_base_req)} queries ({(cos_base_req > 0.80).float().mean()*100:.1f}%)")

    # Compute S_req - S_base for a sample of documents
    # Load doc embeddings
    all_doc_ids = sorted(set(d for docs in candidates.values() for d in docs))
    doc_texts = [corpus[did]["text"] for did in all_doc_ids]

    print(f"\nEncoding {len(all_doc_ids)} documents...")
    doc_emb = engine._encode_queries(doc_texts[:100])  # Just sample first 100 for speed
    doc_emb = F.normalize(doc_emb, p=2, dim=1)

    # Compute S_base and S_req for sample queries
    n_sample = min(10, len(query_ids_changed))
    sample_idx = list(range(n_sample))

    s_base_sample = torch.matmul(q_base_emb[sample_idx], doc_emb.T)
    s_req_sample = torch.matmul(q_req_emb[sample_idx], doc_emb.T)
    s_neg_sample = torch.matmul(q_neg_emb[sample_idx], doc_emb.T)

    delta_req = s_req_sample - s_base_sample

    print(f"\nS_req - S_base distribution (sample of {n_sample} queries × 100 docs):")
    print(f"  Mean: {delta_req.mean():.6f}")
    print(f"  Std: {delta_req.std():.6f}")
    print(f"  Min: {delta_req.min():.6f}")
    print(f"  Max: {delta_req.max():.6f}")
    print(f"  Fraction > 0: {(delta_req > 0).float().mean()*100:.1f}%")
    print(f"  Fraction > 0.01: {(delta_req > 0.01).float().mean()*100:.1f}%")
    print(f"  |S_req - S_base| / |S_base|: {(delta_req.abs().mean() / s_base_sample.abs().mean())*100:.2f}%")

    # Correlation between S_req and S_base
    s_base_flat = s_base_sample.flatten()
    s_req_flat = s_req_sample.flatten()
    correlation = torch.corrcoef(torch.stack([s_base_flat, s_req_flat]))[0, 1]
    print(f"  Corr(S_base, S_req): {correlation:.6f}")

    # Compute reward_scale = 1 - Cos(Q_base, Q+)
    reward_scale = 1 - cos_base_req
    print(f"\nReward scale (1 - Cos(Q_base, Q+)):")
    print(f"  Mean: {reward_scale.mean():.4f}")
    print(f"  Std: {reward_scale.std():.4f}")
    print(f"  Min: {reward_scale.min():.4f}")
    print(f"  Max: {reward_scale.max():.4f}")

    return {
        "task_name": task_name,
        "cos_base_req_mean": float(cos_base_req.mean()),
        "cos_base_req_std": float(cos_base_req.std()),
        "cos_base_neg_mean": float(cos_base_neg[with_neg].mean()) if with_neg.any() else 0,
        "reward_scale_mean": float(reward_scale.mean()),
        "delta_req_mean": float(delta_req.mean()),
        "delta_req_std": float(delta_req.std()),
        "sreq_sbase_corr": float(correlation),
    }


def main():
    device = "cuda"

    datasets = [
        {
            "task_name": "Core17InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_Core17InstructionRetrieval.jsonl",
        },
        {
            "task_name": "Robust04InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_Robust04InstructionRetrieval.jsonl",
        },
        {
            "task_name": "News21InstructionRetrieval",
            "dual_queries_path": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v4_News21InstructionRetrieval.jsonl",
        },
    ]

    results = []
    for ds in datasets:
        r = analyze_dataset(ds["task_name"], ds["dual_queries_path"], device)
        results.append(r)

    print(f"\n{'='*70}")
    print("CROSS-DATASET COMPARISON - REWARD SIGNAL ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Dataset':>30} {'Cos(B,R)':>9} {'Cos(B,N)':>9} {'RwdScale':>9} {'Δ(Sr-Sb)':>10} {'Corr':>7}")
    for r in results:
        print(f"{r['task_name']:>30} {r['cos_base_req_mean']:>9.4f} {r['cos_base_neg_mean']:>9.4f} {r['reward_scale_mean']:>9.4f} {r['delta_req_mean']:>10.6f} {r['sreq_sbase_corr']:>7.4f}")


if __name__ == "__main__":
    main()
