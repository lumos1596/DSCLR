"""Diagnostic script to validate the background leakage assumption in V8.6.

Runs the engine with an added diagnostic hook that collects per-query
S_base, S_neg, R_neg, Ŝ_neg^bg data, then computes diagnostics.

Diagnostics:
  1. Correlation between Ŝ_neg^bg and S_neg on safe docs
  2. Residual R_neg distribution analysis
  3. Trap docs (high S_base + high S_neg) have higher R_neg?
  4. Shuffled neg sanity check
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats as scipy_stats

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default="results/diagnose_bg_leakage")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--margin_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    from eval.engine import FollowIRDataLoader
    data_loader = FollowIRDataLoader(args.task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()
    q_raw_og, q_raw_changed = data_loader.load_raw_queries()
    dual_data_raw = []
    with open(args.dual_queries_path) as f:
        for line in f:
            dual_data_raw.append(json.loads(line.strip()))
    dual_data = {d["qid"]: d for d in dual_data_raw}

    all_doc_ids = list(set(did for doc_list in candidates.values() for did in doc_list))
    doc_id_to_idx = {did: i for i, did in enumerate(all_doc_ids)}

    # Use the project's own encoder for correct prompt template
    from eval.models.encoder import ModelFactory
    print("Loading model and encoding documents...")
    encoder = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    doc_texts = [corpus[did]["text"] for did in all_doc_ids]
    doc_emb = encoder.encode_documents(doc_texts, batch_size=args.batch_size)
    doc_emb_cpu = doc_emb.float().cpu()
    del doc_emb
    print(f"Doc embeddings shape: {doc_emb_cpu.shape}")

    # Free model before queries to save GPU memory
    del encoder
    torch.cuda.empty_cache()
    print("Model freed after document encoding.")

    # Build query lists (changed set)
    query_ids, q_base_list, q_neg_list, has_neg_list = [], [], [], []
    for qid in q_changed.keys():
        raw = q_raw_changed.get(qid, ("", ""))
        query_text, instruction = raw[0], raw[1]
        q_base = f"{query_text} {instruction}".strip() if query_text else q_changed.get(qid, "")
        d = dual_data.get(qid, {})
        q_minus = d.get("q_minus", "")
        q_neg = q_minus if q_minus and q_minus.strip().lower() not in ("none", "n/a", "null", "") else ""
        has_neg = bool(q_neg)
        query_ids.append(qid)
        q_base_list.append(q_base)
        q_neg_list.append(q_neg)
        has_neg_list.append(has_neg)

    # Encode queries (load model again after freeing)
    print("Encoding queries...")
    from eval.models.encoder import ModelFactory
    encoder2 = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    q_base_emb = encoder2.encode_queries(q_base_list, batch_size=args.batch_size).float().cpu()
    q_neg_emb = encoder2.encode_queries(q_neg_list, batch_size=args.batch_size).float().cpu()
    del encoder2
    torch.cuda.empty_cache()
    print(f"Query embeddings: base {q_base_emb.shape}, neg {q_neg_emb.shape}")

    # Compute scores on CPU (more memory efficient)
    print("Computing scores...")
    # Compute in chunks to save memory
    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)

    device = torch.device("cpu")

    # Collect per-query diagnostic data (compute scores on CPU per query)
    diag_data = []
    for i, qid in enumerate(query_ids):
        if not has_neg_list[i]:
            continue
        base_qid = qid.replace("-og", "").replace("-changed", "")
        cand_doc_ids = candidates.get(base_qid, [])
        if not cand_doc_ids:
            continue
        cand_indices = [doc_id_to_idx[did] for did in cand_doc_ids if did in doc_id_to_idx]
        if not cand_indices:
            continue
        idx_tensor = torch.tensor(cand_indices, dtype=torch.long)

        # Compute scores for this query only (on CPU)
        s_base = torch.matmul(q_base_emb[i], doc_emb_cpu[idx_tensor].T)
        s_neg = torch.matmul(q_neg_emb[i], doc_emb_cpu[idx_tensor].T)
        c_q = cos_qbase_qneg[i].item()

        # Background leakage prediction
        mu_n = s_neg.mean()
        sigma_n = s_neg.std()
        mu_b = s_base.mean()
        sigma_b = s_base.std()
        z_b = (s_base - mu_b) / sigma_b if sigma_b > 1e-8 else torch.zeros_like(s_base)
        s_neg_bg = mu_n + sigma_n * c_q * z_b
        R_neg = s_neg - s_neg_bg

        mad_R = (R_neg - R_neg.median()).abs().median() * 1.4826
        m_q = args.margin_scale * mad_R
        overflow = R_neg - m_q
        safe_mask = overflow <= 0

        # Convert to numpy
        s_base_np = s_base.cpu().numpy()
        s_neg_np = s_neg.cpu().numpy()
        s_neg_bg_np = s_neg_bg.cpu().numpy()
        R_neg_np = R_neg.cpu().numpy()
        overflow_np = overflow.cpu().numpy()
        safe_mask_np = safe_mask.cpu().numpy()

        diag_data.append({
            "qid": qid,
            "cos_qbase_qneg": c_q,
            "num_candidates": len(cand_indices),
            "s_base": s_base_np,
            "s_neg": s_neg_np,
            "s_neg_bg": s_neg_bg_np,
            "R_neg": R_neg_np,
            "overflow": overflow_np,
            "safe_mask": safe_mask_np,
            "mad_R": float(mad_R.item()),
            "m_q": float(m_q.item()),
        })

    # =========================================================================
    # DIAGNOSTIC 1: Correlation on safe docs
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: Correlation(Ŝ_neg^bg, S_neg) on safe docs")
    print("=" * 60)

    corr_real = []
    r2_real = []
    for d in diag_data:
        safe = d["safe_mask"]
        if safe.sum() > 2:
            pred = d["s_neg_bg"][safe]
            actual = d["s_neg"][safe]
            r = np.corrcoef(pred, actual)[0, 1]
            corr_real.append(r)
            r2_real.append(r ** 2)

    print(f"  Queries with neg: {len(diag_data)}")
    print(f"  Correlation on safe docs:")
    print(f"    mean r  = {np.mean(corr_real):.4f}, median r = {np.median(corr_real):.4f}")
    print(f"    mean R² = {np.mean(r2_real):.4f}, median R² = {np.median(r2_real):.4f}")
    print(f"    min r   = {np.min(corr_real):.4f}, max r = {np.max(corr_real):.4f}")

    # =========================================================================
    # DIAGNOSTIC 2: Residual distribution
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: Residual R_neg distribution")
    print("=" * 60)

    all_R = np.concatenate([d["R_neg"] for d in diag_data])
    skewness_global = float(((all_R - all_R.mean()) ** 3).mean() / (all_R.std() ** 3 + 1e-12))

    per_query_skew = []
    per_query_frac_pos = []
    per_query_frac_over = []
    for d in diag_data:
        R = d["R_neg"]
        sk = float(((R - R.mean()) ** 3).mean() / (R.std() ** 3 + 1e-12))
        per_query_skew.append(sk)
        per_query_frac_pos.append(float((R > 0).mean()))
        per_query_frac_over.append(float((d["overflow"] > 0).mean()))

    print(f"  Global residual (pooled across all queries):")
    print(f"    mean = {all_R.mean():.6f}, median = {np.median(all_R):.6f}, std = {all_R.std():.6f}")
    print(f"    skewness = {skewness_global:.4f}")
    print(f"    frac(R > 0) = {(all_R > 0).mean():.4f}")
    print(f"  Per-query summary:")
    print(f"    skewness:       mean={np.mean(per_query_skew):.4f}, median={np.median(per_query_skew):.4f}")
    print(f"    frac(R > 0):    mean={np.mean(per_query_frac_pos):.4f}, median={np.median(per_query_frac_pos):.4f}")
    print(f"    at-risk ratio:  mean={np.mean(per_query_frac_over):.4f}, median={np.median(per_query_frac_over):.4f}")

    # =========================================================================
    # DIAGNOSTIC 3: Trap docs vs non-trap docs
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: Trap docs (top 25% S_base AND top 25% S_neg)")
    print("=" * 60)

    R_trap_means = []
    R_non_trap_means = []
    R_trap_fracs = []
    R_non_trap_fracs = []
    per_query_trap = []

    for d in diag_data:
        q75_b = np.percentile(d["s_base"], 75)
        q75_n = np.percentile(d["s_neg"], 75)
        trap = (d["s_base"] > q75_b) & (d["s_neg"] > q75_n)
        non_trap = ~trap

        if trap.sum() > 0 and non_trap.sum() > 0:
            R_trap = d["R_neg"][trap]
            R_non_trap = d["R_neg"][non_trap]
            R_trap_means.append(R_trap.mean())
            R_non_trap_means.append(R_non_trap.mean())
            R_trap_fracs.append((R_trap > 0).mean())
            R_non_trap_fracs.append((R_non_trap > 0).mean())
            per_query_trap.append({
                "qid": d["qid"],
                "R_trap_mean": float(R_trap.mean()),
                "R_non_trap_mean": float(R_non_trap.mean()),
                "R_trap_frac_pos": float((R_trap > 0).mean()),
                "R_non_trap_frac_pos": float((R_non_trap > 0).mean()),
            })

    print(f"  Trap docs (top 25% × top 25%):")
    print(f"    R_neg mean:  trap={np.mean(R_trap_means):.4f}, non-trap={np.mean(R_non_trap_means):.4f}")
    print(f"    frac(R>0):   trap={np.mean(R_trap_fracs):.4f}, non-trap={np.mean(R_non_trap_fracs):.4f}")

    t_stat, p_value = scipy_stats.ttest_rel(R_trap_means, R_non_trap_means)
    print(f"    Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
    sig = "SIGNIFICANTLY" if p_value < 0.05 else "NOT significantly"
    print(f"    → Trap docs have {sig} higher residuals (p={p_value:.4f})")

    # Also test top-10% trap docs
    R_trap10_means = []
    R_non_trap10_means = []
    for d in diag_data:
        q90_b = np.percentile(d["s_base"], 90)
        q90_n = np.percentile(d["s_neg"], 90)
        trap = (d["s_base"] > q90_b) & (d["s_neg"] > q90_n)
        non_trap = ~trap
        if trap.sum() > 0 and non_trap.sum() > 0:
            R_trap10_means.append(d["R_neg"][trap].mean())
            R_non_trap10_means.append(d["R_neg"][non_trap].mean())

    if R_trap10_means:
        t10, p10 = scipy_stats.ttest_rel(R_trap10_means, R_non_trap10_means)
        print(f"  Trap docs (top 10% × top 10%):")
        print(f"    R_neg mean:  trap={np.mean(R_trap10_means):.4f}, non-trap={np.mean(R_non_trap10_means):.4f}")
        print(f"    Paired t-test: t={t10:.4f}, p={p10:.6f}")

    # =========================================================================
    # DIAGNOSTIC 4: Shuffled neg sanity check
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: Shuffled neg sanity check")
    print("=" * 60)

    n_neg_queries = sum(has_neg_list)
    # Only shuffle among neg queries
    neg_indices = [i for i, h in enumerate(has_neg_list) if h]
    perm_neg = np.random.permutation(neg_indices)
    q_neg_emb_shuffled = q_neg_emb.clone()
    q_neg_emb_shuffled[neg_indices] = q_neg_emb[perm_neg]

    cos_shuffled = F.cosine_similarity(q_base_emb, q_neg_emb_shuffled, dim=1)

    corr_shuffled = []
    at_risk_shuffled = []

    for d in diag_data:
        i = query_ids.index(d["qid"])
        base_qid = d["qid"].replace("-og", "").replace("-changed", "")
        cand_doc_ids = candidates.get(base_qid, [])
        cand_indices = [doc_id_to_idx[did] for did in cand_doc_ids if did in doc_id_to_idx]
        idx_tensor = torch.tensor(cand_indices, dtype=torch.long)

        s_base = torch.matmul(q_base_emb[i], doc_emb_cpu[idx_tensor].T)
        s_neg_sh = torch.matmul(q_neg_emb_shuffled[i], doc_emb_cpu[idx_tensor].T)
        c_q_sh = cos_shuffled[i].item()

        mu_n = s_neg_sh.mean()
        sigma_n = s_neg_sh.std()
        mu_b = s_base.mean()
        sigma_b = s_base.std()
        z_b = (s_base - mu_b) / sigma_b if sigma_b > 1e-8 else torch.zeros_like(s_base)
        s_neg_bg_sh = mu_n + sigma_n * c_q_sh * z_b
        R_neg_sh = s_neg_sh - s_neg_bg_sh
        mad_sh = (R_neg_sh - R_neg_sh.median()).abs().median() * 1.4826
        overflow_sh = R_neg_sh - args.margin_scale * mad_sh
        safe_sh = overflow_sh <= 0

        if safe_sh.sum() > 2:
            pred = s_neg_bg_sh[safe_sh].cpu().numpy()
            actual = s_neg_sh[safe_sh].cpu().numpy()
            c = np.corrcoef(pred, actual)[0, 1]
            corr_shuffled.append(c)
        else:
            corr_shuffled.append(float('nan'))

        at_risk_shuffled.append(float((overflow_sh > 0).float().mean().item()))

    corr_shuffled_clean = [c for c in corr_shuffled if not np.isnan(c)]
    real_cos = [d["cos_qbase_qneg"] for d in diag_data]
    sh_cos = [cos_shuffled[query_ids.index(d["qid"])].item() for d in diag_data]

    print(f"  cos(Q_base, Q_neg):")
    print(f"    real:     mean={np.mean(real_cos):.4f}, std={np.std(real_cos):.4f}")
    print(f"    shuffled: mean={np.mean(sh_cos):.4f}, std={np.std(sh_cos):.4f}")
    print(f"  Correlation on safe docs:")
    print(f"    real:     mean={np.mean(corr_real):.4f}, median={np.median(corr_real):.4f}")
    print(f"    shuffled: mean={np.mean(corr_shuffled_clean):.4f}, median={np.median(corr_shuffled_clean):.4f}")
    print(f"  At-risk ratio:")
    print(f"    real:     mean={np.mean(per_query_frac_over):.4f}")
    print(f"    shuffled: mean={np.mean(at_risk_shuffled):.4f}")

    # Paired test for correlation
    paired_real = []
    paired_sh = []
    for i, d in enumerate(diag_data):
        if not np.isnan(corr_shuffled[i]):
            paired_real.append(corr_real[i])
            paired_sh.append(corr_shuffled[i])

    t_sh, p_sh = scipy_stats.ttest_rel(paired_real, paired_sh)
    print(f"  Paired t-test (correlation real vs shuffled): t={t_sh:.4f}, p={p_sh:.6f}")
    sig_sh = "SIGNIFICANTLY" if p_sh < 0.05 else "NOT significantly"
    print(f"  → Real neg correlation is {sig_sh} higher than shuffled (p={p_sh:.4f})")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "diagnostic_1_correlation": {
            "mean_r": float(np.mean(corr_real)),
            "median_r": float(np.median(corr_real)),
            "mean_r_squared": float(np.mean(r2_real)),
            "min_r": float(np.min(corr_real)),
            "max_r": float(np.max(corr_real)),
        },
        "diagnostic_2_residual_distribution": {
            "global_mean": float(all_R.mean()),
            "global_median": float(np.median(all_R)),
            "global_std": float(all_R.std()),
            "global_skewness": skewness_global,
            "global_frac_positive": float((all_R > 0).mean()),
            "mean_per_query_skewness": float(np.mean(per_query_skew)),
            "mean_at_risk_ratio": float(np.mean(per_query_frac_over)),
        },
        "diagnostic_3_trap_docs": {
            "R_trap_mean": float(np.mean(R_trap_means)),
            "R_non_trap_mean": float(np.mean(R_non_trap_means)),
            "R_trap_frac_positive": float(np.mean(R_trap_fracs)),
            "R_non_trap_frac_positive": float(np.mean(R_non_trap_fracs)),
            "paired_t": float(t_stat),
            "paired_p": float(p_value),
            "significant": bool(p_value < 0.05),
            "top10_R_trap_mean": float(np.mean(R_trap10_means)) if R_trap10_means else None,
            "top10_R_non_trap_mean": float(np.mean(R_non_trap10_means)) if R_non_trap10_means else None,
            "top10_paired_p": float(p10) if R_trap10_means else None,
        },
        "diagnostic_4_shuffled_neg": {
            "real_mean_correlation": float(np.mean(corr_real)),
            "shuffled_mean_correlation": float(np.mean(corr_shuffled_clean)),
            "real_mean_cos": float(np.mean(real_cos)),
            "shuffled_mean_cos": float(np.mean(sh_cos)),
            "real_mean_at_risk": float(np.mean(per_query_frac_over)),
            "shuffled_mean_at_risk": float(np.mean(at_risk_shuffled)),
            "paired_t": float(t_sh),
            "paired_p": float(p_sh),
            "significant": bool(p_sh < 0.05),
        },
    }

    out_path = os.path.join(args.output_dir, "diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o))
    print(f"\nResults saved to {out_path}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    d1_ok = np.mean(corr_real) > 0.5
    d2_ok = abs(skewness_global) < 1.0
    d3_ok = p_value < 0.05
    d4_ok = p_sh < 0.05

    print(f"  D1 (correlation > 0.5): {'PASS' if d1_ok else 'FAIL'} (r = {np.mean(corr_real):.4f})")
    print(f"  D2 (residual near-symmetric): {'PASS' if d2_ok else 'FAIL'} (skewness = {skewness_global:.4f})")
    print(f"  D3 (trap docs have higher R): {'PASS' if d3_ok else 'FAIL'} (p = {p_value:.4f})")
    print(f"  D4 (real > shuffled): {'PASS' if d4_ok else 'FAIL'} (p = {p_sh:.4f})")
    all_pass = d1_ok and d2_ok and d3_ok and d4_ok
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")


if __name__ == "__main__":
    main()
