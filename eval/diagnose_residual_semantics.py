"""Diagnostic figure: Does R_neg/MAD distinguish trap docs from relevant docs?

Generates a violin/box plot showing the distribution of R_neg/MAD for three
document categories:
  - Relevant docs (qrel relevance > 0)
  - Trap docs (high S_base AND high S_neg, but qrel relevance = 0)
  - Other non-relevant docs

Usage:
  python -m eval.diagnose_residual_semantics \
    --task_name Core17InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
    --output_dir results/diagnose_residual_semantics \
    --device cuda --batch_size 64
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default="results/diagnose_residual_semantics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--margin_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    from eval.engine import FollowIRDataLoader
    from eval.metrics.evaluator import DataLoader
    data_loader = FollowIRDataLoader(args.task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()
    q_raw_og, q_raw_changed = data_loader.load_raw_queries()
    qrels = DataLoader(args.task_name).load_qrels()
    dual_data_raw = []
    with open(args.dual_queries_path) as f:
        for line in f:
            dual_data_raw.append(json.loads(line.strip()))
    dual_data = {d["qid"]: d for d in dual_data_raw}

    all_doc_ids = list(set(did for doc_list in candidates.values() for did in doc_list))
    doc_id_to_idx = {did: i for i, did in enumerate(all_doc_ids)}

    # Encode documents (try loading from cache first)
    from eval.models.encoder import ModelFactory
    cache_dir = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"
    model_short = args.model_name.replace("samaya-ai/", "").replace("/", "_").replace("-", "_")
    cache_emb_path = os.path.join(cache_dir, model_short, f"{args.task_name}_{model_short}_corpus_embeddings.npy")
    cache_ids_path = os.path.join(cache_dir, model_short, f"{args.task_name}_{model_short}_corpus_ids.json")

    if os.path.exists(cache_emb_path) and os.path.exists(cache_ids_path):
        print(f"Loading cached doc embeddings from {cache_emb_path}")
        doc_emb_np = np.load(cache_emb_path)
        with open(cache_ids_path) as f:
            cached_doc_ids = json.load(f)
        cached_id_to_idx = {did: i for i, did in enumerate(cached_doc_ids)}
        # Build doc_emb in all_doc_ids order
        doc_emb_cpu = torch.zeros(len(all_doc_ids), doc_emb_np.shape[1])
        for i, did in enumerate(all_doc_ids):
            if did in cached_id_to_idx:
                doc_emb_cpu[i] = torch.from_numpy(doc_emb_np[cached_id_to_idx[did]])
        del doc_emb_np
        print(f"Doc embeddings shape: {doc_emb_cpu.shape} (from cache)")
    else:
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
        del doc_emb, encoder
        torch.cuda.empty_cache()
        print(f"Doc embeddings shape: {doc_emb_cpu.shape}")

    # Encode queries
    print("Encoding queries...")
    encoder2 = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    query_ids, q_base_list, q_neg_list, has_neg_list = [], [], [], []
    for qid in q_changed.keys():
        raw = q_raw_changed.get(qid, ("", ""))
        query_text, instruction = raw[0], raw[1]
        q_base = f"{query_text} {instruction}".strip() if query_text else q_changed.get(qid, "")
        d = dual_data.get(qid, {})
        q_minus = d.get("q_minus", "")
        q_neg = q_minus if q_minus and q_minus.strip().lower() not in ("none", "n/a", "null", "") else ""
        query_ids.append(qid)
        q_base_list.append(q_base)
        q_neg_list.append(q_neg)
        has_neg_list.append(bool(q_neg))

    q_base_emb = encoder2.encode_queries(q_base_list, batch_size=args.batch_size).float().cpu()
    q_neg_emb = encoder2.encode_queries(q_neg_list, batch_size=args.batch_size).float().cpu()
    del encoder2
    torch.cuda.empty_cache()
    print(f"Query embeddings: base {q_base_emb.shape}, neg {q_neg_emb.shape}")

    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)

    # Collect per-document R_neg/MAD values with category labels
    relevant_rneg_mad = []      # R_neg/MAD for relevant docs
    trap_rneg_mad = []          # R_neg/MAD for trap docs
    other_rneg_mad = []         # R_neg/MAD for other non-relevant docs

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
        if mad_R < 1e-8:
            continue

        r_neg_mad = (R_neg / mad_R).cpu().numpy()
        s_base_np = s_base.cpu().numpy()
        s_neg_np = s_neg.cpu().numpy()

        # Get qrels for this query (changed version)
        query_qrels = qrels.get(qid, {})
        if not query_qrels:
            query_qrels = qrels.get(base_qid, {})

        # Define trap docs: top 25% S_base AND top 25% S_neg
        q75_b = np.percentile(s_base_np, 75)
        q75_n = np.percentile(s_neg_np, 75)
        trap_mask = (s_base_np > q75_b) & (s_neg_np > q75_n)

        for j, doc_id in enumerate(cand_doc_ids):
            rel = query_qrels.get(doc_id, 0)
            val = float(r_neg_mad[j])
            if rel > 0:
                relevant_rneg_mad.append(val)
            elif trap_mask[j]:
                trap_rneg_mad.append(val)
            else:
                other_rneg_mad.append(val)

    print(f"\nDocument counts:")
    print(f"  Relevant: {len(relevant_rneg_mad)}")
    print(f"  Trap:     {len(trap_rneg_mad)}")
    print(f"  Other:    {len(other_rneg_mad)}")

    # Statistical tests
    from scipy import stats as scipy_stats

    relevant_arr = np.array(relevant_rneg_mad)
    trap_arr = np.array(trap_rneg_mad)
    other_arr = np.array(other_rneg_mad)

    print(f"\nR_neg/MAD statistics:")
    print(f"  Relevant:  mean={relevant_arr.mean():.4f}, median={np.median(relevant_arr):.4f}, std={relevant_arr.std():.4f}")
    print(f"  Trap:      mean={trap_arr.mean():.4f}, median={np.median(trap_arr):.4f}, std={trap_arr.std():.4f}")
    print(f"  Other:     mean={other_arr.mean():.4f}, median={np.median(other_arr):.4f}, std={other_arr.std():.4f}")

    # Mann-Whitney U test: trap vs relevant
    u_stat, u_p = scipy_stats.mannwhitneyu(trap_arr, relevant_arr, alternative="greater")
    print(f"\n  Mann-Whitney U (trap > relevant): U={u_stat:.1f}, p={u_p:.6f}")

    # t-test: trap vs relevant
    t_stat, t_p = scipy_stats.ttest_ind(trap_arr, relevant_arr)
    print(f"  t-test (trap vs relevant): t={t_stat:.4f}, p={t_p:.6f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))

    data = [relevant_arr, trap_arr, other_arr]
    labels = ["Relevant", "Trap", "Other\nnon-rel."]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    # Violin plot
    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for key in ["cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)

    # Overlay box plot for quartiles
    bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"$R_{\mathrm{neg}} / \mathrm{MAD}$", fontsize=12)
    ax.set_title("Residual separates trap docs from relevant docs", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=args.margin_scale, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label=f"penalty threshold (λ={args.margin_scale})")
    ax.legend(fontsize=9, loc="upper right")

    # Add significance annotation
    y_max = max(relevant_arr.max(), trap_arr.max(), other_arr.max())
    y_bar = y_max * 1.05
    ax.plot([1, 1, 2, 2], [y_bar, y_bar * 1.03, y_bar * 1.03, y_bar], "k-", linewidth=1)
    sig_label = "***" if u_p < 0.001 else ("**" if u_p < 0.01 else ("*" if u_p < 0.05 else "n.s."))
    ax.text(1.5, y_bar * 1.05, sig_label, ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "residual_semantics_violin.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")

    # Also save a histogram version
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bins = np.linspace(-3, 6, 50)
    ax2.hist(relevant_arr, bins=bins, alpha=0.5, color=colors[0], label=f"Relevant (n={len(relevant_arr)})", density=True)
    ax2.hist(trap_arr, bins=bins, alpha=0.5, color=colors[1], label=f"Trap (n={len(trap_arr)})", density=True)
    ax2.hist(other_arr, bins=bins, alpha=0.3, color=colors[2], label=f"Other (n={len(other_arr)})", density=True)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axvline(x=args.margin_scale, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
                label=f"penalty threshold (λ={args.margin_scale})")
    ax2.set_xlabel(r"$R_{\mathrm{neg}} / \mathrm{MAD}$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("R_neg/MAD distribution by document category", fontsize=11)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    fig2_path = os.path.join(args.output_dir, "residual_semantics_hist.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {fig2_path}")

    # Save statistics
    output = {
        "task_name": args.task_name,
        "counts": {
            "relevant": len(relevant_rneg_mad),
            "trap": len(trap_rneg_mad),
            "other": len(other_rneg_mad),
        },
        "r_neg_mad_stats": {
            "relevant": {"mean": float(relevant_arr.mean()), "median": float(np.median(relevant_arr)), "std": float(relevant_arr.std())},
            "trap": {"mean": float(trap_arr.mean()), "median": float(np.median(trap_arr)), "std": float(trap_arr.std())},
            "other": {"mean": float(other_arr.mean()), "median": float(np.median(other_arr)), "std": float(other_arr.std())},
        },
        "mannwhitneyu_trap_vs_relevant": {"U": float(u_stat), "p": float(u_p)},
        "ttest_trap_vs_relevant": {"t": float(t_stat), "p": float(t_p)},
        "margin_scale": args.margin_scale,
    }
    stats_path = os.path.join(args.output_dir, "residual_semantics_stats.json")
    with open(stats_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
