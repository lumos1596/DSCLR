import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=200)
    parser.add_argument("--top_k_ratio", type=float, default=0.5,
                        help="Only keep neg docs with S_base in top top_k_ratio percentile")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embeddings_path", type=str,
                        default="dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt")
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--deltas", type=str, default=None)
    parser.add_argument("--gammas", type=str, default=None)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    groups_raw = []
    with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
        for line in f:
            groups_raw.append(json.loads(line.strip()))

    cache = torch.load(args.embeddings_path, map_location=args.device, weights_only=False)
    model_name = cache.get("model_name", "unknown")
    print(f"Using embeddings from: {args.embeddings_path}")
    print(f"Model: {model_name}")
    print(f"Top-K ratio: {args.top_k_ratio}")

    q_base = F.normalize(cache["q_base_embeddings"].float(), p=2, dim=1)
    q_plus = F.normalize(cache["q_plus_embeddings"].float(), p=2, dim=1)
    q_minus = F.normalize(cache["q_minus_embeddings"].float(), p=2, dim=1)
    pos_embs = F.normalize(cache["pos_embeddings"].float(), p=2, dim=1)
    neg_embs = F.normalize(cache["neg_embeddings"].float(), p=2, dim=1)

    n_queries = len(groups_raw)
    has_req_mask = torch.tensor(
        [
            float(
                json.loads(item["output"]).get("Q_plus", "")
                not in ("[NONE]", "NONE", "", None)
            )
            for item in groups_raw
        ],
        dtype=torch.float32,
    )
    has_neg_mask = torch.tensor(
        [
            float(
                json.loads(item["output"]).get("Q_minus", "")
                not in ("[NONE]", "NONE", "", None)
            )
            for item in groups_raw
        ],
        dtype=torch.float32,
    )
    cos_qbase_qneg = F.cosine_similarity(q_base, q_minus, dim=1)
    cos_qbase_qreq = F.cosine_similarity(q_base, q_plus, dim=1)

    print(f"Cos(Q_base, Q_neg): min={cos_qbase_qneg.min():.4f}, max={cos_qbase_qneg.max():.4f}, mean={cos_qbase_qneg.mean():.4f}")
    print(f"Cos(Q_base, Q_req): min={cos_qbase_qreq.min():.4f}, max={cos_qbase_qreq.max():.4f}, mean={cos_qbase_qreq.mean():.4f}")

    np.random.seed(42)
    n_dist = args.n_distractors

    # Compute S_base for all neg docs against all queries
    print("Computing S_base for all neg docs against all queries...")
    S_base_neg_matrix = q_base @ neg_embs.T  # (n_queries, n_neg)
    S_neg_neg_matrix = q_minus @ neg_embs.T   # (n_queries, n_neg)

    # Score-Percentile Filtering: sample distractors with S_base distribution
    # that mimics test set (most docs have low S_base, but top-k have high S_base)
    # Strategy: mix top-k% (high S_base) and bottom-k% (low S_base) distractors
    # to simulate the test set distribution where ~top-5% are "retrieved" candidates
    
    distractor_indices = []
    for q_idx in range(n_queries):
        own_neg_start = q_idx * 15
        candidates = np.array([
            i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_start + 15
        ])
        s_base_vals = S_base_neg_matrix[q_idx, candidates].numpy()
        
        # Sort by S_base descending
        sorted_idx = np.argsort(-s_base_vals)
        
        # Take top_k_ratio from high S_base (simulating retrieved candidates)
        # and rest from low S_base (simulating non-retrieved docs)
        n_high = int(n_dist * args.top_k_ratio)
        n_low = n_dist - n_high
        
        high_idx = candidates[sorted_idx[:n_high]]
        if n_low > 0:
            low_idx = candidates[sorted_idx[-(n_low * 3):]]  # sample from bottom 3x
            low_sampled = np.random.choice(low_idx, size=n_low, replace=False)
            sampled = np.concatenate([high_idx, low_sampled])
        else:
            sampled = high_idx
        
        distractor_indices.append(sampled)

    # Verify at-risk ratio
    at_risk_counts = []
    for q_idx in range(n_queries):
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        s_neg_dist = S_neg_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        at_risk = (s_neg_dist > s_base_dist).mean()
        at_risk_counts.append(at_risk)
    avg_at_risk = np.mean(at_risk_counts)
    print(f"Distractor at-risk ratio (S_neg > S_base): {avg_at_risk:.4f}")

    # Check S_base distribution of distractors
    all_dist_sbase = []
    for q_idx in range(n_queries):
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        all_dist_sbase.extend(s_base_dist.tolist())
    all_dist_sbase = np.array(all_dist_sbase)
    print(f"Distractor S_base: mean={all_dist_sbase.mean():.4f}, std={all_dist_sbase.std():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p}={np.percentile(all_dist_sbase, p):.4f}")

    print(f"\nPrecomputing scores with {n_dist} distractors per query...")
    pos_offset = 0
    all_s_base = []
    all_s_req = []
    all_s_neg = []
    all_pos_count = []
    all_has_req = []
    all_has_neg = []
    all_cos_neg = []
    all_cos_req = []

    for q_idx in range(n_queries):
        item = groups_raw[q_idx]
        pos_count = len(item.get("pos", []))

        s_base = [
            torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item()
            for pi in range(pos_count)
        ]
        s_req = [
            torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item()
            for pi in range(pos_count)
        ]
        s_neg = [
            torch.dot(q_minus[q_idx], pos_embs[pos_offset + pi]).item()
            for pi in range(pos_count)
        ]

        s_base += [
            torch.dot(q_base[q_idx], neg_embs[q_idx * 15 + ni]).item()
            for ni in range(15)
        ]
        s_req += [
            torch.dot(q_plus[q_idx], neg_embs[q_idx * 15 + ni]).item()
            for ni in range(15)
        ]
        s_neg += [
            torch.dot(q_minus[q_idx], neg_embs[q_idx * 15 + ni]).item()
            for ni in range(15)
        ]

        s_base += [
            torch.dot(q_base[q_idx], neg_embs[di]).item()
            for di in distractor_indices[q_idx]
        ]
        s_req += [
            torch.dot(q_plus[q_idx], neg_embs[di]).item()
            for di in distractor_indices[q_idx]
        ]
        s_neg += [
            torch.dot(q_minus[q_idx], neg_embs[di]).item()
            for di in distractor_indices[q_idx]
        ]

        all_s_base.append(np.array(s_base))
        all_s_req.append(np.array(s_req))
        all_s_neg.append(np.array(s_neg))
        all_pos_count.append(pos_count)
        all_has_req.append(has_req_mask[q_idx].item() > 0)
        all_has_neg.append(has_neg_mask[q_idx].item() > 0)
        all_cos_neg.append(cos_qbase_qneg[q_idx].item())
        all_cos_req.append(cos_qbase_qreq[q_idx].item())

        pos_offset += pos_count

    print(f"Precomputed scores for {n_queries} queries")

    alpha_list = [float(x) for x in args.alphas.split(",")] if args.alphas else [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]
    beta_list = [float(x) for x in args.betas.split(",")] if args.betas else [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    delta_list = [float(x) for x in args.deltas.split(",")] if args.deltas else [-0.15, -0.10, -0.05, 0.00, 0.05]
    gamma_list = [float(x) for x in args.gammas.split(",")] if args.gammas else [1.0]

    all_results = []
    total = len(alpha_list) * len(beta_list) * len(delta_list) * len(gamma_list)
    trial = 0

    for alpha in alpha_list:
        for beta in beta_list:
            for delta in delta_list:
                for gamma in gamma_list:
                    trial += 1
                    maps = []
                    ndcgs = []
                    for q_idx in range(n_queries):
                        s_b = all_s_base[q_idx]
                        s_r = all_s_req[q_idx]
                        s_n = all_s_neg[q_idx]
                        pos_count = all_pos_count[q_idx]
                        has_req = all_has_req[q_idx]
                        has_neg = all_has_neg[q_idx]
                        cos_neg = all_cos_neg[q_idx]
                        cos_req = all_cos_req[q_idx]
                        tau = cos_neg + delta

                        if has_req:
                            raw_gate = max(0.0, 1.0 - cos_req)
                            reward_gate = raw_gate ** gamma
                        else:
                            reward_gate = 0.0

                        if not has_neg:
                            s_req_eff = s_r if has_req else np.zeros_like(s_b)
                            s_final = s_b + beta * reward_gate * s_req_eff
                        else:
                            smooth_penalty = np.log1p(np.exp(s_n - tau))
                            gap_w = 1.0 / (1.0 + np.exp(-(s_n - s_b) * 20.0))
                            raw_penalty = alpha * smooth_penalty * gap_w
                            penalty = np.minimum(raw_penalty, s_b * 0.5)
                            safety = 1.0 - 1.0 / (1.0 + np.exp(-(s_n - tau) * 20.0))
                            s_req_eff = s_r if has_req else np.zeros_like(s_b)
                            s_final = s_b + beta * reward_gate * s_req_eff * safety - penalty

                        n_total = len(s_final)
                        rel = np.zeros(n_total)
                        rel[:pos_count] = 1.0
                        sorted_idx = np.argsort(-s_final)

                        ap_sum = 0.0
                        hits = 0
                        for rank, idx in enumerate(sorted_idx):
                            if idx < pos_count:
                                hits += 1
                                ap_sum += hits / (rank + 1)
                        maps.append(ap_sum / pos_count if pos_count > 0 else 0.0)

                        dcg = sum(
                            rel[sorted_idx[r]] / math.log2(r + 2)
                            for r in range(min(5, n_total))
                        )
                        idcg = sum(
                            1.0 / math.log2(i + 2) for i in range(min(5, pos_count))
                        )
                        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

                    avg_map = float(np.mean(maps))
                    avg_ndcg = float(np.mean(ndcgs))
                    all_results.append(
                        {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "delta": float(delta),
                            "gamma": float(gamma),
                            "map": avg_map,
                            "ndcg@5": avg_ndcg,
                        }
                    )
                    if trial % 50 == 0:
                        print(
                            f"[{trial}/{total}] a={alpha} b={beta} d={delta} g={gamma} | MAP={avg_map:.4f} nDCG@5={avg_ndcg:.4f}"
                        )

    all_results.sort(key=lambda x: x["map"], reverse=True)
    print(f"\n=== Top 10 by MAP (top_k_ratio={args.top_k_ratio}, {n_dist} distractors, {model_name}) ===")
    for i, r in enumerate(all_results[:10]):
        print(
            f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.1f} d={r["delta"]:.2f} g={r["gamma"]:.1f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f}'
        )

    all_results.sort(key=lambda x: x["ndcg@5"], reverse=True)
    print(f"\n=== Top 10 by nDCG@5 (top_k_ratio={args.top_k_ratio}, {n_dist} distractors, {model_name}) ===")
    for i, r in enumerate(all_results[:10]):
        print(
            f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.1f} d={r["delta"]:.2f} g={r["gamma"]:.1f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f}'
        )

    suffix = args.output_suffix
    model_name_safe = model_name.replace("/", "_")
    with open(
        f"dataset/FollowIR_train/train/train_param_search_topk{args.top_k_ratio}_{model_name_safe}{suffix}.json",
        "w",
    ) as f:
        json.dump(
            {
                "n_distractors": n_dist,
                "model_name": model_name,
                "top_k_ratio": args.top_k_ratio,
                "actual_at_risk_ratio": float(avg_at_risk),
                "best_by_map": sorted(all_results, key=lambda x: x["map"], reverse=True)[0],
                "best_by_ndcg": sorted(all_results, key=lambda x: x["ndcg@5"], reverse=True)[0],
                "all_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
