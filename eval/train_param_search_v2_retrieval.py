import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=200)
    parser.add_argument("--retrieval_top_k", type=int, default=100,
                        help="Number of top-k documents to retrieve with Q_base (simulating test-set retrieval)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embeddings_path", type=str,
                        default="dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt")
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--deltas", type=str, default=None)
    parser.add_argument("--analytical_beta", action="store_true",
                        help="Use analytical beta estimation instead of grid search for beta")
    parser.add_argument("--analytical_k", type=float, default=0.5,
                        help="k factor for analytical beta: beta = k * mean(S_base_pos) / (mean(reward_gate) * mean(S_req_pos))")
    parser.add_argument("--analytical_beta_v2", action="store_true",
                        help="Use refined analytical beta: median of beta_needed for pos docs in top-1000 where S_req_gap > 0")
    parser.add_argument("--analytical_beta_percentile", type=float, default=50,
                        help="Percentile of beta_needed distribution to use (default: 50=median)")
    parser.add_argument("--beta_from_analytical", action="store_true",
                        help="Fix beta at analytical estimate and search only alpha/delta")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--sparse_pos", action="store_true",
                        help="Use sparse positive evaluation: only keep 1 positive doc per query for nDCG@5")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sparse positive selection")
    args = parser.parse_args()

    groups_raw = []
    with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
        for line in f:
            groups_raw.append(json.loads(line.strip()))

    cache = torch.load(args.embeddings_path, map_location=args.device, weights_only=False)
    model_name = cache.get("model_name", "unknown")
    print(f"Using embeddings from: {args.embeddings_path}")
    print(f"Model: {model_name}")
    print(f"Retrieval top-k: {args.retrieval_top_k}")

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

    # Analytical Beta Estimation
    analytical_beta_v2_value = None
    if args.analytical_beta_v2:
        S_base_neg_all = q_base @ neg_embs.T
        S_req_neg_all = q_plus @ neg_embs.T

        beta_needed_list = []
        pos_in_top1000 = 0
        pos_total_count = 0
        sreq_gap_neg_count = 0

        pos_offset = 0
        for q_idx in range(n_queries):
            item = groups_raw[q_idx]
            pos_count = len(item.get("pos", []))
            pos_total_count += pos_count

            own_neg_start = q_idx * 15
            own_neg_end = q_idx * 15 + 15
            all_neg_indices = np.array([
                i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_end
            ])

            sbase_pos = np.array([torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)])
            sreq_pos = np.array([torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)])
            sbase_neg = S_base_neg_all[q_idx, all_neg_indices].numpy()
            sreq_neg = S_req_neg_all[q_idx, all_neg_indices].numpy()

            all_sbase = np.concatenate([sbase_pos, sbase_neg])
            all_sreq = np.concatenate([sreq_pos, sreq_neg])
            all_is_pos = np.concatenate([np.ones(pos_count, dtype=bool), np.zeros(len(all_neg_indices), dtype=bool)])
            sorted_idx = np.argsort(-all_sbase)

            for pi in range(len(all_sbase)):
                if not all_is_pos[pi]:
                    continue
                sbase_p = all_sbase[pi]
                sreq_p = all_sreq[pi]
                sbase_rank = (all_sbase > sbase_p).sum() + 1
                if sbase_rank > 1000:
                    continue
                pos_in_top1000 += 1

                top5_sbase = []
                top5_sreq = []
                for si in sorted_idx:
                    if all_is_pos[si]:
                        continue
                    if len(top5_sbase) >= 5:
                        break
                    top5_sbase.append(all_sbase[si])
                    top5_sreq.append(all_sreq[si])
                if not top5_sbase:
                    continue

                mean_sbase_top5 = np.mean(top5_sbase)
                mean_sreq_top5 = np.mean(top5_sreq)
                sreq_gap = sreq_p - mean_sreq_top5
                sbase_gap = mean_sbase_top5 - sbase_p

                if sreq_gap > 0 and sbase_gap > 0:
                    beta_needed_list.append(sbase_gap / sreq_gap)
                elif sreq_gap <= 0 and sbase_gap > 0:
                    sreq_gap_neg_count += 1

            pos_offset += pos_count

        beta_needed_arr = np.array(beta_needed_list)
        pctile = args.analytical_beta_percentile
        if len(beta_needed_arr) > 0:
            analytical_beta_v2_value = float(np.percentile(beta_needed_arr, pctile))
        else:
            analytical_beta_v2_value = 1.0

        print(f"\n=== Analytical Beta V2 Estimation ===")
        print(f"  Total pos docs: {pos_total_count}")
        print(f"  Pos in top-1000: {pos_in_top1000} ({pos_in_top1000/pos_total_count*100:.1f}%)")
        print(f"  S_req gap NEGATIVE (beta hurts): {sreq_gap_neg_count}")
        print(f"  Need beta>0: {len(beta_needed_arr)}")
        if len(beta_needed_arr) > 0:
            print(f"  Beta needed: mean={beta_needed_arr.mean():.4f}, median={np.median(beta_needed_arr):.4f}")
            print(f"  p25={np.percentile(beta_needed_arr,25):.4f}, p75={np.percentile(beta_needed_arr,75):.4f}")
            print(f"  p90={np.percentile(beta_needed_arr,90):.4f}")
            for b in [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
                pct = (beta_needed_arr <= b).mean() * 100
                print(f"  beta<={b:.1f}: {pct:.1f}%")
        print(f"  Selected beta (p{pctile:.0f}): {analytical_beta_v2_value:.4f}")

        if args.beta_from_analytical:
            beta_list = [analytical_beta_v2_value]
            print(f"  -> Fixed beta at {analytical_beta_v2_value:.4f}, searching only alpha/delta")
        else:
            beta_list = [float(x) for x in args.betas.split(",")] if args.betas else [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
            print(f"  -> Using full grid search for beta (analytical estimate = {analytical_beta_v2_value:.4f} for reference)")
    elif args.analytical_beta:
        pos_offset = 0
        sbase_pos_list = []
        sreq_pos_list = []
        for q_idx in range(n_queries):
            item = groups_raw[q_idx]
            pos_count = len(item.get("pos", []))
            for pi in range(pos_count):
                sb = torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item()
                sr = torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item()
                sbase_pos_list.append(sb)
                sreq_pos_list.append(sr)
            pos_offset += pos_count

        sbase_pos = np.array(sbase_pos_list)
        sreq_pos = np.array(sreq_pos_list)
        mean_sbase_pos = float(sbase_pos.mean())
        mean_sreq_pos = float(sreq_pos.mean())

        beta_est = args.analytical_k * mean_sbase_pos / (1.0 * mean_sreq_pos)
        print(f"\n=== Analytical Beta Estimation (V1, no cosine gating) ===")
        print(f"  mean(S_base for pos docs) = {mean_sbase_pos:.4f}")
        print(f"  mean(S_req for pos docs) = {mean_sreq_pos:.4f}")
        print(f"  S_req / S_base ratio = {mean_sreq_pos / mean_sbase_pos:.4f}")
        print(f"  k = {args.analytical_k}")
        print(f"  beta_est = {beta_est:.2f}")

        for k in [0.3, 0.5, 0.7, 1.0]:
            b = k * mean_sbase_pos / mean_sreq_pos
            print(f"  k={k}: beta_est = {b:.2f}")

        beta_list = [beta_est]
        print(f"\nUsing analytical beta = {beta_est:.2f}")
    else:
        beta_list = [float(x) for x in args.betas.split(",")] if args.betas else [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]

    # Retrieval-Simulated Distractor Sampling
    print(f"\nComputing S_base for all neg docs against all queries...")
    S_base_neg_matrix = q_base @ neg_embs.T
    S_neg_neg_matrix = q_minus @ neg_embs.T

    n_dist = args.n_distractors
    top_k = args.retrieval_top_k

    distractor_indices = []
    for q_idx in range(n_queries):
        own_neg_start = q_idx * 15
        own_neg_end = q_idx * 15 + 15

        all_neg_indices = np.array([
            i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_end
        ])
        s_base_vals = S_base_neg_matrix[q_idx, all_neg_indices].numpy()

        sorted_idx = np.argsort(-s_base_vals)

        n_retrieved = min(top_k, len(all_neg_indices))
        retrieved_indices = all_neg_indices[sorted_idx[:n_retrieved]]

        if n_retrieved <= n_dist:
            sampled = retrieved_indices
        else:
            sampled = np.random.choice(retrieved_indices, size=n_dist, replace=False)

        distractor_indices.append(sampled)

    at_risk_counts = []
    for q_idx in range(n_queries):
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        s_neg_dist = S_neg_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        at_risk = (s_neg_dist > s_base_dist).mean()
        at_risk_counts.append(at_risk)
    avg_at_risk = np.mean(at_risk_counts)
    print(f"Distractor at-risk ratio (S_neg > S_base): {avg_at_risk:.4f}")

    all_dist_sbase = []
    for q_idx in range(n_queries):
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        all_dist_sbase.extend(s_base_dist.tolist())
    all_dist_sbase = np.array(all_dist_sbase)
    print(f"Distractor S_base: mean={all_dist_sbase.mean():.4f}, std={all_dist_sbase.std():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p}={np.percentile(all_dist_sbase, p):.4f}")

    own_at_risk = []
    for q_idx in range(n_queries):
        s_base_own = S_base_neg_matrix[q_idx, q_idx*15:(q_idx+1)*15].numpy()
        s_neg_own = S_neg_neg_matrix[q_idx, q_idx*15:(q_idx+1)*15].numpy()
        own_at_risk.append((s_neg_own > s_base_own).mean())
    print(f"Own neg at-risk ratio: {np.mean(own_at_risk):.4f}")

    overall_at_risk = []
    for q_idx in range(n_queries):
        s_base_own = S_base_neg_matrix[q_idx, q_idx*15:(q_idx+1)*15].numpy()
        s_neg_own = S_neg_neg_matrix[q_idx, q_idx*15:(q_idx+1)*15].numpy()
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        s_neg_dist = S_neg_neg_matrix[q_idx, distractor_indices[q_idx]].numpy()
        all_sb = np.concatenate([s_base_own, s_base_dist])
        all_sn = np.concatenate([s_neg_own, s_neg_dist])
        overall_at_risk.append((all_sn > all_sb).mean())
    print(f"Overall at-risk ratio (own neg + distractors): {np.mean(overall_at_risk):.4f}")

    print(f"\nPrecomputing scores with {n_dist} distractors per query...")
    pos_offset = 0
    all_s_base = []
    all_s_req = []
    all_s_neg = []
    all_pos_count = []
    all_has_req = []
    all_has_neg = []
    all_cos_neg = []

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

        pos_offset += pos_count

    print(f"Precomputed scores for {n_queries} queries")

    sparse_pos_indices = None
    if args.sparse_pos:
        rng = np.random.RandomState(args.seed)
        sparse_pos_indices = []
        for q_idx in range(n_queries):
            pos_count = all_pos_count[q_idx]
            if pos_count > 0:
                sparse_pos_indices.append(rng.randint(0, pos_count))
            else:
                sparse_pos_indices.append(-1)
        print(f"Sparse positive evaluation enabled (seed={args.seed})")
        pos_counts = [all_pos_count[q_idx] for q_idx in range(n_queries)]
        print(f"Positive doc counts: min={min(pos_counts)}, max={max(pos_counts)}, mean={np.mean(pos_counts):.2f}")
        multi_pos = sum(1 for c in pos_counts if c > 1)
        print(f"Queries with >1 positive doc: {multi_pos}/{n_queries}")

    alpha_list = [float(x) for x in args.alphas.split(",")] if args.alphas else [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]
    delta_list = [float(x) for x in args.deltas.split(",")] if args.deltas else [-0.15, -0.10, -0.05, 0.00, 0.05]

    all_results = []
    total = len(alpha_list) * len(beta_list) * len(delta_list)
    trial = 0

    for alpha in alpha_list:
        for beta in beta_list:
            for delta in delta_list:
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
                    tau = cos_neg + delta

                    # V2 formula: S_final = S_base + beta * S_req * safety - penalty
                    # No cosine gating (reward_gate = 1.0)
                    if not has_neg:
                        s_req_eff = s_r if has_req else np.zeros_like(s_b)
                        s_final = s_b + beta * s_req_eff
                    else:
                        smooth_penalty = np.log1p(np.exp(s_n - tau))
                        gap_w = 1.0 / (1.0 + np.exp(-(s_n - s_b) * 20.0))
                        raw_penalty = alpha * smooth_penalty * gap_w
                        penalty = np.minimum(raw_penalty, s_b * 0.5)
                        safety = 1.0 - 1.0 / (1.0 + np.exp(-(s_n - tau) * 20.0))
                        s_req_eff = s_r if has_req else np.zeros_like(s_b)
                        s_final = s_b + beta * s_req_eff * safety - penalty

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

                    if sparse_pos_indices is not None:
                        sparse_idx = sparse_pos_indices[q_idx]
                        if sparse_idx >= 0:
                            rel_sparse = np.zeros(n_total)
                            rel_sparse[sparse_idx] = 1.0
                            dcg = sum(
                                rel_sparse[sorted_idx[r]] / math.log2(r + 2)
                                for r in range(min(5, n_total))
                            )
                            idcg = 1.0
                            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
                        else:
                            ndcgs.append(0.0)
                    else:
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
                        "map": avg_map,
                        "ndcg@5": avg_ndcg,
                        "target_avg": (2 * avg_map + avg_ndcg) / 3,
                    }
                )
                if trial % 50 == 0:
                    print(
                        f"[{trial}/{total}] a={alpha} b={beta:.2f} d={delta:.2f} | MAP={avg_map:.4f} nDCG@5={avg_ndcg:.4f}"
                    )

    all_results.sort(key=lambda x: x["map"], reverse=True)
    print(f"\n=== Top 10 by MAP (V2, retrieval_top_k={top_k}, {n_dist} distractors, {model_name}) ===")
    for i, r in enumerate(all_results[:10]):
        print(
            f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f}'
        )

    all_results.sort(key=lambda x: x["ndcg@5"], reverse=True)
    print(f"\n=== Top 10 by nDCG@5 (V2, retrieval_top_k={top_k}, {n_dist} distractors, {model_name}) ===")
    for i, r in enumerate(all_results[:10]):
        print(
            f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} target_avg={r["target_avg"]:.4f}'
        )

    all_results.sort(key=lambda x: x["target_avg"], reverse=True)
    print(f"\n=== Top 10 by target_avg = (2*MAP + nDCG@5)/3 (V2, retrieval_top_k={top_k}, {n_dist} distractors, {model_name}) ===")
    for i, r in enumerate(all_results[:10]):
        print(
            f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} target_avg={r["target_avg"]:.4f}'
        )

    suffix = args.output_suffix
    model_name_safe = model_name.replace("/", "_")
    sparse_tag = "_sparse" if args.sparse_pos else ""
    with open(
        f"dataset/FollowIR_train/train/train_param_search_v2_retrieval_topk{top_k}_{model_name_safe}{sparse_tag}{suffix}.json",
        "w",
    ) as f:
        json.dump(
            {
                "engine": "V2",
                "n_distractors": n_dist,
                "retrieval_top_k": top_k,
                "model_name": model_name,
                "sparse_pos": args.sparse_pos,
                "seed": args.seed if args.sparse_pos else None,
                "analytical_beta": args.analytical_beta,
                "analytical_k": args.analytical_k if args.analytical_beta else None,
                "analytical_beta_v2": args.analytical_beta_v2,
                "analytical_beta_v2_value": analytical_beta_v2_value,
                "analytical_beta_percentile": args.analytical_beta_percentile if args.analytical_beta_v2 else None,
                "beta_from_analytical": args.beta_from_analytical,
                "actual_at_risk_ratio": float(avg_at_risk),
                "overall_at_risk_ratio": float(np.mean(overall_at_risk)),
                "best_by_map": sorted(all_results, key=lambda x: x["map"], reverse=True)[0],
                "best_by_ndcg": sorted(all_results, key=lambda x: x["ndcg@5"], reverse=True)[0],
                "best_by_target_avg": sorted(all_results, key=lambda x: x["target_avg"], reverse=True)[0],
                "all_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
