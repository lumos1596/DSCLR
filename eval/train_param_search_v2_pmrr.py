import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import argparse


def rank_score(og_rank, new_rank):
    if og_rank >= new_rank:
        return (1.0 / og_rank) / (1.0 / new_rank) - 1.0
    else:
        return 1.0 - (1.0 / new_rank) / (1.0 / og_rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=200)
    parser.add_argument("--retrieval_top_k", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embeddings_path", type=str,
                        default="dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt")
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--deltas", type=str, default=None)
    parser.add_argument("--analytical_beta_v2", action="store_true")
    parser.add_argument("--analytical_beta_percentile", type=float, default=50)
    parser.add_argument("--beta_from_analytical", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_safety", type=float, default=20.0)
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
        [float(json.loads(item["output"]).get("Q_plus", "") not in ("[NONE]", "NONE", "", None)) for item in groups_raw],
        dtype=torch.float32,
    )
    has_neg_mask = torch.tensor(
        [float(json.loads(item["output"]).get("Q_minus", "") not in ("[NONE]", "NONE", "", None)) for item in groups_raw],
        dtype=torch.float32,
    )
    cos_qbase_qneg = F.cosine_similarity(q_base, q_minus, dim=1)

    analytical_beta_v2_value = None
    if args.analytical_beta_v2:
        S_base_neg_all = q_base @ neg_embs.T
        S_req_neg_all = q_plus @ neg_embs.T
        beta_needed_list = []
        pos_offset = 0
        for q_idx in range(n_queries):
            item = groups_raw[q_idx]
            pos_count = len(item.get("pos", []))
            own_neg_start = q_idx * 15
            own_neg_end = q_idx * 15 + 15
            all_neg_indices = np.array([i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_end])
            sbase_pos = np.array([torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)])
            sreq_pos = np.array([torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)])
            sbase_neg = S_base_neg_all[q_idx, all_neg_indices].cpu().numpy()
            sreq_neg = S_req_neg_all[q_idx, all_neg_indices].cpu().numpy()
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
            pos_offset += pos_count
        beta_needed_arr = np.array(beta_needed_list)
        pctile = args.analytical_beta_percentile
        analytical_beta_v2_value = float(np.percentile(beta_needed_arr, pctile)) if len(beta_needed_arr) > 0 else 1.0
        print(f"Analytical Beta V2: {analytical_beta_v2_value:.4f} (p{pctile:.0f})")
        if args.beta_from_analytical:
            beta_list = [analytical_beta_v2_value]
        else:
            beta_list = [float(x) for x in args.betas.split(",")] if args.betas else [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    else:
        beta_list = [float(x) for x in args.betas.split(",")] if args.betas else [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]

    n_dist = args.n_distractors
    top_k = args.retrieval_top_k

    print(f"\nComputing S_base for all neg docs against all queries...")
    S_base_neg_matrix = q_base @ neg_embs.T
    S_neg_neg_matrix = q_minus @ neg_embs.T

    distractor_indices = []
    rng = np.random.RandomState(args.seed)
    for q_idx in range(n_queries):
        own_neg_start = q_idx * 15
        own_neg_end = q_idx * 15 + 15
        all_neg_indices = np.array([i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_end])
        s_base_vals = S_base_neg_matrix[q_idx, all_neg_indices].cpu().numpy()
        sorted_idx = np.argsort(-s_base_vals)
        n_retrieved = min(top_k, len(all_neg_indices))
        retrieved_indices = all_neg_indices[sorted_idx[:n_retrieved]]
        if n_retrieved <= n_dist:
            sampled = retrieved_indices
        else:
            sampled = rng.choice(retrieved_indices, size=n_dist, replace=False)
        distractor_indices.append(sampled)

    at_risk_counts = []
    for q_idx in range(n_queries):
        s_base_dist = S_base_neg_matrix[q_idx, distractor_indices[q_idx]].cpu().numpy()
        s_neg_dist = S_neg_neg_matrix[q_idx, distractor_indices[q_idx]].cpu().numpy()
        at_risk_counts.append((s_neg_dist > s_base_dist).mean())
    avg_at_risk = np.mean(at_risk_counts)
    print(f"Distractor at-risk ratio: {avg_at_risk:.4f}")

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

        s_base = [torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)]
        s_req = [torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)]
        s_neg = [torch.dot(q_minus[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)]

        s_base += [torch.dot(q_base[q_idx], neg_embs[q_idx * 15 + ni]).item() for ni in range(15)]
        s_req += [torch.dot(q_plus[q_idx], neg_embs[q_idx * 15 + ni]).item() for ni in range(15)]
        s_neg += [torch.dot(q_minus[q_idx], neg_embs[q_idx * 15 + ni]).item() for ni in range(15)]

        s_base += [torch.dot(q_base[q_idx], neg_embs[di]).item() for di in distractor_indices[q_idx]]
        s_req += [torch.dot(q_plus[q_idx], neg_embs[di]).item() for di in distractor_indices[q_idx]]
        s_neg += [torch.dot(q_minus[q_idx], neg_embs[di]).item() for di in distractor_indices[q_idx]]

        all_s_base.append(np.array(s_base))
        all_s_req.append(np.array(s_req))
        all_s_neg.append(np.array(s_neg))
        all_pos_count.append(pos_count)
        all_has_req.append(has_req_mask[q_idx].item() > 0)
        all_has_neg.append(has_neg_mask[q_idx].item() > 0)
        all_cos_neg.append(cos_qbase_qneg[q_idx].item())

        pos_offset += pos_count

    print(f"Precomputed scores for {n_queries} queries")

    alpha_list = [float(x) for x in args.alphas.split(",")] if args.alphas else [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    delta_list = [float(x) for x in args.deltas.split(",")] if args.deltas else [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]

    all_results = []
    total = len(alpha_list) * len(beta_list) * len(delta_list)
    trial = 0

    for alpha in alpha_list:
        for beta in beta_list:
            for delta in delta_list:
                trial += 1
                maps = []
                ndcgs = []
                pmrr_per_query = []

                for q_idx in range(n_queries):
                    s_b = all_s_base[q_idx]
                    s_r = all_s_req[q_idx]
                    s_n = all_s_neg[q_idx]
                    pos_count = all_pos_count[q_idx]
                    has_req = all_has_req[q_idx]
                    has_neg = all_has_neg[q_idx]
                    cos_neg = all_cos_neg[q_idx]
                    tau = cos_neg + delta

                    if not has_neg:
                        s_req_eff = s_r if has_req else np.zeros_like(s_b)
                        s_final = s_b + beta * s_req_eff
                    else:
                        smooth_penalty = np.log1p(np.exp(s_n - tau))
                        raw_penalty = alpha * smooth_penalty
                        safety = 1.0 - 1.0 / (1.0 + np.exp(-(s_n - tau) * args.t_safety))
                        s_req_eff = s_r if has_req else np.zeros_like(s_b)
                        s_final = s_b + beta * s_req_eff * safety - raw_penalty

                    n_total = len(s_final)
                    rel = np.zeros(n_total)
                    rel[:pos_count] = 1.0
                    effective_pos = pos_count

                    og_sorted_idx = np.argsort(-s_b)
                    changed_sorted_idx = np.argsort(-s_final)

                    og_rank_arr = np.empty(n_total, dtype=np.int64)
                    changed_rank_arr = np.empty(n_total, dtype=np.int64)
                    for rank, idx in enumerate(og_sorted_idx):
                        og_rank_arr[idx] = rank + 1
                    for rank, idx in enumerate(changed_sorted_idx):
                        changed_rank_arr[idx] = rank + 1

                    if effective_pos == 0:
                        maps.append(0.0)
                        ndcgs.append(0.0)
                        pmrr_per_query.append(0.0)
                        continue

                    ap_sum = 0.0
                    hits = 0
                    for rank, idx in enumerate(changed_sorted_idx):
                        if rel[idx] > 0:
                            hits += 1
                            ap_sum += hits / (rank + 1)
                    maps.append(ap_sum / effective_pos)

                    dcg = sum(rel[changed_sorted_idx[r]] / math.log2(r + 2) for r in range(min(5, n_total)))
                    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(5, effective_pos)))
                    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

                    if has_neg:
                        at_risk_neg_mask = s_n[pos_count:] > tau
                        at_risk_global_indices = np.where(at_risk_neg_mask)[0] + pos_count

                        if len(at_risk_global_indices) > 0:
                            doc_scores = []
                            for gi in at_risk_global_indices:
                                og_r = int(og_rank_arr[gi])
                                ch_r = int(changed_rank_arr[gi])
                                doc_scores.append(rank_score(og_r, ch_r))
                            pmrr_per_query.append(np.mean(doc_scores))
                        else:
                            pmrr_per_query.append(0.0)
                    else:
                        pmrr_per_query.append(0.0)

                avg_map = float(np.mean(maps))
                avg_ndcg = float(np.mean(ndcgs))
                avg_pmrr = float(np.mean(pmrr_per_query))
                target_avg = (2 * avg_map + avg_ndcg) / 3

                all_results.append({
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "delta": float(delta),
                    "map": avg_map,
                    "ndcg@5": avg_ndcg,
                    "target_avg": target_avg,
                    "approx_pmrr": avg_pmrr,
                    "composite": target_avg + avg_pmrr,
                })
                if trial % 50 == 0:
                    print(f"[{trial}/{total}] a={alpha} b={beta:.2f} d={delta:.2f} | MAP={avg_map:.4f} nDCG@5={avg_ndcg:.4f} pMRR={avg_pmrr:.4f}")

    all_results.sort(key=lambda x: x["target_avg"], reverse=True)
    print(f"\n=== Top 10 by target_avg ===")
    for i, r in enumerate(all_results[:10]):
        print(f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} pMRR={r["approx_pmrr"]:.4f} comp={r["composite"]:.4f}')

    all_results.sort(key=lambda x: x["approx_pmrr"], reverse=True)
    print(f"\n=== Top 10 by approx_pmrr ===")
    for i, r in enumerate(all_results[:10]):
        print(f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} pMRR={r["approx_pmrr"]:.4f} comp={r["composite"]:.4f}')

    all_results.sort(key=lambda x: x["composite"], reverse=True)
    print(f"\n=== Top 10 by composite (target_avg + approx_pmrr) ===")
    for i, r in enumerate(all_results[:10]):
        print(f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} pMRR={r["approx_pmrr"]:.4f} comp={r["composite"]:.4f}')

    # Constrained: target_avg >= threshold, maximize approx_pmrr
    ta_threshold = 0.13
    constrained = [r for r in all_results if r["target_avg"] >= ta_threshold]
    if constrained:
        constrained.sort(key=lambda x: x["approx_pmrr"], reverse=True)
        print(f"\n=== Top 10 constrained (target_avg >= {ta_threshold}) by approx_pmrr ===")
        for i, r in enumerate(constrained[:10]):
            print(f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} pMRR={r["approx_pmrr"]:.4f} comp={r["composite"]:.4f}')

    # Constrained: approx_pmrr >= threshold, maximize target_avg
    pmrr_threshold = 0.05
    constrained2 = [r for r in all_results if r["approx_pmrr"] >= pmrr_threshold]
    if constrained2:
        constrained2.sort(key=lambda x: x["target_avg"], reverse=True)
        print(f"\n=== Top 10 constrained (approx_pmrr >= {pmrr_threshold}) by target_avg ===")
        for i, r in enumerate(constrained2[:10]):
            print(f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.2f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} pMRR={r["approx_pmrr"]:.4f} comp={r["composite"]:.4f}')

    suffix = args.output_suffix
    model_name_safe = model_name.replace("/", "_")
    output_path = f"dataset/FollowIR_train/train/train_param_search_v2_retrieval_topk{top_k}_{model_name_safe}_pmrr{suffix}.json"
    with open(output_path, "w") as f:
        json.dump({
            "engine": "V2-final-pmrr",
            "n_distractors": n_dist,
            "retrieval_top_k": top_k,
            "model_name": model_name,
            "seed": args.seed,
            "analytical_beta_v2": args.analytical_beta_v2,
            "analytical_beta_v2_value": analytical_beta_v2_value,
            "analytical_beta_percentile": args.analytical_beta_percentile if args.analytical_beta_v2 else None,
            "beta_from_analytical": args.beta_from_analytical,
            "actual_at_risk_ratio": float(avg_at_risk),
            "t_safety": args.t_safety,
            "best_by_map": sorted(all_results, key=lambda x: x["map"], reverse=True)[0],
            "best_by_ndcg": sorted(all_results, key=lambda x: x["ndcg@5"], reverse=True)[0],
            "best_by_target_avg": sorted(all_results, key=lambda x: x["target_avg"], reverse=True)[0],
            "best_by_pmrr": sorted(all_results, key=lambda x: x["approx_pmrr"], reverse=True)[0],
            "best_by_composite": sorted(all_results, key=lambda x: x["composite"], reverse=True)[0],
            "all_results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
