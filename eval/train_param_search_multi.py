import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import argparse
from typing import List, Dict, Any, Tuple


def apply_v2_formula(
    s_base: np.ndarray,
    s_req: np.ndarray,
    s_neg: np.ndarray,
    cos_val: float,
    has_req: bool,
    has_neg: bool,
    alpha: float,
    beta: float,
    delta: float,
    t_gap: float = 20.0,
    t_safety: float = 20.0,
    max_penalty_ratio: float = 0.5,
) -> np.ndarray:
    tau = cos_val + delta
    if not has_neg:
        s_req_eff = s_req if has_req else np.zeros_like(s_base)
        return s_base + beta * s_req_eff
    smooth_penalty = np.log1p(np.exp(s_neg - tau))
    gap_w = 1.0 / (1.0 + np.exp(-(s_neg - s_base) * t_gap))
    raw_penalty = alpha * smooth_penalty * gap_w
    penalty = np.minimum(raw_penalty, s_base * max_penalty_ratio)
    safety = 1.0 - 1.0 / (1.0 + np.exp(-(s_neg - tau) * t_safety))
    s_req_eff = s_req if has_req else np.zeros_like(s_base)
    return s_base + beta * s_req_eff * safety - penalty


def compute_map(scores: np.ndarray, pos_count: int) -> float:
    if pos_count == 0:
        return 0.0
    sorted_idx = np.argsort(-scores)
    ap_sum = 0.0
    hits = 0
    for rank, idx in enumerate(sorted_idx):
        if idx < pos_count:
            hits += 1
            ap_sum += hits / (rank + 1)
    return ap_sum / pos_count


def compute_ndcg5(scores: np.ndarray, pos_count: int) -> float:
    if pos_count == 0:
        return 0.0
    sorted_idx = np.argsort(-scores)[:5]
    rel = np.zeros(len(scores))
    rel[:pos_count] = 1.0
    dcg = sum(rel[sorted_idx[r]] / math.log2(r + 2) for r in range(min(5, len(sorted_idx))))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(5, pos_count)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    groups_raw = []
    with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
        for line in f:
            groups_raw.append(json.loads(line.strip()))

    cache = torch.load(
        "dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt",
        map_location=args.device,
        weights_only=False,
    )
    q_base = F.normalize(cache["q_base_embeddings"].float(), p=2, dim=1)
    q_plus = F.normalize(cache["q_plus_embeddings"].float(), p=2, dim=1)
    q_minus = F.normalize(cache["q_minus_embeddings"].float(), p=2, dim=1)
    pos_embs = F.normalize(cache["pos_embeddings"].float(), p=2, dim=1)
    neg_embs = F.normalize(cache["neg_embeddings"].float(), p=2, dim=1)

    n_queries = len(groups_raw)
    n_total_neg = neg_embs.shape[0]
    n_total_pos = pos_embs.shape[0]

    has_req_mask = [
        json.loads(item["output"]).get("Q_plus", "") not in ("[NONE]", "NONE", "", None)
        for item in groups_raw
    ]
    has_neg_mask = [
        json.loads(item["output"]).get("Q_minus", "") not in ("[NONE]", "NONE", "", None)
        for item in groups_raw
    ]
    cos_qbase_qneg = F.cosine_similarity(q_base, q_minus, dim=1)

    alpha_list = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]
    beta_list = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    delta_list = [-0.15, -0.10, -0.05, 0.00, 0.05]

    strategies = [
        {"name": "neg_500", "n_distractors": 500, "source": "neg"},
        {"name": "neg_1000", "n_distractors": 1000, "source": "neg"},
        {"name": "pos_200", "n_distractors": 200, "source": "pos"},
        {"name": "pos_500", "n_distractors": 500, "source": "pos"},
        {"name": "mixed_500", "n_distractors": 500, "source": "mixed"},
    ]

    np.random.seed(42)

    all_strategy_results = {}

    for strat in strategies:
        strat_name = strat["name"]
        n_dist = strat["n_distractors"]
        source = strat["source"]

        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name} ({n_dist} distractors from {source})")
        print(f"{'='*60}")

        distractor_indices = []
        for q_idx in range(n_queries):
            if source == "neg":
                own_neg_start = q_idx * 15
                candidates = [
                    i for i in range(n_total_neg) if i < own_neg_start or i >= own_neg_start + 15
                ]
                sampled = np.random.choice(candidates, size=n_dist, replace=False)
                distractor_indices.append(("neg", sampled))
            elif source == "pos":
                own_pos_start = sum(len(groups_raw[j].get("pos", [])) for j in range(q_idx))
                own_pos_end = own_pos_start + len(groups_raw[q_idx].get("pos", []))
                candidates = [
                    i for i in range(n_total_pos) if i < own_pos_start or i >= own_pos_end
                ]
                if len(candidates) < n_dist:
                    candidates = list(range(n_total_pos))
                sampled = np.random.choice(candidates, size=n_dist, replace=False)
                distractor_indices.append(("pos", sampled))
            elif source == "mixed":
                n_half = n_dist // 2
                own_neg_start = q_idx * 15
                neg_cands = [
                    i for i in range(n_total_neg) if i < own_neg_start or i >= own_neg_start + 15
                ]
                neg_sampled = np.random.choice(neg_cands, size=n_half, replace=False)
                own_pos_start = sum(len(groups_raw[j].get("pos", [])) for j in range(q_idx))
                own_pos_end = own_pos_start + len(groups_raw[q_idx].get("pos", []))
                pos_cands = [
                    i for i in range(n_total_pos) if i < own_pos_start or i >= own_pos_end
                ]
                if len(pos_cands) < n_dist - n_half:
                    pos_cands = list(range(n_total_pos))
                pos_sampled = np.random.choice(pos_cands, size=n_dist - n_half, replace=False)
                distractor_indices.append(("mixed", neg_sampled, pos_sampled))

        print("Precomputing scores...")
        pos_offset = 0
        all_s_base = []
        all_s_req = []
        all_s_neg = []
        all_pos_count = []
        all_has_req = []
        all_has_neg = []
        all_cos = []

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

            di = distractor_indices[q_idx]
            if di[0] == "neg":
                dist_embs = neg_embs[di[1]]
            elif di[0] == "pos":
                dist_embs = pos_embs[di[1]]
            elif di[0] == "mixed":
                dist_embs = torch.cat([neg_embs[di[1]], pos_embs[di[2]]], dim=0)

            s_base += (q_base[q_idx] @ dist_embs.T).tolist()
            s_req += (q_plus[q_idx] @ dist_embs.T).tolist()
            s_neg += (q_minus[q_idx] @ dist_embs.T).tolist()

            all_s_base.append(np.array(s_base))
            all_s_req.append(np.array(s_req))
            all_s_neg.append(np.array(s_neg))
            all_pos_count.append(pos_count)
            all_has_req.append(has_req_mask[q_idx])
            all_has_neg.append(has_neg_mask[q_idx])
            all_cos.append(cos_qbase_qneg[q_idx].item())

            pos_offset += pos_count

        print(f"Precomputed {n_queries} queries")

        results = []
        total = len(alpha_list) * len(beta_list) * len(delta_list)
        trial = 0

        for alpha in alpha_list:
            for beta in beta_list:
                for delta in delta_list:
                    trial += 1
                    maps = []
                    ndcgs = []
                    for q_idx in range(n_queries):
                        s_final = apply_v2_formula(
                            all_s_base[q_idx],
                            all_s_req[q_idx],
                            all_s_neg[q_idx],
                            all_cos[q_idx],
                            all_has_req[q_idx],
                            all_has_neg[q_idx],
                            alpha,
                            beta,
                            delta,
                        )
                        maps.append(compute_map(s_final, all_pos_count[q_idx]))
                        ndcgs.append(compute_ndcg5(s_final, all_pos_count[q_idx]))

                    avg_map = float(np.mean(maps))
                    avg_ndcg = float(np.mean(ndcgs))
                    results.append(
                        {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "delta": float(delta),
                            "map": avg_map,
                            "ndcg@5": avg_ndcg,
                            "target_avg": (avg_map + avg_ndcg) / 2.0,
                        }
                    )

        results.sort(key=lambda x: x["target_avg"], reverse=True)
        print(f"\n=== Top 10 by target_avg (MAP+nDCG@5)/2 ===")
        for i, r in enumerate(results[:10]):
            print(
                f'{i+1}. a={r["alpha"]:.1f} b={r["beta"]:.1f} d={r["delta"]:.2f} | MAP={r["map"]:.4f} nDCG@5={r["ndcg@5"]:.4f} avg={r["target_avg"]:.4f}'
            )

        test_opt = [
            r for r in results if r["alpha"] == 0.5 and r["beta"] == 1.1 and r["delta"] == 0.0
        ]
        if test_opt:
            r = test_opt[0]
            map_rank = sorted(results, key=lambda x: x["map"], reverse=True).index(r) + 1
            ndcg_rank = sorted(results, key=lambda x: x["ndcg@5"], reverse=True).index(r) + 1
            avg_rank = sorted(results, key=lambda x: x["target_avg"], reverse=True).index(r) + 1
            print(f'\nTest-optimal (a=0.5, b=1.1, d=0.0): MAP={r["map"]:.4f} rank={map_rank}/{len(results)}, nDCG@5={r["ndcg@5"]:.4f} rank={ndcg_rank}/{len(results)}, avg rank={avg_rank}/{len(results)}')

        all_strategy_results[strat_name] = {
            "best_by_map": sorted(results, key=lambda x: x["map"], reverse=True)[0],
            "best_by_ndcg": sorted(results, key=lambda x: x["ndcg@5"], reverse=True)[0],
            "best_by_avg": sorted(results, key=lambda x: x["target_avg"], reverse=True)[0],
            "test_optimal_rank": {
                "map": sorted(results, key=lambda x: x["map"], reverse=True).index(test_opt[0]) + 1 if test_opt else -1,
                "ndcg": sorted(results, key=lambda x: x["ndcg@5"], reverse=True).index(test_opt[0]) + 1 if test_opt else -1,
                "avg": sorted(results, key=lambda x: x["target_avg"], reverse=True).index(test_opt[0]) + 1 if test_opt else -1,
            },
        }

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL STRATEGIES")
    print(f"{'='*60}")
    for strat_name, res in all_strategy_results.items():
        best = res["best_by_avg"]
        rank = res["test_optimal_rank"]
        print(
            f'{strat_name}: best_avg a={best["alpha"]:.1f} b={best["beta"]:.1f} d={best["delta"]:.2f} avg={best["target_avg"]:.4f} | test-opt rank: MAP={rank["map"]}, nDCG={rank["ndcg"]}, avg={rank["avg"]}'
        )

    with open(
        "dataset/FollowIR_train/train/train_param_search_multi_strategy_results.json", "w"
    ) as f:
        json.dump(all_strategy_results, f, indent=2)
    print("\nResults saved.")


if __name__ == "__main__":
    main()
