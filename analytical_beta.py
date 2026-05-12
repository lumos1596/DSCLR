import json, torch, torch.nn.functional as F, numpy as np, sys

groups_raw = []
with open('dataset/FollowIR_train/train/dsclr_total_dataset.jsonl') as f:
    for line in f:
        groups_raw.append(json.loads(line.strip()))

results = {}

for model_name, emb_path in [
    ('Repllama', 'dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt'),
    ('Mistral', 'dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt'),
]:
    cache = torch.load(emb_path, map_location='cpu', weights_only=False)
    q_base = F.normalize(cache['q_base_embeddings'].float(), p=2, dim=1)
    q_plus = F.normalize(cache['q_plus_embeddings'].float(), p=2, dim=1)
    q_minus = F.normalize(cache['q_minus_embeddings'].float(), p=2, dim=1)
    pos_embs = F.normalize(cache['pos_embeddings'].float(), p=2, dim=1)
    neg_embs = F.normalize(cache['neg_embeddings'].float(), p=2, dim=1)
    n_queries = len(groups_raw)

    S_base_neg_all = q_base @ neg_embs.T
    S_req_neg_all = q_plus @ neg_embs.T
    S_neg_neg_all = q_minus @ neg_embs.T

    beta_needed_top1000 = []
    beta_needed_top100 = []
    pos_in_top1000_count = 0
    pos_in_top100_count = 0
    pos_total = 0
    pos_ranks_in_top1000 = []
    sreq_gap_positive = []
    sreq_gap_negative = []

    pos_offset = 0
    for q_idx in range(n_queries):
        item = groups_raw[q_idx]
        pos_count = len(item.get('pos', []))
        pos_total += pos_count

        own_neg_start = q_idx * 15
        own_neg_end = q_idx * 15 + 15
        all_neg_indices = np.array([i for i in range(neg_embs.shape[0]) if i < own_neg_start or i >= own_neg_end])

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

            pos_in_top1000_count += 1
            pos_ranks_in_top1000.append(sbase_rank)
            if sbase_rank <= 100:
                pos_in_top100_count += 1

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
                beta_needed_top1000.append(sbase_gap / sreq_gap)
                sreq_gap_positive.append(sreq_gap)
                if sbase_rank <= 100:
                    beta_needed_top100.append(sbase_gap / sreq_gap)
            elif sreq_gap <= 0 and sbase_gap > 0:
                sreq_gap_negative.append(sreq_gap)
                beta_needed_top1000.append(float('inf'))
                if sbase_rank <= 100:
                    beta_needed_top100.append(float('inf'))
            elif sbase_gap <= 0:
                beta_needed_top1000.append(0.0)
                if sbase_rank <= 100:
                    beta_needed_top100.append(0.0)

        pos_offset += pos_count

    beta_arr = np.array(beta_needed_top1000)
    finite_betas = beta_arr[beta_arr < float('inf')]
    inf_count = (beta_arr == float('inf')).sum()
    zero_count = (beta_arr == 0.0).sum()
    positive_betas = finite_betas[finite_betas > 0]

    r = {
        'model': model_name,
        'total_pos': pos_total,
        'pos_in_top1000': pos_in_top1000_count,
        'pos_in_top1000_pct': pos_in_top1000_count / pos_total * 100,
        'pos_in_top100': pos_in_top100_count,
        'sbase_rank_mean': float(np.mean(pos_ranks_in_top1000)) if pos_ranks_in_top1000 else 0,
        'sbase_rank_median': float(np.median(pos_ranks_in_top1000)) if pos_ranks_in_top1000 else 0,
        'sbase_rank_in_top5': int(sum(1 for r in pos_ranks_in_top1000 if r <= 5)),
        'sbase_rank_in_top10': int(sum(1 for r in pos_ranks_in_top1000 if r <= 10)),
        'already_top5': int(zero_count),
        'sreq_gap_negative_count': int(inf_count),
        'sreq_gap_negative_pct': inf_count / len(beta_arr) * 100 if len(beta_arr) > 0 else 0,
        'need_beta_positive': len(positive_betas),
        'beta_mean': float(positive_betas.mean()) if len(positive_betas) > 0 else 0,
        'beta_median': float(np.median(positive_betas)) if len(positive_betas) > 0 else 0,
        'beta_p25': float(np.percentile(positive_betas, 25)) if len(positive_betas) > 0 else 0,
        'beta_p75': float(np.percentile(positive_betas, 75)) if len(positive_betas) > 0 else 0,
        'beta_p90': float(np.percentile(positive_betas, 90)) if len(positive_betas) > 0 else 0,
        'beta_cdf': {str(b): float((positive_betas <= b).mean() * 100) for b in [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]} if len(positive_betas) > 0 else {},
        'mean_sreq_gap_positive': float(np.mean(sreq_gap_positive)) if sreq_gap_positive else 0,
        'mean_sreq_gap_negative': float(np.mean(sreq_gap_negative)) if sreq_gap_negative else 0,
    }

    if beta_needed_top100:
        beta_arr100 = np.array(beta_needed_top100)
        finite100 = beta_arr100[beta_arr100 < float('inf')]
        positive100 = finite100[finite100 > 0]
        r['beta_top100_median'] = float(np.median(positive100)) if len(positive100) > 0 else 0
        r['beta_top100_mean'] = float(positive100.mean()) if len(positive100) > 0 else 0

    results[model_name] = r

with open('dataset/FollowIR_train/train/analytical_beta_results.json', 'w') as f:
    json.dump(results, f, indent=2)

for mn, r in results.items():
    print(f"\n=== {mn} ===")
    print(f"  Total pos docs: {r['total_pos']}")
    print(f"  Pos in top-1000: {r['pos_in_top1000']} ({r['pos_in_top1000_pct']:.1f}%)")
    print(f"  Pos in top-100: {r['pos_in_top100']}")
    print(f"  S_base rank: mean={r['sbase_rank_mean']:.1f}, median={r['sbase_rank_median']:.1f}")
    print(f"    In top-5: {r['sbase_rank_in_top5']}, top-10: {r['sbase_rank_in_top10']}")
    print(f"  Already top-5 (beta=0): {r['already_top5']}")
    print(f"  S_req gap NEGATIVE (beta=inf, beta hurts): {r['sreq_gap_negative_count']} ({r['sreq_gap_negative_pct']:.1f}%)")
    print(f"  Need beta>0: {r['need_beta_positive']}")
    if r['need_beta_positive'] > 0:
        print(f"  Beta: mean={r['beta_mean']:.4f}, median={r['beta_median']:.4f}")
        print(f"    p25={r['beta_p25']:.4f}, p75={r['beta_p75']:.4f}, p90={r['beta_p90']:.4f}")
        for b, pct in sorted(r['beta_cdf'].items()):
            print(f"    beta<={b}: {pct:.1f}%")
    print(f"  Mean S_req gap (positive): {r['mean_sreq_gap_positive']:.4f}")
    print(f"  Mean S_req gap (negative): {r['mean_sreq_gap_negative']:.4f}")
    if 'beta_top100_median' in r:
        print(f"  Beta for pos in top-100: median={r['beta_top100_median']:.4f}, mean={r['beta_top100_mean']:.4f}")
