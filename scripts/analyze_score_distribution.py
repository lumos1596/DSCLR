import torch
import torch.nn.functional as F
import numpy as np
import json

repllama = torch.load('dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt', map_location='cpu', weights_only=False)
mistral = torch.load('dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt', map_location='cpu', weights_only=False)

groups_raw = []
with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
    for line in f:
        groups_raw.append(json.loads(line.strip()))

np.random.seed(42)
n_dist = 200

for model_name, emb in [('Repllama', repllama), ('Mistral', mistral)]:
    q_base = F.normalize(emb["q_base_embeddings"].float(), p=2, dim=1)
    q_plus = F.normalize(emb["q_plus_embeddings"].float(), p=2, dim=1)
    q_minus = F.normalize(emb["q_minus_embeddings"].float(), p=2, dim=1)
    pos_embs = F.normalize(emb["pos_embeddings"].float(), p=2, dim=1)
    neg_embs = F.normalize(emb["neg_embeddings"].float(), p=2, dim=1)

    n_queries = len(groups_raw)

    all_s_base_pos = []
    all_s_base_neg = []
    all_s_base_dist = []
    all_s_neg_pos = []
    all_s_neg_neg = []
    all_s_neg_dist = []
    all_s_req_pos = []
    all_at_risk_ratios = []

    distractor_indices = []
    for q_idx in range(n_queries):
        candidates = [i for i in range(neg_embs.shape[0]) if i < q_idx * 15 or i >= q_idx * 15 + 15]
        sampled = np.random.choice(candidates, size=n_dist, replace=False)
        distractor_indices.append(sampled)

    pos_offset = 0
    for q_idx in range(n_queries):
        item = groups_raw[q_idx]
        pos_count = len(item.get("pos", []))

        for pi in range(pos_count):
            sb = torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item()
            sn = torch.dot(q_minus[q_idx], pos_embs[pos_offset + pi]).item()
            sr = torch.dot(q_plus[q_idx], pos_embs[pos_offset + pi]).item()
            all_s_base_pos.append(sb)
            all_s_neg_pos.append(sn)
            all_s_req_pos.append(sr)

        for ni in range(15):
            sb = torch.dot(q_base[q_idx], neg_embs[q_idx * 15 + ni]).item()
            sn = torch.dot(q_minus[q_idx], neg_embs[q_idx * 15 + ni]).item()
            all_s_base_neg.append(sb)
            all_s_neg_neg.append(sn)

        for di in distractor_indices[q_idx]:
            sb = torch.dot(q_base[q_idx], neg_embs[di]).item()
            sn = torch.dot(q_minus[q_idx], neg_embs[di]).item()
            all_s_base_dist.append(sb)
            all_s_neg_dist.append(sn)

        all_sb = [torch.dot(q_base[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)]
        all_sb += [torch.dot(q_base[q_idx], neg_embs[q_idx * 15 + ni]).item() for ni in range(15)]
        all_sb += [torch.dot(q_base[q_idx], neg_embs[di]).item() for di in distractor_indices[q_idx]]

        all_sn = [torch.dot(q_minus[q_idx], pos_embs[pos_offset + pi]).item() for pi in range(pos_count)]
        all_sn += [torch.dot(q_minus[q_idx], neg_embs[q_idx * 15 + ni]).item() for ni in range(15)]
        all_sn += [torch.dot(q_minus[q_idx], neg_embs[di]).item() for di in distractor_indices[q_idx]]

        at_risk = sum(1 for sb, sn in zip(all_sb, all_sn) if sn > sb) / len(all_sb)
        all_at_risk_ratios.append(at_risk)

        pos_offset += pos_count

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Training Set Score Distribution (200 distractors)")
    print(f"{'=' * 60}")

    for label, sb, sn in [
        ('Positive docs', all_s_base_pos, all_s_neg_pos),
        ('Own Neg docs', all_s_base_neg, all_s_neg_neg),
        ('Distractor docs', all_s_base_dist, all_s_neg_dist),
    ]:
        sb_arr = np.array(sb)
        sn_arr = np.array(sn)
        gap = sn_arr - sb_arr
        print(f"\n  {label}:")
        print(f"    S_base: mean={sb_arr.mean():.4f}, std={sb_arr.std():.4f}, median={np.median(sb_arr):.4f}")
        print(f"    S_neg:  mean={sn_arr.mean():.4f}, std={sn_arr.std():.4f}, median={np.median(sn_arr):.4f}")
        print(f"    S_neg-S_base gap: mean={gap.mean():.4f}, std={gap.std():.4f}")
        print(f"    At-risk (S_neg>S_base): {(gap > 0).mean() * 100:.1f}%")

    print(f"\n  Per-query at-risk ratio: mean={np.mean(all_at_risk_ratios):.4f}, std={np.std(all_at_risk_ratios):.4f}")

    sr_arr = np.array(all_s_req_pos)
    sb_pos_arr = np.array(all_s_base_pos)
    print(f"\n  S_req - S_base (positive docs): mean={(sr_arr - sb_pos_arr).mean():.4f}")
    print(f"  S_req / S_base ratio: mean={sr_arr.mean() / sb_pos_arr.mean():.4f}")

    # Key metric: threat ratio = S_neg/S_base for negative docs
    sn_neg_arr = np.array(all_s_base_neg + all_s_base_dist)
    sb_neg_arr = np.array(all_s_neg_neg + all_s_neg_dist)
    # For negative docs: how close is S_neg to S_base?
    # This determines how much penalty is needed
    neg_sb = np.array(all_s_base_neg + all_s_base_dist)
    neg_sn = np.array(all_s_neg_neg + all_s_neg_dist)
    print(f"\n  Threat ratio (S_neg/S_base for neg docs): mean={np.mean(neg_sn / (neg_sb + 1e-8)):.4f}")
    print(f"  S_neg > S_base for neg docs: {(neg_sn > neg_sb).mean() * 100:.1f}%")
