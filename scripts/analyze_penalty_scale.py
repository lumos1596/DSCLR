import torch
import torch.nn.functional as F
import numpy as np
import json

for model_name, emb_path in [
    ('Repllama', 'dataset/FollowIR_train/embeddings/repllama-reproduced/dsclr_train_embeddings_repllama-reproduced.pt'),
    ('Mistral', 'dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt'),
]:
    cache = torch.load(emb_path, map_location='cpu', weights_only=False)
    q_base = F.normalize(cache["q_base_embeddings"].float(), p=2, dim=1)
    q_plus = F.normalize(cache["q_plus_embeddings"].float(), p=2, dim=1)
    q_minus = F.normalize(cache["q_minus_embeddings"].float(), p=2, dim=1)
    pos_embs = F.normalize(cache["pos_embeddings"].float(), p=2, dim=1)
    neg_embs = F.normalize(cache["neg_embeddings"].float(), p=2, dim=1)

    cos_qbase_qneg = F.cosine_similarity(q_base, q_minus, dim=1)

    groups_raw = []
    with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
        for line in f:
            groups_raw.append(json.loads(line.strip()))

    n_queries = len(groups_raw)

    # Compute S_neg - tau for own neg docs (delta=0)
    all_overflow = []
    all_gap = []
    all_sbase = []
    all_sneg = []

    for q_idx in range(n_queries):
        tau = cos_qbase_qneg[q_idx].item()
        for ni in range(15):
            sb = torch.dot(q_base[q_idx], neg_embs[q_idx * 15 + ni]).item()
            sn = torch.dot(q_minus[q_idx], neg_embs[q_idx * 15 + ni]).item()
            overflow = sn - tau
            gap = sn - sb
            all_overflow.append(overflow)
            all_gap.append(gap)
            all_sbase.append(sb)
            all_sneg.append(sn)

    overflow_arr = np.array(all_overflow)
    gap_arr = np.array(all_gap)
    sbase_arr = np.array(all_sbase)
    sneg_arr = np.array(all_sneg)

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Own Neg Docs Score Statistics")
    print(f"{'=' * 60}")
    print(f"  S_base: mean={sbase_arr.mean():.4f}, std={sbase_arr.std():.4f}")
    print(f"  S_neg:  mean={sneg_arr.mean():.4f}, std={sneg_arr.std():.4f}")
    print(f"  S_neg - tau (delta=0): mean={overflow_arr.mean():.4f}, std={overflow_arr.std():.4f}")
    print(f"  S_neg - S_base: mean={gap_arr.mean():.4f}, std={gap_arr.std():.4f}")
    print(f"  At-risk (S_neg > S_base): {(gap_arr > 0).mean() * 100:.1f}%")
    print(f"  Overflow > 0 (S_neg > tau): {(overflow_arr > 0).mean() * 100:.1f}%")
    
    # Softplus(S_neg - tau) statistics
    sp_vals = np.log1p(np.exp(overflow_arr))
    print(f"  Softplus(S_neg - tau): mean={sp_vals.mean():.4f}, std={sp_vals.std():.4f}")
    
    # gap_w statistics
    gap_w_vals = 1.0 / (1.0 + np.exp(-gap_arr * 10.0))
    print(f"  gap_w: mean={gap_w_vals.mean():.4f}, std={gap_w_vals.std():.4f}")
    
    # Effective penalty per unit alpha
    eff_penalty = sp_vals * gap_w_vals
    print(f"  Effective penalty (per unit alpha): mean={eff_penalty.mean():.4f}, std={eff_penalty.std():.4f}")
    
    # What alpha would be needed to achieve a given penalty level?
    # penalty = alpha * eff_penalty, capped at S_base * 0.5
    # For a target penalty of, say, 0.05 (5% of S_base):
    target_penalty = 0.05
    alpha_needed = target_penalty / (eff_penalty.mean() + 1e-8)
    print(f"  Alpha needed for 5% penalty: {alpha_needed:.2f}")
    
    target_penalty = 0.10
    alpha_needed = target_penalty / (eff_penalty.mean() + 1e-8)
    print(f"  Alpha needed for 10% penalty: {alpha_needed:.2f}")
    
    # Key insight: the scale of effective penalty determines the scale of alpha
    # If we normalize by this, different encoders should have similar optimal alpha
    print(f"\n  Normalization factor (1/mean_eff_penalty): {1.0/eff_penalty.mean():.2f}")
