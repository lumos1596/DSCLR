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
    pos_embs = F.normalize(cache["pos_embeddings"].float(), p=2, dim=1)

    cos_qbase_qreq = F.cosine_similarity(q_base, q_plus, dim=1)
    reward_gate = (1.0 - cos_qbase_qreq).mean().item()

    groups_raw = []
    with open("dataset/FollowIR_train/train/dsclr_total_dataset.jsonl") as f:
        for line in f:
            groups_raw.append(json.loads(line.strip()))

    n_queries = len(groups_raw)
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

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Analytical Beta Estimation")
    print(f"{'=' * 60}")
    print(f"  mean(Cos(Q_base, Q+)) = {cos_qbase_qreq.mean():.4f}")
    print(f"  mean(reward_gate) = {reward_gate:.4f}")
    print(f"  mean(S_base for pos docs) = {sbase_pos.mean():.4f}")
    print(f"  mean(S_req for pos docs) = {sreq_pos.mean():.4f}")
    print(f"  S_req / S_base ratio = {sreq_pos.mean() / sbase_pos.mean():.4f}")

    # Beta estimation: we want beta * reward_gate * S_req to be comparable to S_base
    # For different k values:
    for k in [0.3, 0.5, 0.7, 1.0]:
        beta_est = k * sbase_pos.mean() / (reward_gate * sreq_pos.mean())
        print(f"  k={k}: beta_est = {beta_est:.2f}")

    # Also compute for gamma != 1.0
    for gamma in [0.5, 0.7, 1.0]:
        rg = ((1.0 - cos_qbase_qreq) ** gamma).mean().item()
        for k in [0.5]:
            beta_est = k * sbase_pos.mean() / (rg * sreq_pos.mean())
            print(f"  gamma={gamma}, k={k}: beta_est = {beta_est:.2f} (reward_gate={rg:.4f})")
