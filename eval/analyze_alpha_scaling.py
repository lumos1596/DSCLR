"""
深入分析 α 推导偏差：为什么 α_raw=0.76 而网格最优 α=0.3？

假设：at-risk 文档中混有"误判"文档（实际相关但 S_neg > τ），
     只应对"真正负相关"的文档施加全额惩罚。
     最优 α ≈ α_raw × (neg_docs_ratio_in_at_risk)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
import numpy as np

EMB_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/embeddings/dsclr_train_embeddings_repllama-reproduced_qwen3-4B.pt"
OUTPUT_PATH = "/home/luwa/Documents/DSCLR/results/alpha_scaling_analysis.json"


def main():
    print("Loading training embeddings...")
    data = torch.load(EMB_PATH, map_location="cuda", weights_only=False)

    q_base = F.normalize(data["q_base_embeddings"].cuda(), dim=1)
    q_plus = F.normalize(data["q_plus_embeddings"].cuda(), dim=1)
    q_minus = F.normalize(data["q_minus_embeddings"].cuda(), dim=1)
    pos = F.normalize(data["pos_embeddings"].cuda(), dim=1)
    neg = F.normalize(data["neg_embeddings"].cuda(), dim=1)

    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    doc_pool = torch.cat([pos, neg], dim=0)
    # 文档类型 mask: 0=pos(相关), 1=neg(负相关但相关)
    doc_is_neg = torch.zeros(n_pos + n_neg, dtype=torch.bool, device="cuda")
    doc_is_neg[n_pos:] = True

    S_base = torch.mm(q_base, doc_pool.T)
    S_req = torch.mm(q_plus, doc_pool.T)
    S_neg = torch.mm(q_minus, doc_pool.T)
    cos_qbase_qneg = (q_base * q_minus).sum(dim=1)

    # tau_anchor proxy
    S_base_pos = torch.mm(q_base, pos.T)
    S_neg_pos = torch.mm(q_minus, pos.T)
    _, topk_idx = S_base_pos.topk(5, dim=1)
    S_neg_topk = S_neg_pos.gather(1, topk_idx)
    tau_anchor_proxy = S_neg_topk.max(dim=1).values

    anchor_delta = 0.02
    tau_safe = torch.maximum(tau_anchor_proxy, cos_qbase_qneg) + anchor_delta
    tau_2d = tau_safe.unsqueeze(1)
    at_risk_mask = S_neg > tau_2d
    safe_mask = ~at_risk_mask

    n_at_risk = at_risk_mask.sum().item()
    n_at_risk_neg = (at_risk_mask & doc_is_neg.unsqueeze(0)).sum().item()
    n_at_risk_pos = (at_risk_mask & ~doc_is_neg.unsqueeze(0)).sum().item()

    # 在 at-risk 中，neg 文档的比例（"真正应被惩罚"的比例）
    neg_ratio_in_at_risk = n_at_risk_neg / n_at_risk if n_at_risk > 0 else 0

    # α_raw
    S_base_at_risk = S_base[at_risk_mask]
    S_neg_at_risk = S_neg[at_risk_mask]
    tau_at_risk = tau_2d.expand_as(S_neg)[at_risk_mask]
    E_S_base_at_risk = S_base_at_risk.mean().item()
    softplus_at_risk = F.softplus(S_neg_at_risk - tau_at_risk)
    E_softplus_at_risk = softplus_at_risk.mean().item()
    alpha_raw = E_S_base_at_risk / E_softplus_at_risk

    # 分别计算 neg 和 pos 在 at-risk 中的统计
    at_risk_neg_mask = at_risk_mask & doc_is_neg.unsqueeze(0)
    at_risk_pos_mask = at_risk_mask & ~doc_is_neg.unsqueeze(0)

    if at_risk_neg_mask.sum() > 0:
        E_S_base_neg = S_base[at_risk_neg_mask].mean().item()
        E_softplus_neg = F.softplus(S_neg[at_risk_neg_mask] - tau_2d.expand_as(S_neg)[at_risk_neg_mask]).mean().item()
        alpha_neg = E_S_base_neg / E_softplus_neg if E_softplus_neg > 0 else 0
    else:
        E_S_base_neg = E_softplus_neg = alpha_neg = 0

    if at_risk_pos_mask.sum() > 0:
        E_S_base_pos_ar = S_base[at_risk_pos_mask].mean().item()
        E_softplus_pos_ar = F.softplus(S_neg[at_risk_pos_mask] - tau_2d.expand_as(S_neg)[at_risk_pos_mask]).mean().item()
        alpha_pos = E_S_base_pos_ar / E_softplus_pos_ar if E_softplus_pos_ar > 0 else 0
    else:
        E_S_base_pos_ar = E_softplus_pos_ar = alpha_pos = 0

    # β
    safety = 1 - torch.sigmoid((S_neg - tau_2d) * 20.0)
    E_S_base_safe = S_base[safe_mask].mean().item()
    E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
    beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0

    print("=" * 80)
    print("AT-RISK COMPOSITION ANALYSIS")
    print("=" * 80)
    print(f"Total at-risk docs:        {n_at_risk}")
    print(f"  - neg docs (应惩罚):     {n_at_risk_neg} ({neg_ratio_in_at_risk*100:.1f}%)")
    print(f"  - pos docs (误判/相关):   {n_at_risk_pos} ({(1-neg_ratio_in_at_risk)*100:.1f}%)")
    print()
    print(f"α_raw (all at-risk):        {alpha_raw:.4f}  (E[S_base]={E_S_base_at_risk:.4f}, E[sp]={E_softplus_at_risk:.4f})")
    print(f"α_neg (neg only at-risk):   {alpha_neg:.4f}  (E[S_base]={E_S_base_neg:.4f}, E[sp]={E_softplus_neg:.4f})")
    print(f"α_pos (pos only at-risk):   {alpha_pos:.4f}  (E[S_base]={E_S_base_pos_ar:.4f}, E[sp]={E_softplus_pos_ar:.4f})")
    print()
    print(f"β:                          {beta:.4f}")
    print()

    # 三种 α 修正方案
    alpha_scaled_by_neg_ratio = alpha_raw * neg_ratio_in_at_risk
    # 加权平均: α = (n_neg * α_neg + n_pos * α_pos) / (n_neg + n_pos)
    # 这其实就是 alpha_raw，所以不是新方案

    # 方案: 只对 neg 文档施加惩罚 → α = α_neg × neg_ratio
    # 但这不对，因为 α 是对所有 at-risk 的统一系数

    # 真正的思路: α 应该使得总惩罚能量 = neg 文档的基础分总和
    # α × E[Softplus|all at-risk] = neg_ratio × E[S_base|neg at-risk]
    # α = neg_ratio × E[S_base|neg] / E[Softplus|all]
    alpha_energy_match = neg_ratio_in_at_risk * E_S_base_neg / E_softplus_at_risk if E_softplus_at_risk > 0 else 0

    print("=" * 80)
    print("α CORRECTION SCHEMES")
    print("=" * 80)
    print(f"  α_raw (current V7):                         {alpha_raw:.4f}")
    print(f"  α_raw × neg_ratio:                          {alpha_scaled_by_neg_ratio:.4f}")
    print(f"  α_energy_match (neg_ratio × E[S_base_neg]/E[sp_all]): {alpha_energy_match:.4f}")
    print(f"  α_neg (neg-only at-risk):                   {alpha_neg:.4f}")
    print()
    print(f"  Grid optimal α (target_avg):                0.3000")
    print(f"  Grid optimal α (combined):                  0.7000")

    # 保存
    output = {
        "analysis": "alpha_scaling_with_neg_ratio",
        "anchor_delta": anchor_delta,
        "at_risk_composition": {
            "total": n_at_risk,
            "neg_docs": n_at_risk_neg,
            "pos_docs": n_at_risk_pos,
            "neg_ratio": neg_ratio_in_at_risk,
        },
        "alpha_variants": {
            "alpha_raw": alpha_raw,
            "alpha_scaled_by_neg_ratio": alpha_scaled_by_neg_ratio,
            "alpha_energy_match": alpha_energy_match,
            "alpha_neg_only": alpha_neg,
        },
        "beta": beta,
        "grid_optimal": {"alpha_target_avg": 0.3, "alpha_combined": 0.7, "beta_target_avg": 1.1, "beta_combined": 2.0},
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
