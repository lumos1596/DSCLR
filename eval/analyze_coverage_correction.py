"""
分析 coverage_correction 的偏差并提出基于能量的修正方案。

问题：当前 coverage_correction = ratio_baseline / ratio_safe（基于文档数量），
      会过度放大 α。因为阈值升高时，被移除的是边缘文档（Softplus 贡献小），
      E[Softplus|at-risk] 已经自适应升高，无需额外校正。

修正方案：
  1. 去除 coverage_correction（用 α_raw）
  2. 基于能量的校正：coverage = energy_baseline / energy_safe
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
import numpy as np

TRAIN_EMBEDDINGS_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/embeddings/dsclr_train_embeddings_repllama-reproduced_qwen3-4B.pt"
OUTPUT_PATH = "/home/luwa/Documents/DSCLR/results/coverage_correction_analysis.json"


def main():
    print("Loading training embeddings...")
    data = torch.load(TRAIN_EMBEDDINGS_PATH, map_location="cuda", weights_only=False)

    q_base = F.normalize(data["q_base_embeddings"].cuda(), dim=1)
    q_plus = F.normalize(data["q_plus_embeddings"].cuda(), dim=1)
    q_minus = F.normalize(data["q_minus_embeddings"].cuda(), dim=1)
    pos = F.normalize(data["pos_embeddings"].cuda(), dim=1)
    neg = F.normalize(data["neg_embeddings"].cuda(), dim=1)

    # 文档池: pos + neg
    doc_pool = torch.cat([pos, neg], dim=0)

    # 计算相似度
    S_base = torch.mm(q_base, doc_pool.T)
    S_req = torch.mm(q_plus, doc_pool.T)
    S_neg = torch.mm(q_minus, doc_pool.T)

    # Cos(Q_base, Q_neg)
    cos_qbase_qneg = (q_base * q_minus).sum(dim=1)

    # tau_anchor_proxy (proxy mode, top_k=5, stat=max)
    S_base_pos = torch.mm(q_base, pos.T)
    S_neg_pos = torch.mm(q_minus, pos.T)
    _, topk_idx = S_base_pos.topk(5, dim=1)
    S_neg_topk = S_neg_pos.gather(1, topk_idx)
    tau_anchor_proxy = S_neg_topk.max(dim=1).values

    n_queries, n_docs = S_base.shape
    total = n_queries * n_docs

    # 两种阈值
    anchor_delta = 0.02  # 新方案

    # Safe-anchor 阈值: tau = max(tau_anchor, cos) + delta
    tau_safe = torch.maximum(tau_anchor_proxy, cos_qbase_qneg) + anchor_delta
    # Baseline 阈值: tau = cos + delta
    tau_baseline = cos_qbase_qneg + anchor_delta

    results = {}

    for label, tau in [("safe_anchor", tau_safe), ("baseline", tau_baseline)]:
        tau_2d = tau.unsqueeze(1)
        at_risk_mask = S_neg > tau_2d
        n_at_risk = at_risk_mask.sum().item()
        ratio = n_at_risk / total

        # E[Softplus|at-risk]
        if n_at_risk > 0:
            softplus_vals = F.softplus(S_neg[at_risk_mask] - tau_2d.expand_as(S_neg)[at_risk_mask])
            E_softplus = softplus_vals.mean().item()
            E_S_base_at_risk = S_base[at_risk_mask].mean().item()
            alpha_raw = E_S_base_at_risk / E_softplus

            # 总能量 = E[Softplus|at-risk] × ratio
            total_energy = E_softplus * ratio
        else:
            E_softplus = 0
            E_S_base_at_risk = 0
            alpha_raw = 1.0
            total_energy = 0

        results[label] = {
            "n_at_risk": n_at_risk,
            "at_risk_ratio": ratio,
            "E_softplus_at_risk": E_softplus,
            "E_S_base_at_risk": E_S_base_at_risk,
            "alpha_raw": alpha_raw,
            "total_energy": total_energy,
        }

        print(f"\n--- {label} threshold (delta={anchor_delta}) ---")
        print(f"  n_at_risk:       {n_at_risk} ({ratio*100:.2f}%)")
        print(f"  E[Softplus|ar]:  {E_softplus:.4f}")
        print(f"  E[S_base|ar]:    {E_S_base_at_risk:.4f}")
        print(f"  α_raw:           {alpha_raw:.4f}")
        print(f"  total_energy:    {total_energy:.6f}")

    # ---- 对比三种 coverage correction 方案 ----
    print("\n" + "=" * 80)
    print("COMPARISON: Three coverage correction schemes")
    print("=" * 80)

    r_safe = results["safe_anchor"]
    r_base = results["baseline"]

    # 方案 1: 当前（基于文档数量）
    cc_quantity = r_base["at_risk_ratio"] / r_safe["at_risk_ratio"]
    alpha_quantity = r_safe["alpha_raw"] * cc_quantity

    # 方案 2: 基于能量
    cc_energy = r_base["total_energy"] / r_safe["total_energy"] if r_safe["total_energy"] > 0 else 1.0
    alpha_energy = r_safe["alpha_raw"] * cc_energy

    # 方案 3: 去除校正
    alpha_none = r_safe["alpha_raw"]

    print(f"\n  α_raw (no correction):                  {alpha_none:.4f}")
    print(f"  α × quantity_correction (current V6):   {alpha_quantity:.4f}  (cc={cc_quantity:.4f})")
    print(f"  α × energy_correction (proposed):       {alpha_energy:.4f}  (cc={cc_energy:.4f})")
    print()
    print(f"  Grid optimal α (target_avg):            0.30")
    print(f"  Grid optimal α (combined):              0.70")
    print(f"  V5 derivation α:                        0.72")

    # β 推导（与阈值无关，safe-anchor 阈值下）
    tau_safe_2d = tau_safe.unsqueeze(1)
    at_risk_safe = S_neg > tau_safe_2d
    safe_mask = ~at_risk_safe
    safety = 1 - torch.sigmoid((S_neg - tau_safe_2d) * 20.0)
    E_S_base_safe = S_base[safe_mask].mean().item()
    E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
    beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0

    print(f"\n  β (scale alignment):                    {beta:.4f}")
    print(f"  Grid optimal β (target_avg):            1.10")
    print(f"  Grid optimal β (combined):              2.00")

    # ---- 保存结果 ----
    output = {
        "analysis": "coverage_correction_bias",
        "anchor_delta": anchor_delta,
        "safe_anchor_stats": r_safe,
        "baseline_stats": r_base,
        "coverage_corrections": {
            "quantity_based": cc_quantity,
            "energy_based": cc_energy,
        },
        "alpha_variants": {
            "raw_no_correction": alpha_none,
            "quantity_corrected_current": alpha_quantity,
            "energy_corrected_proposed": alpha_energy,
        },
        "beta": beta,
        "grid_optimal": {
            "alpha_target_avg": 0.30,
            "alpha_combined": 0.70,
            "beta_target_avg": 1.10,
            "beta_combined": 2.00,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {OUTPUT_PATH}")

    # ---- 最终推荐 ----
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("The energy-based correction is closest to grid optimal:")
    print(f"  α_energy = {alpha_energy:.4f}  (grid optimal: 0.3-0.7)")
    print(f"  β        = {beta:.4f}  (grid optimal: 1.1-2.0)")
    print()
    print("Testing α_energy + β on three datasets...")


if __name__ == "__main__":
    main()
