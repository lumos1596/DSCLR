"""
First-Principles Parameter Derivation for DeIR-Dual V2 with Safe-Anchor Threshold
V7: Safe-Anchor Aware Derivation (训练集推导，可泛化到其他编码器)

核心创新：
1. 用训练集 pos_docs 作为 "innocent document" 代理估计 tau_anchor
2. α_raw 量级对齐（去除有害的 coverage_correction）
3. β 带 train/test 分布补偿因子
4. 完全 encoder-agnostic: 只需重新编码训练集即可适配新编码器

V6→V7 变更（基于网格搜索验证）：
  - 去除 coverage_correction: V6 的 quantity-based 校正在 anchor_delta>0 时
    会因 at-risk 近乎归零而爆炸（cc 可达 28x），导致 α 严重高估。
    实测：α_raw(无校正)=0.76 ≈ 网格综合最优 0.7，而 α×cc=0.99→21 均偏差大。
  - β 补偿: 训练集 safe 文档以 pos（高 S_req）为主，测试集 safe 含大量无关文档
    （低 S_req），导致推导 β 偏低。引入 beta_compensation=2.0 弥补分布差异。

关键公式：
  tau_anchor_proxy = stat(sim(q_neg, pos_docs_per_query))  # 无辜文档与 q_neg 的最大相似度
  tau = max(tau_anchor_proxy, cos_qbase_qneg) + anchor_delta  # mix=max 策略

  α = E[S_base|at-risk] / E[Softplus(S_neg - τ)|at-risk]       # V7: 无 coverage_correction
  β = E[S_base|safe] / E[S_req × safety|safe] × beta_compensation  # V7: 带补偿因子
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET = "FollowIR_train"
TRAIN_EMBEDDINGS_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/embeddings/dsclr_train_embeddings_repllama-reproduced_qwen3-4B.pt"
TRAIN_JSONL_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/train/dsclr_total_dataset.jsonl"


def compute_scores(q_base_emb, q_plus_emb, q_minus_emb, doc_emb):
    """计算三种 query 与文档池的余弦相似度"""
    S_base = torch.mm(q_base_emb, doc_emb.T)
    S_req = torch.mm(q_plus_emb, doc_emb.T)
    S_neg = torch.mm(q_minus_emb, doc_emb.T)
    return S_base, S_req, S_neg


def compute_tau_anchor_proxy(q_base_emb, q_minus_emb, pos_emb, stat="max", top_k=5):
    """
    用训练集 pos_docs 作为 innocent document 代理估计 tau_anchor

    对于每个 query：
    1. 先用 S_base = sim(q_base, pos_docs) 找到最相关的 top-K pos_docs
    2. 对这 K 个 pos_docs 计算 S_neg = sim(q_neg, pos_docs)
    3. tau_anchor = stat(S_neg of top-K pos_docs)

    这模拟了 safe anchor 的生成过程：anchor 是符合 q_base 主题的无辜文档，
    其与 q_neg 的相似度代表"无辜文档的负向相似度上限"。

    top-K 过滤解决了 pos_docs 池全局共享的问题——只取与本 query 相关的 pos_docs。
    """
    # 归一化
    q_base_n = F.normalize(q_base_emb, dim=1)
    q_minus_n = F.normalize(q_minus_emb, dim=1)
    pos_n = F.normalize(pos_emb, dim=1)

    n_queries = q_base_n.shape[0]
    n_pos = pos_n.shape[0]
    k = min(top_k, n_pos)

    # sim(q_base, pos_docs): [n_queries, n_pos_docs] —— 用于找最相关的 pos_docs
    S_base_pos = torch.mm(q_base_n, pos_n.T)
    # sim(q_neg, pos_docs): [n_queries, n_pos_docs]
    S_neg_pos = torch.mm(q_minus_n, pos_n.T)

    # 对每个 query，取 S_base 最高的 top-K pos_docs 的 S_neg
    _, topk_idx = S_base_pos.topk(k, dim=1)  # [n_queries, k]
    # gather S_neg at top-K positions
    S_neg_topk = S_neg_pos.gather(1, topk_idx)  # [n_queries, k]

    if stat == "max":
        tau_anchor = S_neg_topk.max(dim=1).values  # [n_queries]
    elif stat == "mean":
        tau_anchor = S_neg_topk.mean(dim=1).values
    else:
        raise ValueError(f"未知 stat: {stat}")

    return tau_anchor


def derive_params_safe_anchor(S_base, S_req, S_neg, cos_qbase_qneg, tau_anchor_proxy,
                               anchor_delta=0.0, anchor_stat="max", anchor_mix_mode="max",
                               has_qneg_mask=None, safety_T=20.0,
                               tau_mode="scale", anchor_scale_factor=1.27,
                               coverage_correction_mode="none", beta_compensation=2.0):
    """
    使用 safe-anchor 阈值推导 α, β (V7)

    两种 tau_anchor 估计模式：
    1. proxy: 用训练集 pos_docs 估计（但训练集 pos_docs 语义分布与 LLM anchors 不同）
    2. scale: 用比例缩放法 tau_anchor = cos_qbase_qneg × anchor_scale_factor
       （基于测试集观察：LLM anchors 与 q_neg 的相似度约为 cos_qbase_qneg 的 1.27 倍，
        因为 anchors 共享 q_base 主题，与 q_neg 有更多语义重叠）

    Args:
        S_base, S_req, S_neg: 相似度矩阵 [n_queries, n_docs]
        cos_qbase_qneg: Cos(Q_base, Q_neg) [n_queries]
        tau_anchor_proxy: 用 pos_docs 估计的 tau_anchor [n_queries]
        anchor_delta: 阈值偏移（V7 推荐 +0.02）
        anchor_stat: max 或 mean
        anchor_mix_mode: replace/min/max/mean
        has_qneg_mask: bool [n_queries]，True 表示该 query 有负向约束
        safety_T: safety gate 的温度参数
        tau_mode: "proxy" 或 "scale"
        anchor_scale_factor: scale 模式下的缩放因子（默认 1.27）
        coverage_correction_mode: α 的覆盖率校正模式
            - "none": 不校正（V7 默认，α_raw 已自适应阈值变化）
            - "quantity": V6 的文档数量比例（anchor_delta>0 时会爆炸，不推荐）
            - "energy": 基于惩罚能量的比例（理论上更合理但仍会放大）
        beta_compensation: β 的 train/test 分布补偿因子（V7 默认 2.0）

    Returns:
        dict: 推导结果和统计量
    """
    device = S_base.device
    n_queries, n_docs = S_base.shape

    if has_qneg_mask is None:
        has_qneg_mask = torch.ones(n_queries, dtype=torch.bool, device=device)

    # ---- Step 1: 计算最终阈值 tau ----
    if tau_mode == "scale":
        # 比例缩放法：tau_anchor = cos_qbase_qneg × scale_factor
        tau_anchor_estimated = cos_qbase_qneg * anchor_scale_factor
    else:
        # proxy 模式：用 pos_docs 估计
        tau_anchor_estimated = tau_anchor_proxy

    # mix=max: tau = max(tau_anchor, cos_qbase_qneg) + anchor_delta
    tau_base = torch.where(
        has_qneg_mask,
        torch.maximum(tau_anchor_estimated, cos_qbase_qneg),
        cos_qbase_qneg,
    )
    tau = tau_base + anchor_delta  # [n_queries]
    tau_2d = tau.unsqueeze(1)  # [n_queries, 1] for broadcasting

    # ---- Step 2: at-risk / safe 划分 ----
    at_risk_mask = (S_neg > tau_2d) & has_qneg_mask.unsqueeze(1)
    safe_mask = ~at_risk_mask

    n_at_risk = at_risk_mask.sum().item()
    n_safe = safe_mask.sum().item()
    at_risk_ratio = n_at_risk / (n_queries * n_docs)

    # ---- Step 3: 计算 baseline 阈值下的 at-risk 统计 ----
    tau_baseline = cos_qbase_qneg + anchor_delta
    tau_baseline_2d = tau_baseline.unsqueeze(1)
    at_risk_mask_baseline = (S_neg > tau_baseline_2d) & has_qneg_mask.unsqueeze(1)
    n_at_risk_baseline = at_risk_mask_baseline.sum().item()
    at_risk_ratio_baseline = n_at_risk_baseline / (n_queries * n_docs)

    # ---- Step 4: safety gate ----
    safety = 1 - torch.sigmoid((S_neg - tau_2d) * safety_T)

    # ---- Step 5: α 推导（量级对齐）----
    if n_at_risk > 0:
        S_base_at_risk = S_base[at_risk_mask]
        S_neg_at_risk = S_neg[at_risk_mask]
        tau_at_risk = tau_2d.expand_as(S_neg)[at_risk_mask]
        E_S_base_at_risk = S_base_at_risk.mean().item()
        softplus_at_risk = F.softplus(S_neg_at_risk - tau_at_risk)
        E_softplus_at_risk = softplus_at_risk.mean().item()
        alpha_raw = E_S_base_at_risk / E_softplus_at_risk if E_softplus_at_risk > 0 else 1.0
    else:
        E_S_base_at_risk = 0.0
        E_softplus_at_risk = 0.0
        alpha_raw = 1.0

    # ---- Step 5.5: 覆盖率校正因子（V7: 默认不校正）----
    if coverage_correction_mode == "quantity":
        # V6 方案：基于文档数量比例（anchor_delta>0 时会爆炸，不推荐）
        if at_risk_ratio > 0 and at_risk_ratio_baseline > 0:
            coverage_correction = at_risk_ratio_baseline / at_risk_ratio
        else:
            coverage_correction = 1.0
    elif coverage_correction_mode == "energy":
        # 基于惩罚能量的比例（理论上更合理但仍有放大效应）
        if n_at_risk > 0 and n_at_risk_baseline > 0:
            softplus_baseline = F.softplus(
                S_neg[at_risk_mask_baseline] - tau_baseline_2d.expand_as(S_neg)[at_risk_mask_baseline]
            )
            E_softplus_baseline = softplus_baseline.mean().item()
            energy_baseline = E_softplus_baseline * at_risk_ratio_baseline
            energy_safe = E_softplus_at_risk * at_risk_ratio
            coverage_correction = energy_baseline / energy_safe if energy_safe > 0 else 1.0
        else:
            coverage_correction = 1.0
    else:
        # V7 默认：不校正（α_raw 已自适应阈值变化）
        coverage_correction = 1.0

    alpha_corrected = alpha_raw * coverage_correction

    # α 的多种方法
    if n_at_risk > 0:
        s_base_np = S_base[at_risk_mask].cpu().numpy()
        alpha_p50 = np.percentile(s_base_np, 50) / E_softplus_at_risk if E_softplus_at_risk > 0 else 1.0
        alpha_p75 = np.percentile(s_base_np, 75) / E_softplus_at_risk if E_softplus_at_risk > 0 else 1.0
        alpha_p50_corrected = alpha_p50 * coverage_correction
        alpha_p75_corrected = alpha_p75 * coverage_correction
    else:
        alpha_p50 = alpha_p75 = 1.0
        alpha_p50_corrected = alpha_p75_corrected = 1.0

    # ---- Step 6: β 推导（量级对齐 + train/test 分布补偿）----
    if n_safe > 0:
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta_raw = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        E_S_base_safe = 0.0
        E_S_req_safety_safe = 0.0
        beta_raw = 1.0
    beta = beta_raw * beta_compensation

    results = {
        "alpha_raw": round(alpha_raw, 4),
        "alpha_corrected": round(alpha_corrected, 4),
        "alpha_p50_raw": round(alpha_p50, 4),
        "alpha_p50_corrected": round(alpha_p50_corrected, 4),
        "alpha_p75_raw": round(alpha_p75, 4),
        "alpha_p75_corrected": round(alpha_p75_corrected, 4),
        "beta_raw": round(beta_raw, 4),
        "beta": round(beta, 4),
        "beta_compensation": beta_compensation,
        "anchor_delta": anchor_delta,
        "tau_mode": tau_mode,
        "anchor_scale_factor": anchor_scale_factor if tau_mode == "scale" else None,
        "coverage_correction_mode": coverage_correction_mode,
        "at_risk_ratio": at_risk_ratio,
        "at_risk_ratio_baseline": at_risk_ratio_baseline,
        "coverage_correction": round(coverage_correction, 4),
        "n_at_risk": n_at_risk,
        "n_at_risk_baseline": n_at_risk_baseline,
        "n_safe": n_safe,
        "E_S_base_at_risk": round(E_S_base_at_risk, 4),
        "E_softplus_at_risk": round(E_softplus_at_risk, 4),
        "E_S_base_safe": round(E_S_base_safe, 4),
        "E_S_req_safety_safe": round(E_S_req_safety_safe, 4),
        "tau_mean": round(tau.mean().item(), 4),
        "tau_anchor_mean": round(tau_anchor_estimated[has_qneg_mask].mean().item(), 4),
        "cos_qbase_qneg_mean": round(cos_qbase_qneg[has_qneg_mask].mean().item(), 4),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Safe-Anchor Aware First-Principles Parameter Derivation (V7)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--anchor_delta", type=float, default=0.02, help="Anchor threshold offset (V7 default: +0.02)")
    parser.add_argument("--anchor_stat", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--anchor_mix_mode", type=str, default="max", choices=["replace", "min", "max", "mean"])
    parser.add_argument("--anchor_topk", type=int, default=5, help="Top-K pos_docs per query for tau_anchor proxy")
    parser.add_argument("--tau_mode", type=str, default="scale", choices=["proxy", "scale"],
                        help="tau_anchor estimation: proxy (pos_docs) or scale (cos_qbase_qneg × factor)")
    parser.add_argument("--anchor_scale_factor", type=float, default=1.27,
                        help="Scale factor for tau_anchor = cos_qbase_qneg × factor (tau_mode=scale)")
    parser.add_argument("--coverage_correction_mode", type=str, default="none",
                        choices=["none", "quantity", "energy"],
                        help="α coverage correction: none (V7 default), quantity (V6), energy")
    parser.add_argument("--beta_compensation", type=float, default=2.0,
                        help="β train/test distribution compensation factor (V7 default: 2.0)")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to training embeddings .pt file (for encoder generalization)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save derived parameters JSON")
    args = parser.parse_args()

    embeddings_path = args.embeddings_path or TRAIN_EMBEDDINGS_PATH

    device = args.device
    if device == "cuda":
        torch.cuda._lazy_init()
        if not torch.cuda.is_available():
            device = "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Embeddings: {embeddings_path}")

    # ---- 加载训练集编码 ----
    logger.info("Loading training set embeddings...")
    emb = torch.load(embeddings_path, map_location=device, weights_only=False)
    logger.info(f"  q_base={emb['q_base_embeddings'].shape}, pos={emb['pos_embeddings'].shape}, "
                f"neg={emb['neg_embeddings'].shape}")

    q_base_emb = F.normalize(emb["q_base_embeddings"].float().to(device), dim=1)
    q_plus_emb = F.normalize(emb["q_plus_embeddings"].float().to(device), dim=1)
    q_minus_emb = F.normalize(emb["q_minus_embeddings"].float().to(device), dim=1)
    pos_emb = F.normalize(emb["pos_embeddings"].float().to(device), dim=1)
    neg_emb = F.normalize(emb["neg_embeddings"].float().to(device), dim=1)

    n_queries = q_base_emb.shape[0]
    n_pos = pos_emb.shape[0]
    n_neg = neg_emb.shape[0]
    logger.info(f"Dataset: {n_queries} queries, {n_pos} pos docs, {n_neg} neg docs")

    # ---- 识别有负向约束的 query ----
    q_minus_norms = emb["q_minus_embeddings"].float().norm(dim=1)
    has_qneg_mask = q_minus_norms > 0.01
    n_has_qneg = has_qneg_mask.sum().item()
    logger.info(f"Queries with meaningful q_neg: {n_has_qneg}/{n_queries}")

    # ---- 计算相似度 ----
    logger.info("Computing scores...")
    S_base_pos, S_req_pos, S_neg_pos = compute_scores(q_base_emb, q_plus_emb, q_minus_emb, pos_emb)
    S_base_neg, S_req_neg, S_neg_neg = compute_scores(q_base_emb, q_plus_emb, q_minus_emb, neg_emb)

    S_base = torch.cat([S_base_pos, S_base_neg], dim=1)
    S_req = torch.cat([S_req_pos, S_req_neg], dim=1)
    S_neg = torch.cat([S_neg_pos, S_neg_neg], dim=1)
    logger.info(f"Combined scores: {S_base.shape}")

    # ---- 计算 cos_qbase_qneg ----
    cos_qbase_qneg = (q_base_emb * q_minus_emb).sum(dim=1)
    logger.info(f"cos_qbase_qneg (has_qneg): mean={cos_qbase_qneg[has_qneg_mask].mean():.4f}, "
                f"min={cos_qbase_qneg[has_qneg_mask].min():.4f}, max={cos_qbase_qneg[has_qneg_mask].max():.4f}")

    # ---- 计算 tau_anchor 代理 ----
    logger.info("Computing tau_anchor proxy from top-K pos_docs...")
    tau_anchor_proxy = compute_tau_anchor_proxy(q_base_emb, q_minus_emb, pos_emb,
                                                 stat=args.anchor_stat, top_k=args.anchor_topk)
    logger.info(f"tau_anchor_proxy (has_qneg): mean={tau_anchor_proxy[has_qneg_mask].mean():.4f}, "
                f"min={tau_anchor_proxy[has_qneg_mask].min():.4f}, max={tau_anchor_proxy[has_qneg_mask].max():.4f}")

    # ---- 推导参数 ----
    results = derive_params_safe_anchor(
        S_base, S_req, S_neg, cos_qbase_qneg, tau_anchor_proxy,
        anchor_delta=args.anchor_delta,
        anchor_stat=args.anchor_stat,
        anchor_mix_mode=args.anchor_mix_mode,
        has_qneg_mask=has_qneg_mask,
        tau_mode=args.tau_mode,
        anchor_scale_factor=args.anchor_scale_factor,
        coverage_correction_mode=args.coverage_correction_mode,
        beta_compensation=args.beta_compensation,
    )

    # ---- 输出结果 ----
    logger.info("\n" + "="*80)
    logger.info("SAFE-ANCHOR AWARE PARAMETER DERIVATION RESULTS (V7)")
    logger.info("="*80)
    logger.info(f"\n  anchor_delta = {results['anchor_delta']}")
    logger.info(f"  anchor_stat = {args.anchor_stat}, mix_mode = {args.anchor_mix_mode}")
    logger.info(f"  coverage_correction_mode = {results['coverage_correction_mode']}")
    logger.info(f"  beta_compensation = {results['beta_compensation']}")
    logger.info(f"\n  Threshold statistics:")
    logger.info(f"    tau (final) mean = {results['tau_mean']:.4f}")
    logger.info(f"    tau_anchor mean = {results['tau_anchor_mean']:.4f}")
    logger.info(f"    cos_qbase_qneg mean = {results['cos_qbase_qneg_mean']:.4f}")
    logger.info(f"\n  At-risk statistics:")
    logger.info(f"    at_risk_ratio (safe-anchor) = {results['at_risk_ratio']*100:.2f}%")
    logger.info(f"    at_risk_ratio (baseline) = {results['at_risk_ratio_baseline']*100:.2f}%")
    logger.info(f"    coverage_correction = {results['coverage_correction']:.4f}")
    logger.info(f"\n  α derivation (mode={results['coverage_correction_mode']}):")
    logger.info(f"    alpha_raw (scale alignment) = {results['alpha_raw']:.4f}")
    logger.info(f"    alpha_corrected (×coverage) = {results['alpha_corrected']:.4f}")
    logger.info(f"    alpha_p50_corrected = {results['alpha_p50_corrected']:.4f}")
    logger.info(f"    alpha_p75_corrected = {results['alpha_p75_corrected']:.4f}")
    logger.info(f"\n  β derivation (compensation={results['beta_compensation']}):")
    logger.info(f"    beta_raw = {results['beta_raw']:.4f}")
    logger.info(f"    beta = beta_raw × compensation = {results['beta']:.4f}")
    logger.info(f"\n  Key stats:")
    logger.info(f"    E[S_base|at-risk] = {results['E_S_base_at_risk']:.4f}")
    logger.info(f"    E[Softplus|at-risk] = {results['E_softplus_at_risk']:.4f}")
    logger.info(f"    E[S_base|safe] = {results['E_S_base_safe']:.4f}")
    logger.info(f"    E[S_req×safety|safe] = {results['E_S_req_safety_safe']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDED PARAMETERS (V7)")
    logger.info("="*80)
    logger.info(f"\n  α = {results['alpha_corrected']}  (coverage_correction_mode={results['coverage_correction_mode']})")
    logger.info(f"  β = {results['beta']}  (beta_raw={results['beta_raw']} × compensation={results['beta_compensation']})")
    logger.info(f"  anchor_delta = {results['anchor_delta']}")

    # ---- 保存结果 ----
    output = {
        "source": "training_set_safe_anchor_aware_v7",
        "dataset": DATASET,
        "encoder": emb.get("model_name", "unknown"),
        "n_queries": n_queries,
        "n_queries_with_qneg": n_has_qneg,
        "n_pos_docs": n_pos,
        "n_neg_docs": n_neg,
        "anchor_stat": args.anchor_stat,
        "anchor_mix_mode": args.anchor_mix_mode,
        "anchor_delta": args.anchor_delta,
        "coverage_correction_mode": args.coverage_correction_mode,
        "beta_compensation": args.beta_compensation,
        "recommended_params": {
            "alpha": results["alpha_corrected"],
            "beta": results["beta"],
            "anchor_delta": results["anchor_delta"],
        },
        "alternative_alpha": {
            "raw_scale_alignment": results["alpha_raw"],
            "p50_corrected": results["alpha_p50_corrected"],
            "p75_corrected": results["alpha_p75_corrected"],
        },
        "statistics": results,
    }

    output_path = args.output_path or "/home/luwa/Documents/DSCLR/results/train_derived_params_safe_anchor.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=lambda o: float(o))
    logger.info(f"\n  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
