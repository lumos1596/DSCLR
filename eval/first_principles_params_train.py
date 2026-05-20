"""
First-Principles Parameter Derivation for DeIR-Dual V2
V5: Training-Set Based Derivation (学术规范版)

使用训练集文档编码推导 α, β, δ，然后在测试集上验证效果。
关键：仅使用训练集的编码数据，不使用测试集编码。
"""

import sys
import json
import torch
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
    S_base = torch.mm(q_base_emb, doc_emb.T)
    S_req = torch.mm(q_plus_emb, doc_emb.T)
    S_neg = torch.mm(q_minus_emb, doc_emb.T)
    return S_base, S_req, S_neg


def compute_penalty(softplus_at_risk, alpha):
    return alpha * softplus_at_risk


def compute_alpha_scale_alignment(S_base, S_neg, tau, at_risk_mask, E_softplus_at_risk):
    E_S_base_at_risk = S_base[at_risk_mask].mean().item()
    alpha = E_S_base_at_risk / E_softplus_at_risk
    return alpha


def compute_alpha_percentile(S_base, S_neg, tau, at_risk_mask, percentile=50):
    s_base_np = S_base[at_risk_mask].cpu().numpy()
    p = np.percentile(s_base_np, percentile)
    E_softplus = torch.nn.functional.softplus(S_neg[at_risk_mask] - tau).mean().item()
    alpha = p / E_softplus
    return alpha


def compute_beta_scale_alignment(S_base, S_req, tau, safe_mask, safety):
    s_base_safe = S_base[safe_mask]
    s_req_safe = S_req[safe_mask]
    safety_safe = safety[safe_mask]
    E_S_base_safe = s_base_safe.mean().item()
    E_S_req_safety_safe = (s_req_safe * safety_safe).mean().item()
    if E_S_req_safety_safe > 0:
        beta = E_S_base_safe / E_S_req_safety_safe
    else:
        beta = 1.0
    return beta


def compute_delta_npmle(S_neg):
    return 0.0


def compute_first_principles_params_from_scores(S_base, S_req, S_neg, device, delta_k=0.0):
    sigma_random = S_neg.std().item()
    delta = delta_k * sigma_random
    tau = S_neg + delta
    softplus_at_risk = torch.nn.functional.softplus(S_neg - tau)

    E_softplus_at_risk = softplus_at_risk.mean().item()
    E_softplus_all = torch.nn.functional.softplus(S_neg - delta).mean().item()

    sigma_random = S_neg.std().item()
    E_tau = tau.mean().item()

    n_docs = S_base.shape[1]
    has_req_mask = torch.ones_like(S_base, dtype=torch.bool).to(device)
    has_neg_mask = torch.ones_like(S_base, dtype=torch.bool).to(device)

    at_risk_mask = (S_neg > tau).float()
    safe_mask = 1 - at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * 20.0)

    at_risk_ratio = at_risk_mask.mean().item()
    n_at_risk = at_risk_mask.sum().item()
    n_safe = safe_mask.sum().item()

    results = {
        "delta": delta,
        "E_tau": E_tau,
        "sigma_random": sigma_random,
        "at_risk_ratio": at_risk_ratio,
        "n_at_risk": n_at_risk,
        "n_safe": n_safe,
    }

    if n_at_risk > 0:
        results["E_softplus_at_risk"] = E_softplus_at_risk
        results["E_S_base_at_risk"] = S_base[at_risk_mask.bool()].mean().item()
    else:
        results["E_softplus_at_risk"] = E_softplus_all
        results["E_S_base_at_risk"] = S_base.mean().item()

    if n_safe > 0:
        results["E_S_base_safe"] = S_base[safe_mask.bool()].mean().item()
    else:
        results["E_S_base_safe"] = S_base.mean().item()

    std_S_pool = torch.cat([S_base.flatten(), S_req.flatten(), S_neg.flatten()]).std().item()
    results["std_S_pool"] = std_S_pool

    if n_at_risk > 0:
        alpha_scale = compute_alpha_scale_alignment(S_base, S_neg, tau, at_risk_mask.bool(), E_softplus_at_risk)
        alpha_p50 = compute_alpha_percentile(S_base, S_neg, tau, at_risk_mask.bool(), percentile=50)
        alpha_p75 = compute_alpha_percentile(S_base, S_neg, tau, at_risk_mask.bool(), percentile=75)
    else:
        alpha_scale = 1.0
        alpha_p50 = 1.0
        alpha_p75 = 1.0

    results["alpha_methods"] = {
        "scale_alignment": {
            "alpha": round(alpha_scale, 4),
            "method": "E[S_base|at-risk] / E[Softplus|at-risk]",
            "meaning": "惩罚量级对齐基础分量级",
        },
        "percentile_50": {
            "alpha": round(alpha_p50, 4),
            "method": "P50[S_base|at-risk] / E[Softplus|at-risk]",
            "meaning": "中位数对齐",
        },
        "percentile_75": {
            "alpha": round(alpha_p75, 4),
            "method": "P75[S_base|at-risk] / E[Softplus|at-risk]",
            "meaning": "75分位数对齐",
        },
    }

    if n_safe > 0:
        beta_scale = compute_beta_scale_alignment(S_base, S_req, tau, safe_mask.bool(), safety)
    else:
        beta_scale = 1.29
    results["beta"] = round(beta_scale, 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="First-Principles Parameter Derivation from Training Set")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--delta_k", type=float, default=0.0, help="Coverage factor k for δ = k × σ_random")
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        torch.cuda._lazy_init()
        if not torch.cuda.is_available():
            device = "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading training set embeddings...")
    emb = torch.load(TRAIN_EMBEDDINGS_PATH, map_location=device, weights_only=False)
    logger.info(f"  Loaded embeddings: q_base={emb['q_base_embeddings'].shape}, "
                f"pos={emb['pos_embeddings'].shape}, neg={emb['neg_embeddings'].shape}")

    logger.info("Loading training set metadata...")
    with open(TRAIN_JSONL_PATH) as f:
        train_data = [json.loads(line) for line in f]
    logger.info(f"  Loaded {len(train_data)} training queries")

    q_base_emb = emb["q_base_embeddings"].to(device)
    q_plus_emb = emb["q_plus_embeddings"].to(device)
    q_minus_emb = emb["q_minus_embeddings"].to(device)
    pos_emb = emb["pos_embeddings"].to(device)
    neg_emb = emb["neg_embeddings"].to(device)

    n_queries = q_base_emb.shape[0]
    n_pos = pos_emb.shape[0]
    n_neg = neg_emb.shape[0]
    logger.info(f"\nDataset sizes: {n_queries} queries, {n_pos} pos docs, {n_neg} neg docs")

    logger.info("\nComputing scores for all document pools...")

    S_base_pos, S_req_pos, S_neg_pos = compute_scores(q_base_emb, q_plus_emb, q_minus_emb, pos_emb)
    logger.info(f"  S_base_pos: shape={S_base_pos.shape}, mean={S_base_pos.mean():.4f}")

    S_base_neg, S_req_neg, S_neg_neg = compute_scores(q_base_emb, q_plus_emb, q_minus_emb, neg_emb)
    logger.info(f"  S_base_neg: shape={S_base_neg.shape}, mean={S_base_neg.mean():.4f}")

    S_base = torch.cat([S_base_pos, S_base_neg], dim=1)
    S_req = torch.cat([S_req_pos, S_req_neg], dim=1)
    S_neg = torch.cat([S_neg_pos, S_neg_neg], dim=1)
    logger.info(f"  Combined: S_base={S_base.shape}, S_req={S_req.shape}, S_neg={S_neg.shape}")

    results = compute_first_principles_params_from_scores(S_base, S_req, S_neg, device, delta_k=args.delta_k)

    logger.info("\n" + "="*80)
    logger.info("TRAINING SET PARAMETER DERIVATION RESULTS")
    logger.info("="*80)

    logger.info(f"\n  δ = {results['delta']:.4f} (k={args.delta_k})")
    logger.info(f"  β = {results['beta']:.4f}")
    logger.info(f"  At-risk ratio: {results['at_risk_ratio']*100:.2f}%")
    logger.info(f"  σ_random (std of S_neg): {results['sigma_random']:.4f}")

    logger.info(f"\n  α derivation methods:")
    for method, info in results["alpha_methods"].items():
        logger.info(f"    {method:20s}: α = {info['alpha']:.4f}")
        logger.info(f"      Method: {info['method']}")
        logger.info(f"      Meaning: {info['meaning']}")

    logger.info("\n  Key statistics:")
    logger.info(f"    E[S_base|at-risk] = {results['E_S_base_at_risk']:.4f}")
    logger.info(f"    E[Softplus|at-risk] = {results['E_softplus_at_risk']:.4f}")
    logger.info(f"    E[S_base|safe] = {results['E_S_base_safe']:.4f}")
    logger.info(f"    std(S_pool) = {results['std_S_pool']:.4f}")

    recommended_alpha = results["alpha_methods"]["scale_alignment"]["alpha"]
    recommended_beta = results["beta"]
    recommended_delta = results["delta"]

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDED PARAMETERS (from training set only)")
    logger.info("="*80)
    logger.info(f"\n  α = {recommended_alpha}")
    logger.info(f"  β = {recommended_beta}")
    logger.info(f"  δ = {recommended_delta}")
    logger.info(f"\n  完整参数组合: α={recommended_alpha}, β={recommended_beta}, δ={recommended_delta}")
    logger.info(f"  注意: 仅使用训练集编码推导，符合学术规范")

    output = {
        "source": "training_set_only",
        "dataset": DATASET,
        "n_queries": n_queries,
        "n_pos_docs": n_pos,
        "n_neg_docs": n_neg,
        "delta_k": args.delta_k,
        "recommended_params": {
            "alpha": recommended_alpha,
            "beta": recommended_beta,
            "delta": recommended_delta,
        },
        "alpha_methods": results["alpha_methods"],
        "statistics": {
            "at_risk_ratio": results["at_risk_ratio"],
            "sigma_random": results["sigma_random"],
            "E_S_base_at_risk": results["E_S_base_at_risk"],
            "E_softplus_at_risk": results["E_softplus_at_risk"],
            "E_S_base_safe": results["E_S_base_safe"],
            "std_S_pool": results["std_S_pool"],
        },
    }

    output_path = "/home/luwa/Documents/DSCLR/results/train_derived_params.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
