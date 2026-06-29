"""
阈值方案对比实验

问题：当前 τ = Cos(Q_base, Q_neg) + δ 中，Cos(Q_base, Q_neg) 是 QQ 相似度，
      S_neg = Cos(Q_neg, D) 是 QD 相似度，两者可能不在同一尺度。

方案：
  A. 仿射校准: τ = a × Cos(Q_base, Q_neg) + b + δ
     - 用训练集线性回归将 QQ 空间映射到 QD 空间
  B. 纯QD百分位: τ = P_q(S_neg) + δ
     - 完全在 QD 空间内，用 S_neg 分布的百分位
  C. S_base调制: τ_d = S_base(d) × Cos(Q_base, Q_neg) + δ
     - 用 S_base (QD空间) 作基底，cos 作无量纲调制

其它条件完全不变：engine_v2 公式、RepLLaMA 编码器、V5 推导原理（量级对齐）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============ 路径常量 ============
TRAIN_EMB_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/embeddings/dsclr_train_embeddings_repllama-reproduced_qwen3-4B.pt"
TRAIN_JSONL_PATH = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/train/dsclr_total_dataset.jsonl"
DUAL_QUERIES_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01"
CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"
T_SAFETY = 20.0
TASKS = ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]


# ============ 阈值方案实现 ============

def compute_tau_original(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """原始方案: τ = Cos(Q_base, Q_neg) + δ (基线)"""
    return cos_qbase_qneg + delta


def compute_tau_affine(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """方案A: 仿射校准 τ = a × Cos(Q_base, Q_neg) + b + δ"""
    a = params.get("affine_a", 1.0)
    b = params.get("affine_b", 0.0)
    return a * cos_qbase_qneg + b + delta


def compute_tau_percentile(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """方案B: 纯QD百分位 τ = P_q(S_neg) + δ (per-query)"""
    q = params.get("percentile_q", 95.0)
    # S_neg shape: (n_docs,) for a single query
    tau = float(np.percentile(S_neg.cpu().numpy(), q))
    return tau + delta


def compute_tau_sbase_modulated(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """方案C: S_base调制 τ_d = S_base(d) × Cos(Q_base, Q_neg) + δ (per-document)"""
    # 返回 per-document 阈值
    return S_base * cos_qbase_qneg + delta


def compute_tau_qd_anchored(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """方案D: QD空间锚定 τ = mean(S_neg) + cos(Q_base, Q_neg) × σ(S_neg) + δ
    将QQ相似度作为无量纲调制子，乘以QD空间的σ，加上QD空间的均值。
    """
    mu = params.get("qd_mu", 0.0)
    sigma = params.get("qd_sigma", 1.0)
    return mu + cos_qbase_qneg * sigma + delta


def compute_tau_qd_max(cos_qbase_qneg, delta, S_neg=None, S_base=None, params=None):
    """方案E: QD空间下界保护 τ = max(cos, μ(S_neg) + k·σ(S_neg)) + δ
    QD空间统计阈值作为下界，cos作为语义调制，取max确保安全。
    理论干净：不混尺度，两个候选阈值各自在自己的空间内定义。
    注意：当 cos=0 ([NONE] 查询) 时，q_minus 为零向量，S_neg 无意义，
    此时 τ 的值不影响实际惩罚（因为 S_neg=0），但影响 safety/softplus 的推导。
    为保持与原始方案一致的推导行为，cos=0 时不应用 qd_floor。
    """
    mu = params.get("qd_mu", 0.0)
    sigma = params.get("qd_sigma", 1.0)
    k = params.get("qd_k", 0.5)
    qd_floor = float(mu + k * sigma)
    # cos_qbase_qneg 可能是 float (评分时) 或 tensor (推导时)
    if isinstance(cos_qbase_qneg, torch.Tensor):
        # 对 cos=0 的位置不应用 qd_floor (保持原始行为)
        result = cos_qbase_qneg.clone()
        nonzero_mask = cos_qbase_qneg > 0
        if nonzero_mask.any():
            result[nonzero_mask] = torch.clamp(cos_qbase_qneg[nonzero_mask], min=qd_floor)
        return result + delta
    else:
        cos_val = float(cos_qbase_qneg)
        if cos_val > 0:
            return max(cos_val, qd_floor) + delta
        else:
            return cos_val + delta


TAU_SCHEMES = {
    "original": compute_tau_original,
    "affine": compute_tau_affine,
    "percentile": compute_tau_percentile,
    "sbase_modulated": compute_tau_sbase_modulated,
    "qd_anchored": compute_tau_qd_anchored,
    "qd_max": compute_tau_qd_max,
}


# ============ 训练集参数推导 ============

def derive_params_original(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """原始方案的 V5 推导"""
    sigma = S_neg.std().item()
    delta = delta_k * sigma

    tau = cos_qbase_qneg.unsqueeze(1) + delta
    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    # α: Scale Alignment
    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    # β: Scale Alignment for Enhancement
    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


def derive_params_affine(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """方案A: 仿射校准推导
    用线性回归将 Cos(Q_base, Q_neg) 映射到 S_neg 的均值空间
    target: 对于每个 query, mean(S_neg) 是该查询的 QD 空间中心
    """
    sigma = S_neg.std().item()
    delta = delta_k * sigma

    # 线性回归: mean(S_neg) ~ a * cos_qbase_qneg + b
    # S_neg shape: (n_queries, n_docs), cos_qbase_qneg shape: (n_queries,)
    mean_S_neg = S_neg.mean(dim=1)  # (n_queries,)
    x = cos_qbase_qneg.cpu().double().numpy()
    y = mean_S_neg.cpu().double().numpy()

    # 最小二乘: y = a*x + b
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) > 1e-10:
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
    else:
        a, b = 1.0, 0.0

    logger.info(f"  [Affine] 线性回归: a={a:.4f}, b={b:.4f}")
    logger.info(f"  [Affine] Cos(Q_base,Q_neg) 范围: [{x.min():.4f}, {x.max():.4f}], mean={x.mean():.4f}")
    logger.info(f"  [Affine] mean(S_neg) 范围: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")
    logger.info(f"  [Affine] 校准后 τ 范围: [{a*x.min()+b+delta:.4f}, {a*x.max()+b+delta:.4f}]")

    # 用校准后的 τ 推导 α, β
    tau = (a * cos_qbase_qneg + b + delta).unsqueeze(1)
    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "affine_a": a, "affine_b": b,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


def derive_params_percentile(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """方案B: 纯QD百分位推导
    找到百分位 q 使得 at-risk ratio ≈ 5%（与原始方案一致）
    """
    sigma = S_neg.std().item()
    delta = delta_k * sigma

    # 搜索 q 使得 at-risk ratio ≈ 5%
    # at-risk: S_neg > P_q(S_neg) + delta, 即 S_neg > P_q(S_neg) 的比例为 (100-q)%
    # 但因为 delta > 0, 实际 at-risk ratio 会略低于 (100-q)%
    # 我们用二分搜索找到合适的 q

    best_q = 95.0
    best_diff = float('inf')
    target_ratio = 0.004  # 目标与原始方案一致 (0.4%)

    for q in np.arange(80.0, 99.9, 0.1):
        # 对每个 query 计算 percentile, 然后统计整体 at-risk
        at_risk_count = 0
        total_count = 0
        for i in range(S_neg.shape[0]):
            threshold = np.percentile(S_neg[i].cpu().numpy(), q) + delta
            at_risk_count += (S_neg[i] > threshold).sum().item()
            total_count += S_neg.shape[1]

        ratio = at_risk_count / total_count
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_q = q

    q = best_q
    logger.info(f"  [Percentile] 最优百分位 q={q:.1f}%, at-risk ratio={best_diff+target_ratio:.4f}")

    # 用百分位 τ 推导 α, β
    # 对于训练集, 计算每个 query 的 percentile threshold
    tau_list = []
    for i in range(S_neg.shape[0]):
        tau_i = np.percentile(S_neg[i].cpu().numpy(), q) + delta
        tau_list.append(tau_i)
    tau = torch.tensor(tau_list, device=S_neg.device).unsqueeze(1)

    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "percentile_q": q,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


def derive_params_sbase_modulated(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """方案C: S_base调制推导
    τ_d = S_base(d) × Cos(Q_base, Q_neg) + δ (per-document)
    """
    sigma = S_neg.std().item()
    delta = delta_k * sigma

    # τ is per-document: S_base * cos + delta
    # S_base shape: (n_queries, n_docs), cos_qbase_qneg shape: (n_queries,)
    tau = S_base * cos_qbase_qneg.unsqueeze(1) + delta

    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


DERIVE_FUNCS = {
    "original": derive_params_original,
    "affine": derive_params_affine,
    "percentile": derive_params_percentile,
    "sbase_modulated": derive_params_sbase_modulated,
    "qd_anchored": None,  # 占位，下方填充
}


def derive_params_qd_anchored(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """方案D: QD空间锚定推导
    τ = mean(S_neg) + cos × σ(S_neg) + δ
    QQ相似度作为无量纲调制子，σ(S_neg)将其映射到QD空间尺度。
    """
    sigma = S_neg.std().item()
    delta = delta_k * sigma
    mu = S_neg.mean().item()

    tau = (mu + cos_qbase_qneg * sigma + delta).unsqueeze(1)
    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    logger.info(f"  [QD-Anchored] μ={mu:.4f}, σ={sigma:.4f}")
    logger.info(f"  [QD-Anchored] τ range: [{mu + cos_qbase_qneg.min().item()*sigma + delta:.4f}, {mu + cos_qbase_qneg.max().item()*sigma + delta:.4f}]")

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "qd_mu": mu, "qd_sigma": sigma,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


DERIVE_FUNCS["qd_anchored"] = derive_params_qd_anchored


def derive_params_qd_max(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=0.09):
    """方案E: QD空间下界保护推导
    τ = max(cos, μ(S_neg) + k·σ(S_neg)) + δ
    k 从训练集推导：选择使 at-risk ratio 最接近原始方案的 k。
    """
    sigma = S_neg.std().item()
    delta = delta_k * sigma
    mu = S_neg.mean().item()

    # 原始方案的 at-risk ratio
    tau_orig = cos_qbase_qneg.unsqueeze(1) + delta
    at_risk_orig = (S_neg > tau_orig).float().mean().item()

    # 搜索 k 使得 at-risk 最接近原始方案
    # 注意：cos=0 的 query 不应用 qd_floor (保持与原始方案一致)
    best_k = 0.5
    best_diff = float('inf')
    for k in np.arange(0.0, 3.0, 0.1):
        qd_floor = float(mu + k * sigma)
        tau_k = cos_qbase_qneg.clone()
        nonzero_mask = cos_qbase_qneg > 0
        if nonzero_mask.any():
            tau_k[nonzero_mask] = torch.clamp(cos_qbase_qneg[nonzero_mask], min=qd_floor)
        tau_k = tau_k.unsqueeze(1) + delta
        at_risk_k = (S_neg > tau_k).float().mean().item()
        diff = abs(at_risk_k - at_risk_orig)
        if diff < best_diff:
            best_diff = diff
            best_k = k

    k = best_k
    qd_floor = float(mu + k * sigma)
    tau = cos_qbase_qneg.clone()
    nonzero_mask = cos_qbase_qneg > 0
    if nonzero_mask.any():
        tau[nonzero_mask] = torch.clamp(cos_qbase_qneg[nonzero_mask], min=qd_floor)
    tau = tau.unsqueeze(1) + delta
    at_risk_mask = (S_neg > tau)
    safe_mask = ~at_risk_mask

    safety = 1 - torch.sigmoid((S_neg - tau) * T_SAFETY)
    softplus_all = F.softplus(S_neg - tau)

    if at_risk_mask.any():
        E_S_base_at_risk = S_base[at_risk_mask].mean().item()
        E_softplus = softplus_all.mean().item()
        alpha = E_S_base_at_risk / E_softplus if E_softplus > 0 else 1.0
    else:
        alpha = 1.0

    if safe_mask.any():
        E_S_base_safe = S_base[safe_mask].mean().item()
        E_S_req_safety_safe = (S_req[safe_mask] * safety[safe_mask]).mean().item()
        beta = E_S_base_safe / E_S_req_safety_safe if E_S_req_safety_safe > 0 else 1.0
    else:
        beta = 1.0

    logger.info(f"  [QD-Max] k={k:.2f}, qd_floor={qd_floor:.4f}")
    logger.info(f"  [QD-Max] at-risk={at_risk_mask.float().mean().item():.4f} (原始={at_risk_orig:.4f})")

    return {"alpha": alpha, "beta": beta, "delta": delta,
            "qd_mu": mu, "qd_sigma": sigma, "qd_k": k,
            "at_risk_ratio": at_risk_mask.float().mean().item()}


DERIVE_FUNCS["qd_max"] = derive_params_qd_max


# ============ 打分函数 ============

def score_deir_dual_v2(S_base, S_req, S_neg, cos_qbase_qneg, has_req, has_neg,
                       alpha, beta, delta, tau_scheme, params):
    """DeIR-Dual V2 打分，支持不同阈值方案

    Returns: S_final, avg_penalty
    """
    if not has_neg:
        s_req_eff = S_req if has_req else torch.zeros_like(S_base)
        s_final = S_base + beta * s_req_eff
        return s_final, 0.0

    # 计算阈值
    tau_fn = TAU_SCHEMES[tau_scheme]
    tau = tau_fn(cos_qbase_qneg, delta, S_neg=S_neg, S_base=S_base, params=params)

    # 打分
    overflow = S_neg - tau
    smooth_penalty = F.softplus(overflow)
    raw_penalty = alpha * smooth_penalty
    safety = 1.0 - torch.sigmoid((S_neg - tau) * T_SAFETY)

    s_req_eff = S_req if has_req else torch.zeros_like(S_base)
    s_final = S_base + beta * s_req_eff * safety - raw_penalty

    avg_penalty = float(raw_penalty.mean().item())
    return s_final, avg_penalty


# ============ 测试集评测 ============

def load_test_data(task_name, dual_queries_path):
    """加载测试集数据"""
    from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings

    engine = DSCLREvaluatorEngine(
        model_name=MODEL_NAME,
        task_name=task_name,
        output_dir=f"/tmp/tau_experiment_{task_name}",
        device="cuda",
        batch_size=64,
        use_cache=True,
    )

    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    # 加载 dual queries
    dual_data = {}
    with open(dual_queries_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            dual_data[item["qid"]] = item

    # 加载缓存的文档向量
    cached_data = load_cached_embeddings(CACHE_DIR, task_name, MODEL_NAME)
    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
    else:
        all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        engine.retriever.index_documents(all_doc_ids, doc_texts, engine.batch_size)

    return engine, corpus, q_og, q_changed, q_raw_og, q_raw_changed, candidates, dual_data


def encode_and_compute_scores(engine, q_raw_og, q_raw_changed, candidates, dual_data):
    """编码查询并计算相似度矩阵"""
    def is_none(text):
        if not text:
            return True
        return str(text).strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", "")

    def prepare_queries(q_raw, dual_data_ref):
        qids = list(q_raw.keys())
        q_base_list, q_req_list, q_neg_list = [], [], []
        has_req_list, has_neg_list = [], []

        for qid in qids:
            raw = q_raw.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_base = f"{query_text} {instruction}".strip() if query_text else ""
            q_base_list.append(q_base)

            d = dual_data_ref.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            q_req_list.append(q_plus if not is_none(q_plus) else "")
            q_neg_list.append(q_minus if not is_none(q_minus) else "")
            has_req_list.append(0.0 if is_none(q_plus) else 1.0)
            has_neg_list.append(0.0 if is_none(q_minus) else 1.0)

        return qids, q_base_list, q_req_list, q_neg_list, has_req_list, has_neg_list

    # OG queries
    qids_og, q_base_og, q_req_og, q_neg_og, has_req_og, has_neg_og = prepare_queries(q_raw_og, dual_data)
    # Changed queries
    qids_ch, q_base_ch, q_req_ch, q_neg_ch, has_req_ch, has_neg_ch = prepare_queries(q_raw_changed, dual_data)

    # 编码
    logger.info(f"  编码 OG queries ({len(q_base_og)} 条)...")
    emb_base_og = engine._encode_queries(q_base_og)
    emb_req_og = engine._encode_queries(q_req_og)
    emb_neg_og = engine._encode_queries(q_neg_og)

    logger.info(f"  编码 Changed queries ({len(q_base_ch)} 条)...")
    emb_base_ch = engine._encode_queries(q_base_ch)
    emb_req_ch = engine._encode_queries(q_req_ch)
    emb_neg_ch = engine._encode_queries(q_neg_ch)

    device = engine.retriever.doc_embeddings.device
    emb_base_og = emb_base_og.to(device)
    emb_req_og = emb_req_og.to(device)
    emb_neg_og = emb_neg_og.to(device)
    emb_base_ch = emb_base_ch.to(device)
    emb_req_ch = emb_req_ch.to(device)
    emb_neg_ch = emb_neg_ch.to(device)

    # 计算相似度
    doc_emb = engine.retriever.doc_embeddings
    S_base_og = torch.matmul(emb_base_og, doc_emb.T)
    S_req_og = torch.matmul(emb_req_og, doc_emb.T)
    S_neg_og = torch.matmul(emb_neg_og, doc_emb.T) * torch.tensor(has_neg_og, device=device).unsqueeze(1)

    S_base_ch = torch.matmul(emb_base_ch, doc_emb.T)
    S_req_ch = torch.matmul(emb_req_ch, doc_emb.T)
    S_neg_ch = torch.matmul(emb_neg_ch, doc_emb.T) * torch.tensor(has_neg_ch, device=device).unsqueeze(1)

    # Cos(Q_base, Q_neg) — 处理零向量导致的 NaN
    cos_qbase_qneg_og = torch.nan_to_num(F.cosine_similarity(emb_base_og, emb_neg_og, dim=1), nan=0.0)
    cos_qbase_qneg_ch = torch.nan_to_num(F.cosine_similarity(emb_base_ch, emb_neg_ch, dim=1), nan=0.0)

    has_req_og_t = torch.tensor(has_req_og, device=device)
    has_neg_og_t = torch.tensor(has_neg_og, device=device)
    has_req_ch_t = torch.tensor(has_req_ch, device=device)
    has_neg_ch_t = torch.tensor(has_neg_ch, device=device)

    return {
        "qids_og": qids_og, "qids_ch": qids_ch,
        "S_base_og": S_base_og, "S_req_og": S_req_og, "S_neg_og": S_neg_og,
        "S_base_ch": S_base_ch, "S_req_ch": S_req_ch, "S_neg_ch": S_neg_ch,
        "cos_og": cos_qbase_qneg_og, "cos_ch": cos_qbase_qneg_ch,
        "has_req_og": has_req_og_t, "has_neg_og": has_neg_og_t,
        "has_req_ch": has_req_ch_t, "has_neg_ch": has_neg_ch_t,
    }


def evaluate_scheme(engine, scores_data, candidates, alpha, beta, delta, tau_scheme, params):
    """评测单个方案"""
    from eval.metrics import FollowIREvaluator

    qids_og = scores_data["qids_og"]
    qids_ch = scores_data["qids_ch"]

    # 构建 candidate indices
    doc_id_to_idx = {did: i for i, did in enumerate(engine.retriever.doc_ids)}

    # OG: 不应用惩罚（原始查询）
    S_final_og = scores_data["S_base_og"].clone()

    # Changed: 应用 DeIR-Dual V2 打分
    S_final_ch = scores_data["S_base_ch"].clone()
    total_penalty = 0.0
    n_neg_queries = 0

    for q_idx, qid in enumerate(qids_ch):
        base_qid = qid.replace("-og", "").replace("-changed", "")
        cand_indices = candidates.get(base_qid, [])
        if not cand_indices:
            continue

        idx_tensor = torch.tensor(
            [doc_id_to_idx[d] for d in cand_indices if d in doc_id_to_idx],
            device=S_final_ch.device, dtype=torch.long
        )
        if len(idx_tensor) == 0:
            continue

        s_b = scores_data["S_base_ch"][q_idx].index_select(0, idx_tensor)
        s_r = scores_data["S_req_ch"][q_idx].index_select(0, idx_tensor)
        s_n = scores_data["S_neg_ch"][q_idx].index_select(0, idx_tensor)

        has_req = bool(scores_data["has_req_ch"][q_idx].item() > 0)
        has_neg = bool(scores_data["has_neg_ch"][q_idx].item() > 0)
        cos_val = float(scores_data["cos_ch"][q_idx].item())

        s_final_local, avg_penalty = score_deir_dual_v2(
            s_b, s_r, s_n, cos_val, has_req, has_neg,
            alpha, beta, delta, tau_scheme, params
        )

        s_final_local = s_final_local.to(dtype=S_final_ch.dtype)
        S_final_ch[q_idx, idx_tensor] = s_final_local
        if has_neg:
            total_penalty += avg_penalty
            n_neg_queries += 1

    # 提取结果
    def extract_results(S_final, qids, top_k=1000):
        results = {}
        for idx, qid in enumerate(qids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand = candidates.get(base_qid, [])
            scores = {}
            for did in cand:
                if did in doc_id_to_idx:
                    scores[did] = float(S_final[idx, doc_id_to_idx[did]].item())
            sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])
            results[qid] = sorted_scores
        return results

    results_og = extract_results(S_final_og, qids_og)
    results_ch = extract_results(S_final_ch, qids_ch)

    # 评测
    evaluator = FollowIREvaluator(engine.task_name)
    metrics = evaluator.evaluate(results_og, results_ch)

    return metrics, total_penalty / max(n_neg_queries, 1)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="阈值方案对比实验")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta_k", type=float, default=0.09, help="δ = delta_k × σ(S_neg)")
    parser.add_argument("--schemes", nargs="+", default=["original", "qd_max"])
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        torch.cuda._lazy_init()
        if not torch.cuda.is_available():
            device = "cpu"
    logger.info(f"Using device: {device}")

    # ============ Step 1: 训练集参数推导 ============
    logger.info("=" * 80)
    logger.info("Step 1: 训练集参数推导")
    logger.info("=" * 80)

    logger.info("加载训练集嵌入...")
    emb = torch.load(TRAIN_EMB_PATH, map_location=device, weights_only=False)
    q_base_emb = emb["q_base_embeddings"].to(device)
    q_plus_emb = emb["q_plus_embeddings"].to(device)
    q_minus_emb = emb["q_minus_embeddings"].to(device)
    pos_emb = emb["pos_embeddings"].to(device)
    neg_emb = emb["neg_embeddings"].to(device)

    # 计算训练集分数
    S_base_pos = torch.mm(q_base_emb, pos_emb.T)
    S_req_pos = torch.mm(q_plus_emb, pos_emb.T)
    S_neg_pos = torch.mm(q_minus_emb, pos_emb.T)

    S_base_neg = torch.mm(q_base_emb, neg_emb.T)
    S_req_neg = torch.mm(q_plus_emb, neg_emb.T)
    S_neg_neg = torch.mm(q_minus_emb, neg_emb.T)

    S_base = torch.cat([S_base_pos, S_base_neg], dim=1)
    S_req = torch.cat([S_req_pos, S_req_neg], dim=1)
    S_neg = torch.cat([S_neg_pos, S_neg_neg], dim=1)

    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_minus_emb, dim=1)
    # 处理零向量导致的 NaN（[NONE] 查询的 q_minus 为零向量）
    cos_qbase_qneg = torch.nan_to_num(cos_qbase_qneg, nan=0.0)

    logger.info(f"训练集: {S_base.shape[0]} queries, {S_base.shape[1]} docs")
    logger.info(f"Cos(Q_base, Q_neg): mean={cos_qbase_qneg.mean():.4f}, std={cos_qbase_qneg.std():.4f}")
    logger.info(f"S_neg: mean={S_neg.mean():.4f}, std={S_neg.std():.4f}")

    # 推导各方案参数
    all_params = {}
    for scheme in args.schemes:
        logger.info(f"\n--- 推导方案: {scheme} ---")
        derive_fn = DERIVE_FUNCS[scheme]
        params = derive_fn(S_base, S_req, S_neg, cos_qbase_qneg, delta_k=args.delta_k)
        all_params[scheme] = params
        logger.info(f"  α={params.get('alpha', 'N/A'):.4f}")
        logger.info(f"  β={params.get('beta', 'N/A'):.4f}")
        logger.info(f"  δ={params.get('delta', 'N/A'):.4f}")
        logger.info(f"  at-risk ratio={params.get('at_risk_ratio', 'N/A'):.4f}")
        if "affine_a" in params:
            logger.info(f"  affine_a={params['affine_a']:.4f}, affine_b={params['affine_b']:.4f}")
        if "percentile_q" in params:
            logger.info(f"  percentile_q={params['percentile_q']:.1f}")

    # ============ Step 2: 测试集评测 ============
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: 测试集评测")
    logger.info("=" * 80)

    all_results = {}

    for task in TASKS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task}")
        logger.info(f"{'='*60}")

        dual_path = os.path.join(DUAL_QUERIES_DIR, f"dual_queries_TSC_BALANCED_t01_{task}.jsonl")
        if not os.path.exists(dual_path):
            logger.warning(f"Dual queries not found: {dual_path}")
            continue

        engine, corpus, q_og, q_changed, q_raw_og, q_raw_changed, candidates, dual_data = \
            load_test_data(task, dual_path)

        scores_data = encode_and_compute_scores(engine, q_raw_og, q_raw_changed, candidates, dual_data)

        task_results = {}
        for scheme in args.schemes:
            logger.info(f"\n  评测方案: {scheme}")
            params = all_params[scheme]
            alpha = params["alpha"]
            beta = params["beta"]
            delta = params["delta"]

            metrics, avg_penalty = evaluate_scheme(
                engine, scores_data, candidates,
                alpha, beta, delta, scheme, params
            )

            p_mrr = metrics.get("p-MRR", 0.0)
            changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
            changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)
            og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
            og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)

            logger.info(f"    p-MRR={p_mrr:.4f}, changed_MAP={changed_map:.4f}, changed_nDCG@5={changed_ndcg5:.4f}")
            logger.info(f"    og_MAP={og_map:.4f}, og_nDCG@5={og_ndcg5:.4f}, avg_penalty={avg_penalty:.4f}")

            task_results[scheme] = {
                "p-MRR": p_mrr,
                "changed_MAP@1000": changed_map,
                "changed_nDCG@5": changed_ndcg5,
                "og_MAP@1000": og_map,
                "og_nDCG@5": og_ndcg5,
                "avg_penalty": avg_penalty,
                "params": {k: v for k, v in params.items() if k != "at_risk_ratio"},
            }

        all_results[task] = task_results

        # 清理 GPU
        del engine
        del scores_data
        torch.cuda.empty_cache()

    # ============ Step 3: 汇总报告 ============
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: 汇总报告")
    logger.info("=" * 80)

    # 计算 target_avg
    print("\n" + "=" * 80)
    print("阈值方案对比结果")
    print("=" * 80)

    header = f"{'方案':<20} {'α':<8} {'β':<8} {'δ':<8} {'p-MRR':<10} {'target_avg':<12} {'Core17_cMAP':<14} {'R04_cMAP':<12} {'News21_cnDCG5':<16}"
    print(header)
    print("-" * len(header))

    for scheme in args.schemes:
        params = all_params[scheme]
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]

        pmrrs = []
        target_avg_components = []

        for task in TASKS:
            if task in all_results and scheme in all_results[task]:
                r = all_results[task][scheme]
                pmrrs.append(r["p-MRR"])
                if "Core17" in task:
                    target_avg_components.append(r["changed_MAP@1000"])
                elif "Robust04" in task:
                    target_avg_components.append(r["changed_MAP@1000"])
                elif "News21" in task:
                    target_avg_components.append(r["changed_nDCG@5"])

        mean_pmrr = sum(pmrrs) / len(pmrrs) if pmrrs else 0
        target_avg = sum(target_avg_components) / len(target_avg_components) if target_avg_components else 0

        # 逐数据集
        c17_map = all_results.get("Core17InstructionRetrieval", {}).get(scheme, {}).get("changed_MAP@1000", 0)
        r04_map = all_results.get("Robust04InstructionRetrieval", {}).get(scheme, {}).get("changed_MAP@1000", 0)
        n21_ndcg = all_results.get("News21InstructionRetrieval", {}).get(scheme, {}).get("changed_nDCG@5", 0)

        extra = ""
        if "affine_a" in params:
            extra = f" (a={params['affine_a']:.2f}, b={params['affine_b']:.2f})"
        elif "percentile_q" in params:
            extra = f" (q={params['percentile_q']:.1f}%)"

        print(f"{scheme:<20} {alpha:<8.4f} {beta:<8.4f} {delta:<8.4f} {mean_pmrr:<10.4f} {target_avg:<12.4f} {c17_map:<14.4f} {r04_map:<12.4f} {n21_ndcg:<16.4f}{extra}")

    print("\n" + "=" * 80)
    print("基线 (V5 δ=0.02): p-MRR=0.1687, target_avg=0.2841")
    print("=" * 80)

    # 保存结果
    output_path = "/home/luwa/Documents/DSCLR/results/tau_scheme_comparison.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "delta_k": args.delta_k,
        "params": {k: {kk: vv for kk, vv in v.items()} for k, v in all_params.items()},
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=lambda o: float(o))
    logger.info(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
