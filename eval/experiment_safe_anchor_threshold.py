"""
Safe Anchor Threshold 实验

核心思想：用 LLM（本实验手动）生成的"无辜文档锚点"估计负向惩罚阈值 τ，
替代原始的 τ = Cos(Q_base, Q_neg) + δ。

    tau_anchor = max_q( sim(q_neg, anchor) )   # anchor 遍历 safe_anchors
    τ = tau_anchor (+ δ 可选)
    其余 DeIR-Dual V2 公式（Softplus 惩罚、safety 门控、α/β）保持不变

Safe anchors 只用于估计 τ，不参与召回、不参与 rerank（非 HyDE/Query2Doc）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn.functional as F

from eval.engine_deir_dual_v2 import DeIRDualV2Evaluator
from eval.engine_dscrl import load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模块化函数
# ---------------------------------------------------------------------------
def load_safe_anchors(path: str) -> Dict[str, List[str]]:
    """加载 safe anchors 文件。返回 {qid: [anchor_text, ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def compute_safe_anchor_threshold(
    q_neg_emb: torch.Tensor,          # [num_queries, dim]
    query_ids: List[str],
    anchors_map: Dict[str, List[str]],
    encoder_fn,                        # callable: List[str] -> Tensor
    stat: str = "max",
) -> tuple:
    """对每个 query 计算 tau_anchor = stat_q( sim(q_neg, anchor) )。

    返回 (override_tensor, per_query_scores)：
      - override_tensor: [num_queries]，无 anchor 的 query 为 -inf
      - per_query_scores: List[List[float]]，每个 query 的各 anchor 相似度（无 anchor 为 []）
    """
    device = q_neg_emb.device
    num_q = len(query_ids)
    override = torch.full((num_q,), float("-inf"), device=device)
    per_query_scores: List[List[float]] = [[] for _ in range(num_q)]

    # 收集所有 anchor 文本并记录 (query_idx, pool_start, pool_end)
    all_anchor_texts: List[str] = []
    anchor_assignments: List[tuple] = []  # (query_idx, pool_start, pool_end)
    for q_idx, qid in enumerate(query_ids):
        anchors = anchors_map.get(qid, [])
        if not anchors:
            anchor_assignments.append((q_idx, -1, -1))
            continue
        start = len(all_anchor_texts)
        all_anchor_texts.extend(anchors)
        anchor_assignments.append((q_idx, start, start + len(anchors)))

    if not all_anchor_texts:
        return override, per_query_scores

    # 编码所有 anchor（批量）
    logger.info(f"🪝 编码 {len(all_anchor_texts)} 个 safe anchors...")
    anchor_emb = encoder_fn(all_anchor_texts).to(device)  # [num_anchors, dim]
    anchor_emb = F.normalize(anchor_emb, p=2, dim=1)

    for q_idx, start, end in anchor_assignments:
        if start < 0:
            continue
        q_neg_vec = q_neg_emb[q_idx].unsqueeze(0)         # [1, dim]
        anchor_vecs = anchor_emb[start:end]               # [k, dim]
        sims = F.cosine_similarity(q_neg_vec, anchor_vecs)  # [k]
        scores = [float(x) for x in sims.tolist()]
        per_query_scores[q_idx] = scores
        if stat == "max":
            val = max(scores)
        elif stat == "mean":
            val = sum(scores) / len(scores)
        elif stat == "min":
            val = min(scores)
        else:
            raise ValueError(f"未知 stat: {stat}")
        override[q_idx] = val

    return override, per_query_scores


def apply_safe_anchor_penalty(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau_anchor: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """独立版惩罚计算（仅供调试/参考，实际实验复用引擎的 Softplus 公式）。

    penalty(d) = max(s_neg(d) - tau_anchor, 0)
    score_final = S_base - alpha * penalty
    """
    tau_exp = tau_anchor.unsqueeze(1)
    penalty = torch.relu(S_neg - tau_exp)
    return S_base - alpha * penalty


# ---------------------------------------------------------------------------
# 子类化引擎：注入 tau_anchor
# ---------------------------------------------------------------------------
class SafeAnchorDeIREvaluator(DeIRDualV2Evaluator):
    """在 DeIR-Dual V2 基础上，用 safe anchor 阈值替代 Cos(Q_base,Q_neg)。"""

    def __init__(
        self,
        *args,
        safe_anchors_path: Optional[str] = None,
        anchor_stat: str = "max",
        anchor_delta: float = 0.0,   # 附加在 tau_anchor 上的偏移
        anchor_mix_mode: str = "replace",  # replace/min/max/mean
        safety_tau_mode: str = "coupled",  # coupled|cos_delta|off|add_margin|req_gated|req_thresh
        safety_margin: float = 0.0,
        req_threshold: float = 0.25,
        per_query_ab: bool = False,  # V8: per-query test-time α/β derivation
        ab_clip_alpha: tuple = (0.1, 5.0),
        ab_clip_beta: tuple = (0.0, 10.0),
        beta_derive_mode: str = "mean",  # V8: mean|std|range|topk_mean
        penalty_tau_mode: str = "anchor",  # V8.1: anchor|s_neg_pctl|hybrid|hybrid_floor
        penalty_percentile: float = 90.0,  # V8.1: percentile for s_neg_pctl/hybrid modes
        penalty_func: str = "softplus",  # V8.2: softplus|linear|scaled_linear|quadratic
        penalty_scale: float = 1.0,  # V8.2: scale factor for linear/scaled_linear/quadratic
        adaptive_t_safety: bool = False,  # V8.3: per-query 自适应 t_safety
        adaptive_safety_threshold: float = 0.92,  # safety_mean 低于此值时触发降 t_safety
        adaptive_t_safety_min: float = 3.0,  # 自适应 t_safety 下限保护
        **kwargs,
):
        self.safe_anchors_path = safe_anchors_path
        self.anchor_stat = anchor_stat
        self.anchor_delta = anchor_delta
        self.anchor_mix_mode = anchor_mix_mode
        self.safety_tau_mode = safety_tau_mode
        self.safety_margin = safety_margin
        self.req_threshold = req_threshold
        self.per_query_ab = per_query_ab
        self.ab_clip_alpha = ab_clip_alpha
        self.ab_clip_beta = ab_clip_beta
        self.beta_derive_mode = beta_derive_mode
        self.penalty_tau_mode = penalty_tau_mode
        self.penalty_percentile = penalty_percentile
        self.penalty_func = penalty_func
        self.penalty_scale = penalty_scale
        self.adaptive_t_safety = adaptive_t_safety
        self.adaptive_safety_threshold = adaptive_safety_threshold
        self.adaptive_t_safety_min = adaptive_t_safety_min
        self._current_t_safety_q: float = 0.0  # 记录当前 query 的自适应 t_safety
        self._tau_anchor_override: Optional[torch.Tensor] = None
        self._cos_qbase_qneg_orig: Optional[torch.Tensor] = None  # 原始 cos，供解耦 safety 用
        self._current_q_idx: int = 0
        self._debug_records: List[Dict[str, Any]] = []
        self._per_query_alphas: List[float] = []
        self._per_query_betas: List[float] = []
        self._per_query_stats: List[Dict[str, Any]] = []  # V8: detailed per-query statistics
        super().__init__(*args, **kwargs)

    def _compute_adaptive_t_safety(
        self,
        s_neg: torch.Tensor,
        tau_safety: float,
    ) -> float:
        """V8.3: 计算自适应 t_safety。

        原理：先用基础 t_safety 计算 safety_init，若 safety_mean 过低（大量文档被
        门控抑制），按比例降低 t_safety 使 sigmoid 更平滑，避免过度抑制 Q_plus 增强。

        公式：
            safety_init = 1 - sigmoid((S_neg - τ) × t_safety_base)
            safety_mean_init = mean(safety_init)
            if safety_mean_init < threshold:
                t_safety_q = t_safety_base × (safety_mean_init / threshold)
                t_safety_q = max(t_safety_q, t_safety_min)
            else:
                t_safety_q = t_safety_base

        物理意义：
        - safety_mean 低 → 大量文档 S_neg 接近或超过 τ → 高 t_safety 硬切换误伤文档
        - 降低 t_safety → sigmoid 平滑 → 接近阈值的文档不被完全抑制 → Q_plus 保留更多
        """
        if not self.adaptive_t_safety:
            self._current_t_safety_q = self.t_safety
            return self.t_safety

        with torch.no_grad():
            safety_init = 1.0 - torch.sigmoid((s_neg - tau_safety) * self.t_safety)
            safety_mean_init = float(safety_init.mean().item())

        if safety_mean_init < self.adaptive_safety_threshold:
            # 按立方比例降低 t_safety：safety_mean 越低，t_safety 降越多（立方使调整更激进）
            ratio = safety_mean_init / self.adaptive_safety_threshold
            t_safety_q = self.t_safety * (ratio ** 3)
            t_safety_q = max(t_safety_q, self.adaptive_t_safety_min)
        else:
            t_safety_q = self.t_safety

        self._current_t_safety_q = t_safety_q
        return t_safety_q

    def _derive_beta_q(
        self,
        s_base_safe: torch.Tensor,
        s_req_safe: torch.Tensor,
        safety_safe: torch.Tensor,
        beta_fallback: float,
    ) -> float:
        """V8: Derive per-query β from safe documents' distribution.

        Modes:
        - mean:       β = E[S_base] / E[S_req·safety]  (scale alignment, V7 approach)
        - std:        β = std(S_base) / std(S_req·safety)  (spread alignment for ranking)
        - range:      β = range(S_base) / range(S_req·safety)
        - topk_mean:  β = mean(S_base, top-20) / mean(S_req·safety, top-20 by S_base)
        - max_mean:   β = max(S_base) / mean(S_req·safety)  (peak-calibrated)
        - p90_mean:   β = percentile(S_base, 90) / mean(S_req·safety)  (robust peak)
        - peak_comp:  β = topk_mean_ratio × (max(S_base) / mean(S_base))  (self-compensated)
        """
        s_reward = s_req_safe * safety_safe
        if s_base_safe.numel() == 0 or s_reward.numel() == 0:
            return beta_fallback

        mode = self.beta_derive_mode
        if mode == "std":
            std_b = s_base_safe.std()
            std_r = s_reward.std()
            return float((std_b / std_r).item()) if std_r > 1e-8 else beta_fallback
        elif mode == "range":
            range_b = s_base_safe.max() - s_base_safe.min()
            range_r = s_reward.max() - s_reward.min()
            return float((range_b / range_r).item()) if range_r > 1e-8 else beta_fallback
        elif mode == "topk_mean":
            k = min(20, s_base_safe.numel())
            topk_idx = torch.topk(s_base_safe, k).indices
            mean_b = s_base_safe[topk_idx].mean()
            mean_r = s_reward[topk_idx].mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        elif mode == "max_mean":
            max_b = s_base_safe.max()
            mean_r = s_reward.mean()
            return float((max_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        elif mode == "p90_mean":
            p90_b = torch.quantile(s_base_safe.float(), 0.9)
            mean_r = s_reward.mean()
            return float((p90_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        elif mode == "peak_comp":
            k = min(20, s_base_safe.numel())
            topk_idx = torch.topk(s_base_safe, k).indices
            mean_b_topk = s_base_safe[topk_idx].mean()
            mean_r_topk = s_reward[topk_idx].mean()
            ratio = float((mean_b_topk / mean_r_topk).item()) if mean_r_topk > 1e-8 else beta_fallback
            mean_b_all = s_base_safe.mean()
            max_b = s_base_safe.max()
            comp = float((max_b / mean_b_all).item()) if mean_b_all > 1e-8 else 1.0
            return ratio * comp
        elif mode == "max_comp":
            # β = max(S_base) / mean(S_req·safety) × (max(S_base) / mean(S_base))
            # = max² / (mean(S_base) × mean(S_req·safety))
            # Double peak calibration: reward must overcome peak base signal
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((max_b * max_b / (mean_b * mean_r)).item())
            return beta_fallback
        elif mode == "cubed_comp":
            # β = max(S_base)³ / (mean(S_base)² × mean(S_req·safety))
            # Triple peak calibration: stronger than max_comp for peaked distributions
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((max_b ** 3 / (mean_b ** 2 * mean_r)).item())
            return beta_fallback
        elif mode == "p95_comp":
            # β = P95(S_base)² / (mean(S_base) × mean(S_req·safety))
            # Robust peak calibration: P95 less sensitive to outlier max
            p95_b = torch.quantile(s_base_safe.float(), 0.95)
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((p95_b * p95_b / (mean_b * mean_r)).item())
            return beta_fallback
        elif mode == "topk_comp":
            # β = mean(S_base, top-K)² / (mean(S_base) × mean(S_req·safety))
            # K=5: stable peak using top-5 mean instead of max
            k = min(5, s_base_safe.numel())
            topk_idx = torch.topk(s_base_safe, k).indices
            peak_b = s_base_safe[topk_idx].mean()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((peak_b * peak_b / (mean_b * mean_r)).item())
            return beta_fallback
        elif mode == "req_gap_comp":
            # β = max_comp × (1 + |mean(S_base) - mean(S_req·safety)| / mean(S_base))
            # Instruction sensitivity aware: when S_base >> S_req·safety (large gap),
            # Q_plus adds little new info, need stronger β to compensate
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b * max_b / (mean_b * mean_r)).item())
                gap_factor = 1.0 + abs(float(mean_b.item()) - float(mean_r.item())) / float(mean_b.item())
                return base_comp * gap_factor
            return beta_fallback
        elif mode == "variance_comp":
            # β = max_comp × (1 + std(S_base) / mean(S_base))
            # Distribution shape aware: higher cv (spread) → stronger β for top docs
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            std_b = s_base_safe.std()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b * max_b / (mean_b * mean_r)).item())
                cv = float((std_b / mean_b).item())
                return base_comp * (1.0 + cv)
            return beta_fallback
        elif mode == "at_risk_comp":
            # β = max_comp × (1 + at_risk_ratio)
            # At-risk ratio aware: queries with more at-risk docs need stronger enhancement
            # Note: at_risk_ratio is passed via self._current_at_risk_ratio
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b * max_b / (mean_b * mean_r)).item())
                at_risk_ratio = getattr(self, '_current_at_risk_ratio', 0.0)
                return base_comp * (1.0 + at_risk_ratio)
            return beta_fallback
        elif mode == "multi_signal":
            # β = max_comp × (1 + cv(S_base) × at_risk_ratio)
            # Multi-signal: combines peak calibration with distribution shape and at-risk awareness
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            std_b = s_base_safe.std()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b * max_b / (mean_b * mean_r)).item())
                cv = float((std_b / mean_b).item())
                at_risk_ratio = getattr(self, '_current_at_risk_ratio', 0.0)
                return base_comp * (1.0 + cv * at_risk_ratio)
            return beta_fallback
        elif mode == "quartic_comp":
            # β = max(S_base)⁴ / (mean(S_base)² × mean(S_req·safety)²)
            # Quartic peak calibration: stronger emphasis on distribution peak
            # Expected β ≈ 2.2-2.8, closer to V7's encoder-level β=2.55
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((max_b ** 4 / (mean_b ** 2 * mean_r ** 2)).item())
            return beta_fallback
        elif mode == "quartic_gap":
            # β = quartic_comp × (1 + |mean(S_base) - mean(S_req·safety)| / mean(S_base))
            # Quartic peak + instruction sensitivity gap
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b ** 4 / (mean_b ** 2 * mean_r ** 2)).item())
                gap_factor = 1.0 + abs(float(mean_b.item()) - float(mean_r.item())) / float(mean_b.item())
                return base_comp * gap_factor
            return beta_fallback
        else:  # mean
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback

    def _score_query_dual_v2(
        self,
        s_base: torch.Tensor,
        s_req: torch.Tensor,
        s_neg: torch.Tensor,
        cos_qbase_qneg: float,
        has_req: bool,
        has_neg: bool,
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, float]:
        """解耦版打分：penalty 用 safe-anchor τ，safety 用独立 τ_safety。

        - coupled: τ_safety = τ_penalty (原 V6 行为)
        - cos_delta: τ_safety = 原始 cos(Q_base,Q_neg) + delta (V5 风格高阈值)
        - off: safety = 1 (移除 safety 项)
        - add_margin: τ_safety = τ_penalty + safety_margin
        - req_gated: safety 由 S_req 门控，高 S_req 文档不被误伤
        - req_thresh: 奖励项加 Softplus 阈值 τ_req，聚焦强相关文档
        """
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            if self.per_query_ab and has_req and s_base.numel() > 0:
                # V8: 无 Q_neg 时所有文档都是 safe
                ones_safety = torch.ones_like(s_base)
                beta_q = self._derive_beta_q(s_base, s_req, ones_safety, beta)
                beta_q = max(self.ab_clip_beta[0], min(beta_q, self.ab_clip_beta[1]))
                self._per_query_betas.append(beta_q)
                beta = beta_q
            s_final = s_base + beta * s_req_eff
            return s_final, 0.0

        # V8.1: 计算惩罚阈值 tau_penalty
        # - anchor: 传统 safe-anchor 阈值 cos_qbase_qneg(=max(tau_anchor,cos)+anchor_delta) + delta
        # - s_neg_pctl: τ = P_percentile(S_neg)，确保 top (100-p)% 文档 at-risk
        # - hybrid: τ = min(anchor_threshold, P_percentile(S_neg))，保守上限
        # - hybrid_floor: τ = max(min(anchor_threshold, P_percentile(S_neg)), cos_qbase_qneg_orig)
        anchor_tau = cos_qbase_qneg + delta  # 原 safe-anchor 阈值
        if self.penalty_tau_mode == "s_neg_pctl" and s_neg.numel() > 1:
            p = self.penalty_percentile / 100.0
            tau_penalty = float(torch.quantile(s_neg.float(), p).item())
        elif self.penalty_tau_mode == "hybrid" and s_neg.numel() > 1:
            p = self.penalty_percentile / 100.0
            s_neg_pctl = float(torch.quantile(s_neg.float(), p).item())
            tau_penalty = min(anchor_tau, s_neg_pctl)
        elif self.penalty_tau_mode == "hybrid_floor" and s_neg.numel() > 1:
            p = self.penalty_percentile / 100.0
            s_neg_pctl = float(torch.quantile(s_neg.float(), p).item())
            cos_orig = 0.0
            if self._cos_qbase_qneg_orig is not None:
                cos_orig = float(self._cos_qbase_qneg_orig[self._current_q_idx].item())
            tau_penalty = max(min(anchor_tau, s_neg_pctl), cos_orig)
        else:  # anchor (default)
            tau_penalty = anchor_tau

        # 计算 τ_safety
        # 注意：当 penalty_tau_mode != "anchor" 时，tau_penalty 被降低（基于 S_neg 分位数）
        # 此时 coupled 模式应使用原始 anchor 阈值，避免 safety gate 过度抑制 Q_plus 增强
        if self.safety_tau_mode == "off":
            safety = torch.ones_like(s_neg)
            tau_safety = tau_penalty  # 不使用，仅占位
            self._current_t_safety_q = self.t_safety
        elif self.safety_tau_mode == "cos_delta":
            cos_orig = 0.0
            if self._cos_qbase_qneg_orig is not None:
                cos_orig = float(self._cos_qbase_qneg_orig[self._current_q_idx].item())
            tau_safety = cos_orig + delta
            t_safety_q = self._compute_adaptive_t_safety(s_neg, tau_safety)
            safety = 1.0 - torch.sigmoid((s_neg - tau_safety) * t_safety_q)
        elif self.safety_tau_mode == "add_margin":
            tau_safety = tau_penalty + self.safety_margin
            t_safety_q = self._compute_adaptive_t_safety(s_neg, tau_safety)
            safety = 1.0 - torch.sigmoid((s_neg - tau_safety) * t_safety_q)
        elif self.safety_tau_mode == "coupled":
            # 当 penalty 阈值被降低时，safety 仍用原始 anchor 阈值（解耦）
            if self.penalty_tau_mode != "anchor":
                tau_safety = anchor_tau
            else:
                tau_safety = tau_penalty
            t_safety_q = self._compute_adaptive_t_safety(s_neg, tau_safety)
            safety = 1.0 - torch.sigmoid((s_neg - tau_safety) * t_safety_q)
        else:  # coupled_strict (legacy fallback)
            tau_safety = tau_penalty
            t_safety_q = self._compute_adaptive_t_safety(s_neg, tau_safety)
            safety = 1.0 - torch.sigmoid((s_neg - tau_safety) * t_safety_q)

        overflow = s_neg - tau_penalty
        # V8.2: penalty function selection
        if self.penalty_func == "linear":
            # 线性惩罚：max(0, overflow)，penalty_scale 控制整体强度
            smooth_penalty = F.relu(overflow) * self.penalty_scale
        elif self.penalty_func == "scaled_linear":
            # 缩放线性：将 overflow 归一化到 S_base 量级后再乘 penalty_scale
            # overflow 范围通常 0~0.05，S_base 范围通常 0.4~0.8
            # scale = mean(S_base) / mean(overflow|at-risk) 让惩罚能量与 S_base 同量级
            at_risk_tmp = overflow > 0
            if at_risk_tmp.any():
                mean_overflow = overflow[at_risk_tmp].mean()
                mean_sbase = s_base[at_risk_tmp].mean()
                if mean_overflow > 1e-8:
                    auto_scale = (mean_sbase / mean_overflow).item() * self.penalty_scale
                else:
                    auto_scale = self.penalty_scale
            else:
                auto_scale = self.penalty_scale
            smooth_penalty = F.relu(overflow) * auto_scale
        elif self.penalty_func == "quadratic":
            # 二次惩罚：max(0, overflow)² × scale，对高 S_neg 文档施加更强惩罚
            smooth_penalty = (F.relu(overflow) ** 2) * self.penalty_scale
        else:  # softplus (default)
            smooth_penalty = F.softplus(overflow)

        # V8: per-query α/β derivation from candidate document distribution
        if self.per_query_ab:
            at_risk_mask = overflow > 0
            safe_mask = ~at_risk_mask
            # Store at_risk_ratio for at_risk_comp / multi_signal modes
            self._current_at_risk_ratio = float(at_risk_mask.float().mean().item())
            num_at_risk = int(at_risk_mask.sum().item())
            num_safe = int(safe_mask.sum().item())
            # α_q: scale alignment on at-risk docs
            if at_risk_mask.any():
                E_sb_risk = s_base[at_risk_mask].mean()
                E_sp_risk = smooth_penalty[at_risk_mask].mean()
                alpha_q = float((E_sb_risk / E_sp_risk).item()) if E_sp_risk > 1e-8 else alpha
            else:
                alpha_q = alpha  # 无 at-risk 时用 fallback（惩罚为零，α 不影响）
            # β_q: scale alignment on safe docs
            if has_req and safe_mask.any():
                beta_q = self._derive_beta_q(
                    s_base[safe_mask], s_req[safe_mask], safety[safe_mask], beta
                )
            else:
                beta_q = beta
            # Clip for stability
            alpha_q = max(self.ab_clip_alpha[0], min(alpha_q, self.ab_clip_alpha[1]))
            beta_q = max(self.ab_clip_beta[0], min(beta_q, self.ab_clip_beta[1]))
            self._per_query_alphas.append(alpha_q)
            self._per_query_betas.append(beta_q)
            # V8: record detailed per-query statistics for analysis
            s_reward_eff = (s_req if has_req else torch.zeros_like(s_base)) * safety
            self._per_query_stats.append({
                "q_idx": self._current_q_idx,
                "has_req": has_req,
                "has_neg": has_neg,
                "cos_qbase_qneg": cos_qbase_qneg,
                "tau_penalty": tau_penalty,
                "alpha_q": alpha_q,
                "beta_q": beta_q,
                "at_risk_ratio": self._current_at_risk_ratio,
                "num_at_risk": num_at_risk,
                "num_safe": num_safe,
                "num_candidates": int(s_base.numel()),
                "s_base_mean": float(s_base.mean().item()),
                "s_base_max": float(s_base.max().item()),
                "s_base_std": float(s_base.std().item()) if s_base.numel() > 1 else 0.0,
                "s_req_mean": float(s_req.mean().item()) if has_req else 0.0,
                "s_req_max": float(s_req.max().item()) if has_req else 0.0,
                "s_neg_mean": float(s_neg.mean().item()),
                "s_neg_max": float(s_neg.max().item()),
                "s_reward_mean": float(s_reward_eff.mean().item()),
                "safety_mean": float(safety.mean().item()),
                "gap_sbase_sreward": float(s_base.mean().item() - s_reward_eff.mean().item()),
                "t_safety_q": float(self._current_t_safety_q),  # V8.3: 自适应 t_safety
                "tau_safety": float(tau_safety) if self.safety_tau_mode != "off" else 0.0,
            })
            alpha = alpha_q
            beta = beta_q

        raw_penalty = alpha * smooth_penalty

        s_req_eff = s_req if has_req else torch.zeros_like(s_base)

        if self.safety_tau_mode == "req_gated":
            # S_req 门控：高 S_req 文档（强相关）safety→1，低 S_req 文档保持原 safety
            gate_req = torch.sigmoid((s_req_eff - self.req_threshold) * self.t_safety)
            safety = safety * (1.0 - gate_req) + gate_req
            s_final = s_base + beta * s_req_eff * safety - raw_penalty
        elif self.safety_tau_mode == "req_thresh":
            # 奖励项加 Softplus 阈值：β × Softplus(S_req - τ_req) × safety
            reward = beta * F.softplus(s_req_eff - self.req_threshold) * safety
            s_final = s_base + reward - raw_penalty
        else:
            s_final = s_base + beta * s_req_eff * safety - raw_penalty

        avg_penalty = float(raw_penalty.mean().item())
        return s_final, avg_penalty

    def compute_deir_dual_v2_scores(
        self,
        S_base: torch.Tensor,
        S_req: torch.Tensor,
        S_neg: torch.Tensor,
        cos_qbase_qneg: torch.Tensor,
        has_req_mask: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """重写以跟踪 _current_q_idx（供解耦 safety 取原始 cos）。"""
        S_final = S_base.clone()
        penalty_scores = torch.zeros(len(query_ids), device=S_base.device)

        for q_idx, qid in enumerate(query_ids):
            self._current_q_idx = q_idx
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            idx_tensor = torch.tensor(cand_indices, device=S_base.device, dtype=torch.long)
            s_b = S_base[q_idx].index_select(0, idx_tensor)
            s_r = S_req[q_idx].index_select(0, idx_tensor)
            s_n = S_neg[q_idx].index_select(0, idx_tensor)

            has_req = bool(has_req_mask[q_idx].item() > 0)
            has_neg = bool(has_neg_mask[q_idx].item() > 0)
            cos_val = float(cos_qbase_qneg[q_idx].item())

            s_final_local, avg_penalty = self._score_query_dual_v2(
                s_base=s_b,
                s_req=s_r,
                s_neg=s_n,
                cos_qbase_qneg=cos_val,
                has_req=has_req,
                has_neg=has_neg,
                alpha=alpha,
                beta=beta,
                delta=delta,
            )

            s_final_local = s_final_local.to(dtype=S_final.dtype)
            S_final[q_idx, idx_tensor] = s_final_local
            penalty_scores[q_idx] = avg_penalty

        return S_final, penalty_scores

    def run(
        self,
        alphas: Optional[List[float]] = None,
        betas: Optional[List[float]] = None,
        deltas: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """重写 run：在 cos_qbase_qneg 计算后注入 tau_anchor，其余完全一致。"""
        logger.info("=" * 60)
        logger.info("🚀 开始 Safe-Anchor-Threshold 评测 (DeIR-Dual V2)")
        logger.info("=" * 60)

        start_time = time.time()

        alpha_list = alphas if alphas else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        beta_list = betas if betas else [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        delta_list = deltas if deltas else [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
        logger.info(f"🔬 网格搜索规模: {total_trials} 组")
        logger.info(f"   α: {alpha_list}, β: {beta_list}, δ: {delta_list}")
        logger.info(f"   anchor_stat={self.anchor_stat}, anchor_delta={self.anchor_delta}, anchor_mix_mode={self.anchor_mix_mode}")

        dual_data = self.load_dual_queries()
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ 使用缓存文档向量 ({len(cached_doc_ids)} 个)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("⚠️ 缓存文档ID不匹配，重新编码...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        # 构建 og / changed 查询列表（与基类完全一致）
        def build_qids(qid_dict, q_raw):
            ids, base_list, req_list, neg_list, req_mask, neg_mask = [], [], [], [], [], []
            for qid in qid_dict.keys():
                ids.append(qid)
                raw = q_raw.get(qid, ("", ""))
                query_text, instruction = raw[0], raw[1]
                q_base = f"{query_text} {instruction}".strip() if query_text else qid_dict.get(qid, "")
                base_list.append(q_base)
                d = dual_data.get(qid, {})
                q_plus = d.get("q_plus", "")
                q_minus = d.get("q_minus", "")
                req_list.append(q_plus if not self._is_none_query(q_plus) else "")
                neg_list.append(q_minus if not self._is_none_query(q_minus) else "")
                req_mask.append(0.0 if self._is_none_query(q_plus) else 1.0)
                neg_mask.append(0.0 if self._is_none_query(q_minus) else 1.0)
            return ids, base_list, req_list, neg_list, req_mask, neg_mask

        (query_ids_og, q_base_list_og, q_req_list_og, q_neg_list_og,
         has_req_og, has_neg_og) = build_qids(q_og, q_raw_og)
        (query_ids_changed, q_base_list_ch, q_req_list_ch, q_neg_list_ch,
         has_req_ch, has_neg_ch) = build_qids(q_changed, q_raw_changed)

        has_req_mask_og = torch.tensor(has_req_og, dtype=torch.float32)
        has_neg_mask_og = torch.tensor(has_neg_og, dtype=torch.float32)
        has_req_mask_ch = torch.tensor(has_req_ch, dtype=torch.float32)
        has_neg_mask_ch = torch.tensor(has_neg_ch, dtype=torch.float32)

        # 编码查询时使用 batch_size=1，消除 batch padding 导致的 float16 编码噪声
        # 确保相同输入永远产生相同输出，无论其他 query 文本如何变化
        _orig_batch_size = self.batch_size
        self.batch_size = 1
        logger.info("📊 编码 OG Q_base/Q_req/Q_neg (batch_size=1, 消除编码噪声)...")
        q_base_emb_og = self._encode_queries(q_base_list_og)
        q_req_emb_og = self._encode_queries(q_req_list_og)
        q_neg_emb_og = self._encode_queries(q_neg_list_og)
        logger.info("📊 编码 Changed Q_base/Q_req/Q_neg (batch_size=1, 消除编码噪声)...")
        q_base_emb_ch = self._encode_queries(q_base_list_ch)
        q_req_emb_ch = self._encode_queries(q_req_list_ch)
        q_neg_emb_ch = self._encode_queries(q_neg_list_ch)
        self.batch_size = _orig_batch_size

        device = self.retriever.doc_embeddings.device
        q_base_emb_og = q_base_emb_og.to(device)
        q_req_emb_og = q_req_emb_og.to(device)
        q_neg_emb_og = q_neg_emb_og.to(device)
        q_base_emb_ch = q_base_emb_ch.to(device)
        q_req_emb_ch = q_req_emb_ch.to(device)
        q_neg_emb_ch = q_neg_emb_ch.to(device)
        has_req_mask_og = has_req_mask_og.to(device)
        has_neg_mask_og = has_neg_mask_og.to(device)
        has_req_mask_ch = has_req_mask_ch.to(device)
        has_neg_mask_ch = has_neg_mask_ch.to(device)

        S_base_og = torch.matmul(q_base_emb_og, self.retriever.doc_embeddings.T)
        S_req_og = torch.matmul(q_req_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = torch.matmul(q_neg_emb_og, self.retriever.doc_embeddings.T) * has_neg_mask_og.unsqueeze(1)
        S_base_ch = torch.matmul(q_base_emb_ch, self.retriever.doc_embeddings.T)
        S_req_ch = torch.matmul(q_req_emb_ch, self.retriever.doc_embeddings.T)
        S_neg_ch = torch.matmul(q_neg_emb_ch, self.retriever.doc_embeddings.T) * has_neg_mask_ch.unsqueeze(1)

        # ---- 原始 Cos(Q_base, Q_neg) ----
        cos_qbase_qneg_og = F.cosine_similarity(q_base_emb_og, q_neg_emb_og, dim=1)
        cos_qbase_qneg_changed = F.cosine_similarity(q_base_emb_ch, q_neg_emb_ch, dim=1)
        # 保存原始 cos，供解耦 safety 模式使用（在 threshold_base 覆盖前）
        self._cos_qbase_qneg_orig = cos_qbase_qneg_changed.clone()

        # ============ 关键注入：Safe Anchor Threshold ============
        anchors_map = load_safe_anchors(self.safe_anchors_path) if self.safe_anchors_path else {}
        tau_anchor, per_query_anchor_scores = compute_safe_anchor_threshold(
            q_neg_emb_ch, query_ids_changed, anchors_map,
            encoder_fn=self._encode_queries, stat=self.anchor_stat,
        )
        anchor_mask = tau_anchor > float("-inf")
        # 对有 anchor 的 query 用 tau_anchor；无 anchor 的回退到 cos_qbase_qneg
        # mix_mode 控制 anchor 与 cos_qbase_qneg 的融合方式
        if self.anchor_mix_mode == "replace":
            threshold_base = torch.where(
                anchor_mask, tau_anchor, cos_qbase_qneg_changed,
            )
        elif self.anchor_mix_mode == "min":
            threshold_base = torch.where(
                anchor_mask, torch.minimum(tau_anchor, cos_qbase_qneg_changed), cos_qbase_qneg_changed,
            )
        elif self.anchor_mix_mode == "max":
            threshold_base = torch.where(
                anchor_mask, torch.maximum(tau_anchor, cos_qbase_qneg_changed), cos_qbase_qneg_changed,
            )
        elif self.anchor_mix_mode == "mean":
            mixed = 0.5 * (tau_anchor + cos_qbase_qneg_changed)
            threshold_base = torch.where(anchor_mask, mixed, cos_qbase_qneg_changed)
        else:
            raise ValueError(f"未知 anchor_mix_mode: {self.anchor_mix_mode}")
        # 加上 anchor_delta（作为可选偏移，默认 0）
        threshold_base = threshold_base + self.anchor_delta
        self._tau_anchor_override = threshold_base

        logger.info(f"   Safe-anchor 覆盖 query 数: {anchor_mask.sum().item()} / {len(query_ids_changed)}")
        if anchor_mask.any():
            ta = tau_anchor[anchor_mask]
            logger.info(f"   tau_anchor 统计 (覆盖query): min={ta.min().item():.4f}, "
                        f"max={ta.max().item():.4f}, mean={ta.mean().item():.4f}")
        logger.info(f"   Cos(Q_base,Q_neg) 统计: min={cos_qbase_qneg_changed.min().item():.4f}, "
                    f"max={cos_qbase_qneg_changed.max().item():.4f}, "
                    f"mean={cos_qbase_qneg_changed.mean().item():.4f}")

        # ---- Debug logging ----
        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        for q_idx, qid in enumerate(query_ids_changed):
            if not bool(anchor_mask[q_idx].item()):
                continue
            cand = candidates.get(qid.replace("-changed", ""), [])
            cand_idx = [doc_id_to_col_idx[d] for d in cand if d in doc_id_to_col_idx]
            s_neg_cand = S_neg_ch[q_idx, cand_idx] if cand_idx else torch.tensor([], device=device)
            self._debug_records.append({
                "query_id": qid,
                "q_neg": q_neg_list_ch[q_idx],
                "safe_anchors": anchors_map.get(qid, []),
                "anchor_neg_scores": per_query_anchor_scores[q_idx],
                "tau_anchor": float(tau_anchor[q_idx].item()),
                "cos_qbase_qneg": float(cos_qbase_qneg_changed[q_idx].item()),
                "threshold_base_used": float(threshold_base[q_idx].item()),
                "candidate_s_neg_min": float(s_neg_cand.min().item()) if s_neg_cand.numel() else None,
                "candidate_s_neg_max": float(s_neg_cand.max().item()) if s_neg_cand.numel() else None,
                "num_penalized_docs": int((s_neg_cand > threshold_base[q_idx]).sum().item()) if s_neg_cand.numel() else 0,
            })

        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        best_per_query_stats: List[Dict[str, Any]] = []
        all_results: List[Dict[str, Any]] = []
        trial_idx = 0

        for alpha in alpha_list:
            for beta in beta_list:
                for delta in delta_list:
                    trial_idx += 1
                    # V8: reset per-query tracking
                    self._per_query_alphas = []
                    self._per_query_betas = []
                    self._per_query_stats = []
                    # 用 threshold_base（已含 anchor_delta）替代 cos_qbase_qneg
                    S_final_changed, penalty_scores = self.compute_deir_dual_v2_scores(
                        S_base=S_base_ch,
                        S_req=S_req_ch,
                        S_neg=S_neg_ch,
                        cos_qbase_qneg=threshold_base,
                        has_req_mask=has_req_mask_ch,
                        has_neg_mask=has_neg_mask_ch,
                        query_ids=query_ids_changed,
                        qid_to_candidate_indices=qid_to_candidate_indices,
                        alpha=alpha,
                        beta=beta,
                        delta=delta,
                    )

                    results_og = self._extract_results(S_base_og, query_ids_og, candidates)
                    results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)

                    evaluator = FollowIREvaluator(self.task_name)
                    metrics = evaluator.evaluate(results_og, results_changed)

                    p_mrr = metrics.get("p-MRR", 0.0)
                    og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                    changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                    changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    changed_mrr = metrics.get("changed", {}).get("mrr_at_10", 0.0)

                    logger.info(
                        "[%d/%d] α=%.1f, β=%.1f, δ=%.2f: p-MRR=%.4f, OG_MAP=%.4f, CH_MAP=%.4f, CH_MRR=%.4f",
                        trial_idx, total_trials, alpha, beta, delta,
                        p_mrr, og_map, changed_map, changed_mrr,
                    )

                    all_results.append({
                        "alpha": alpha, "beta": beta, "delta": delta,
                        "t_safety": self.t_safety,
                        "anchor_stat": self.anchor_stat,
                        "anchor_delta": self.anchor_delta,
                        "anchor_mix_mode": self.anchor_mix_mode,
                        "per_query_ab": self.per_query_ab,
                        "p-MRR": p_mrr,
                        "og_MAP@1000": og_map,
                        "og_nDCG@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
                        "changed_MAP@1000": changed_map,
                        "changed_nDCG@5": changed_ndcg,
                        "changed_MRR@10": changed_mrr,
                        "avg_penalty": float(penalty_scores.mean().item()),
                    })

                    # V8: log per-query α/β statistics
                    if self.per_query_ab and self._per_query_alphas:
                        import numpy as np
                        a_arr = np.array(self._per_query_alphas)
                        b_arr = np.array(self._per_query_betas)
                        logger.info(
                            "   V8 per-query α: mean=%.2f, std=%.2f, min=%.2f, max=%.2f | "
                            "β: mean=%.2f, std=%.2f, min=%.2f, max=%.2f (n=%d)",
                            a_arr.mean(), a_arr.std(), a_arr.min(), a_arr.max(),
                            b_arr.mean(), b_arr.std(), b_arr.min(), b_arr.max(),
                            len(a_arr),
                        )

                    composite = p_mrr + changed_map + changed_ndcg
                    if best_metrics is None or composite > (
                        best_metrics.get("p-MRR", 0.0)
                        + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                        + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    ):
                        best_metrics = metrics
                        best_params = {"alpha": alpha, "beta": beta, "delta": delta}
                        best_results_og = results_og
                        best_results_changed = results_changed
                        # V8: save per-query stats for the best trial
                        best_per_query_stats = list(self._per_query_stats)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("📊 Safe-Anchor-Threshold 搜索完成")
        logger.info("   最佳参数: %s", best_params)
        logger.info("   p-MRR: %.4f", best_metrics.get("p-MRR", 0.0))
        logger.info("   OG MAP@1000: %.4f", best_metrics.get("original", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MAP@1000: %.4f", best_metrics.get("changed", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MRR@10: %.4f", best_metrics.get("changed", {}).get("mrr_at_10", 0.0))
        logger.info("   耗时: %.1f 秒", elapsed)
        logger.info("=" * 60)

        # 保存结果
        os.makedirs(self.output_dir, exist_ok=True)
        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "Safe-Anchor-Threshold",
            "dual_queries_source": self.dual_queries_path,
            "safe_anchors_source": self.safe_anchors_path,
            "anchor_stat": self.anchor_stat,
            "anchor_delta": self.anchor_delta,
            "anchor_mix_mode": self.anchor_mix_mode,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
                "full_scores": best_metrics.get("full_scores", {}),
            },
        }
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "debug_anchor_logs.json"), "w", encoding="utf-8") as f:
            json.dump(self._debug_records, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_og.json"), "w", encoding="utf-8") as f:
            json.dump(best_results_og, f, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_changed.json"), "w", encoding="utf-8") as f:
            json.dump(best_results_changed, f, ensure_ascii=False)
        # V8: save per-query statistics for analysis
        if best_per_query_stats:
            with open(os.path.join(self.output_dir, "per_query_stats.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "beta_derive_mode": self.beta_derive_mode,
                    "safety_tau_mode": self.safety_tau_mode,
                    "anchor_delta": self.anchor_delta,
                    "t_safety": self.t_safety,
                    "adaptive_t_safety": self.adaptive_t_safety,
                    "adaptive_safety_threshold": self.adaptive_safety_threshold,
                    "adaptive_t_safety_min": self.adaptive_t_safety_min,
                    "best_params": best_params,
                    "query_ids_changed": query_ids_changed,
                    "per_query_stats": best_per_query_stats,
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 V8 per-query stats 已保存: {len(best_per_query_stats)} queries")
        logger.info(f"💾 结果已保存: {self.output_dir}")

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "debug_records": self._debug_records,
            "elapsed": elapsed,
        }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run_safe_anchor_experiment(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    safe_anchors_path: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    anchor_stat: str = "max",
    anchor_delta: float = 0.0,
    anchor_mix_mode: str = "replace",
    safety_tau_mode: str = "coupled",
    safety_margin: float = 0.0,
    req_threshold: float = 0.25,
    t_safety: float = 20.0,
    device: str = "auto",
    batch_size: int = 64,
    use_cache: bool = True,
    per_query_ab: bool = False,
    beta_derive_mode: str = "mean",
    penalty_tau_mode: str = "anchor",
    penalty_percentile: float = 90.0,
    penalty_func: str = "softplus",
    penalty_scale: float = 1.0,
    adaptive_t_safety: bool = False,
    adaptive_safety_threshold: float = 0.92,
    adaptive_t_safety_min: float = 3.0,
) -> Dict[str, Any]:
    if not dual_queries_path:
        raise ValueError("dual_queries_path 不能为空")
    if not safe_anchors_path:
        raise ValueError("safe_anchors_path 不能为空")
    if not output_dir:
        output_dir = f"results/safe_anchor/{task_name}"

    engine = SafeAnchorDeIREvaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        safe_anchors_path=safe_anchors_path,
        anchor_stat=anchor_stat,
        anchor_delta=anchor_delta,
        anchor_mix_mode=anchor_mix_mode,
        safety_tau_mode=safety_tau_mode,
        safety_margin=safety_margin,
        req_threshold=req_threshold,
        t_safety=t_safety,
        per_query_ab=per_query_ab,
        beta_derive_mode=beta_derive_mode,
        penalty_tau_mode=penalty_tau_mode,
        penalty_percentile=penalty_percentile,
        penalty_func=penalty_func,
        penalty_scale=penalty_scale,
        adaptive_t_safety=adaptive_t_safety,
        adaptive_safety_threshold=adaptive_safety_threshold,
        adaptive_t_safety_min=adaptive_t_safety_min,
        device=device,
        batch_size=batch_size,
        use_cache=use_cache,
    )
    return engine.run(alphas=alphas, betas=betas, deltas=deltas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Safe Anchor Threshold 实验")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--safe_anchors_path", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="1.0")
    parser.add_argument("--betas", type=str, default="1.5")
    parser.add_argument("--deltas", type=str, default="0.0")
    parser.add_argument("--anchor_stat", type=str, default="max", choices=["max", "mean", "min"])
    parser.add_argument("--anchor_delta", type=float, default=0.0)
    parser.add_argument("--anchor_mix_mode", type=str, default="replace",
                        choices=["replace", "min", "max", "mean"])
    parser.add_argument("--safety_tau_mode", type=str, default="coupled",
                        choices=["coupled", "cos_delta", "off", "add_margin", "req_gated", "req_thresh"],
                        help="safety 项的 τ 解耦模式")
    parser.add_argument("--safety_margin", type=float, default=0.0,
                        help="add_margin 模式下 τ_safety = τ_penalty + margin")
    parser.add_argument("--req_threshold", type=float, default=0.25,
                        help="req_gated/req_thresh 模式下的 S_req 阈值 τ_req")
    parser.add_argument("--t_safety", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--per_query_ab", type=str, default="false",
                        help="V8: 启用 per-query test-time α/β 推导")
    parser.add_argument("--beta_derive_mode", type=str, default="mean",
                        choices=["mean", "std", "range", "topk_mean", "max_mean", "p90_mean", "peak_comp", "max_comp",
                                 "cubed_comp", "p95_comp", "topk_comp", "req_gap_comp", "variance_comp",
                                 "at_risk_comp", "multi_signal", "quartic_comp", "quartic_gap"],
                        help="V8: β 推导模式")
    parser.add_argument("--penalty_tau_mode", type=str, default="anchor",
                        choices=["anchor", "s_neg_pctl", "hybrid", "hybrid_floor"],
                        help="V8.1: 惩罚阈值模式 (anchor=传统safe-anchor, s_neg_pctl=S_neg分位数, hybrid=min两者, hybrid_floor=带cos下限)")
    parser.add_argument("--penalty_percentile", type=float, default=90.0,
                        help="V8.1: s_neg_pctl/hybrid 模式下的 S_neg 分位数 (默认 P90)")
    parser.add_argument("--penalty_func", type=str, default="softplus",
                        choices=["softplus", "linear", "scaled_linear", "quadratic"],
                        help="V8.2: 惩罚函数 (softplus=默认, linear=线性, scaled_linear=自动缩放线性, quadratic=二次)")
    parser.add_argument("--penalty_scale", type=float, default=1.0,
                        help="V8.2: linear/scaled_linear/quadratic 模式下的惩罚缩放因子")
    parser.add_argument("--adaptive_t_safety", type=str, default="false",
                        help="V8.3: 启用 per-query 自适应 t_safety")
    parser.add_argument("--adaptive_safety_threshold", type=float, default=0.92,
                        help="V8.3: safety_mean 低于此值时触发降 t_safety (推荐 0.92)")
    parser.add_argument("--adaptive_t_safety_min", type=float, default=3.0,
                        help="V8.3: 自适应 t_safety 下限保护")
    args = parser.parse_args()

    result = run_safe_anchor_experiment(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        safe_anchors_path=args.safe_anchors_path,
        alphas=[float(x) for x in args.alphas.split(",") if x.strip()],
        betas=[float(x) for x in args.betas.split(",") if x.strip()],
        deltas=[float(x) for x in args.deltas.split(",") if x.strip()],
        anchor_stat=args.anchor_stat,
        anchor_delta=args.anchor_delta,
        anchor_mix_mode=args.anchor_mix_mode,
        safety_tau_mode=args.safety_tau_mode,
        safety_margin=args.safety_margin,
        req_threshold=args.req_threshold,
        t_safety=args.t_safety,
        device=args.device,
        batch_size=args.batch_size,
        use_cache=args.use_cache.lower() == "true",
        per_query_ab=args.per_query_ab.lower() == "true",
        beta_derive_mode=args.beta_derive_mode,
        penalty_tau_mode=args.penalty_tau_mode,
        penalty_percentile=args.penalty_percentile,
        penalty_func=args.penalty_func,
        penalty_scale=args.penalty_scale,
        adaptive_t_safety=args.adaptive_t_safety.lower() == "true",
        adaptive_safety_threshold=args.adaptive_safety_threshold,
        adaptive_t_safety_min=args.adaptive_t_safety_min,
    )

    m = result["best_metrics"]
    print(f"\n最终 p-MRR: {m.get('p-MRR', 0.0):.4f}")
    print(f"最佳参数: {result['best_params']}")
