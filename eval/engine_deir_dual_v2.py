"""
DeIR-Dual V2: 免训练奖惩双轨制评估引擎

##在conda环境dsclr中运行

## 评测相关：两个重要指标
### target_avg 定义（重要！）
**target_avg = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3**
除非用户特别说明，否则"平均指标"均指此定义，而非三个数据集 MAP 的简单平均。

### pMRR: 衡量指令敏感度

核心公式:
    τ = Cos(Q_base, Q_neg) + δ                           (动态语义阈值)
    safety = 1 - sigmoid((S_neg - τ) × T_safety)          (安全门控)
    penalty = α × Softplus(S_neg - τ)                      (平滑惩罚)
    S_final = S_base + β × S_req × safety - penalty        (条件性奖励)

V1 → V2 三大升级:
    1. 动态语义阈值: τ = Cos(Q_base, Q_neg) + δ (替代 mean(S_neg) + δ)
    2. Softplus 平滑惩罚: 替代 ReLU 硬截断
    3. 条件性奖励: safety 门控防止踩雷文档被推高
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator
from eval.residual_boundary import compute_background_residual_boundary

logger = logging.getLogger(__name__)


class DeIRDualV2Evaluator(DSCLREvaluatorEngine):
    """DeIR-Dual V2 免训练奖惩双轨制评估引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        t_safety: float = 20.0,
        boundary_mode: str = "semantic",
        residual_margin_scale: float = 1.0,
        safety_kappa: float = 0.0,
        beta_raw: bool = False,
        per_query_ab: bool = False,
        beta_derive_mode: str = "max_mean",
        ab_clip_alpha: Tuple[float, float] = (0.05, 5.0),
        ab_clip_beta: Tuple[float, float] = (0.05, 5.0),
        ablation_mode: str = "full",
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        self.t_safety = t_safety
        self.boundary_mode = boundary_mode
        self.residual_margin_scale = residual_margin_scale
        self.safety_kappa = safety_kappa
        self.per_query_ab = per_query_ab
        self.beta_raw = beta_raw
        self.beta_derive_mode = beta_derive_mode
        self.ab_clip_alpha = ab_clip_alpha
        self.ab_clip_beta = ab_clip_beta
        self.ablation_mode = ablation_mode
        self._per_query_alphas: List[float] = []
        self._per_query_betas: List[float] = []
        self._per_query_stats: List[Dict[str, Any]] = []
        self._current_qid: str = ""
        if device == "auto":
            import torch as _torch
            try:
                _torch.cuda._lazy_init()
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        kwargs.setdefault("device", device)
        kwargs.setdefault("batch_size", kwargs.pop("batch_size", 64))
        super().__init__(model_name, task_name, output_dir, **kwargs)

        logger.info("🏛️ DeIR-Dual V2 奖惩双轨制模式已启用")
        logger.info(f"📁 Dual queries 数据路径: {self.dual_queries_path}")
        logger.info(f"⚙️ T_safety=%.1f", self.t_safety)
        logger.info(
            "🧪 boundary_mode=%s, residual_margin_scale=%.3f, per_query_ab=%s, beta_derive_mode=%s",
            self.boundary_mode,
            self.residual_margin_scale,
            self.per_query_ab,
            self.beta_derive_mode,
        )

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path:
            raise ValueError("必须提供 dual_queries_path")

        dual_data: Dict[str, Dict[str, Any]] = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item

        logger.info(f"✅ 加载 dual queries 数据: {len(dual_data)} 条")
        return dual_data

    def _build_candidate_indices(
        self,
        candidates: Dict[str, List[str]],
        doc_id_to_col_idx: Dict[str, int],
    ) -> Dict[str, List[int]]:
        qid_to_indices: Dict[str, List[int]] = {}
        for qid, doc_ids in candidates.items():
            indices = [doc_id_to_col_idx[d] for d in doc_ids if d in doc_id_to_col_idx]
            qid_to_indices[qid] = indices
        return qid_to_indices

    def _is_none_query(self, text: str) -> bool:
        if not text:
            return True
        t = str(text).strip().upper()
        return t in ("[NONE]", "NONE", "NULL", "N/A", "")

    def _clip(self, value: float, bounds: Tuple[float, float]) -> float:
        return max(bounds[0], min(value, bounds[1]))

    def _derive_beta_q(
        self,
        s_base_safe: torch.Tensor,
        s_req_safe: torch.Tensor,
        safety_safe: torch.Tensor,
        beta_fallback: float,
    ) -> float:
        """V8-style per-query beta derivation from safe candidates."""
        s_reward = s_req_safe * safety_safe
        if s_base_safe.numel() == 0 or s_reward.numel() == 0:
            return beta_fallback

        mode = self.beta_derive_mode
        if mode == "mean":
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        if mode == "topk_mean":
            k = min(20, s_base_safe.numel())
            topk_idx = torch.topk(s_base_safe, k).indices
            mean_b = s_base_safe[topk_idx].mean()
            mean_r = s_reward[topk_idx].mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        if mode == "p90_mean":
            p90_b = torch.quantile(s_base_safe.float(), 0.9)
            mean_r = s_reward.mean()
            return float((p90_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        if mode == "max_comp":
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                return float((max_b * max_b / (mean_b * mean_r)).item())
            return beta_fallback
        if mode == "quartic_gap":
            max_b = s_base_safe.max()
            mean_b = s_base_safe.mean()
            mean_r = s_reward.mean()
            if mean_b > 1e-8 and mean_r > 1e-8:
                base_comp = float((max_b ** 4 / (mean_b ** 2 * mean_r ** 2)).item())
                gap_factor = 1.0 + abs(float(mean_b.item()) - float(mean_r.item())) / float(mean_b.item())
                return base_comp * gap_factor
            return beta_fallback

        # max_mean: peak-calibrated V8 default.
        max_b = s_base_safe.max()
        mean_r = s_reward.mean()
        return float((max_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback

    def _derive_beta_q_raw(
        self,
        s_base_safe: torch.Tensor,
        s_req_safe: torch.Tensor,
        beta_fallback: float,
    ) -> float:
        """V8.6 beta derivation WITHOUT safety — β only handles scale alignment.

        β establishes the raw enhancement scale: β × S_req ≈ max(S_base).
        Safety independently modulates: β × S_req × safety.
        This avoids β inflating to compensate for safety < 1, which would
        partially undo safety's suppression of documents with partial negative evidence.
        """
        if s_base_safe.numel() == 0 or s_req_safe.numel() == 0:
            return beta_fallback

        mode = self.beta_derive_mode
        s_reward_raw = s_req_safe  # No safety multiplication

        if mode == "mean":
            mean_b = s_base_safe.mean()
            mean_r = s_reward_raw.mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback
        if mode == "topk_mean":
            k = min(20, s_base_safe.numel())
            topk_idx = torch.topk(s_base_safe, k).indices
            mean_b = s_base_safe[topk_idx].mean()
            mean_r = s_reward_raw[topk_idx].mean()
            return float((mean_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback

        # max_mean (default): peak-calibrated, raw S_req only
        max_b = s_base_safe.max()
        mean_r = s_reward_raw.mean()
        return float((max_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback

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
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """DeIR-Dual V2 核心打分函数。

        τ = Cos(Q_base, Q_neg) + δ
        safety = 1 - sigmoid((S_neg - τ) × T_safety)
        penalty = α × Softplus(S_neg - τ)
        S_final = S_base + β × S_req × safety - penalty
        """
        stats: Dict[str, Any] = {}
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            if self.per_query_ab and has_req and s_base.numel() > 0:
                safety = torch.ones_like(s_base)
                if self.beta_raw:
                    beta = self._clip(
                        self._derive_beta_q_raw(s_base, s_req, beta),
                        self.ab_clip_beta,
                    )
                else:
                    beta = self._clip(
                        self._derive_beta_q(s_base, s_req, safety, beta),
                        self.ab_clip_beta,
                    )
                self._per_query_betas.append(beta)
                stats.update({
                    "qid": self._current_qid,
                    "has_req": has_req,
                    "has_neg": has_neg,
                    "alpha_q": alpha,
                    "beta_q": beta,
                    "num_candidates": int(s_base.numel()),
                    "at_risk_ratio": 0.0,
                })
                self._per_query_stats.append(stats)
            s_final = s_base + beta * s_req_eff
            if self.ablation_mode == "base_only":
                s_final = s_base.clone()
            elif self.ablation_mode == "linear":
                s_final = s_base + beta * s_req_eff  # no neg, same as no_safety
            stats["ablation_mode"] = self.ablation_mode
            return s_final, 0.0, stats

        if self.boundary_mode == "residual_bg":
            boundary = compute_background_residual_boundary(
                s_base=s_base,
                s_neg=s_neg,
                cos_qbase_qneg=cos_qbase_qneg,
                margin_scale=self.residual_margin_scale,
            )
            # 惩罚项：残差超出 MAD 阈值的部分，不加 delta
            overflow = boundary.overflow  # R_neg - m_q
            smooth_penalty = F.softplus(overflow)
            stats.update(boundary.stats)
            stats["boundary_mode"] = self.boundary_mode
            # safety gate：基于残差的 MAD 归一化 safety
            # κ > 0: safety = 1 - sigmoid(R_neg / MAD × κ)
            #   可解释性：R_neg/MAD 度量残差是"几个 MAD"，κ 控制过渡锐度
            #   R_neg = 0 → safety = 0.5（负向证据恰等于背景泄漏预期）
            #   R_neg = MAD → safety = 1 - sigmoid(κ)
            # κ = 0: 回退到传统 τ = cos(Q_base, Q_neg) + δ
            tau = cos_qbase_qneg + delta
            if self.safety_kappa > 0 and boundary.mad > 1e-8:
                safety = 1.0 - torch.sigmoid(
                    boundary.residual / boundary.mad * self.safety_kappa
                )
                stats["safety_mode"] = "residual_mad"
                stats["safety_kappa"] = self.safety_kappa
                stats["residual_mad"] = boundary.mad
            else:
                safety = 1.0 - torch.sigmoid((s_neg - tau) * self.t_safety)
                stats["safety_mode"] = "semantic_tau"
        else:
            tau = cos_qbase_qneg + delta
            overflow = s_neg - tau
            smooth_penalty = F.softplus(overflow)
            stats.update({
                "boundary_mode": self.boundary_mode,
                "boundary_tau": float(tau),
                "boundary_at_risk_ratio": float((overflow > 0).float().mean().item()),
            })
            safety = 1.0 - torch.sigmoid(overflow * self.t_safety)

        if self.per_query_ab:
            at_risk_mask = overflow > 0
            safe_mask = ~at_risk_mask
            if at_risk_mask.any():
                mean_base_risk = s_base[at_risk_mask].mean()
                mean_penalty_risk = smooth_penalty[at_risk_mask].mean()
                if mean_penalty_risk > 1e-8:
                    alpha = float((mean_base_risk / mean_penalty_risk).item())
            if has_req and safe_mask.any():
                if self.beta_raw:
                    beta = self._derive_beta_q_raw(
                        s_base[safe_mask],
                        s_req[safe_mask],
                        beta,
                    )
                else:
                    beta = self._derive_beta_q(
                        s_base[safe_mask],
                        s_req[safe_mask],
                        safety[safe_mask],
                        beta,
                    )
            alpha = self._clip(alpha, self.ab_clip_alpha)
            beta = self._clip(beta, self.ab_clip_beta)
            self._per_query_alphas.append(alpha)
            self._per_query_betas.append(beta)
            stats.update({
                "qid": self._current_qid,
                "has_req": has_req,
                "has_neg": has_neg,
                "cos_qbase_qneg": cos_qbase_qneg,
                "alpha_q": alpha,
                "beta_q": beta,
                "num_candidates": int(s_base.numel()),
                "num_at_risk": int(at_risk_mask.sum().item()),
                "num_safe": int(safe_mask.sum().item()),
                "at_risk_ratio": float(at_risk_mask.float().mean().item()),
                "s_base_mean": float(s_base.mean().item()),
                "s_base_max": float(s_base.max().item()),
                "s_req_mean": float(s_req.mean().item()) if has_req else 0.0,
                "s_req_max": float(s_req.max().item()) if has_req else 0.0,
                "s_neg_mean": float(s_neg.mean().item()),
                "s_neg_max": float(s_neg.max().item()),
                "safety_mean": float(safety.mean().item()),
                "tau": float(tau),
            })
            self._per_query_stats.append(stats)

        raw_penalty = alpha * smooth_penalty

        s_req_eff = s_req if has_req else torch.zeros_like(s_base)

        # Ablation mode: control which components contribute to final score
        # full:       S_base + β·S_req·safety - penalty
        # base_only:  S_base
        # no_pos:     S_base - penalty
        # no_neg:     S_base + β·S_req·safety
        # no_safety:  S_base + β·S_req - penalty  (safety=1)
        # linear:     S_base + β·S_req - α·S_neg   (linear fusion, no softplus/sigmoid)
        mode = self.ablation_mode
        if mode == "base_only":
            s_final = s_base.clone()
        elif mode == "no_pos":
            s_final = s_base - raw_penalty
        elif mode == "no_neg":
            s_final = s_base + beta * s_req_eff * safety
        elif mode == "no_safety":
            s_final = s_base + beta * s_req_eff - raw_penalty
        elif mode == "linear":
            s_final = s_base + beta * s_req_eff - alpha * s_neg
        else:  # full
            s_final = s_base + beta * s_req_eff * safety - raw_penalty

        stats["ablation_mode"] = mode
        avg_penalty = float(raw_penalty.mean().item())
        return s_final, avg_penalty, stats

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
        S_final = S_base.clone()
        penalty_scores = torch.zeros(len(query_ids), device=S_base.device)

        for q_idx, qid in enumerate(query_ids):
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
            self._current_qid = qid

            s_final_local, avg_penalty, _stats = self._score_query_dual_v2(
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
        logger.info("=" * 60)
        logger.info("🚀 开始 DeIR-Dual V2 奖惩双轨制评测")
        logger.info("=" * 60)

        start_time = time.time()

        alpha_list = alphas if alphas else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        beta_list = betas if betas else [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        delta_list = deltas if deltas else [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
        logger.info(f"🔬 DeIR-Dual V2 3D 网格搜索规模: {total_trials} 组")
        logger.info(f"   α 范围: {alpha_list}")
        logger.info(f"   β 范围: {beta_list}")
        logger.info(f"   δ 范围: {delta_list}")
        logger.info(f"   T_safety=%.1f", self.t_safety)

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

        query_ids_og: List[str] = []
        q_base_list_og: List[str] = []
        q_req_list_og: List[str] = []
        q_neg_list_og: List[str] = []
        has_req_mask_og_list: List[float] = []
        has_neg_mask_og_list: List[float] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_base = f"{query_text} {instruction}".strip() if query_text else q_og.get(qid, "")
            q_base_list_og.append(q_base)

            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            q_req_list_og.append(q_plus if not self._is_none_query(q_plus) else "")
            q_neg_list_og.append(q_minus if not self._is_none_query(q_minus) else "")
            has_req_mask_og_list.append(0.0 if self._is_none_query(q_plus) else 1.0)
            has_neg_mask_og_list.append(0.0 if self._is_none_query(q_minus) else 1.0)

        query_ids_changed: List[str] = []
        q_base_list_changed: List[str] = []
        q_req_list_changed: List[str] = []
        q_neg_list_changed: List[str] = []
        has_req_mask_changed_list: List[float] = []
        has_neg_mask_changed_list: List[float] = []

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_base = f"{query_text} {instruction}".strip() if query_text else q_changed.get(qid, "")
            q_base_list_changed.append(q_base)

            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            q_req_list_changed.append(q_plus if not self._is_none_query(q_plus) else "")
            q_neg_list_changed.append(q_minus if not self._is_none_query(q_minus) else "")
            has_req_mask_changed_list.append(0.0 if self._is_none_query(q_plus) else 1.0)
            has_neg_mask_changed_list.append(0.0 if self._is_none_query(q_minus) else 1.0)

        has_req_mask_og = torch.tensor(has_req_mask_og_list, dtype=torch.float32)
        has_neg_mask_og = torch.tensor(has_neg_mask_og_list, dtype=torch.float32)
        has_req_mask_changed = torch.tensor(has_req_mask_changed_list, dtype=torch.float32)
        has_neg_mask_changed = torch.tensor(has_neg_mask_changed_list, dtype=torch.float32)

        logger.info("📊 编码 OG Q_base...")
        q_base_emb_og = self._encode_queries(q_base_list_og)
        logger.info("📊 编码 OG Q_req (Q+)...")
        q_req_emb_og = self._encode_queries(q_req_list_og)
        logger.info("📊 编码 OG Q_neg (Q-)...")
        q_neg_emb_og = self._encode_queries(q_neg_list_og)

        logger.info("📊 编码 Changed Q_base...")
        q_base_emb_changed = self._encode_queries(q_base_list_changed)
        logger.info("📊 编码 Changed Q_req (Q+)...")
        q_req_emb_changed = self._encode_queries(q_req_list_changed)
        logger.info("📊 编码 Changed Q_neg (Q-)...")
        q_neg_emb_changed = self._encode_queries(q_neg_list_changed)

        device = self.retriever.doc_embeddings.device
        score_dtype = self.retriever.doc_embeddings.dtype
        q_base_emb_og = q_base_emb_og.to(device=device, dtype=score_dtype)
        q_req_emb_og = q_req_emb_og.to(device=device, dtype=score_dtype)
        q_neg_emb_og = q_neg_emb_og.to(device=device, dtype=score_dtype)
        q_base_emb_changed = q_base_emb_changed.to(device=device, dtype=score_dtype)
        q_req_emb_changed = q_req_emb_changed.to(device=device, dtype=score_dtype)
        q_neg_emb_changed = q_neg_emb_changed.to(device=device, dtype=score_dtype)
        has_req_mask_og = has_req_mask_og.to(device)
        has_neg_mask_og = has_neg_mask_og.to(device)
        has_req_mask_changed = has_req_mask_changed.to(device)
        has_neg_mask_changed = has_neg_mask_changed.to(device)

        logger.info("📊 计算 S_base / S_req / S_neg...")
        S_base_og = torch.matmul(q_base_emb_og, self.retriever.doc_embeddings.T)
        S_req_og = torch.matmul(q_req_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = torch.matmul(q_neg_emb_og, self.retriever.doc_embeddings.T) * has_neg_mask_og.unsqueeze(1)

        S_base_changed = torch.matmul(q_base_emb_changed, self.retriever.doc_embeddings.T)
        S_req_changed = torch.matmul(q_req_emb_changed, self.retriever.doc_embeddings.T)
        S_neg_changed = torch.matmul(q_neg_emb_changed, self.retriever.doc_embeddings.T) * has_neg_mask_changed.unsqueeze(1)

        logger.info("📊 计算 Cos(Q_base, Q_neg) 动态语义阈值...")
        cos_qbase_qneg_og = F.cosine_similarity(q_base_emb_og, q_neg_emb_og, dim=1)
        cos_qbase_qneg_changed = F.cosine_similarity(q_base_emb_changed, q_neg_emb_changed, dim=1)

        logger.info(f"   Cos(Q_base, Q_neg) 统计 (changed): min=%.4f, max=%.4f, mean=%.4f",
                     cos_qbase_qneg_changed.min().item(),
                     cos_qbase_qneg_changed.max().item(),
                     cos_qbase_qneg_changed.mean().item())

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
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
                    self._per_query_alphas = []
                    self._per_query_betas = []
                    self._per_query_stats = []

                    S_final_changed, penalty_scores = self.compute_deir_dual_v2_scores(
                        S_base=S_base_changed,
                        S_req=S_req_changed,
                        S_neg=S_neg_changed,
                        cos_qbase_qneg=cos_qbase_qneg_changed,
                        has_req_mask=has_req_mask_changed,
                        has_neg_mask=has_neg_mask_changed,
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
                    og_ndcg = metrics.get("original", {}).get("ndcg_at_5", 0.0)
                    changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                    changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                    og_mrr = metrics.get("original", {}).get("mrr_at_10", 0.0)
                    changed_mrr = metrics.get("changed", {}).get("mrr_at_10", 0.0)

                    logger.info(
                        "[%d/%d] α=%.1f, β=%.1f, δ=%.2f: "
                        "p-MRR=%.4f, OG_MAP=%.4f, Changed_MAP=%.4f, Changed_MRR=%.4f",
                        trial_idx, total_trials,
                        alpha, beta, delta,
                        p_mrr, og_map, changed_map, changed_mrr,
                    )

                    all_results.append({
                        "alpha": alpha,
                        "beta": beta,
                        "delta": delta,
                        "t_safety": self.t_safety,
                        "boundary_mode": self.boundary_mode,
                        "residual_margin_scale": self.residual_margin_scale,
                        "per_query_ab": self.per_query_ab,
                        "beta_derive_mode": self.beta_derive_mode,
                        "p-MRR": p_mrr,
                        "og_nDCG@5": og_ndcg,
                        "og_nDCG@10": metrics.get("original", {}).get("ndcg_at_10", 0.0),
                        "og_MAP@1000": og_map,
                        "og_MRR@10": og_mrr,
                        "changed_nDCG@5": changed_ndcg,
                        "changed_nDCG@10": metrics.get("changed", {}).get("ndcg_at_10", 0.0),
                        "changed_MAP@1000": changed_map,
                        "changed_MRR@10": changed_mrr,
                        "avg_penalty": float(penalty_scores.mean().item()),
                        "alpha_q_mean": (
                            sum(self._per_query_alphas) / len(self._per_query_alphas)
                            if self._per_query_alphas else alpha
                        ),
                        "beta_q_mean": (
                            sum(self._per_query_betas) / len(self._per_query_betas)
                            if self._per_query_betas else beta
                        ),
                    })

                    composite_score = p_mrr + changed_map + changed_ndcg
                    if best_metrics is None:
                        best_metrics = metrics
                        best_params = {"alpha": alpha, "beta": beta, "delta": delta}
                        best_results_og = results_og
                        best_results_changed = results_changed
                        best_per_query_stats = list(self._per_query_stats)
                    else:
                        best_composite = (
                            best_metrics.get("p-MRR", 0.0)
                            + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                            + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                        )
                        if composite_score > best_composite:
                            best_metrics = metrics
                            best_params = {"alpha": alpha, "beta": beta, "delta": delta}
                            best_results_og = results_og
                            best_results_changed = results_changed
                            best_per_query_stats = list(self._per_query_stats)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("📊 DeIR-Dual V2 3D 搜索完成")
        logger.info(
            "   最佳参数: α=%.1f, β=%.1f, δ=%.2f",
            best_params["alpha"], best_params["beta"], best_params["delta"],
        )
        logger.info("   p-MRR: %.4f", best_metrics.get("p-MRR", 0.0))
        logger.info("   OG MAP@1000: %.4f", best_metrics.get("original", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MAP@1000: %.4f", best_metrics.get("changed", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MRR@10: %.4f", best_metrics.get("changed", {}).get("mrr_at_10", 0.0))
        logger.info("   耗时: %.1f 秒", elapsed)
        logger.info("=" * 60)

        self._save_results(
            best_params=best_params,
            best_metrics=best_metrics,
            all_results=all_results,
            results_og=best_results_og,
            results_changed=best_results_changed,
            per_query_stats=best_per_query_stats,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _save_results(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        per_query_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "DeIR-Dual-V2",
            "dual_queries_source": self.dual_queries_path,
            "fixed_params": {
                "t_safety": self.t_safety,
                "boundary_mode": self.boundary_mode,
                "residual_margin_scale": self.residual_margin_scale,
                "per_query_ab": self.per_query_ab,
                "beta_derive_mode": self.beta_derive_mode,
                "ab_clip_alpha": self.ab_clip_alpha,
                "ab_clip_beta": self.ab_clip_beta,
            },
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
                "full_scores": best_metrics.get("full_scores", {}),
            },
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        metrics_path = os.path.join(self.output_dir, "metrics_deir_dual_v2.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        if per_query_stats:
            per_query_path = os.path.join(self.output_dir, "per_query_stats.json")
            with open(per_query_path, "w", encoding="utf-8") as f:
                json.dump({
                    "boundary_mode": self.boundary_mode,
                    "per_query_ab": self.per_query_ab,
                    "beta_derive_mode": self.beta_derive_mode,
                    "stats": per_query_stats,
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 per-query 统计已保存: {per_query_path}")

        logger.info(f"💾 指标已保存: {metrics_path}")
        logger.info(f"💾 参数明细已保存: {all_results_path}")


def run_deir_dual_v2(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    t_safety: float = 20.0,
    boundary_mode: str = "semantic",
    residual_margin_scale: float = 1.0,
    safety_kappa: float = 0.0,
    beta_raw: bool = False,
    per_query_ab: bool = False,
    beta_derive_mode: str = "max_mean",
    ablation_mode: str = "full",
    use_cache: bool = True,
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/deir_dual_v2/{task_name}"

    if not dual_queries_path:
        raise ValueError("dual_queries_path 不能为空")

    engine = DeIRDualV2Evaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        t_safety=t_safety,
        boundary_mode=boundary_mode,
        residual_margin_scale=residual_margin_scale,
        safety_kappa=safety_kappa,
        beta_raw=beta_raw,
        per_query_ab=per_query_ab,
        beta_derive_mode=beta_derive_mode,
        ablation_mode=ablation_mode,
        use_cache=use_cache,
        device=device,
        batch_size=batch_size,
    )

    return engine.run(
        alphas=alphas,
        betas=betas,
        deltas=deltas,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeIR-Dual V2 奖惩双轨制评估引擎")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)

    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--betas", type=str, default="0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--deltas", type=str, default="-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30")

    parser.add_argument("--t_safety", type=float, default=20.0)
    parser.add_argument("--boundary_mode", type=str, default="semantic",
                        choices=["semantic", "residual_bg"],
                        help="semantic=τ=cos(Qbase,Qneg)+δ; residual_bg=background-calibrated residual boundary")
    parser.add_argument("--residual_margin_scale", type=float, default=1.0,
                        help="MAD multiplier for residual_bg boundary margin")
    parser.add_argument("--safety_kappa", type=float, default=0.0,
                        help="Residual-based safety gate sharpness (0=traditional τ, >0=MAD-normalized residual)")
    parser.add_argument("--beta_raw", type=str, default="false",
                        help="If true, derive β from raw S_req without safety (V8.6)")
    parser.add_argument("--per_query_ab", type=str, default="false",
                        help="Enable V8-style per-query test-time alpha/beta derivation")
    parser.add_argument("--beta_derive_mode", type=str, default="max_mean",
                        choices=["mean", "topk_mean", "p90_mean", "max_mean", "max_comp", "quartic_gap"],
                        help="V8 beta derivation mode")
    parser.add_argument("--ablation_mode", type=str, default="full",
                        choices=["full", "base_only", "no_pos", "no_neg", "no_safety", "linear"],
                        help="Ablation: full=complete method, base_only=S_base only, "
                             "no_pos=w/o positive refinement, no_neg=w/o negative residual branch, "
                             "no_safety=w/o safety gate, linear=linear fusion")

    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas_list = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    deltas_list = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    use_cache = args.use_cache.lower() == "true"

    result = run_deir_dual_v2(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        alphas=alphas_list,
        betas=betas_list,
        deltas=deltas_list,
        t_safety=args.t_safety,
        boundary_mode=args.boundary_mode,
        residual_margin_scale=args.residual_margin_scale,
        safety_kappa=args.safety_kappa,
        beta_raw=args.beta_raw.lower() == "true",
        per_query_ab=args.per_query_ab.lower() == "true",
        beta_derive_mode=args.beta_derive_mode,
        ablation_mode=args.ablation_mode,
        use_cache=use_cache,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\n最终 p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
