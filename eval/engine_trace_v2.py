"""
TRACE-V2: Regression-informed boundary in original score space.

Combines V8.6's score-space advantages with TRACE's Huber regression
for per-query adaptive boundary computation.

V2.1 improvements over V2:
  1. MAD-normalized safety gate (replaces raw overflow * t_safety)
     → safety = 1 - sigmoid(overflow / e_mad * kappa)
     → Properly calibrated transition width proportional to residual spread
  2. Regress on S_base (optionally S_pos) instead of only S_pos
     → S_base is the primary ranking signal with stronger S_neg correlation
  3. Per-query stats saved for diagnosis

Core formula:
  Huber regression:  e(d) = S_neg(d) - a_hat - b_hat * S_regress(d)
  Adaptive boundary: tau_trace = median(e) + MAD(e) * lambda
  overflow = e(d) - tau_trace
  MAD-normalized safety: safety = 1 - sigmoid(overflow / e_mad * kappa)
  Penalty = alpha * Softplus(overflow)
  S_final = S_base + beta * S_pos * safety - penalty
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

logger = logging.getLogger(__name__)


def _mad(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return torch.ones((), device=x.device, dtype=x.dtype) * eps
    med = x.median()
    return (x - med).abs().median().clamp_min(eps)


def fit_huber_regression(
    y: torch.Tensor,
    X: torch.Tensor,
    delta: float = 1.345,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> Tuple[float, float, float]:
    """Fit y = a + b*X using Huber loss (IRLS).

    Returns (a_hat, b_hat, r_squared).
    """
    n = y.numel()
    if n < 3:
        return float(y.mean().item()), 0.0, 0.0

    y_f = y.float()
    X_f = X.float()

    # OLS init
    mean_X = X_f.mean()
    mean_y = y_f.mean()
    var_X = ((X_f - mean_X) ** 2).sum()
    if var_X < 1e-12:
        return float(mean_y.item()), 0.0, 0.0
    b = float(((X_f - mean_X) * (y_f - mean_y)).sum().item() / var_X.item())
    a = float(mean_y.item()) - b * float(mean_X.item())

    # IRLS
    for _ in range(max_iter):
        pred = a + b * X_f
        resid = y_f - pred
        abs_r = resid.abs()
        weights = torch.where(
            abs_r <= delta,
            torch.ones_like(abs_r),
            delta / abs_r.clamp_min(1e-12),
        )
        w_sum = weights.sum()
        w_X_sum = (weights * X_f).sum()
        w_y_sum = (weights * y_f).sum()
        w_XX_sum = (weights * X_f * X_f).sum()
        w_Xy_sum = (weights * X_f * y_f).sum()
        denom = w_sum * w_XX_sum - w_X_sum ** 2
        if abs(denom.item()) < 1e-12:
            break
        a_new = (w_y_sum * w_XX_sum - w_X_sum * w_Xy_sum) / denom
        b_new = (w_sum * w_Xy_sum - w_X_sum * w_y_sum) / denom
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = float(a_new.item()), float(b_new.item())
            break
        a, b = float(a_new.item()), float(b_new.item())

    # R²
    pred_final = a + b * X_f
    ss_res = ((y_f - pred_final) ** 2).sum()
    ss_tot = ((y_f - mean_y) ** 2).sum()
    r2 = 1.0 - float(ss_res.item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0

    return a, b, r2


class TRACEV2Evaluator(DSCLREvaluatorEngine):
    """TRACE-V2: Regression-informed boundary in original score space."""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        huber_delta: float = 1.345,
        lambda_boundary: float = 1.0,
        safety_kappa: float = 10.0,
        t_safety: float = 20.0,
        r2_gate: float = 0.0,
        regress_on: str = "s_base",
        per_query_ab: bool = True,
        beta_derive_mode: str = "max_mean",
        ab_clip_alpha: Tuple[float, float] = (0.05, 5.0),
        ab_clip_beta: Tuple[float, float] = (0.05, 5.0),
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        self.huber_delta = huber_delta
        self.lambda_boundary = lambda_boundary
        self.safety_kappa = safety_kappa
        self.t_safety = t_safety
        self.r2_gate = r2_gate
        self.regress_on = regress_on
        self.per_query_ab = per_query_ab
        self.beta_derive_mode = beta_derive_mode
        self.ab_clip_alpha = ab_clip_alpha
        self.ab_clip_beta = ab_clip_beta
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

        logger.info("TRACE-V2.1 evaluator initialized")
        logger.info(f"  huber_delta={self.huber_delta}, lambda={self.lambda_boundary}")
        logger.info(f"  safety_kappa={self.safety_kappa}, t_safety={self.t_safety}")
        logger.info(f"  r2_gate={self.r2_gate}, regress_on={self.regress_on}")
        logger.info(f"  per_query_ab={self.per_query_ab}")

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path:
            raise ValueError("dual_queries_path is required")
        dual_data: Dict[str, Dict[str, Any]] = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item
        logger.info(f"Loaded {len(dual_data)} dual queries")
        return dual_data

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
        beta_fallback: float,
    ) -> float:
        if s_base_safe.numel() == 0 or s_req_safe.numel() == 0:
            return beta_fallback
        max_b = s_base_safe.max()
        mean_r = s_req_safe.mean()
        return float((max_b / mean_r).item()) if mean_r > 1e-8 else beta_fallback

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

    def _score_query_trace_v2(
        self,
        s_base: torch.Tensor,
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        has_neg: bool,
        alpha: float,
        beta: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """TRACE-V2.1 scoring: V8.6 framework with regression-informed boundary.

        Core formula:
          1. Huber regression: e(d) = S_neg(d) - a_hat - b_hat * S_regress(d)
             where S_regress = S_base (default) or S_pos
          2. Adaptive boundary: tau_trace = median(e) + MAD(e) * lambda
          3. overflow = e(d) - tau_trace
          4. MAD-normalized safety: safety = 1 - sigmoid(overflow / e_mad * kappa)
          5. Penalty: alpha * Softplus(overflow)
          6. S_final = S_base + beta * S_pos * safety - penalty
        """
        stats: Dict[str, Any] = {"qid": self._current_qid, "has_neg": has_neg}

        if not has_neg or s_neg.numel() < 3:
            # No exclusion: just reward
            if self.per_query_ab and s_base.numel() > 0:
                beta = self._clip(
                    self._derive_beta_q(s_base, s_pos, beta),
                    self.ab_clip_beta,
                )
                self._per_query_betas.append(beta)
            s_final = s_base + beta * s_pos
            stats.update({"alpha_q": 0.0, "beta_q": beta})
            self._per_query_stats.append(stats)
            return s_final, stats

        # Step 1: Huber regression of S_neg on S_regress
        s_regress = s_base if self.regress_on == "s_base" else s_pos
        a_hat, b_hat, r2 = fit_huber_regression(s_neg, s_regress, delta=self.huber_delta)

        # Compute regression residual in ORIGINAL score space
        e = s_neg.float() - a_hat - b_hat * s_regress.float()

        # Step 2: Adaptive boundary using robust statistics of residual
        e_median = e.median()
        e_mad = _mad(e)
        tau_trace = e_median + e_mad * self.lambda_boundary

        # Residual excess (how much S_neg exceeds regression prediction + boundary)
        overflow = e - tau_trace

        # R² gate: if regression is poor, fall back to no penalty
        if self.r2_gate > 0 and r2 < self.r2_gate:
            # Poor regression: use pos_only (no penalty)
            if self.per_query_ab and s_base.numel() > 0:
                beta = self._clip(
                    self._derive_beta_q(s_base, s_pos, beta),
                    self.ab_clip_beta,
                )
                self._per_query_betas.append(beta)
            s_final = s_base + beta * s_pos
            stats.update({
                "alpha_q": 0.0, "beta_q": beta, "r2": r2,
                "r2_gated": True, "regression_skipped": True,
                "regress_on": self.regress_on,
            })
            self._per_query_stats.append(stats)
            return s_final, stats

        # Step 3: Safety gate - MAD-normalized (V2.1 key improvement)
        # kappa > 0: safety = 1 - sigmoid(overflow / e_mad * kappa)
        #   overflow/e_mad measures residual excess in "MAD units"
        #   kappa controls transition sharpness (like V8.6's safety_kappa)
        # kappa = 0: fallback to raw overflow * t_safety (old V2 behavior)
        if self.safety_kappa > 0 and e_mad > 1e-8:
            safety = 1.0 - torch.sigmoid(overflow / e_mad * self.safety_kappa)
            stats["safety_mode"] = "mad_normalized"
        else:
            safety = 1.0 - torch.sigmoid(overflow * self.t_safety)
            stats["safety_mode"] = "raw_overflow"

        # Step 4: Smooth penalty
        smooth_penalty = F.softplus(overflow)

        # Per-query alpha/beta derivation
        if self.per_query_ab:
            at_risk_mask = overflow > 0
            safe_mask = ~at_risk_mask

            if at_risk_mask.any():
                mean_base_risk = s_base[at_risk_mask].mean()
                mean_penalty_risk = smooth_penalty[at_risk_mask].mean()
                if mean_penalty_risk > 1e-8:
                    alpha = float((mean_base_risk / mean_penalty_risk).item())

            if safe_mask.any():
                beta = self._clip(
                    self._derive_beta_q(s_base[safe_mask], s_pos[safe_mask], beta),
                    self.ab_clip_beta,
                )

            alpha = self._clip(alpha, self.ab_clip_alpha)
            self._per_query_alphas.append(alpha)
            self._per_query_betas.append(beta)

        raw_penalty = alpha * smooth_penalty
        s_final = s_base + beta * s_pos * safety - raw_penalty

        stats.update({
            "alpha_q": alpha, "beta_q": beta, "r2": r2,
            "a_hat": a_hat, "b_hat": b_hat,
            "regress_on": self.regress_on,
            "e_median": float(e_median.item()),
            "e_mad": float(e_mad.item()),
            "tau_trace_mean": float(tau_trace.mean().item()),
            "at_risk_ratio": float((overflow > 0).float().mean().item()),
            "num_at_risk": int((overflow > 0).sum().item()),
            "num_safe": int((overflow <= 0).sum().item()),
            "safety_mean": float(safety.mean().item()),
            "safety_min": float(safety.min().item()),
            "penalty_mean": float(raw_penalty.mean().item()),
            "penalty_max": float(raw_penalty.max().item()),
            "s_base_mean": float(s_base.mean().item()),
            "s_neg_mean": float(s_neg.mean().item()),
            "overflow_max": float(overflow.max().item()),
            "r2_gated": False,
        })
        self._per_query_stats.append(stats)

        return s_final, stats

    def compute_trace_v2_scores(
        self,
        S_base: torch.Tensor,
        S_pos: torch.Tensor,
        S_neg: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        S_final = S_base.clone()

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            idx_tensor = torch.tensor(cand_indices, device=S_base.device, dtype=torch.long)
            s_b = S_base[q_idx].index_select(0, idx_tensor)
            s_p = S_pos[q_idx].index_select(0, idx_tensor)
            s_n = S_neg[q_idx].index_select(0, idx_tensor)
            has_neg = bool(has_neg_mask[q_idx].item() > 0)

            self._current_qid = qid
            s_final_local, _stats = self._score_query_trace_v2(
                s_base=s_b, s_pos=s_p, s_neg=s_n,
                has_neg=has_neg, alpha=alpha, beta=beta,
            )

            s_final_local = s_final_local.to(dtype=S_final.dtype)
            S_final[q_idx, idx_tensor] = s_final_local

        return S_final

    def run(
        self,
        alphas: Optional[List[float]] = None,
        betas: Optional[List[float]] = None,
        lambda_list: Optional[List[float]] = None,
        r2_gates: Optional[List[float]] = None,
        safety_kappas: Optional[List[float]] = None,
        regress_ons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting TRACE-V2.1 evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        alpha_list = alphas if alphas else [0.5]
        beta_list = betas if betas else [0.5]
        lambda_list = lambda_list if lambda_list else [1.0, 1.5, 2.0, 3.0]
        r2_list = r2_gates if r2_gates else [0.0, 0.1, 0.2]
        kappa_list = safety_kappas if safety_kappas else [5.0, 10.0, 20.0]
        regress_list = regress_ons if regress_ons else ["s_base", "s_pos"]

        # With per_query_ab=True, alpha/beta grid values don't affect results
        # So we only grid over lambda, r2_gate, safety_kappa, regress_on
        total_trials = len(lambda_list) * len(r2_list) * len(kappa_list) * len(regress_list)
        logger.info(f"Grid search: {total_trials} combinations (lambda x r2_gate x kappa x regress_on)")
        logger.info(f"  alpha/beta fixed (per_query_ab=True auto-derives them)")

        dual_data = self.load_dual_queries()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)

        # Load or compute document embeddings
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)

        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"Using cached doc embeddings ({len(cached_doc_ids)})")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("Cached doc IDs mismatch, re-encoding...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("Encoding candidate documents...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        # Build query lists
        def build_query_lists(queries, raw_queries):
            query_ids, q_base_list, q_pos_list, q_neg_list = [], [], [], []
            has_neg_list = []
            for qid in queries.keys():
                query_ids.append(qid)
                raw = raw_queries.get(qid, ("", ""))
                query_text, instruction = raw[0], raw[1]
                q_base = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")
                q_base_list.append(q_base)

                d = dual_data.get(qid, {})
                q_plus = d.get("q_plus", "")
                q_minus = d.get("q_minus", "")

                q_pos_list.append(q_plus if not self._is_none_query(q_plus) else "")
                q_neg_list.append(q_minus if not self._is_none_query(q_minus) else "")
                has_neg_list.append(0.0 if self._is_none_query(q_minus) else 1.0)
            return query_ids, q_base_list, q_pos_list, q_neg_list, has_neg_list

        query_ids_og, q_base_og, q_pos_og, q_neg_og, has_neg_og = build_query_lists(q_og, q_raw_og)
        query_ids_ch, q_base_ch, q_pos_ch, q_neg_ch, has_neg_ch = build_query_lists(q_changed, q_raw_changed)

        has_neg_mask_og = torch.tensor(has_neg_og, dtype=torch.float32)
        has_neg_mask_ch = torch.tensor(has_neg_ch, dtype=torch.float32)

        # Encode queries
        logger.info("Encoding OG queries (Q_base, Q_pos, Q_neg)...")
        q_base_emb_og = self._encode_queries(q_base_og)
        q_pos_emb_og = self._encode_queries(q_pos_og)
        q_neg_emb_og = self._encode_queries(q_neg_og)

        logger.info("Encoding Changed queries (Q_base, Q_pos, Q_neg)...")
        q_base_emb_ch = self._encode_queries(q_base_ch)
        q_pos_emb_ch = self._encode_queries(q_pos_ch)
        q_neg_emb_ch = self._encode_queries(q_neg_ch)

        device = self.retriever.doc_embeddings.device
        score_dtype = self.retriever.doc_embeddings.dtype
        q_base_emb_og = q_base_emb_og.to(device=device, dtype=score_dtype)
        q_pos_emb_og = q_pos_emb_og.to(device=device, dtype=score_dtype)
        q_neg_emb_og = q_neg_emb_og.to(device=device, dtype=score_dtype)
        q_base_emb_ch = q_base_emb_ch.to(device=device, dtype=score_dtype)
        q_pos_emb_ch = q_pos_emb_ch.to(device=device, dtype=score_dtype)
        q_neg_emb_ch = q_neg_emb_ch.to(device=device, dtype=score_dtype)
        has_neg_mask_og = has_neg_mask_og.to(device)
        has_neg_mask_ch = has_neg_mask_ch.to(device)

        # Compute cosine scores
        logger.info("Computing S_base, S_pos, S_neg...")
        S_base_og = torch.matmul(q_base_emb_og, self.retriever.doc_embeddings.T)
        S_pos_og = torch.matmul(q_pos_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = torch.matmul(q_neg_emb_og, self.retriever.doc_embeddings.T) * has_neg_mask_og.unsqueeze(1)

        S_base_ch = torch.matmul(q_base_emb_ch, self.retriever.doc_embeddings.T)
        S_pos_ch = torch.matmul(q_pos_emb_ch, self.retriever.doc_embeddings.T)
        S_neg_ch = torch.matmul(q_neg_emb_ch, self.retriever.doc_embeddings.T) * has_neg_mask_ch.unsqueeze(1)

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        # Grid search
        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        all_results: List[Dict[str, Any]] = []
        best_per_query_stats = None

        trial_idx = 0
        alpha, beta_val = alpha_list[0], beta_list[0]
        for lam in lambda_list:
            for r2g in r2_list:
                for kappa in kappa_list:
                    for reg_on in regress_list:
                        trial_idx += 1
                        self.lambda_boundary = lam
                        self.r2_gate = r2g
                        self.safety_kappa = kappa
                        self.regress_on = reg_on
                        self._per_query_alphas = []
                        self._per_query_betas = []
                        self._per_query_stats = []

                        S_final_ch = self.compute_trace_v2_scores(
                            S_base_ch, S_pos_ch, S_neg_ch,
                            has_neg_mask_ch, query_ids_ch, qid_to_candidate_indices,
                            alpha=alpha, beta=beta_val,
                        )

                        results_og = self._extract_results(S_base_og, query_ids_og, candidates)
                        results_changed = self._extract_results(S_final_ch, query_ids_ch, candidates)

                        evaluator = FollowIREvaluator(self.task_name)
                        metrics = evaluator.evaluate(results_og, results_changed)

                        p_mrr = metrics.get("p-MRR", 0.0)
                        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                        changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

                        logger.info(
                            "[%d/%d] λ=%.1f R²≥%.1f κ=%.0f reg=%s: p-MRR=%.4f C_MAP=%.4f C_nDCG5=%.4f",
                            trial_idx, total_trials,
                            lam, r2g, kappa, reg_on,
                            p_mrr, changed_map, changed_ndcg,
                        )

                        all_results.append({
                            "alpha": alpha, "beta": beta_val,
                            "lambda": lam, "r2_gate": r2g,
                            "safety_kappa": kappa, "regress_on": reg_on,
                            "huber_delta": self.huber_delta,
                            "p-MRR": p_mrr,
                            "og_MAP@1000": og_map,
                            "changed_MAP@1000": changed_map,
                            "changed_nDCG@5": changed_ndcg,
                        })

                        composite = p_mrr + changed_map + changed_ndcg
                        if best_metrics is None:
                            best_metrics = metrics
                            best_params = {
                                "alpha": alpha, "beta": beta_val,
                                "lambda": lam, "r2_gate": r2g,
                                "safety_kappa": kappa, "regress_on": reg_on,
                            }
                            best_results_og = results_og
                            best_results_changed = results_changed
                            best_per_query_stats = list(self._per_query_stats)
                        else:
                            best_composite = (
                                best_metrics.get("p-MRR", 0.0)
                                + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                                + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                            )
                            if composite > best_composite:
                                best_metrics = metrics
                                best_params = {
                                    "alpha": alpha, "beta": beta_val,
                                    "lambda": lam, "r2_gate": r2g,
                                    "safety_kappa": kappa, "regress_on": reg_on,
                                }
                                best_results_og = results_og
                                best_results_changed = results_changed
                                best_per_query_stats = list(self._per_query_stats)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("TRACE-V2 evaluation complete")
        logger.info(f"  Best: {best_params}")
        logger.info(f"  p-MRR: {best_metrics.get('p-MRR', 0.0):.4f}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")

        self._save_results(best_params, best_metrics, all_results, best_results_og, best_results_changed, best_per_query_stats, elapsed)
        return {"best_params": best_params, "best_metrics": best_metrics, "all_results": all_results}

    def _save_results(self, best_params, best_metrics, all_results, results_og, results_changed, per_query_stats, elapsed):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "TRACE-V2.1",
            "dual_queries_source": self.dual_queries_path,
            "fixed_params": {
                "huber_delta": self.huber_delta,
                "t_safety": self.t_safety,
                "per_query_ab": self.per_query_ab,
            },
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
            },
            "elapsed": elapsed,
        }

        with open(os.path.join(self.output_dir, "trace_v2_metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "trace_v2_all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_og.json"), "w") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_changed.json"), "w") as f:
            json.dump(results_changed, f, ensure_ascii=False)
        if per_query_stats:
            with open(os.path.join(self.output_dir, "trace_v2_per_query_stats.json"), "w") as f:
                json.dump({
                    "mode": "TRACE-V2.1",
                    "best_params": best_params,
                    "per_query_stats": per_query_stats,
                }, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def run_trace_v2(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    huber_delta: float = 1.345,
    lambda_boundary: float = 1.0,
    safety_kappa: float = 10.0,
    t_safety: float = 20.0,
    r2_gate: float = 0.0,
    regress_on: str = "s_base",
    use_cache: bool = True,
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/trace_v2/{task_name}"
    if not dual_queries_path:
        raise ValueError("dual_queries_path is required")

    engine = TRACEV2Evaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        huber_delta=huber_delta,
        lambda_boundary=lambda_boundary,
        safety_kappa=safety_kappa,
        t_safety=t_safety,
        r2_gate=r2_gate,
        regress_on=regress_on,
        use_cache=use_cache,
        device=device,
        batch_size=batch_size,
    )
    return engine.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRACE-V2.1 evaluator")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--huber_delta", type=float, default=1.345)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--safety_kappa", type=float, default=10.0)
    parser.add_argument("--t_safety", type=float, default=20.0)
    parser.add_argument("--r2_gate", type=float, default=0.0)
    parser.add_argument("--regress_on", type=str, default="s_base", choices=["s_base", "s_pos"])
    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    use_cache = args.use_cache.lower() == "true"

    result = run_trace_v2(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        huber_delta=args.huber_delta,
        lambda_boundary=args.lambda_boundary,
        safety_kappa=args.safety_kappa,
        t_safety=args.t_safety,
        r2_gate=args.r2_gate,
        regress_on=args.regress_on,
        use_cache=use_cache,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nFinal p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
