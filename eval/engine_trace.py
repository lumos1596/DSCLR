"""
TRACE: Candidate-Conditional Exclusion Correction for Instruction-Following Retrieval

Paper: "TRACE: Candidate-Conditional Exclusion Correction for Instruction-Following Retrieval"

Core method (Eqs. 3-8):
  1. Robust standardization:  z_x(d) = (S_x(d) - median) / (MAD + eps)
  2. Huber regression:        fit z_neg(d) = a + b * z_pos(d) on candidate set
  3. Residual normalization:  r(d) = (e(d) - median_e) / (MAD_e + eps)
     where e(d) = z_neg(d) - a_hat - b_hat * z_pos(d)
  4. Monotone composition:
     p(d) = [z_pos(d)]+                                  (nonneg reward)
     h(d) = [r(d) - lambda]+                              (residual excess)
     g(d) = exp(-h(d) / tau_decay)                        (gate)
     S_final(d) = z_full(d) + p(d)*g(d) - h(d)           (monotone in r)

Usage:
  python -m eval.engine_trace \
    --task_name Core17InstructionRetrieval \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings, save_embeddings_cache
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


@dataclass
class HuberFitResult:
    """Result of Huber regression fit on a single query's candidate set."""
    a_hat: float          # intercept
    b_hat: float          # slope
    r_squared: float      # pseudo R^2 of the fit
    residual_median: float
    residual_mad: float
    n_candidates: int
    n_above_boundary: int  # number of docs with r(d) > lambda


@dataclass
class TRACEQueryResult:
    """TRACE scoring result for a single query."""
    s_final: torch.Tensor
    z_full: torch.Tensor
    z_pos: torch.Tensor
    z_neg: torch.Tensor
    r: torch.Tensor       # normalized residual
    p: torch.Tensor       # positive reward
    h: torch.Tensor       # residual excess
    g: torch.Tensor       # gate
    huber_fit: HuberFitResult
    stats: Dict[str, Any] = field(default_factory=dict)


def _mad(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median Absolute Deviation."""
    if x.numel() == 0:
        return torch.ones((), device=x.device, dtype=x.dtype) * eps
    med = x.median()
    return (x - med).abs().median().clamp_min(eps)


def robust_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust standardization using median and MAD (Eq. 3)."""
    if x.numel() <= 1:
        return torch.zeros_like(x)
    med = x.median()
    mad_val = _mad(x, eps)
    return (x - med) / mad_val


def _mean_std_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean/std standardization (diagnostic ablation variant)."""
    if x.numel() <= 1:
        return torch.zeros_like(x)
    mean_val = x.mean()
    std_val = x.std()
    if std_val < eps:
        return torch.zeros_like(x)
    return (x - mean_val) / std_val


def huber_loss(residual: torch.Tensor, delta: float = 1.345) -> torch.Tensor:
    """Huber loss function.

    For |r| <= delta: 0.5 * r^2
    For |r| > delta:  delta * (|r| - 0.5 * delta)
    """
    abs_r = residual.abs()
    quadratic = torch.min(abs_r, torch.tensor(delta, device=residual.device, dtype=residual.dtype))
    return 0.5 * quadratic ** 2 + delta * (abs_r - quadratic)


def fit_huber_regression(
    z_neg: torch.Tensor,
    z_pos: torch.Tensor,
    delta: float = 1.345,
    lr: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> Tuple[float, float]:
    """Fit z_neg = a + b * z_pos using Huber loss (Eq. 4).

    Uses iterative reweighted least squares (IRLS) for robustness.

    Returns (a_hat, b_hat).
    """
    n = z_neg.numel()
    if n < 3:
        # Not enough data; return OLS fallback
        b_hat = 0.0
        a_hat = float(z_neg.mean().item())
        return a_hat, b_hat

    z_n = z_neg.float()
    z_p = z_pos.float()

    # Initialize with OLS
    mean_p = z_p.mean()
    mean_n = z_n.mean()
    var_p = ((z_p - mean_p) ** 2).sum()
    if var_p < 1e-12:
        return float(mean_n.item()), 0.0
    b = float(((z_p - mean_p) * (z_n - mean_n)).sum().item()) / float(var_p.item())
    a = float(mean_n.item()) - b * float(mean_p.item())

    # IRLS iterations
    for _ in range(max_iter):
        pred = a + b * z_p
        resid = z_n - pred
        abs_r = resid.abs()

        # Huber weights: w = 1 if |r| <= delta, else delta / |r|
        weights = torch.where(
            abs_r <= delta,
            torch.ones_like(abs_r),
            delta / abs_r.clamp_min(1e-12)
        )

        # Weighted least squares
        w_sum = weights.sum()
        w_p_sum = (weights * z_p).sum()
        w_n_sum = (weights * z_n).sum()
        w_pp_sum = (weights * z_p * z_p).sum()
        w_pn_sum = (weights * z_p * z_n).sum()

        denom = w_sum * w_pp_sum - w_p_sum ** 2
        if abs(denom.item()) < 1e-12:
            break

        a_new = (w_n_sum * w_pp_sum - w_p_sum * w_pn_sum) / denom
        b_new = (w_sum * w_pn_sum - w_p_sum * w_n_sum) / denom

        # Check convergence
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = float(a_new.item()), float(b_new.item())
            break
        a, b = float(a_new.item()), float(b_new.item())

    return a, b


class TRACEEvaluator(DSCLREvaluatorEngine):
    """TRACE: Candidate-Conditional Exclusion Correction evaluator."""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        huber_delta: float = 1.345,
        lambda_boundary: float = 1.0,
        tau_decay: float = 0.2,
        eps: float = 1e-6,
        regression_mode: str = "huber",   # "huber" or "ols"
        normalization_mode: str = "median_mad",  # "median_mad" or "mean_std"
        uncentered_residual: bool = False,       # skip residual recentering
        constrained_slope: bool = False,         # enforce b_hat >= 0
        raw_score_fit: bool = False,             # fit regression before channel normalization
        ablation: str = "full",           # "full", "no_regression", "no_gate", "pos_only", "linear"
        candidate_depth: int = 0,         # truncate candidates to top-K (0 = use all)
        residual_pooling: str = "max",    # "max", "mean", "independent" — multi-exclusion aggregation
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        self.huber_delta = huber_delta
        self.lambda_boundary = lambda_boundary
        self.tau_decay = tau_decay
        self.eps = eps
        self.regression_mode = regression_mode
        self.normalization_mode = normalization_mode
        self.uncentered_residual = uncentered_residual
        self.constrained_slope = constrained_slope
        self.raw_score_fit = raw_score_fit
        self.ablation = ablation
        self.candidate_depth = candidate_depth
        self.residual_pooling = residual_pooling

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

        logger.info("TRACE evaluator initialized")
        logger.info(f"  huber_delta={self.huber_delta}, lambda={self.lambda_boundary}, tau={self.tau_decay}")
        logger.info(f"  regression_mode={self.regression_mode}, normalization_mode={self.normalization_mode}")
        logger.info(f"  uncentered_residual={self.uncentered_residual}, constrained_slope={self.constrained_slope}")
        logger.info(f"  raw_score_fit={self.raw_score_fit}, ablation={self.ablation}")
        logger.info(f"  residual_pooling={self.residual_pooling}")

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

    def trace_score_query(
        self,
        s_full: torch.Tensor,
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        has_neg: bool,
    ) -> TRACEQueryResult:
        """Apply TRACE scoring for a single query's candidate set.

        Implements Eqs. 3-8 from the paper.
        """
        stats: Dict[str, Any] = {}
        n = s_full.numel()

        # Step 1: Channel normalization (Eq. 3)
        if self.normalization_mode == "mean_std":
            # Mean/std scaling variant: replace median/MAD with mean/std
            z_full = _mean_std_standardize(s_full.float(), self.eps)
            z_pos = _mean_std_standardize(s_pos.float(), self.eps)
            z_neg = _mean_std_standardize(s_neg.float(), self.eps)
        else:
            # Default: robust standardization (median/MAD)
            z_full = robust_standardize(s_full.float(), self.eps)
            z_pos = robust_standardize(s_pos.float(), self.eps)
            z_neg = robust_standardize(s_neg.float(), self.eps)

        stats["n_candidates"] = n
        stats["s_full_median"] = float(s_full.float().median().item())
        stats["s_pos_median"] = float(s_pos.float().median().item())
        stats["s_neg_median"] = float(s_neg.float().median().item())
        stats["s_full_mad"] = float(_mad(s_full.float(), self.eps).item())
        stats["s_pos_mad"] = float(_mad(s_pos.float(), self.eps).item())
        stats["s_neg_mad"] = float(_mad(s_neg.float(), self.eps).item())

        if not has_neg or n < 3:
            # No exclusion: rank by z_full + [z_pos]+ (unless z_full_only)
            p = torch.clamp(z_pos, min=0)
            if self.ablation == "z_full_only":
                s_final = z_full
            else:
                s_final = z_full + p
            return TRACEQueryResult(
                s_final=s_final,
                z_full=z_full,
                z_pos=z_pos,
                z_neg=z_neg,
                r=torch.zeros_like(z_neg),
                p=p,
                h=torch.zeros_like(z_neg),
                g=torch.ones_like(z_neg),
                huber_fit=HuberFitResult(0, 0, 0, 0, 0, n, 0),
                stats=stats,
            )

        # Step 2: Fit regression (Eq. 4)
        # raw_score_fit variant: fit on raw scores before channel normalization
        if self.raw_score_fit:
            # Fit regression on raw (un-normalized) scores
            if self.regression_mode == "ols":
                s_p = s_pos.float()
                s_n = s_neg.float()
                mean_p = s_p.mean()
                mean_n = s_n.mean()
                var_p = ((s_p - mean_p) ** 2).sum()
                if var_p < 1e-12:
                    a_hat_raw, b_hat_raw = float(mean_n.item()), 0.0
                else:
                    b_hat_raw = float(((s_p - mean_p) * (s_n - mean_n)).sum().item() / var_p.item())
                    a_hat_raw = float(mean_n.item()) - b_hat_raw * float(mean_p.item())
            else:
                a_hat_raw, b_hat_raw = fit_huber_regression(
                    s_neg.float(), s_pos.float(), delta=self.huber_delta
                )
            # Enforce nonnegative slope if constrained_slope is set
            if self.constrained_slope and b_hat_raw < 0:
                b_hat_raw = 0.0
                a_hat_raw = float(s_neg.float().mean().item())
            # Compute residual on raw scores, then normalize it using mean/std of residual
            e_raw = s_neg.float() - a_hat_raw - b_hat_raw * s_pos.float()
            e_mean = e_raw.mean()
            e_std = e_raw.std()
            if e_std < 1e-12:
                e_std = torch.tensor(1.0, device=e_raw.device)
            r = (e_raw - e_mean) / e_std
            a_hat = a_hat_raw
            b_hat = b_hat_raw
            # Pseudo R^2 on raw scores
            ss_res = (e_raw ** 2).sum()
            ss_tot = ((s_neg.float() - s_neg.float().mean()) ** 2).sum()
            r_squared = 1.0 - float(ss_res.item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0
        else:
            # Default: fit on standardized scores z_neg = a + b * z_pos
            if self.regression_mode == "ols":
                z_p = z_pos.float()
                z_n = z_neg.float()
                mean_p = z_p.mean()
                mean_n = z_n.mean()
                var_p = ((z_p - mean_p) ** 2).sum()
                if var_p < 1e-12:
                    a_hat, b_hat = float(mean_n.item()), 0.0
                else:
                    b_hat = float(((z_p - mean_p) * (z_n - mean_n)).sum().item() / var_p.item())
                    a_hat = float(mean_n.item()) - b_hat * float(mean_p.item())
            else:
                a_hat, b_hat = fit_huber_regression(
                    z_neg, z_pos, delta=self.huber_delta
                )

            # Enforce nonnegative slope if constrained_slope is set
            if self.constrained_slope and b_hat < 0:
                b_hat = 0.0
                a_hat = float(z_neg.float().mean().item())

            # Compute raw residual e(d) = z_neg(d) - a_hat - b_hat * z_pos(d)
            e = z_neg.float() - a_hat - b_hat * z_pos.float()

            # Compute pseudo R^2
            ss_res = (e ** 2).sum()
            ss_tot = ((z_neg.float() - z_neg.float().mean()) ** 2).sum()
            r_squared = 1.0 - float(ss_res.item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0

            # Step 3: Re-standardize the residual (Eq. 5)
            if self.normalization_mode == "mean_std":
                # Mean/std variant for residual normalization
                e_mean = e.mean()
                e_std_val = e.std()
                if e_std_val < 1e-12:
                    e_std_val = torch.tensor(1.0, device=e.device)
                r = (e - e_mean) / e_std_val
            else:
                # Default: robust re-standardization (median/MAD)
                e_median = e.median()
                e_mad = _mad(e, self.eps)
                if self.uncentered_residual:
                    # Uncentered residual variant: skip recentering, keep MAD scaling only
                    r = e / e_mad
                else:
                    r = (e - e_median) / e_mad

        # Step 4: Monotone residual-aware composition (Eqs. 6-8)
        p = torch.clamp(z_pos, min=0)          # Eq. 6: nonneg reward
        h = torch.clamp(r - self.lambda_boundary, min=0)  # Eq. 7: residual excess
        g = torch.exp(-h / self.tau_decay)     # Eq. 7: gate

        n_above = int((r > self.lambda_boundary).sum().item())

        # Eq. 8: final score
        if self.ablation == "z_full_only":
            # No positive, no negative: just z_full
            s_final = z_full
        elif self.ablation == "full":
            s_final = z_full + p * g - h
        elif self.ablation == "no_regression":
            # Skip regression: use raw z_neg directly
            r_raw = z_neg
            h_raw = torch.clamp(r_raw - self.lambda_boundary, min=0)
            g_raw = torch.exp(-h_raw / self.tau_decay)
            s_final = z_full + p * g_raw - h_raw
        elif self.ablation == "no_gate":
            # Remove gate: S = z_full + p - h
            s_final = z_full + p - h
        elif self.ablation == "pos_only":
            # Only positive channel: S = z_full + p
            s_final = z_full + p
        elif self.ablation == "linear":
            # Linear fusion: S = z_full + p - r (no gate, no boundary)
            s_final = z_full + p - r
        elif self.ablation == "raw_neg_subtract":
            # Raw negative subtraction (no regression, no gate): S = z_full + p - z_neg
            s_final = z_full + p - z_neg
        elif self.ablation == "raw_neg_fusion":
            # Raw negative penalty (no regression): S = z_full + p - [z_neg - lambda]+
            h_raw = torch.clamp(z_neg - self.lambda_boundary, min=0)
            s_final = z_full + p - h_raw
        elif self.ablation == "gate_only":
            # Attenuation only (no penalty): S = z_full + p*g
            s_final = z_full + p * g
        else:
            s_final = z_full + p * g - h

        # Compute residual stats for HuberFitResult (may differ by variant)
        _resid_median = float(e_median.item()) if 'e_median' in locals() else 0.0
        _resid_mad = float(e_mad.item()) if 'e_mad' in locals() else 0.0

        huber_fit = HuberFitResult(
            a_hat=a_hat,
            b_hat=b_hat,
            r_squared=r_squared,
            residual_median=_resid_median,
            residual_mad=_resid_mad,
            n_candidates=n,
            n_above_boundary=n_above,
        )

        stats.update({
            "a_hat": a_hat,
            "b_hat": b_hat,
            "r_squared": r_squared,
            "residual_mad": _resid_mad,
            "n_above_boundary": n_above,
            "above_boundary_ratio": n_above / max(n, 1),
            "p_mean": float(p.mean().item()),
            "p_max": float(p.max().item()),
            "h_mean": float(h.mean().item()),
            "h_max": float(h.max().item()),
            "g_mean": float(g.mean().item()),
            "r_mean": float(r.mean().item()),
            "r_std": float(r.std().item()) if n > 1 else 0.0,
            "r_min": float(r.min().item()),
            "r_max": float(r.max().item()),
        })

        return TRACEQueryResult(
            s_final=s_final,
            z_full=z_full,
            z_pos=z_pos,
            z_neg=z_neg,
            r=r,
            p=p,
            h=h,
            g=g,
            huber_fit=huber_fit,
            stats=stats,
        )

    def compute_trace_scores(
        self,
        S_full: torch.Tensor,
        S_pos: torch.Tensor,
        S_neg_list_per_query: List[Optional[List[torch.Tensor]]],
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Apply TRACE scoring to all queries.

        Args:
            S_neg_list_per_query: list of length n_queries; each element is either
                None (no exclusion) or a list of M_i tensors each of shape (n_docs,).
                M_i=1 degenerates to original TRACE; M_i>1 uses max-residual aggregation.
        """
        S_final = S_full.clone()
        all_stats: List[Dict[str, Any]] = []

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            idx_tensor = torch.tensor(cand_indices, device=S_full.device, dtype=torch.long)
            s_f = S_full[q_idx].index_select(0, idx_tensor)
            s_p = S_pos[q_idx].index_select(0, idx_tensor)
            has_neg = bool(has_neg_mask[q_idx].item() > 0)

            s_neg_list = S_neg_list_per_query[q_idx]
            n_excl = len(s_neg_list) if s_neg_list else 0

            if not has_neg or n_excl == 0:
                # No exclusion: call with dummy s_neg (has_neg=False)
                result = self.trace_score_query(s_f, s_p, s_f, has_neg=False)
            elif n_excl == 1:
                # Single exclusion: original TRACE behavior (unchanged)
                s_n = s_neg_list[0].index_select(0, idx_tensor)
                result = self.trace_score_query(s_f, s_p, s_n, has_neg=True)
            else:
                # Multi-exclusion (M>1): configurable aggregation
                # For each exclusion unit j, run trace_score_query to get r_j
                per_unit_results = []
                for s_neg_j in s_neg_list:
                    s_n_j = s_neg_j.index_select(0, idx_tensor)
                    result_j = self.trace_score_query(s_f, s_p, s_n_j, has_neg=True)
                    per_unit_results.append(result_j)

                # z_full, z_pos, p are identical across units (they don't depend on s_neg)
                z_full = per_unit_results[0].z_full
                z_pos = per_unit_results[0].z_pos
                p = per_unit_results[0].p

                r_stack = torch.stack([r.r for r in per_unit_results])  # (M, n_candidates)

                if self.residual_pooling == "max":
                    # Max-residual: r_agg = max_j r_j(d) — most aggressive
                    r_agg = r_stack.max(dim=0).values
                    h_agg = torch.clamp(r_agg - self.lambda_boundary, min=0)
                    g_agg = torch.exp(-h_agg / self.tau_decay)
                elif self.residual_pooling == "mean":
                    # Mean-residual: r_agg = mean_j r_j(d) — moderate, avoids over-penalization
                    r_agg = r_stack.mean(dim=0)
                    h_agg = torch.clamp(r_agg - self.lambda_boundary, min=0)
                    g_agg = torch.exp(-h_agg / self.tau_decay)
                elif self.residual_pooling == "independent":
                    # Independent: g_agg = prod_j g_j, h_agg = sum_j h_j
                    # Each exclusion unit independently gates reward; penalties are additive
                    h_stack = torch.stack([r_j.h for r_j in per_unit_results])  # (M, n_candidates)
                    g_stack = torch.stack([r_j.g for r_j in per_unit_results])  # (M, n_candidates)
                    h_agg = h_stack.sum(dim=0)   # additive penalties
                    g_agg = g_stack.prod(dim=0)  # multiplicative gates (any exclusion kills reward)
                    r_agg = r_stack.max(dim=0).values  # for stats reporting only
                else:
                    raise ValueError(f"Unknown residual_pooling: {self.residual_pooling}")

                # Compose final score
                if self.ablation == "z_full_only":
                    s_final = z_full
                elif self.ablation in ("full", "no_regression", "no_gate", "linear",
                                        "raw_neg_subtract", "raw_neg_fusion", "gate_only"):
                    if self.ablation == "full":
                        s_final = z_full + p * g_agg - h_agg
                    elif self.ablation == "no_gate":
                        s_final = z_full + p - h_agg
                    elif self.ablation == "linear":
                        s_final = z_full + p - r_agg
                    elif self.ablation == "gate_only":
                        s_final = z_full + p * g_agg
                    else:
                        s_final = z_full + p * g_agg - h_agg
                elif self.ablation == "pos_only":
                    s_final = z_full + p
                else:
                    s_final = z_full + p * g_agg - h_agg

                # Build aggregated result (use first unit's z_neg for stats compatibility)
                result = TRACEQueryResult(
                    s_final=s_final,
                    z_full=z_full,
                    z_pos=z_pos,
                    z_neg=per_unit_results[0].z_neg,
                    r=r_agg,
                    p=p,
                    h=h_agg,
                    g=g_agg,
                    huber_fit=per_unit_results[0].huber_fit,
                    stats=per_unit_results[0].stats,
                )
                # Augment stats with multi-exclusion info
                result.stats["n_exclusion_units"] = n_excl
                result.stats["multi_exclusion"] = True
                result.stats["residual_pooling"] = self.residual_pooling
                result.stats["per_unit_r_squared"] = [
                    float(r_j.huber_fit.r_squared) for r_j in per_unit_results
                ]

            s_final_local = result.s_final.to(dtype=S_final.dtype)
            S_final[q_idx, idx_tensor] = s_final_local

            stat = result.stats
            stat["qid"] = qid
            stat["has_neg"] = has_neg
            stat["n_exclusion_units"] = n_excl
            all_stats.append(stat)

        return S_final, all_stats

    def run(
        self,
        lambda_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run TRACE evaluation on FollowIR."""
        logger.info("=" * 60)
        logger.info("Starting TRACE evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        if lambda_list is None:
            lambda_list = [0.5, 1.0, 1.5, 2.0]
        if tau_list is None:
            tau_list = [0.1, 0.2, 0.5, 1.0]

        total_trials = len(lambda_list) * len(tau_list)
        logger.info(f"Grid search: {total_trials} combinations (lambda x tau)")

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
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name,
                                      self.retriever.doc_embeddings, self.retriever.doc_ids)
        else:
            logger.info("Encoding candidate documents...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
            if self.use_cache:
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name,
                                      self.retriever.doc_embeddings, self.retriever.doc_ids)

        # Build query lists for OG and changed
        def build_query_lists(queries, raw_queries):
            """Build query lists supporting both v6 (q_minus string) and v7 (q_minus_list).

            Returns q_neg_list_per_query: List[List[str]] where each inner list holds
            M_i exclusion queries (M_i=0 means no exclusion).
            """
            query_ids, q_base_list, q_pos_list = [], [], []
            q_neg_list_per_query: List[List[str]] = []
            has_neg_list = []
            for qid in queries.keys():
                query_ids.append(qid)
                raw = raw_queries.get(qid, ("", ""))
                query_text, instruction = raw[0], raw[1]
                q_base = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")
                q_base_list.append(q_base)

                d = dual_data.get(qid, {})
                q_plus = d.get("q_plus", "")
                q_pos_list.append(q_plus if not self._is_none_query(q_plus) else "")

                # Support q_minus_list (v7) or q_minus (v6) for backward compatibility
                q_minus_list_field = d.get("q_minus_list", None)
                if q_minus_list_field is not None:
                    # v7 format: list of exclusion units
                    valid_q_minus = [
                        qm for qm in q_minus_list_field
                        if qm and not self._is_none_query(qm)
                    ]
                    q_neg_list_per_query.append(valid_q_minus)
                    has_neg_list.append(1.0 if valid_q_minus else 0.0)
                else:
                    # v6 format: single string
                    q_minus = d.get("q_minus", "")
                    if not self._is_none_query(q_minus):
                        q_neg_list_per_query.append([q_minus])
                        has_neg_list.append(1.0)
                    else:
                        q_neg_list_per_query.append([])
                        has_neg_list.append(0.0)
            return query_ids, q_base_list, q_pos_list, q_neg_list_per_query, has_neg_list

        query_ids_og, q_base_og, q_pos_og, q_neg_list_per_query_og, has_neg_og = build_query_lists(q_og, q_raw_og)
        query_ids_ch, q_base_ch, q_pos_ch, q_neg_list_per_query_ch, has_neg_ch = build_query_lists(q_changed, q_raw_changed)

        has_neg_mask_og = torch.tensor(has_neg_og, dtype=torch.float32)
        has_neg_mask_ch = torch.tensor(has_neg_ch, dtype=torch.float32)

        # Encode Q_full and Q_pos (single string per query, unchanged)
        logger.info("Encoding OG queries (Q_full, Q_pos)...")
        q_full_emb_og = self._encode_queries(q_base_og)
        q_pos_emb_og = self._encode_queries(q_pos_og)

        logger.info("Encoding Changed queries (Q_full, Q_pos)...")
        q_full_emb_ch = self._encode_queries(q_base_ch)
        q_pos_emb_ch = self._encode_queries(q_pos_ch)

        # Encode Q_minus: flatten all exclusion units across queries, then group by query
        def encode_and_group_q_minus(q_neg_list_per_query, label):
            """Encode all q_minus texts and group embeddings by query.

            Returns:
                S_neg_list_per_query: List[Optional[List[Tensor]]] where each Tensor
                    has shape (n_docs,) and length equals n_queries.
                S_neg_legacy: Tensor of shape (n_queries, n_docs) using the first
                    exclusion unit per query (for backward-compatible diagnostics).
            """
            all_q_minus_texts: List[str] = []
            q_minus_ranges: List[Optional[Tuple[int, int]]] = []
            for q_minus_list in q_neg_list_per_query:
                if q_minus_list:
                    start = len(all_q_minus_texts)
                    all_q_minus_texts.extend(q_minus_list)
                    end = len(all_q_minus_texts)
                    q_minus_ranges.append((start, end))
                else:
                    q_minus_ranges.append(None)

            if not all_q_minus_texts:
                # No exclusion across all queries: return empty structures
                n_queries = len(q_neg_list_per_query)
                dummy = torch.zeros(
                    (n_queries, self.retriever.doc_embeddings.shape[0]),
                    device=self.retriever.doc_embeddings.device,
                    dtype=self.retriever.doc_embeddings.dtype,
                )
                return [None] * n_queries, dummy

            logger.info(f"Encoding {label} Q_minus ({len(all_q_minus_texts)} exclusion units)...")
            q_neg_emb_all = self._encode_queries(all_q_minus_texts)
            q_neg_emb_all = q_neg_emb_all.to(
                device=self.retriever.doc_embeddings.device,
                dtype=self.retriever.doc_embeddings.dtype,
            )

            S_neg_list_per_query: List[Optional[List[torch.Tensor]]] = []
            S_neg_first_unit: List[torch.Tensor] = []  # for legacy S_neg
            doc_emb = self.retriever.doc_embeddings
            for i, rng in enumerate(q_minus_ranges):
                if rng is None:
                    S_neg_list_per_query.append(None)
                    S_neg_first_unit.append(torch.zeros(
                        doc_emb.shape[0], device=doc_emb.device, dtype=doc_emb.dtype
                    ))
                else:
                    start, end = rng
                    q_emb_i = q_neg_emb_all[start:end]  # (M_i, dim)
                    S_neg_i = torch.matmul(q_emb_i, doc_emb.T)  # (M_i, n_docs)
                    S_neg_list_per_query.append([S_neg_i[m] for m in range(S_neg_i.shape[0])])
                    S_neg_first_unit.append(S_neg_i[0])  # first unit for legacy

            S_neg_legacy = torch.stack(S_neg_first_unit)  # (n_queries, n_docs)
            return S_neg_list_per_query, S_neg_legacy

        S_neg_list_per_query_og, S_neg_og = encode_and_group_q_minus(
            q_neg_list_per_query_og, "OG"
        )
        S_neg_list_per_query_ch, S_neg_ch = encode_and_group_q_minus(
            q_neg_list_per_query_ch, "Changed"
        )

        device = self.retriever.doc_embeddings.device
        score_dtype = self.retriever.doc_embeddings.dtype
        q_full_emb_og = q_full_emb_og.to(device=device, dtype=score_dtype)
        q_pos_emb_og = q_pos_emb_og.to(device=device, dtype=score_dtype)
        q_full_emb_ch = q_full_emb_ch.to(device=device, dtype=score_dtype)
        q_pos_emb_ch = q_pos_emb_ch.to(device=device, dtype=score_dtype)
        has_neg_mask_og = has_neg_mask_og.to(device)
        has_neg_mask_ch = has_neg_mask_ch.to(device)

        # Compute cosine scores: S_full, S_pos (S_neg already computed above)
        logger.info("Computing S_full, S_pos...")
        S_full_og = torch.matmul(q_full_emb_og, self.retriever.doc_embeddings.T)
        S_pos_og = torch.matmul(q_pos_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = S_neg_og * has_neg_mask_og.unsqueeze(1)

        S_full_ch = torch.matmul(q_full_emb_ch, self.retriever.doc_embeddings.T)
        S_pos_ch = torch.matmul(q_pos_emb_ch, self.retriever.doc_embeddings.T)
        S_neg_ch = S_neg_ch * has_neg_mask_ch.unsqueeze(1)

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        # Truncate candidate indices for TRACE scoring to top-K
        # Keep full candidates for evaluation; only limit TRACE scoring depth
        if self.candidate_depth > 0:
            truncated_indices = {}
            for qid, indices in qid_to_candidate_indices.items():
                truncated_indices[qid] = indices[:self.candidate_depth]
            qid_to_candidate_indices = truncated_indices
            logger.info(f"TRACE scoring limited to top-{self.candidate_depth} candidates per query")

        # ====== Regression Diagnostic Analysis ======
        logger.info("\n" + "=" * 60)
        logger.info("REGRESSION DIAGNOSTIC ANALYSIS")
        logger.info("=" * 60)

        regression_diagnostics = self._diagnose_regression(
            S_full_ch, S_pos_ch, S_neg_ch,
            has_neg_mask_ch, query_ids_ch, qid_to_candidate_indices
        )

        # Print summary
        self._print_regression_summary(regression_diagnostics)

        # ====== Grid Search ======
        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        best_per_query_stats: List[Dict[str, Any]] = []
        all_results: List[Dict[str, Any]] = []

        trial_idx = 0
        for lam in lambda_list:
            for tau_d in tau_list:
                trial_idx += 1
                self.lambda_boundary = lam
                self.tau_decay = tau_d

                S_final_ch, per_query_stats = self.compute_trace_scores(
                    S_full_ch, S_pos_ch, S_neg_list_per_query_ch,
                    has_neg_mask_ch, query_ids_ch, qid_to_candidate_indices
                )

                results_og = self._extract_results(S_full_og, query_ids_og, candidates)
                results_changed = self._extract_results(S_final_ch, query_ids_ch, candidates)

                evaluator = FollowIREvaluator(self.task_name)
                metrics = evaluator.evaluate(results_og, results_changed)

                p_mrr = metrics.get("p-MRR", 0.0)
                og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

                logger.info(
                    "[%d/%d] lambda=%.1f, tau=%.2f: p-MRR=%.4f, Changed_MAP=%.4f, Changed_nDCG@5=%.4f",
                    trial_idx, total_trials,
                    lam, tau_d,
                    p_mrr, changed_map, changed_ndcg,
                )

                # target_avg
                target_avg = metrics.get("target_avg", 0.0)

                all_results.append({
                    "lambda": lam,
                    "tau_decay": tau_d,
                    "huber_delta": self.huber_delta,
                    "regression_mode": self.regression_mode,
                    "ablation": self.ablation,
                    "p-MRR": p_mrr,
                    "target_avg": target_avg,
                    "og_MAP@1000": og_map,
                    "changed_MAP@1000": changed_map,
                    "changed_nDCG@5": changed_ndcg,
                })

                composite = p_mrr + changed_map + changed_ndcg
                if best_metrics is None:
                    best_metrics = metrics
                    best_params = {"lambda": lam, "tau_decay": tau_d}
                    best_results_og = results_og
                    best_results_changed = results_changed
                    best_per_query_stats = list(per_query_stats)
                else:
                    best_composite = (
                        best_metrics.get("p-MRR", 0.0)
                        + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                        + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    )
                    if composite > best_composite:
                        best_metrics = metrics
                        best_params = {"lambda": lam, "tau_decay": tau_d}
                        best_results_og = results_og
                        best_results_changed = results_changed
                        best_per_query_stats = list(per_query_stats)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("TRACE evaluation complete")
        logger.info(f"  Best: lambda={best_params['lambda']}, tau={best_params['tau_decay']}")
        logger.info(f"  p-MRR: {best_metrics.get('p-MRR', 0.0):.4f}")
        logger.info(f"  Changed MAP@1000: {best_metrics.get('changed', {}).get('map_at_1000', 0.0):.4f}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            best_params, best_metrics, all_results,
            best_results_og, best_results_changed,
            best_per_query_stats, regression_diagnostics,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "regression_diagnostics": regression_diagnostics,
            "elapsed": elapsed,
        }

    def _diagnose_regression(
        self,
        S_full: torch.Tensor,
        S_pos: torch.Tensor,
        S_neg: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        """Diagnose regression quality for each query's candidate set."""
        diagnostics = []

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            has_neg = bool(has_neg_mask[q_idx].item() > 0)
            if not has_neg:
                continue

            idx_tensor = torch.tensor(cand_indices, device=S_full.device, dtype=torch.long)
            s_f = S_full[q_idx].index_select(0, idx_tensor).float()
            s_p = S_pos[q_idx].index_select(0, idx_tensor).float()
            s_n = S_neg[q_idx].index_select(0, idx_tensor).float()

            n = s_f.numel()
            if n < 5:
                continue

            # Raw correlations
            full_pos_corr = float(torch.corrcoef(torch.stack([s_f, s_p]))[0, 1].item()) if n > 2 else 0.0
            full_neg_corr = float(torch.corrcoef(torch.stack([s_f, s_n]))[0, 1].item()) if n > 2 else 0.0
            pos_neg_corr = float(torch.corrcoef(torch.stack([s_p, s_n]))[0, 1].item()) if n > 2 else 0.0

            # After robust standardization
            z_p = robust_standardize(s_p, self.eps)
            z_n = robust_standardize(s_n, self.eps)
            pos_neg_corr_z = float(torch.corrcoef(torch.stack([z_p, z_n]))[0, 1].item()) if n > 2 else 0.0

            # Huber fit
            a_hat, b_hat = fit_huber_regression(z_n, z_p, delta=self.huber_delta)

            # OLS for comparison
            mean_p = z_p.mean()
            mean_n = z_n.mean()
            var_p = ((z_p - mean_p) ** 2).sum()
            if var_p > 1e-12:
                b_ols = float(((z_p - mean_p) * (z_n - mean_n)).sum().item() / var_p.item())
                a_ols = float(mean_n.item()) - b_ols * float(mean_p.item())
            else:
                b_ols = 0.0
                a_ols = float(mean_n.item())

            # Residuals
            e_huber = z_n - a_hat - b_hat * z_p
            e_ols = z_n - a_ols - b_ols * z_p

            # R^2 for both
            ss_tot = ((z_n - z_n.mean()) ** 2).sum()
            r2_huber = 1.0 - float((e_huber ** 2).sum().item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0
            r2_ols = 1.0 - float((e_ols ** 2).sum().item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0

            # Residual MAD
            e_huber_mad = float(_mad(e_huber, self.eps).item())
            e_ols_mad = float(_mad(e_ols, self.eps).item())

            diagnostics.append({
                "qid": qid,
                "n_candidates": n,
                "s_full_range": f"[{s_f.min():.4f}, {s_f.max():.4f}]",
                "s_pos_range": f"[{s_p.min():.4f}, {s_p.max():.4f}]",
                "s_neg_range": f"[{s_n.min():.4f}, {s_n.max():.4f}]",
                "s_pos_mad": float(_mad(s_p, self.eps).item()),
                "s_neg_mad": float(_mad(s_n, self.eps).item()),
                "corr_full_pos": full_pos_corr,
                "corr_full_neg": full_neg_corr,
                "corr_pos_neg": pos_neg_corr,
                "corr_pos_neg_z": pos_neg_corr_z,
                "huber_a": a_hat,
                "huber_b": b_hat,
                "huber_r2": r2_huber,
                "huber_residual_mad": e_huber_mad,
                "ols_a": a_ols,
                "ols_b": b_ols,
                "ols_r2": r2_ols,
                "ols_residual_mad": e_ols_mad,
            })

        return diagnostics

    def _print_regression_summary(self, diagnostics: List[Dict[str, Any]]) -> None:
        """Print a summary of regression diagnostics."""
        if not diagnostics:
            logger.info("No queries with exclusion conditions found.")
            return

        n = len(diagnostics)
        avg_r2_huber = sum(d["huber_r2"] for d in diagnostics) / n
        avg_r2_ols = sum(d["ols_r2"] for d in diagnostics) / n
        avg_b_huber = sum(d["huber_b"] for d in diagnostics) / n
        avg_b_ols = sum(d["ols_b"] for d in diagnostics) / n
        avg_corr = sum(d["corr_pos_neg"] for d in diagnostics) / n
        avg_corr_z = sum(d["corr_pos_neg_z"] for d in diagnostics) / n

        # Distribution of b_hat
        b_huber_vals = [d["huber_b"] for d in diagnostics]
        b_positive = sum(1 for b in b_huber_vals if b > 0)
        b_negative = sum(1 for b in b_huber_vals if b <= 0)

        logger.info(f"\n  Queries with exclusion: {n}")
        logger.info(f"  Avg corr(S_pos, S_neg) raw:     {avg_corr:.4f}")
        logger.info(f"  Avg corr(z_pos, z_neg) robust:  {avg_corr_z:.4f}")
        logger.info(f"  Avg Huber R^2:  {avg_r2_huber:.4f}")
        logger.info(f"  Avg OLS R^2:    {avg_r2_ols:.4f}")
        logger.info(f"  Avg Huber b_hat: {avg_b_huber:.4f} (positive: {b_positive}, negative: {b_negative})")
        logger.info(f"  Avg OLS b:       {avg_b_ols:.4f}")
        logger.info(f"  Huber b range: [{min(b_huber_vals):.4f}, {max(b_huber_vals):.4f}]")

        # R^2 distribution
        r2_bins = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, ">0.7": 0}
        for d in diagnostics:
            r2 = d["huber_r2"]
            if r2 < 0.1:
                r2_bins["<0.1"] += 1
            elif r2 < 0.3:
                r2_bins["0.1-0.3"] += 1
            elif r2 < 0.5:
                r2_bins["0.3-0.5"] += 1
            elif r2 < 0.7:
                r2_bins["0.5-0.7"] += 1
            else:
                r2_bins[">0.7"] += 1
        logger.info(f"  R^2 distribution: {r2_bins}")

        # Feasibility assessment
        if avg_r2_huber > 0.3:
            logger.info("  ASSESSMENT: Regression is FEASIBLE - S_pos explains significant variance in S_neg")
        elif avg_r2_huber > 0.1:
            logger.info("  ASSESSMENT: Regression is PARTIALLY feasible - weak but present S_pos->S_neg relationship")
        else:
            logger.info("  ASSESSMENT: Regression is WEAK - S_pos does not explain much variance in S_neg")

    def _save_results(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        per_query_stats: List[Dict[str, Any]],
        regression_diagnostics: List[Dict[str, Any]],
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "TRACE",
            "dual_queries_source": self.dual_queries_path,
            "fixed_params": {
                "huber_delta": self.huber_delta,
                "regression_mode": self.regression_mode,
                "normalization_mode": self.normalization_mode,
                "uncentered_residual": self.uncentered_residual,
                "constrained_slope": self.constrained_slope,
                "raw_score_fit": self.raw_score_fit,
                "ablation": self.ablation,
            },
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
            },
        }

        summary_path = os.path.join(self.output_dir, "trace_metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "trace_all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        diag_path = os.path.join(self.output_dir, "trace_regression_diagnostics.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(regression_diagnostics, f, indent=2, ensure_ascii=False)

        if per_query_stats:
            pqs_path = os.path.join(self.output_dir, "trace_per_query_stats.json")
            with open(pqs_path, "w", encoding="utf-8") as f:
                json.dump(per_query_stats, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def run_trace(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    huber_delta: float = 1.345,
    lambda_boundary: float = 1.0,
    tau_decay: float = 0.2,
    regression_mode: str = "huber",
    normalization_mode: str = "median_mad",
    uncentered_residual: bool = False,
    constrained_slope: bool = False,
    raw_score_fit: bool = False,
    ablation: str = "full",
    candidate_depth: int = 0,
    eps: float = 1e-6,
    use_cache: bool = True,
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/trace/{task_name}"
    if not dual_queries_path:
        raise ValueError("dual_queries_path is required")

    engine = TRACEEvaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        huber_delta=huber_delta,
        lambda_boundary=lambda_boundary,
        tau_decay=tau_decay,
        eps=eps,
        regression_mode=regression_mode,
        normalization_mode=normalization_mode,
        uncentered_residual=uncentered_residual,
        constrained_slope=constrained_slope,
        raw_score_fit=raw_score_fit,
        ablation=ablation,
        candidate_depth=candidate_depth,
        use_cache=use_cache,
        device=device,
        batch_size=batch_size,
    )

    return engine.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRACE evaluator")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--huber_delta", type=float, default=1.345)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--tau_decay", type=float, default=0.2)
    parser.add_argument("--regression_mode", type=str, default="huber",
                        choices=["huber", "ols"])
    parser.add_argument("--normalization_mode", type=str, default="median_mad",
                        choices=["median_mad", "mean_std"])
    parser.add_argument("--uncentered_residual", type=str, default="false",
                        help="Skip residual recentering (diagnostic ablation)")
    parser.add_argument("--constrained_slope", type=str, default="false",
                        help="Enforce nonnegative fitted slope (diagnostic ablation)")
    parser.add_argument("--raw_score_fit", type=str, default="false",
                        help="Fit regression before channel normalization (diagnostic ablation)")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "z_full_only", "no_regression", "no_gate", "pos_only", "linear", "raw_neg_subtract", "raw_neg_fusion", "gate_only"])
    parser.add_argument("--candidate_depth", type=int, default=0,
                        help="Truncate candidates to top-K (0 = use all)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Numerical floor for MAD standardization")
    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    use_cache = args.use_cache.lower() == "true"
    uncentered_residual = args.uncentered_residual.lower() == "true"
    constrained_slope = args.constrained_slope.lower() == "true"
    raw_score_fit = args.raw_score_fit.lower() == "true"

    result = run_trace(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        huber_delta=args.huber_delta,
        lambda_boundary=args.lambda_boundary,
        tau_decay=args.tau_decay,
        regression_mode=args.regression_mode,
        normalization_mode=args.normalization_mode,
        uncentered_residual=uncentered_residual,
        constrained_slope=constrained_slope,
        raw_score_fit=raw_score_fit,
        ablation=args.ablation,
        candidate_depth=args.candidate_depth,
        eps=args.eps,
        use_cache=use_cache,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nFinal p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
