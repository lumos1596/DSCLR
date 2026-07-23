"""Runtime residual boundary for DeIR-Dual.

This module converts the query-query background/exclusion overlap into a
query-document negative-score baseline, then penalizes only residual
negative evidence above that baseline.

Boundary modes:
  "cos_transfer" (default):
      tau_bg(d) = mean(S_neg) + std(S_neg) * cos(Q_base, Q_neg) * z_base(d)
      Uses query-query cosine as heuristic scale-transfer coefficient.

  "regression_bg":
      Fits Huber regression S_neg = a_hat + b_hat * S_base per query.
      If R^2 > r2_threshold: uses regression-based tau_bg = a_hat + b_hat * S_base
      If R^2 <= r2_threshold: falls back to cos_transfer
      Data-driven replacement of the heuristic cos transfer when regression
      is reliable; preserves V8.6's margin/safety/penalty machinery unchanged.

Adaptive margin (target_at_risk > 0):
    Uses quantile-based margin instead of margin_scale * MAD.
    Controls the fraction of at-risk docs to target_at_risk, ensuring
    consistent conservativeness across queries and boundary modes.
    Particularly important for regression_bg: regression residuals are more
    concentrated (smaller MAD), so fixed margin_scale over-flags at-risk docs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ResidualBoundaryOutput:
    tau: torch.Tensor
    residual: torch.Tensor
    overflow: torch.Tensor
    penalty_signal: torch.Tensor
    margin: float
    mad: float  # MAD(R_neg), used for safety gate normalization
    stats: Dict[str, float]


def _safe_std(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.ones((), device=x.device, dtype=x.dtype) * eps
    return x.std(unbiased=False).clamp_min(eps)


def _mad(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.numel() == 0:
        return torch.ones((), device=x.device, dtype=x.dtype) * eps
    med = x.median()
    return (x - med).abs().median().clamp_min(eps)


def _fit_huber(
    y: torch.Tensor,
    X: torch.Tensor,
    delta: float = 1.345,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> Tuple[float, float, float]:
    """Fit y = a + b*X using Huber loss (IRLS). Returns (a_hat, b_hat, R^2)."""
    n = y.numel()
    if n < 3:
        return float(y.mean().item()), 0.0, 0.0
    y_f = y.float()
    X_f = X.float()
    mean_X = X_f.mean()
    mean_y = y_f.mean()
    var_X = ((X_f - mean_X) ** 2).sum()
    if var_X < 1e-12:
        return float(mean_y.item()), 0.0, 0.0
    b = float(((X_f - mean_X) * (y_f - mean_y)).sum().item() / var_X.item())
    a = float(mean_y.item()) - b * float(mean_X.item())
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
    pred_final = a + b * X_f
    ss_res = ((y_f - pred_final) ** 2).sum()
    ss_tot = ((y_f - mean_y) ** 2).sum()
    r2 = 1.0 - float(ss_res.item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0
    return a, b, r2


def compute_background_residual_boundary(
    s_base: torch.Tensor,
    s_neg: torch.Tensor,
    cos_qbase_qneg: float,
    *,
    boundary_mode: str = "cos_transfer",
    margin_scale: float = 1.0,
    r2_threshold: float = 0.3,
    huber_delta: float = 1.345,
    compute_r2: bool = False,
    target_at_risk: float = 0.0,
    r2_tar_scale: float = 0.0,
    blend_weight: float = 1.0,
    regression_type: str = "huber",
    quantile_level: float = 0.95,
    eps: float = 1e-6,
) -> ResidualBoundaryOutput:
    """Compute a background-calibrated residual exclusion boundary.

    Args:
        boundary_mode:
            "cos_transfer" - original heuristic: tau = mean(S_neg) + std(S_neg) * c_q * z_base
            "regression_bg" - Huber regression boundary when R^2 > r2_threshold,
                else cos_transfer fallback. Set r2_threshold=0.0 for pure regression.
        r2_threshold: minimum R^2 to use regression boundary (only for "regression_bg").
        huber_delta: Huber loss delta parameter (only for "regression_bg")
        compute_r2: if True, always compute R^2 and include in stats.
        target_at_risk: base target fraction of candidates with overflow > 0.
            When > 0, uses quantile-based adaptive margin instead of margin_scale * MAD.
            When r2_tar_scale > 0, the effective target_at_risk is modulated per query:
                effective_tar = target_at_risk * (1 + r2_tar_scale * R^2)
            This means high-R^2 queries get more aggressive boundaries (boundary is
            reliable, residual is genuine signal), while low-R^2 queries stay conservative.
        r2_tar_scale: R^2 scaling factor for per-query target_at_risk adjustment.
            When > 0, high R^2 queries get higher effective target_at_risk.
            E.g., r2_tar_scale=1.0 means R^2=1.0 queries get 2x the base target_at_risk.
            This is the key mechanism that makes regression useful: R^2 provides a
            per-query quality diagnostic that calibrates boundary aggressiveness.
        blend_weight: weight for regression boundary vs cos_transfer (regression_bg only).
        regression_type: type of regression to fit.
        quantile_level: quantile for quantile regression.
    """
    if s_base.shape != s_neg.shape:
        raise ValueError("s_base and s_neg must have the same shape")

    dtype = s_base.dtype
    device = s_base.device
    s_base_f = s_base.float()
    s_neg_f = s_neg.float()

    mean_b = s_base_f.mean()
    std_b = _safe_std(s_base_f, eps)
    mean_n = s_neg_f.mean()
    std_n = _safe_std(s_neg_f, eps)

    c_q = float(max(-1.0, min(1.0, cos_qbase_qneg)))
    z_base = (s_base_f - mean_b) / std_b

    used_regression = False
    a_hat, b_hat, r2 = 0.0, 0.0, 0.0

    if boundary_mode == "regression_bg" or compute_r2:
        if regression_type == "quantile":
            a_hat, b_hat, r2 = _fit_quantile_regression(s_neg_f, s_base_f, quantile=quantile_level)
        elif regression_type == "ols":
            a_hat, b_hat, r2 = _fit_huber(s_neg_f, s_base_f, delta=1e6)  # OLS = Huber with large delta
        else:  # "huber" (default)
            a_hat, b_hat, r2 = _fit_huber(s_neg_f, s_base_f, delta=huber_delta)

    tau_cos = mean_n + std_n * c_q * z_base
    if boundary_mode == "regression_bg" and r2 > r2_threshold:
        tau_reg = a_hat + b_hat * s_base_f
        if blend_weight >= 1.0:
            tau_bg = tau_reg
        else:
            tau_bg = blend_weight * tau_reg + (1.0 - blend_weight) * tau_cos
        used_regression = True
    else:
        tau_bg = tau_cos

    residual = s_neg_f - tau_bg
    mad_val = _mad(residual, eps)

    # Adaptive margin: use quantile to control at-risk ratio
    # R²-gated: high R² → more aggressive (reliable boundary), low R² → conservative
    effective_tar = target_at_risk
    if target_at_risk > 0 and r2_tar_scale > 0 and r2 > 0:
        effective_tar = target_at_risk * (1.0 + r2_tar_scale * r2)

    if effective_tar > 0 and residual.numel() > 10:
        margin_val = torch.quantile(residual, 1.0 - effective_tar)
        residual_margin = margin_val
        mad_f = float(mad_val.item())
        effective_margin_scale = float((margin_val / mad_val).item()) if mad_f > eps else margin_scale
    else:
        residual_margin = margin_scale * mad_val
        effective_margin_scale = margin_scale

    overflow = residual - residual_margin
    penalty_signal = F.softplus(overflow)

    stats = {
        "boundary_mean_base": float(mean_b.item()),
        "boundary_std_base": float(std_b.item()),
        "boundary_mean_neg": float(mean_n.item()),
        "boundary_std_neg": float(std_n.item()),
        "boundary_cos": c_q,
        "boundary_margin": float(residual_margin.item()),
        "boundary_effective_margin_scale": effective_margin_scale,
        "boundary_tau_mean": float(tau_bg.mean().item()),
        "boundary_tau_min": float(tau_bg.min().item()),
        "boundary_tau_max": float(tau_bg.max().item()),
        "boundary_residual_mean": float(residual.mean().item()),
        "boundary_residual_max": float(residual.max().item()),
        "boundary_at_risk_ratio": float((overflow > 0).float().mean().item()),
        "boundary_used_regression": used_regression,
        "boundary_target_at_risk": target_at_risk,
        "boundary_effective_tar": effective_tar,
        "boundary_r2_tar_scale": r2_tar_scale,
        "boundary_blend_weight": blend_weight,
        "boundary_regression_type": regression_type,
    }
    if boundary_mode == "regression_bg" or compute_r2:
        stats["boundary_r2"] = r2
        stats["boundary_a_hat"] = a_hat
        stats["boundary_b_hat"] = b_hat

    return ResidualBoundaryOutput(
        tau=tau_bg.to(dtype=dtype, device=device),
        residual=residual.to(dtype=dtype, device=device),
        overflow=overflow.to(dtype=dtype, device=device),
        penalty_signal=penalty_signal.to(dtype=dtype, device=device),
        margin=float(residual_margin.item()),
        mad=float(mad_val.item()),
        stats=stats,
    )


def _fit_quantile_regression(
    y: torch.Tensor,
    X: torch.Tensor,
    quantile: float = 0.95,
    max_iter: int = 500,
    tol: float = 1e-7,
    lr: float = 0.01,
) -> Tuple[float, float, float]:
    """Fit y = a + b*X using quantile (pinball) loss via gradient descent.
    
    Directly estimates the conditional quantile Q_{quantile}(y | X),
    avoiding the need for a separate margin calibration step.
    
    Returns (a_hat, b_hat, pseudo_R2).
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
    
    # Gradient descent on pinball loss
    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    
    for _ in range(max_iter):
        pred = a_t + b_t * X_f
        err = y_f - pred
        # Pinball loss gradient
        grad = torch.where(err >= 0, quantile, quantile - 1.0)
        ga = -grad.mean()
        gb = -(grad * X_f).mean()
        
        a_t = a_t - lr * ga
        b_t = b_t - lr * gb
        
        if abs(ga.item()) < tol and abs(gb.item()) < tol:
            break
    
    a_final = float(a_t.item())
    b_final = float(b_t.item())
    
    # Pseudo R²: fraction of y below the estimated quantile
    pred_final = a_final + b_final * X_f
    pseudo_r2 = float((y_f <= pred_final).float().mean().item())
    
    return a_final, b_final, pseudo_r2
