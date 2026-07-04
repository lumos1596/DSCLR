"""Runtime residual boundary for DeIR-Dual.

This module converts the query-query background/exclusion overlap into a
query-document negative-score baseline, then penalizes only residual
negative evidence above that baseline.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class ResidualBoundaryOutput:
    tau: torch.Tensor
    residual: torch.Tensor
    overflow: torch.Tensor
    penalty_signal: torch.Tensor
    margin: float
    mad: float  # MAD(R_neg)，用于 safety gate 归一化
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


def compute_background_residual_boundary(
    s_base: torch.Tensor,
    s_neg: torch.Tensor,
    cos_qbase_qneg: float,
    *,
    margin_scale: float = 1.0,
    eps: float = 1e-6,
) -> ResidualBoundaryOutput:
    """Compute a background-calibrated residual exclusion boundary.

    The query-query cosine is used as a scale-transfer coefficient rather than
    as a direct query-document threshold. For each candidate document, we
    estimate the negative-channel score explainable by base-topic relevance:

        tau_bg(d) = mean(S_neg) + std(S_neg) * c_q * z_base(d)

    where z_base(d) is the standardized base score in the current candidate
    set. The residual S_neg(d) - tau_bg(d) is treated as exclusion evidence
    only when it exceeds a robust residual margin.
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

    # Cosine can be slightly outside [-1, 1] due to numeric noise.
    c_q = float(max(-1.0, min(1.0, cos_qbase_qneg)))
    z_base = (s_base_f - mean_b) / std_b
    tau_bg = mean_n + std_n * c_q * z_base

    residual = s_neg_f - tau_bg
    mad_val = _mad(residual, eps)
    residual_margin = margin_scale * mad_val
    overflow = residual - residual_margin
    penalty_signal = F.softplus(overflow)

    stats = {
        "boundary_mean_base": float(mean_b.item()),
        "boundary_std_base": float(std_b.item()),
        "boundary_mean_neg": float(mean_n.item()),
        "boundary_std_neg": float(std_n.item()),
        "boundary_cos": c_q,
        "boundary_margin": float(residual_margin.item()),
        "boundary_tau_mean": float(tau_bg.mean().item()),
        "boundary_tau_min": float(tau_bg.min().item()),
        "boundary_tau_max": float(tau_bg.max().item()),
        "boundary_residual_mean": float(residual.mean().item()),
        "boundary_residual_max": float(residual.max().item()),
        "boundary_at_risk_ratio": float((overflow > 0).float().mean().item()),
    }

    return ResidualBoundaryOutput(
        tau=tau_bg.to(dtype=dtype, device=device),
        residual=residual.to(dtype=dtype, device=device),
        overflow=overflow.to(dtype=dtype, device=device),
        penalty_signal=penalty_signal.to(dtype=dtype, device=device),
        margin=float(residual_margin.item()),
        mad=float(mad_val.item()),
        stats=stats,
    )
