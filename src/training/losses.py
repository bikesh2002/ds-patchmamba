"""
Loss functions for DS-PatchMamba training.

Total loss:
    L = alpha * MSE_fine + (1 - alpha) * MSE_coarse + beta * L_cons

L_cons is the per-timestep scale-normalized consistency regulariser.
It penalises the two streams for disagreeing on WHICH timesteps are
anomalous (high error at different positions), not just for different magnitudes.

Scale-normalization removes the natural amplitude difference between
the fine (P_f=12) and coarse (P_c=48) reconstruction streams.
Distribution-free and robust to heavy-tailed reconstruction errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error between input and reconstruction.
    x, x_hat: (B, T, V)  — time-aligned after patch back-projection
    Returns scalar.
    """
    return F.mse_loss(x_hat, x, reduction='mean')


def consistency_loss(
    e_fine: torch.Tensor,
    e_coarse: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-timestep scale-normalized consistency regulariser.

    Penalises scale disagreement at the timestep level:
        L_cons = (1/BL) * sum_{b,t} [ e_f_norm(b,t) - e_c_norm(b,t) ]^2

    where:
        e_f_norm(b,t) = e_f(b,t) / (batch_mean(e_f) + eps)
        e_c_norm(b,t) = e_c(b,t) / (batch_mean(e_c) + eps)

    e_fine, e_coarse: (B, T) — per-timestep reconstruction errors
                               averaged across channels (V-dim already reduced)
    Returns scalar.
    """
    mean_f = e_fine.mean() + eps      # scalar — batch-level normalization
    mean_c = e_coarse.mean() + eps

    e_f_norm = e_fine  / mean_f       # (B, T)
    e_c_norm = e_coarse / mean_c      # (B, T)

    return F.mse_loss(e_f_norm, e_c_norm, reduction='mean')


class DSPatchMambaLoss(nn.Module):
    """
    Combined loss for DS-PatchMamba.

    Args:
        alpha: weight for fine-stream MSE (1-alpha for coarse)
        beta:  weight for consistency regulariser
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        x: torch.Tensor,           # (B, L, V) — original input window
        x_hat_fine: torch.Tensor,  # (B, L, V) — fine reconstruction (time-aligned)
        x_hat_coarse: torch.Tensor,# (B, L, V) — coarse reconstruction (time-aligned)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, dict of components for logging).
        """
        loss_fine   = reconstruction_loss(x, x_hat_fine)
        loss_coarse = reconstruction_loss(x, x_hat_coarse)

        # Per-timestep, per-channel squared errors → reduce to (B, L) per-timestep errors
        e_fine   = ((x - x_hat_fine)  ** 2).mean(dim=-1)    # (B, L)
        e_coarse = ((x - x_hat_coarse)** 2).mean(dim=-1)    # (B, L)

        loss_cons = consistency_loss(e_fine, e_coarse)

        total = (
            self.alpha       * loss_fine
            + (1 - self.alpha) * loss_coarse
            + self.beta        * loss_cons
        )

        return total, {
            "loss_total":   total.item(),
            "loss_fine":    loss_fine.item(),
            "loss_coarse":  loss_coarse.item(),
            "loss_cons":    loss_cons.item(),
        }
