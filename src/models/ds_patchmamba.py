"""
DS-PatchMamba — Full Model Architecture

Dual-Scale Patch Mamba for Multivariate Time-Series Anomaly Detection.

Architecture pipeline (from WORKFLOW.md Section 3):
    Input (B, L, V)
        ↓
    DualScaleTokenizer  →  fine tokens (B, V, N_f, d) + coarse tokens (B, V, N_c, d)
        ↓
    MambaEncoder (fine)  →  (B, V, N_f, d)
    MambaEncoder (coarse)→  (B, V, N_c, d)
        ↓
    PartialChannelDependence (CP module)
        ↓
    Fine recon head   d → d'=64 → P_f×V  →  x_hat_fine   (B, L, V)
    Coarse recon head d → d'=64 → P_c×V  →  x_hat_coarse (B, L, V)
        ↓
    Per-timestep errors e_f, e_c → fused score s = w*e_f + (1-w)*e_c
        ↓
    DSPOT threshold → binary anomaly labels

Total parameters: ~7M | VRAM: <2 GB at batch=64 on T4

Ablation variants are controlled via the AblationConfig dataclass:
    - use_coarse=False        → A1 (fine-only)
    - use_fine=False          → A2 (coarse-only)
    - single_scale=True       → A3 (single scale P=24)
    - use_attention=True      → A4 (attention backbone instead of Mamba)
    - use_cd=False            → A5 (CI-only, no cross-variable module)
    - use_bottleneck=False    → A6 (no bottleneck, d'=d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional

from .patch_tokenizer import DualScaleTokenizer, PatchTokenizer
from .mamba_encoder import MambaEncoder, AttentionEncoder
from .partial_cd import PartialChannelDependence


@dataclass
class ModelConfig:
    """All architecture hyperparameters in one place."""
    # Patching
    patch_size_fine:   int   = 12
    patch_size_coarse: int   = 48
    window_length:     int   = 300

    # Mamba-2
    d_model:           int   = 128
    d_state:           int   = 64
    expand:            int   = 2
    d_conv:            int   = 4
    num_heads:         int   = 4
    num_mamba_blocks:  int   = 2

    # Partial CD
    cd_num_heads:      int   = 4
    gumbel_tau_init:   float = 2.0
    gumbel_tau_min:    float = 0.1
    gumbel_decay:      float = 0.997

    # Bottleneck decoder
    d_bottleneck:      int   = 64    # d' < d_model


@dataclass
class AblationConfig:
    """Switches for ablation variants A1–A10."""
    use_fine:        bool = True    # A1: set False for coarse-only
    use_coarse:      bool = True    # A2: set False for fine-only
    single_scale:    bool = False   # A3: single scale with P=24
    single_patch:    int  = 24      # used when single_scale=True
    use_attention:   bool = False   # A4: replace Mamba with attention
    use_cd:          bool = True    # A5: set False for CI-only
    use_bottleneck:  bool = True    # A6: set False for d'=d (no bottleneck)


class ReconHead(nn.Module):
    """
    Reconstruction head with optional bottleneck.
    Projects each patch token back to its original P×V values.
    Bottleneck (d→d'<d→P*V) prevents over-generalisation by forcing
    the model to compress normal patterns into a compact manifold.
    """
    def __init__(
        self,
        d_model:       int,
        patch_size:    int,
        V:             int,
        d_bottleneck:  int,
        use_bottleneck: bool = True,
    ):
        super().__init__()
        d_out = patch_size  # output per channel per patch = P timesteps
        if use_bottleneck and d_bottleneck < d_model:
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_bottleneck),
                nn.GELU(),
                nn.Linear(d_bottleneck, d_out),
            )
        else:
            self.proj = nn.Linear(d_model, d_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, V, N, d_model)
        Returns: (B, V, N, P) — reconstructed patches per channel
        """
        return self.proj(tokens)


def patches_to_series(patches: torch.Tensor, L: int) -> torch.Tensor:
    """
    Back-project patch reconstruction to per-timestep values.
    patches: (B, V, N, P)
    Returns: (B, L, V) — cropped to original window length L
    """
    B, V, N, P = patches.shape
    x = patches.permute(0, 2, 3, 1)    # (B, N, P, V)
    x = x.reshape(B, N * P, V)         # (B, N*P, V)
    return x[:, :L, :]                 # crop padding


class DSPatchMamba(nn.Module):
    """
    Full DS-PatchMamba model.

    Args:
        V:         number of input channels (set at runtime from data)
        cfg:       ModelConfig with architecture hyperparameters
        abl:       AblationConfig for ablation variants
    """
    def __init__(
        self,
        V:   int,
        cfg: ModelConfig     = None,
        abl: AblationConfig  = None,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        abl = abl or AblationConfig()
        self.cfg = cfg
        self.abl = abl
        self.V   = V

        d = cfg.d_model

        # ── Tokenizer ──────────────────────────────────────────────────
        if abl.single_scale:
            # A3: single scale with P=24
            self.tokenizer = DualScaleTokenizer(
                patch_size_fine=abl.single_patch,
                patch_size_coarse=abl.single_patch,
                d_model=d,
            )
        else:
            self.tokenizer = DualScaleTokenizer(
                patch_size_fine=cfg.patch_size_fine,
                patch_size_coarse=cfg.patch_size_coarse,
                d_model=d,
            )

        # ── Encoders ───────────────────────────────────────────────────
        EncoderCls = AttentionEncoder if abl.use_attention else MambaEncoder

        if abl.use_fine or abl.single_scale:
            self.encoder_fine = EncoderCls(
                num_blocks=cfg.num_mamba_blocks,
                d_model=d,
                **(dict(d_state=cfg.d_state, expand=cfg.expand,
                        d_conv=cfg.d_conv, num_heads=cfg.num_heads)
                   if not abl.use_attention else dict(num_heads=cfg.num_heads)),
            )

        if abl.use_coarse or abl.single_scale:
            self.encoder_coarse = EncoderCls(
                num_blocks=cfg.num_mamba_blocks,
                d_model=d,
                **(dict(d_state=cfg.d_state, expand=cfg.expand,
                        d_conv=cfg.d_conv, num_heads=cfg.num_heads)
                   if not abl.use_attention else dict(num_heads=cfg.num_heads)),
            )

        # ── Partial CD module ──────────────────────────────────────────
        self.cd = PartialChannelDependence(
            d_model=d,
            num_heads=min(cfg.cd_num_heads, V),
            tau_init=cfg.gumbel_tau_init,
            tau_min=cfg.gumbel_tau_min,
            decay=cfg.gumbel_decay,
            use_cd=abl.use_cd and not abl.single_scale,
        )

        # ── Reconstruction heads ───────────────────────────────────────
        p_f = abl.single_patch if abl.single_scale else cfg.patch_size_fine
        p_c = abl.single_patch if abl.single_scale else cfg.patch_size_coarse

        if abl.use_fine or abl.single_scale:
            self.head_fine = ReconHead(
                d, p_f, V, cfg.d_bottleneck, abl.use_bottleneck
            )

        if abl.use_coarse or abl.single_scale:
            self.head_coarse = ReconHead(
                d, p_c, V, cfg.d_bottleneck, abl.use_bottleneck
            )

    def forward(
        self,
        x: torch.Tensor,    # (B, L, V)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            x_hat_fine:   (B, L, V)  — fine reconstruction (None if coarse-only)
            x_hat_coarse: (B, L, V)  — coarse reconstruction (None if fine-only)
        """
        B, L, V = x.shape
        abl = self.abl

        fine_tokens, coarse_tokens = self.tokenizer(x)  # (B,V,N_f,d), (B,V,N_c,d)

        # Encode
        if abl.use_fine or abl.single_scale:
            fine_enc = self.encoder_fine(fine_tokens)
        else:
            fine_enc = fine_tokens

        if abl.use_coarse or abl.single_scale:
            coarse_enc = self.encoder_coarse(coarse_tokens)
        else:
            coarse_enc = coarse_tokens

        # Partial CD (no-op if use_cd=False or single_scale)
        fine_enc, coarse_enc = self.cd(fine_enc, coarse_enc)

        # Reconstruct
        result = {}

        if abl.use_fine or abl.single_scale:
            patches_f = self.head_fine(fine_enc)          # (B, V, N_f, P_f)
            result["x_hat_fine"] = patches_to_series(patches_f, L)

        if abl.use_coarse or abl.single_scale:
            patches_c = self.head_coarse(coarse_enc)      # (B, V, N_c, P_c)
            result["x_hat_coarse"] = patches_to_series(patches_c, L)

        return result

    def step_gumbel_temperature(self):
        """Advance Gumbel annealing — call once per training step."""
        self.cd.step_temperature()

    def compute_errors(
        self,
        x: torch.Tensor,
        outputs: dict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute per-timestep, per-channel MSE errors for fine and coarse streams.
        Returns (e_fine, e_coarse) each of shape (B, L, V), or None if stream absent.
        """
        e_fine   = None
        e_coarse = None

        if "x_hat_fine" in outputs:
            e_fine   = (x - outputs["x_hat_fine"])   ** 2   # (B, L, V)

        if "x_hat_coarse" in outputs:
            e_coarse = (x - outputs["x_hat_coarse"]) ** 2   # (B, L, V)

        return e_fine, e_coarse

    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        outputs: dict,
        w: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute per-timestep anomaly scores for a batch.
        s_t = w * mean_V(e_f,t) + (1-w) * mean_V(e_c,t)

        w is a hyperparameter selected on the Tuning set (not learned).
        If only one stream exists (A1/A2), uses that stream's error directly.

        Returns: (B, L) anomaly scores
        """
        e_fine, e_coarse = self.compute_errors(x, outputs)

        if e_fine is not None and e_coarse is not None:
            score = w * e_fine.mean(-1) + (1.0 - w) * e_coarse.mean(-1)
        elif e_fine is not None:
            score = e_fine.mean(-1)
        else:
            score = e_coarse.mean(-1)

        return score  # (B, L)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
