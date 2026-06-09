"""
Dual-scale patch tokenizer for DS-PatchMamba.

Converts a (B, L, V) input window into two sequences of patch tokens:
    Fine stream:   (B, V, L/P_f, d_model)   — targets point/contextual anomalies
    Coarse stream: (B, V, L/P_c, d_model)   — targets collective/sequence anomalies

Each patch (P consecutive timesteps × V channels, flattened per channel)
is embedded via a shared learnable linear projection to d_model.

Weight sharing across channels: the same projection is applied to every channel.
This reduces parameters by V× and improves robustness — analogous to how
PatchTST shares patch embedding across channels (CI backbone).

Design motivation (from WORKFLOW.md Section 2):
    P_f=12 aligns with 1-second industrial sensor data: anomalies span 5-30s → 2-3 patches
    P_c=48 captures collective events spanning minutes (matches MITDB cardiac avg len 1847)
    At L=300: fine produces 25 tokens, coarse produces 6 tokens (minimum for temporal context)
    At L=100: coarse produces 2 tokens (functionally disabled — used only as ablation A3)
"""

import torch
import torch.nn as nn
import math


class PatchTokenizer(nn.Module):
    """
    Single-scale patch tokenizer.

    Args:
        patch_size: number of consecutive timesteps per patch
        d_model:    output embedding dimension
    """
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model    = d_model
        # Maps one patch of a single channel (P,) → d_model
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, V)
        Returns: (B, V, N_patches, d_model)
        """
        B, L, V = x.shape
        P = self.patch_size

        # Pad L to be divisible by P (right-pad with zeros)
        remainder = L % P
        if remainder != 0:
            pad = P - remainder
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))  # pad time dim

        N = x.shape[1] // P  # number of patches

        # Reshape to (B, N, P, V) then extract per-channel patches
        x = x.reshape(B, N, P, V)          # (B, N, P, V)
        x = x.permute(0, 3, 1, 2)          # (B, V, N, P)
        x = x.reshape(B * V, N, P)         # (B*V, N, P)

        tokens = self.proj(x)               # (B*V, N, d_model)
        tokens = tokens.reshape(B, V, N, self.d_model)  # (B, V, N, d_model)
        return tokens


class DualScaleTokenizer(nn.Module):
    """
    Dual-scale patch tokenizer producing fine and coarse token sequences.

    Args:
        patch_size_fine:   P_f (default 12)
        patch_size_coarse: P_c (default 48)
        d_model:           embedding dimension
    """
    def __init__(
        self,
        patch_size_fine:   int = 12,
        patch_size_coarse: int = 48,
        d_model:           int = 128,
    ):
        super().__init__()
        self.tokenizer_fine   = PatchTokenizer(patch_size_fine,   d_model)
        self.tokenizer_coarse = PatchTokenizer(patch_size_coarse, d_model)
        self.patch_size_fine   = patch_size_fine
        self.patch_size_coarse = patch_size_coarse

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, V)
        Returns:
            fine_tokens:   (B, V, N_f, d_model)
            coarse_tokens: (B, V, N_c, d_model)
        """
        return self.tokenizer_fine(x), self.tokenizer_coarse(x)

    @staticmethod
    def get_n_patches(L: int, P: int) -> int:
        return math.ceil(L / P)
