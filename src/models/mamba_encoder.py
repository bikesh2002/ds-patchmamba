"""
Mamba-2 temporal encoder for DS-PatchMamba.

Each scale (fine, coarse) uses its own encoder stack. The encoder is applied
independently to each channel (weight sharing across channels — CI backbone).

Mamba-2 hyperparameters (from WORKFLOW.md Section 4):
    d_model = 128, d_state = 64, expand = 2, d_conv = 4, num_heads = 4

Why Mamba over attention:
    - O(L/P) vs O((L/P)²) time complexity — critical for long traces (GHL 199K, MITDB 336K)
    - Input-dependent selective state: intuitively compresses normal patterns,
      fails on anomalies that break learned dynamics
    - Hardware-aware parallel scan (mamba-ssm) — efficient on T4

Fallback:
    If mamba-ssm is not available (CUDA version mismatch on Kaggle),
    we fall back to a pure-PyTorch mamba-minimal implementation or to
    a 1D-CNN + Attention hybrid. The fallback is registered automatically.
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Mamba-2 availability check
# ─────────────────────────────────────────────

MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    pass

if not MAMBA_AVAILABLE:
    try:
        # mamba-minimal: pure-PyTorch reference implementation
        from mamba_minimal import Mamba as MambaMinimal  # type: ignore
        MAMBA_AVAILABLE = True
        _USE_MINIMAL = True
    except ImportError:
        _USE_MINIMAL = False
else:
    _USE_MINIMAL = False


# ─────────────────────────────────────────────
# Fallback: 1D-CNN + GRU block (runs on any hardware)
# ─────────────────────────────────────────────

class CNNGRUFallback(nn.Module):
    """
    Lightweight 1D-CNN + GRU fallback when neither mamba-ssm nor mamba-minimal
    is available. Used for CI and A4 (attention backbone) ablation comparisons.
    Not the primary model — used only when Mamba is unavailable.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=2)
        self.gru  = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, :x.shape[1], :]
        out, _ = self.gru(h)
        return self.norm(out + x)


# ─────────────────────────────────────────────
# Single Mamba-2 block wrapper (uniform interface)
# ─────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    One Mamba-2 block with pre-norm and residual connection.
    Interface: (B, N, d_model) → (B, N, d_model)
    """
    def __init__(
        self,
        d_model:  int = 128,
        d_state:  int = 64,
        expand:   int = 2,
        d_conv:   int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        if MAMBA_AVAILABLE and not _USE_MINIMAL:
            self.ssm = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=d_model // num_heads,
            )
        elif MAMBA_AVAILABLE and _USE_MINIMAL:
            self.ssm = MambaMinimal(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = CNNGRUFallback(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ─────────────────────────────────────────────
# Multi-block encoder (weight-shared across channels)
# ─────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """
    Stack of Mamba-2 blocks applied independently to each channel.
    Weight sharing: the same blocks process every channel → CI backbone.

    Input:  (B, V, N, d_model)  — patch tokens per channel
    Output: (B, V, N, d_model)  — encoded patch representations

    Args:
        num_blocks: number of Mamba-2 blocks (default 2)
        d_model, d_state, expand, d_conv, num_heads: Mamba-2 hyperparameters
    """
    def __init__(
        self,
        num_blocks: int = 2,
        d_model:    int = 128,
        d_state:    int = 64,
        expand:     int = 2,
        d_conv:     int = 4,
        num_heads:  int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state,
                       expand=expand, d_conv=d_conv, num_heads=num_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, V, N, d_model)
        Returns: (B, V, N, d_model)
        """
        B, V, N, D = tokens.shape

        # Reshape to (B*V, N, D) — process all channels with shared weights
        x = tokens.reshape(B * V, N, D)

        for block in self.blocks:
            x = block(x)

        return x.reshape(B, V, N, D)


# ─────────────────────────────────────────────
# Attention backbone (used in Ablation A4)
# ─────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """
    Standard multi-head self-attention block for Ablation A4.
    Replaces Mamba to test: 'does Mamba specifically add value over attention?'
    Cost: O(N²) where N = L/P — demonstrates quadratic scaling.
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + h
        return x + self.ffn(self.norm2(x))


class AttentionEncoder(nn.Module):
    """
    Attention-based encoder for Ablation A4 (replaces MambaEncoder).
    Same interface as MambaEncoder.
    """
    def __init__(self, num_blocks: int = 2, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, num_heads) for _ in range(num_blocks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, V, N, D = tokens.shape
        x = tokens.reshape(B * V, N, D)
        for block in self.blocks:
            x = block(x)
        return x.reshape(B, V, N, D)
