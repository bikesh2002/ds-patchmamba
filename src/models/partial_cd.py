"""
Partial Channel Dependence (CP) module for DS-PatchMamba.

Implements selective cross-variable attention over compact patch summary tokens.

Design (from WORKFLOW.md Section 3 / Architecture):
    1. Average-pool Mamba output across patch (N) dimension → 1 summary token per channel per scale
    2. Add learnable 2-class scale embedding to distinguish fine vs coarse tokens
    3. Concatenate all V×2 summary tokens
    4. Multi-head cross-channel attention with a learned sparse correlation mask
    5. Mask uses Gumbel-softmax relaxation with temperature annealing for binary convergence
    6. Residual-broadcast: add attention-enriched summaries back to all patch positions

Key novelties:
    - Scale embedding: BERT-style segment embedding that lets the attention
      distinguish which scale (fine=0 or coarse=1) a summary token belongs to.
      Without this, cross-channel attention at the scale-summary level is
      ambiguous — it cannot exploit the dual-scale structure.

    - Gumbel-softmax with temperature annealing: converges the correlation mask
      to a hard binary mask over training (τ: 2.0 → 0.1). Without annealing,
      the mask stays as a continuous soft approximation, making the "sparsity"
      claim meaningless. Hard argmax at inference.
      Schedule: τ_t = max(τ_min, τ_init × decay^step)

    - Summary-token design ensures cost is O(V²) CONSTANT in L — unlike full-CD
      which costs O(L × V²). At V=248 (OPP in TSB-AD-M): 496² × 128 ≈ 31M FLOPs,
      negligible even on CPU. This is the scalability advantage.

Ablation A5: set use_cd=False to remove this module and run CI-only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaleEmbedding(nn.Module):
    """
    Learnable 2-class embedding to distinguish fine (0) vs coarse (1) summary tokens.
    Analogous to BERT's segment (sentence A/B) embeddings.
    Total parameters: 2 × d_model = 256 (negligible).
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)

    def forward(
        self,
        fine_tokens:   torch.Tensor,   # (B, V, d_model)
        coarse_tokens: torch.Tensor,   # (B, V, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, D = fine_tokens.shape
        device = fine_tokens.device

        fine_ids   = torch.zeros(B, V, dtype=torch.long, device=device)   # class 0
        coarse_ids = torch.ones(B, V,  dtype=torch.long, device=device)   # class 1

        fine_out   = fine_tokens   + self.embedding(fine_ids)
        coarse_out = coarse_tokens + self.embedding(coarse_ids)
        return fine_out, coarse_out


class GumbelSoftmaxMask(nn.Module):
    """
    Learnable sparse correlation mask for cross-channel attention.

    Maintains a logit matrix M ∈ R^(2V × 2V). During training, uses
    Gumbel-softmax to produce a differentiable approximation of a binary mask.
    At inference, uses hard argmax (straight-through estimator).

    Temperature annealing schedule (from WORKFLOW.md):
        τ_t = max(τ_min, τ_init × decay^step)
    This drives the soft mask toward binary over training, making the
    learned sparsity meaningful rather than decorative.

    Args:
        n_tokens:     total number of summary tokens = V × 2
        tau_init:     initial temperature (default 2.0)
        tau_min:      minimum temperature (default 0.1)
        decay:        per-step decay factor (default 0.997)
    """
    def __init__(
        self,
        n_tokens:  int   = 10,     # placeholder; resized at first forward pass
        tau_init:  float = 2.0,
        tau_min:   float = 0.1,
        decay:     float = 0.997,
    ):
        super().__init__()
        self.tau_init  = tau_init
        self.tau_min   = tau_min
        self.decay     = decay
        self.register_buffer("tau",   torch.tensor(tau_init))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.logits    = None
        self._n_tokens = n_tokens

    def _ensure_logits(self, n: int, device: torch.device):
        if self.logits is None or self.logits.shape[0] != n:
            self.logits = nn.Parameter(torch.zeros(n, n, device=device))
            self._n_tokens = n

    def step_temperature(self):
        """Call once per training step to anneal temperature."""
        if self.training:
            new_tau = max(
                self.tau_min,
                self.tau_init * (self.decay ** self.steps.item()),
            )
            self.tau.fill_(new_tau)
            self.steps += 1

    def forward(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Returns binary-approx mask of shape (n, n).
        During training: differentiable Gumbel-softmax.
        During inference: hard argmax.
        """
        self._ensure_logits(n, device)

        if self.training:
            # Gumbel-softmax relaxation: treat each row as a 2-way choice (keep/drop)
            # Stack logits with their negative as a 2-class distribution
            logit_2 = torch.stack([self.logits, -self.logits], dim=-1)  # (n, n, 2)
            gumbel  = F.gumbel_softmax(logit_2, tau=self.tau.item(), hard=False, dim=-1)
            mask    = gumbel[..., 0]  # (n, n) — soft probability of "keep"
        else:
            mask = (self.logits > 0).float()  # hard binary mask at inference

        return mask


class PartialChannelDependence(nn.Module):
    """
    Partial channel-dependence module (CP) over patch summary tokens.

    Forward pass:
        1. Avg-pool fine/coarse Mamba outputs → summary tokens (B, V, d)
        2. Add scale embedding (distinguishes fine vs coarse)
        3. Concatenate → (B, 2V, d) sequence for cross-channel attention
        4. Apply MHA with sparse Gumbel-softmax mask
        5. Split back to fine/coarse; residual-broadcast to all patch positions

    Args:
        d_model:    embedding dimension
        num_heads:  MHA heads (runtime: min(num_heads, V))
        tau_init, tau_min, decay: Gumbel temperature schedule
        use_cd:     if False, module is a no-op (CI-only, Ablation A5)
    """
    def __init__(
        self,
        d_model:   int   = 128,
        num_heads: int   = 4,
        tau_init:  float = 2.0,
        tau_min:   float = 0.1,
        decay:     float = 0.997,
        use_cd:    bool  = True,
    ):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.use_cd    = use_cd

        if use_cd:
            self.scale_emb = ScaleEmbedding(d_model)
            self.attn      = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)
            self.mask_gen  = GumbelSoftmaxMask(
                tau_init=tau_init, tau_min=tau_min, decay=decay
            )
            self.out_norm  = nn.LayerNorm(d_model)

    def step_temperature(self):
        """Call once per training step to advance Gumbel annealing."""
        if self.use_cd:
            self.mask_gen.step_temperature()

    def forward(
        self,
        fine_enc:   torch.Tensor,   # (B, V, N_f, d_model)
        coarse_enc: torch.Tensor,   # (B, V, N_c, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fine_enriched:   (B, V, N_f, d_model)
            coarse_enriched: (B, V, N_c, d_model)
        """
        if not self.use_cd:
            return fine_enc, coarse_enc

        B, V, N_f, D = fine_enc.shape
        _, _, N_c, _  = coarse_enc.shape

        # 1. Average-pool over patch dimension → summary tokens
        fine_summary   = fine_enc.mean(dim=2)    # (B, V, D)
        coarse_summary = coarse_enc.mean(dim=2)  # (B, V, D)

        # 2. Add scale embeddings
        fine_emb, coarse_emb = self.scale_emb(fine_summary, coarse_summary)

        # 3. Concatenate V fine + V coarse → (B, 2V, D)
        tokens = torch.cat([fine_emb, coarse_emb], dim=1)  # (B, 2V, D)

        # 4. Build correlation mask (2V × 2V), apply MHA
        n_tokens = 2 * V
        mask = self.mask_gen(n_tokens, tokens.device)  # (2V, 2V)

        # Convert mask to additive attention bias:
        # 0 (drop) → -inf, 1 (keep) → 0
        attn_bias = (1.0 - mask) * (-1e9)   # (2V, 2V)

        normed = self.attn_norm(tokens)
        enriched, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_bias,
        )
        tokens = tokens + enriched   # residual connection

        # 5. Split back to fine / coarse summaries, apply norm
        fine_out   = self.out_norm(tokens[:, :V, :])   # (B, V, D)
        coarse_out = self.out_norm(tokens[:, V:, :])   # (B, V, D)

        # 6. Broadcast back to all patch positions (residual inject)
        fine_enriched   = fine_enc   + fine_out.unsqueeze(2)    # (B, V, N_f, D)
        coarse_enriched = coarse_enc + coarse_out.unsqueeze(2)  # (B, V, N_c, D)

        return fine_enriched, coarse_enriched
