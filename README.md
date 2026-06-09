# DS-PatchMamba

**Dual-Scale Patch Mamba for Multivariate Time-Series Anomaly Detection**

Target venue: Engineering Applications of AI (Q1, IF ~8.0) / Expert Systems with Applications (Q1, IF ~7.5)

---

## Project Summary

DS-PatchMamba is a reconstruction-based anomaly detector for multivariate time series that closes 5 documented limitations of present literature simultaneously:

| Gap | Solution | Proved by |
|---|---|---|
| O(L²) attention can't scale | Mamba-2 backbone: O(L/P) | Ablation A4 + latency analysis |
| Point-wise SSMs lose semantics | Dual-scale patching (P_f=12, P_c=48) | Ablation A1/A2/A3 |
| Channel-independent models miss correlation breaks | Partial CD over summary tokens | Ablation A5 + gain-vs-V plot |
| Static thresholds fail under drift | DSPOT adaptive threshold (GPD) | Ablation A7 + drift test |
| PA-F1 evaluation is gameable | VUS-PR + DQE + TSB-AD-M primary | Full evaluation tables |

---

## Quickstart (Kaggle)

```bash
# Cell 1: Download TSB-AD-M
wget https://www.thedatum.org/datasets/TSB-AD-M.zip && unzip TSB-AD-M.zip

# Cell 2: Install packages
pip install TSB-AD statsmodels scikit-posthocs PyWavelets mamba-ssm causal-conv1d

# Cell 3: Fallback if mamba-ssm fails
pip install git+https://github.com/johnma2006/mamba-minimal
```

Run notebooks in order:
1. `session_S1_setup_baselines.ipynb`    — setup + CPU baselines
2. `session_S2_modern_baselines.ipynb`  — GPU deep baselines
3. `session_S3_train_legacy.ipynb`      — DS-PatchMamba on SMD/PSM
4. `session_S4_train_legacy2_hptuning.ipynb` — SWaT + HP tuning
5. `session_S5_tsbadm_1_60.ipynb`       — TSB-AD-M series 1–60
6. `session_S6_tsbadm_61_120.ipynb`     — TSB-AD-M series 61–120
7. `session_S7_tsbadm_121_170.ipynb`    — TSB-AD-M series 121–170
8. `session_S8_ablations_A1_A5.ipynb`   — Ablations
9. `session_S9_ablations_A6_A10_analyses.ipynb` — Ablations + analyses
10. `session_S10_postprocessing.ipynb`  — All post-processing analyses

---

## Architecture

```
Input (B, L=300, V)
├─ Fine patches (P_f=12, 25 tokens) → Mamba-2 ×2 → summary tokens
├─ Coarse patches (P_c=48, 6 tokens) → Mamba-2 ×2 → summary tokens
│                          ↓
│         Partial CD: V×2 tokens + scale embeddings
│         + Gumbel-softmax mask (τ: 2.0→0.1) cross-channel MHA
│                          ↓
├─ Fine recon head (d→64→P_f×V) → x_hat_fine
└─ Coarse recon head (d→64→P_c×V) → x_hat_coarse
                           ↓
          Score: s_t = w·e_f,t + (1-w)·e_c,t
                           ↓
              DSPOT threshold (GPD, q=1e-4)
```

**Size:** ~7M params | <2 GB VRAM at batch=64 | <2h/dataset on T4

---

## Competitive Target

| Method | VUS-PR (TSB-AD-M) | Type |
|---|---|---|
| xLSTMAD | 0.370 | Unsupervised LSTM — must beat |
| CNN-AE | 0.313 | Simple neural — must beat |
| **DS-PatchMamba target** | **≥ 0.38** | This work |

---

## Minimum Viable Results (Q1 threshold)

- VUS-PR ≥ 0.38 on TSB-AD-M Eval set
- DS-PatchMamba > T²-VAR on average rank
- All 10 ablations statistically significant
- Code + checkpoints released on GitHub

---

## Kaggle Session Budget

| Session | GPU-hrs | Content |
|---|---|---|
| S1 | 6–8h | Setup + CPU/classic baselines |
| S2 | 8–10h | Modern deep baselines |
| S3–S4 | 16–20h | DS-PatchMamba legacy + HP tuning |
| S5–S7 | 28–36h | TSB-AD-M Eval full run |
| S8–S9 | 16–20h | Ablations + analyses |
| S10 | 2–4h | Post-processing |
| **Total** | **~100–115h** | Fits 120h Kaggle budget |
