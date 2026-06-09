# DS-PatchMamba — Master Project Workflow

**Project:** Dual-Scale Patch Mamba for Multivariate Time-Series Anomaly Detection  
**Target Venue:** Engineering Applications of AI (Q1, IF ~8.0) / Expert Systems with Applications (Q1, IF ~7.5)  
**Timeline:** 4 weeks experiments + 2–3 weeks writing  
**Compute budget:** 120 GPU-hours (Kaggle free tier, single T4/P100 16 GB VRAM, 30h/week)

---

## 1. Project Purpose in One Paragraph

DS-PatchMamba is a reconstruction-based multivariate time-series anomaly detector (MTSAD). It combines dual-scale patch tokenization (fine P_f=12, coarse P_c=48) with a Mamba-2 selective SSM encoder and a partial channel-dependence (CP) module operating over compact patch summary tokens. Evaluated under rigorous non-point-adjusted protocols (VUS-PR primary, DQE supplementary) on TSB-AD-M (NeurIPS 2024, 200 series, 9 domains) plus legacy benchmarks. The novelty is the simultaneous closure of 5 documented limitations that no single prior work closes together.

---

## 2. Five Documented Gaps We Close

| # | Gap in literature | What we do | Proved by |
|---|---|---|---|
| L1 | O(L²) attention can't scale | Mamba-2 backbone: O(L/P) | Ablation A4 + Analysis 3 latency curve |
| L2 | Point-wise SSMs lose temporal semantics | Dual-scale patching (P_f=12, P_c=48) | Ablation A1/A2/A3 |
| L3 | Channel-independent models miss correlation breaks | Partial CD over summary tokens + Gumbel-softmax mask | Ablation A5 + Analysis 4 gain-vs-V |
| L4 | Static thresholds degrade under drift | DSPOT adaptive thresholding (GPD on score tail) | Ablation A7 + Analysis 5 drift test |
| L5 | PA-F1 evaluation is gameable | VUS-PR + DQE + TSB-AD-M (no PA-F1 in main table) | All experiment tables |

---

## 3. Architecture Summary

```
Input X ∈ R^(T×V)
    ↓  sliding windows L=300 (80/20 normal train/val split)
    ↓  per-channel z-score normalisation (train stats only)

Fine stream (P_f=12):  [L/12=25 tokens] → Linear embed d=128 → Mamba-2 ×2 blocks → Avg-pool → V fine summary tokens
Coarse stream (P_c=48):[L/48=6 tokens]  → Linear embed d=128 → Mamba-2 ×2 blocks → Avg-pool → V coarse summary tokens

                     V×2 summary tokens
                         + scale embedding (2-class learnable, 256 params)
                         ↓
              Cross-channel MHA
              + Gumbel-softmax correlation mask (τ: 2.0→0.1, decay 0.997/step)
                         ↓
         Residual inject → enrich both stream outputs

Fine recon head:   d→d'=64→P_f×V  (bottleneck prevents over-generalisation)
Coarse recon head: d→d'=64→P_c×V

Per-timestep errors e_f,t and e_c,t
Fused score: s_t = w·e_f,t + (1-w)·e_c,t     [w = hyperparameter from Tuning set]
                 ↓
DSPOT threshold (GPD fit, q=1e-4) → binary anomaly label
                 ↓
Threshold-free metrics: VUS-PR (primary), DQE (supplementary)
```

**Key numbers:** ~7M params | <2 GB VRAM at batch=64 | trains in <2h/dataset on T4

---

## 4. Hyperparameters (Final, All Fixed Before Eval Set Is Touched)

| Parameter | Value | How chosen |
|---|---|---|
| Window length L | 300 (primary), 100/1000 (ablation only) | 300 = minimum for ≥6 coarse patches |
| Fine patch P_f | 12 | Tuning set sweep {6,12,24} |
| Coarse patch P_c | 48 | Tuning set sweep {24,48,96} |
| d_model | 128 | Model capacity vs. VRAM trade-off |
| d_state | 64 | Tuning set sweep {16,32,64,128} (ablation A9) |
| Mamba-2 blocks | 2 per scale | Ablation checked |
| Bottleneck d' | 64 | Fixed (d/2); ablated via A6 |
| Gumbel τ_init | 2.0 | Standard (Jang et al. ICLR 2017) |
| Gumbel decay α | 0.997 per step | Converges to binary by epoch 30 |
| Gumbel τ_min | 0.1 | Standard lower bound |
| Fusion weight w | {0.3, 0.5, 0.7} → best on Tuning set | Hyperparameter (not learned) |
| Loss weights α, β | α=0.5, β=0.01 (swept 0.3–0.7, 0–0.1) | Tuning set |
| Optimizer | AdamW (lr=1e-3, β1=0.9, β2=0.999, wd=1e-4) | Standard |
| LR schedule | 5-epoch warmup + CosineAnnealing | Standard |
| Batch size | 64 | VRAM constraint |
| Precision | BF16 (FP16 fallback) | Halves VRAM |
| Early stopping | patience=7 on val MSE (normal data only) | Label-free stopping |
| Seeds | 3 seeds {42, 0, 1} main; 1 seed ablations | 120h budget |
| Channel mask aug | k = max(1, floor(V×0.15)) channels/batch | Prevents CD overfitting |
| DSPOT q | 1e-4 | Tuning set sweep {1e-3, 1e-4, 1e-5} |
| DSPOT stride | L/2 (stride=50 for short series avg_len<5000) | MSL fix |

---

## 5. Datasets

### Primary: TSB-AD-M (NeurIPS 2024)
- 200 multivariate series, 17 collections, 9 domains, V=2 to V=248
- **Eval set:** TSB-AD-M-Eva.csv (~170 series, 85%) — used ONLY for final results
- **Tuning set:** TSB-AD-M-Tuning.csv (~30 series, 15%) — used ONLY for HP selection
- Download: `wget https://www.thedatum.org/datasets/TSB-AD-M.zip && unzip TSB-AD-M.zip`

### Legacy (comparability)
- SMD (38 channels, 28 machines, IT servers)
- PSM (25 channels, service mesh)
- SWaT (51 channels, water treatment)

### Domain breakdown (for Analysis 9)
| Domain | Key collections | V range | Primary anomaly type |
|---|---|---|---|
| Medical | MITDB, SVDB, LTDB | 2 | Sequence (long) |
| IT servers | Exathlon, SMD | 21–38 | Sequence (medium) |
| Industrial | GHL, Genesis | 18–19 | Sequence (very long) |
| Human activity | OPP, Daphnet | 9–248 | Sequence |
| Water | SWaT, PSM | 25–59 | Sequence (long) |
| Satellite | SMAP, MSL | 25–55 | Mixed |
| Finance | CreditCard | 29 | Point |
| Oceanography | TAO | 3 | Point |
| Environment | GECCO | 9 | Short collective |

---

## 6. Baselines

### Experimental (run on GPU)
| Tier | Method | Type | Est. VUS-PR |
|---|---|---|---|
| Sanity | Random score | — | ~0.05 |
| Sanity | Untrained network | — | ~0.08 |
| Simple neural | CNN-AE | Conv autoencoder | ~0.31 |
| Simple neural | LSTMAD | LSTM autoencoder | ~0.31 |
| Classic deep | OmniAnomaly | VAE + POT | ~0.31 |
| Classic deep | USAD | Dual-AE | ~0.30 |
| Modern attention | Anomaly Transformer | Assoc-discrepancy | ~0.12 |
| Modern SSM | MAAT | Point-wise Mamba | TBD |
| Modern patch | PatchTrAD | Patch Transformer | TBD |
| Concurrent | Patched-DeltaNet | Patch + DeltaNet | TBD |

### Cross-domain linear (CPU, mandatory)
| Method | Domain | Library |
|---|---|---|
| OLS-RRR | Econometrics | numpy/sklearn (analytical, <1 min) |
| T²-VAR | Quality engineering / SPC | statsmodels + scipy |
| PCA reconstruction | Statistics | sklearn |
| IForest | ML | sklearn |

---

## 7. Ablation Studies (10 variants, 1 seed each)

| ID | Variant | Component removed | Limitation tested |
|---|---|---|---|
| A1 | Fine-only | Coarse stream | Collective anomaly detection |
| A2 | Coarse-only | Fine stream | Point anomaly detection |
| A3 | Single-scale P=24 | Dual-scale architecture | Coverage vs one patch size |
| A4 | Attention backbone | Mamba (replaced with MHA) | Scalability / L1 |
| A5 | CI-only | Partial CD module | Cross-variable modeling / L3 |
| A6 | No bottleneck d'=d | Latent compression | Over-generalisation / L5 |
| A7 | Static threshold | DSPOT | Drift robustness / L4 |
| A8 | w=0.5 (no tuning) | Gate selection | Dual-scale fusion value |
| A9 | d_state sweep 16/32/64/128 | SSM capacity | Optimal state dimension |
| A10 | Full DS-PatchMamba | — | Baseline for ablation comparison |

---

## 8. Analyses (post-processing, free or low-cost)

| ID | What | Proves |
|---|---|---|
| Analysis 1 | VUS-PR stratified by anomaly type (point/collective/contextual) | Fine=point, coarse=collective |
| Analysis 2 | FLOPs vs VUS-PR Pareto scatter | Efficiency claim |
| Analysis 3 | Latency vs L curve {100,300,1000,5000,10000} | Linear-time claim (long-trace regime) |
| Analysis 4 | CD gain (full − A5 CI) vs V per dataset | Partial-CD value concentrated at high V |
| Analysis 5 | Drift injection: DSPOT vs static threshold | L4 closure |
| Analysis 6 | HP sensitivity grid | Robustness to HP choice |
| Analysis 7 | Friedman + Wilcoxon-Holm + CD diagram (3 seeds × 170 series) | Statistical significance |
| Analysis 8 | 2+ qualitative failure cases (plot score + GT + threshold) | Honest limitations |
| Analysis 9 | Per-domain + per-V-bin VUS-PR breakdown | Where partial-CD adds value |
| Analysis 10 | VUS-PR vs DQE cross-check on anomaly-ratio strata | VUS-PR reliability confirmation |

---

## 9. Minimum Viable Results (Q1 acceptance threshold)

- VUS-PR ≥ 0.38 on TSB-AD-M Eval set (beats xLSTMAD 0.370, the best unsupervised method)
- VUS-PR and DQE rankings agree on ≥80% of TSB-AD-M series
- DS-PatchMamba > T²-VAR on average rank (proves non-linear CD necessary)
- OPP (V=248) runs without modification and yields competitive VUS-PR
- All 10 ablations show statistically significant contribution from their component
- Per-domain breakdown shows advantage concentrated on high-V industrial/IT domains
- Statistical tests: Friedman p<0.05 + DS-PatchMamba significantly better than top-2 baselines

---

## 10. Kaggle Session Plan (12-hour cap enforcement)

| Session | Content | Est. GPU-hrs |
|---|---|---|
| S1 | Setup + data download + CPU baselines + classic deep baselines (CNN-AE, LSTMAD, OmniAnomaly, USAD) | 6–8h |
| S2 | Modern deep baselines (Anomaly Transformer, MAAT, PatchTrAD, Patched-DeltaNet) | 8–10h |
| S3 | DS-PatchMamba: SMD × 3 seeds + PSM × 3 seeds | 10–12h |
| S4 | DS-PatchMamba: SWaT × 3 seeds + HP tuning on Tuning set | 6–8h |
| S5 | TSB-AD-M Eval series 1–60 (DS-PatchMamba + top 4 baselines) | 10–12h |
| S6 | TSB-AD-M Eval series 61–120 | 10–12h |
| S7 | TSB-AD-M Eval series 121–170 | 8–10h |
| S8 | Ablations A1–A5 (legacy × 1 seed) | 8–10h |
| S9 | Ablations A6–A10 + Analysis 3 (latency) + Analysis 5 (drift) | 8–10h |
| S10 | All post-processing analyses (1–2, 4, 6–10) + figures | 2–4h |
| **Total** | | **~100–115h** |

### The 7 Mandatory Code Rules (session-death-proof)
1. **Checkpoint after every epoch** — save `{dataset}_{seed}_epoch{e}.pt`
2. **Save anomaly scores as .npy immediately after inference** — before computing metrics
3. **Append results to CSV row-by-row** — never accumulate in memory
4. **Skip already-completed runs** — check results CSV before starting each (method, dataset, seed)
5. **Each cell ≤ 10h of compute** — leave 2h buffer per session
6. **Persist outputs via Kaggle dataset between sessions** — `Save Version` → next session loads as input
7. **Wall-clock guard in training loop** — `if time.time() - start > 10*3600: save and break`

---

## 11. Project Structure

```
ds-patchmamba/
├── WORKFLOW.md                  ← this file
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/
│   │   ├── loader.py            ← TSB-AD-M + legacy dataset loader
│   │   └── preprocessing.py    ← normalisation, sliding windows, augmentation
│   │
│   ├── models/
│   │   ├── ds_patchmamba.py     ← full DS-PatchMamba architecture
│   │   ├── mamba_encoder.py     ← Mamba-2 blocks (weight-shared across channels)
│   │   ├── patch_tokenizer.py   ← dual-scale patching + linear embedding
│   │   ├── partial_cd.py        ← cross-channel attention + Gumbel-softmax mask
│   │   └── baselines/
│   │       ├── cpu_baselines.py ← OLS-RRR, T2-VAR, PCA, IForest
│   │       ├── omnianomaly.py
│   │       ├── usad.py
│   │       ├── anomaly_transformer.py
│   │       ├── maat.py
│   │       └── patchtrad.py
│   │
│   ├── training/
│   │   ├── trainer.py           ← training loop with checkpoint/resume
│   │   ├── losses.py            ← dual-scale MSE + per-timestep consistency loss
│   │   └── dspot.py             ← DSPOT adaptive threshold (GPD)
│   │
│   ├── evaluation/
│   │   ├── metrics.py           ← VUS-PR, DQE, AUROC, AUPR, all 10 metrics
│   │   ├── harness.py           ← shared no-PA evaluation harness
│   │   └── statistical.py       ← Friedman + Wilcoxon-Holm + CD diagram
│   │
│   └── analyses/
│       ├── analysis_1_anomaly_type.py
│       ├── analysis_2_flops_pareto.py
│       ├── analysis_3_latency.py
│       ├── analysis_4_cd_gain_vs_V.py
│       ├── analysis_5_drift.py
│       ├── analysis_6_sensitivity.py
│       ├── analysis_7_statistics.py
│       ├── analysis_8_failures.py
│       ├── analysis_9_per_domain.py
│       └── analysis_10_vuspr_dqe.py
│
├── notebooks/
│   ├── session_S1_setup_baselines.ipynb
│   ├── session_S2_modern_baselines.ipynb
│   ├── session_S3_train_legacy.ipynb
│   ├── session_S4_train_legacy2_hptuning.ipynb
│   ├── session_S5_tsbadm_1_60.ipynb
│   ├── session_S6_tsbadm_61_120.ipynb
│   ├── session_S7_tsbadm_121_170.ipynb
│   ├── session_S8_ablations_A1_A5.ipynb
│   ├── session_S9_ablations_A6_A10_analyses.ipynb
│   └── session_S10_postprocessing.ipynb
│
├── configs/
│   └── default_config.yaml      ← all hyperparameters in one place
│
├── results/
│   ├── main_results.csv         ← row-by-row appended results
│   ├── ablation_results.csv
│   └── scores/                  ← .npy anomaly scores per (method, dataset, seed)
│
└── checkpoints/                 ← .pt checkpoints per (dataset, seed, epoch)
```

---

## 12. Target Journals (Priority Order)

1. **Engineering Applications of AI** — IF ~8.0, Q1 — best fit (IoT/industrial anomaly). Est. 75–85% accept probability with full execution.
2. **Expert Systems with Applications** — IF ~7.5, Q1 — safe fallback. Est. 70–80%.
3. **Neurocomputing** — IF ~6.0, Q1 — fallback. Est. 80–88%.
4. **IEEE TNNLS** — IF ~10.2, Q1 — only if a strong theoretical section is added. Est. 25–35%.

**Mandatory for acceptance:** public GitHub release with code, configs, and trained checkpoints.

---

## 13. Competitive Target

| Method | VUS-PR (TSB-AD-M) | Type | Our position |
|---|---|---|---|
| AxonAD | ~0.493 | Unknown | Ceiling |
| xLSTMAD | ~0.370 | Unsupervised LSTM | **Must beat** |
| TSPulse ZS | ~0.361 | Zero-shot LLM | Must beat |
| MMPAD | ~0.354 | Non-neural | Must beat |
| CNN-AE | ~0.313 | Simple neural | Must beat |
| **DS-PatchMamba target** | **≥ 0.38** | Unsupervised | **Minimum for Q1 claim** |

---

## 14. All Issues Resolved Before Implementation (from planning phase)

All 19 issues identified across 3 review rounds are resolved:

| # | Issue | Resolution |
|---|---|---|
| M1 | Early stopping on "val AUPR" — impossible without labels | → val MSE on held-out normal data |
| M2 | L=100 disables coarse stream (only 2 patches) | → L=300 primary; L=100 as ablation A3 |
| M3 | Channel-grouping text persisted incorrectly | → summary-token scalability framing |
| M4 | 5 seeds exceeds 120h compute budget | → 3 seeds main, 1 seed ablations |
| M5 | Consistency loss KL between Gaussians — unsound | → distribution-free per-timestep MSE |
| M6 | TMLR 2026 evaluation paper not cited | → added to literature Thread C |
| M7 | TSB-AD-M download path unclear | → wget URL confirmed |
| M8 | d_state=64 not validated | → KambaAD confirms; ablation A9 sweeps |
| M9 | Reviewer challenge still referenced channel-grouping | → updated to summary-token framing |
| M10 | Seeds inconsistent across document | → unified to 3 seeds / 1 seed |
| M11 | Gumbel-softmax had no temperature annealing | → τ_init=2.0, decay 0.997/step, τ_min=0.1 |
| M12 | Scale embedding missing — attention couldn't distinguish scales | → 2-class learnable embedding added |
| M13 | Fusion gate learned via backprop — near-zero gradient | → changed to hyperparameter w on Tuning set |
| M14 | HP selection protocol unspecified — potential leakage | → explicit Tuning set only; Eval set locked |
| M15 | Custom stratified split less rigorous than official | → official TSB-AD-M-Eva.csv + Tuning.csv |
| M16 | Consistency loss batch-mean too coarse | → per-timestep scale-normalized MSE |
| M17 | Channel masking edge case: floor(2×0.15)=0 | → max(1, floor(V×0.15)) |
| M18 | Leaderboard outdated | → updated June 2026; target ≥0.38 |
| M19 | Paper limitations section not planned | → Section 5.11 with 5 specific limitations |
