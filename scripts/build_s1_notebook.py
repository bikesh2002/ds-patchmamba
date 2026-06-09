"""Regenerate notebooks/session_S1_setup_baselines.ipynb with consistent GitHub sync."""
import json
from pathlib import Path

GITHUB_COMMIT = "63b573f"
NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "session_S1_setup_baselines.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


CELL0 = """# Session S1 — Setup + CPU Baselines

**DS-PatchMamba | WORKFLOW.md Session S1**

### What this session does
1. Installs packages and downloads TSB-AD-M
2. Loads datasets and verifies shapes and label alignment
3. Runs 5 CPU baselines: Random, PCA, IForest, OLS-RRR, T²-VAR
4. Validates the evaluation harness against known VUS-PR ranges

### Before running on Kaggle
**Add these three datasets via Input panel (right sidebar):**
- **SMD:** `smd-onmiad` by mgusat
- **PSM:** `pooled-server-metrics-psm` by ljolm08
- **SWaT:** `swat-dataset-secure-water-treatment-system` by vishala28

Cell 1 auto-detects mount paths. **Run cells in order** (1 → 6) after **Restart Session**.

### Session-death protection
- Results appended row-by-row to `results/main_results.csv`
- Scores saved as `.npy` immediately after each series
- Click **Save Version** → name `ds-patchmamba-run1` before session ends
"""

CELL1 = f'''# =============================================================================
# CELL 1 — Environment setup (run once per session, after Restart Session)
# Expected time: ~10-15 minutes
# =============================================================================

import json
import os
import shutil
import subprocess
import sys
import urllib.request

# --- SINGLE SOURCE OF TRUTH for GitHub code version ---
GITHUB_USER   = "bikesh2002"
GITHUB_REPO   = "ds-patchmamba"
GITHUB_COMMIT = "{GITHUB_COMMIT}"   # update this hash after each git push
REPO_DIR      = "/kaggle/working/repo"
SRC_LINK      = "/kaggle/working/src"
SESSION_META  = "/kaggle/working/session_meta.json"

SYNC_FILES = [
    "src/data/loader.py",
    "src/data/preprocessing.py",
    "src/evaluation/metrics.py",
    "src/evaluation/harness.py",
    "src/models/baselines/cpu_baselines.py",
]


def github_raw_url(rel_path: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{{GITHUB_USER}}/{{GITHUB_REPO}}/"
        f"{{GITHUB_COMMIT}}/{{rel_path}}"
    )


def sync_github_files(rel_paths):
    """Download pinned files into REPO_DIR. Single sync entry-point."""
    synced = []
    for rel in rel_paths:
        dest = os.path.join(REPO_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(github_raw_url(rel), dest)
        synced.append(dest)
        print(f"  synced {{rel}} ({{GITHUB_COMMIT}})")
    return synced


def verify_loader(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    compile(src, path, "exec")
    if "def _load_swat_style(" not in src:
        raise RuntimeError(
            "loader.py is missing _load_swat_style. "
            "Push latest code and update GITHUB_COMMIT in Cell 1."
        )
    if "def _load_swat_test_period(" not in src:
        raise RuntimeError(
            "loader.py is missing _load_swat_test_period. "
            "Push latest code and update GITHUB_COMMIT in Cell 1."
        )

# --- Restore previous session outputs (optional) ---
for prev_run in [
    "/kaggle/input/ds-patchmamba-run2",
    "/kaggle/input/ds-patchmamba-run1",
]:
    if os.path.exists(prev_run):
        shutil.copytree(prev_run, "/kaggle/working/", dirs_exist_ok=True)
        print(f"Restored previous session outputs from: {{prev_run}}")
        break

# --- Fresh git clone (directory structure + non-synced files) ---
if os.path.islink(SRC_LINK):
    os.unlink(SRC_LINK)
elif os.path.exists(SRC_LINK):
    shutil.rmtree(SRC_LINK)
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", f"https://github.com/{{GITHUB_USER}}/{{GITHUB_REPO}}.git", REPO_DIR],
    check=True,
)
os.symlink(f"{{REPO_DIR}}/src", SRC_LINK)
print(f"Cloned {{GITHUB_REPO}} -> {{REPO_DIR}}")

# --- Overwrite critical files from pinned commit ---
print("Syncing pinned source files from GitHub:")
sync_github_files(SYNC_FILES)

LOADER_PATH = os.path.join(REPO_DIR, "src/data/loader.py")
verify_loader(LOADER_PATH)
print(f"loader.py verified OK: {{LOADER_PATH}}")

with open(SESSION_META, "w", encoding="utf-8") as f:
    json.dump({{
        "commit": GITHUB_COMMIT,
        "repo_dir": REPO_DIR,
        "loader_path": LOADER_PATH,
    }}, f)

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

for d in ["results/scores", "checkpoints", "data"]:
    os.makedirs(d, exist_ok=True)

print("Installing packages...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "statsmodels", "scikit-posthocs", "PyWavelets"],
    check=False,
)
print("Packages installed (VUS-PR uses built-in fallback, not TSB-AD package).")

# --- TSB-AD-M ---
tsb_eva    = "data/TSB-AD-M-Eva.csv"
tsb_tuning = "data/TSB-AD-M-Tuning.csv"
FILE_LIST_BASE = "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/Datasets/File_List"

_series_present = (
    os.path.isdir("data/TSB-AD-M")
    or any(
        f.endswith(".csv")
        for f in os.listdir("data")
        if os.path.isfile(os.path.join("data", f))
        and f not in ("TSB-AD-M-Eva.csv", "TSB-AD-M-Tuning.csv")
    )
)
if not _series_present:
    print("Downloading TSB-AD-M series zip...")
    subprocess.run(["wget", "-q", "https://www.thedatum.org/datasets/TSB-AD-M.zip", "-O", "data/TSB-AD-M.zip"], check=True)
    subprocess.run(["unzip", "-o", "-q", "data/TSB-AD-M.zip", "-d", "data/"], check=True)
    print("TSB-AD-M series files extracted.")
else:
    print("TSB-AD-M series files already present.")

for name, dest in [("TSB-AD-M-Eva", tsb_eva), ("TSB-AD-M-Tuning", tsb_tuning)]:
    if not os.path.exists(dest):
        print(f"Downloading {{name}}.csv from TSB-AD GitHub...")
        subprocess.run(["wget", "-q", f"{{FILE_LIST_BASE}}/{{name}}.csv", "-O", dest], check=True)
    print(f"  {{name}}: {{dest}} OK")

import pandas as pd
assert os.path.exists(tsb_eva), f"Missing: {{tsb_eva}}"
assert os.path.exists(tsb_tuning), f"Missing: {{tsb_tuning}}"
for f, name in [(tsb_eva, "Eva"), (tsb_tuning, "Tuning")]:
    idx = pd.read_csv(f)
    path_col = "filepath" if "filepath" in idx.columns else "file_name"
    assert path_col in idx.columns
    print(f"TSB-AD-M {{name}} set: {{len(idx)}} series listed (column: {{path_col}}).")

# --- Legacy datasets: auto-detect Kaggle mount paths ---
KAGGLE_INPUT = "/kaggle/input"
LEGACY_CANDIDATES = {{
    "SMD": [
        "/kaggle/input/datasets/mgusat/smd-onmiad/ServerMachineDataset",
        "/kaggle/input/smd-onmiad/ServerMachineDataset",
    ],
    "PSM": [
        "/kaggle/input/datasets/ljolm08/pooled-server-metrics-psm/data",
        "/kaggle/input/pooled-server-metrics-psm/data",
    ],
    "SWaT": [
        "/kaggle/input/datasets/vishala28/swat-dataset-secure-water-treatment-system",
        "/kaggle/input/swat-dataset-secure-water-treatment-system",
    ],
}}
LEGACY_MARKERS = {{
    "SMD":  ["train"],
    "PSM":  ["train.csv", "test.csv"],
    "SWaT": ["normal.csv", "attack.csv"],
}}

def _has_markers(path, markers):
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, m)) for m in markers)

def resolve_legacy_path(candidates, markers):
    for path in candidates:
        if _has_markers(path, markers):
            return path
    for root, dirs, _ in os.walk(KAGGLE_INPUT):
        if root[len(KAGGLE_INPUT):].count(os.sep) > 5:
            dirs.clear()
            continue
        if _has_markers(root, markers):
            return root
    return None

LEGACY_PATHS = {{}}
for ds, candidates in LEGACY_CANDIDATES.items():
    resolved = resolve_legacy_path(candidates, LEGACY_MARKERS[ds])
    if resolved:
        dest = f"data/{{ds}}"
        if os.path.islink(dest):
            os.remove(dest)
        elif not os.path.exists(dest):
            os.symlink(resolved, dest)
        LEGACY_PATHS[ds] = resolved
        print(f"{{ds}}: found at {{resolved}}")
    else:
        print(f"[WARNING] {{ds}} not found — add as Kaggle Input dataset.")

if not LEGACY_PATHS:
    print("\\n/kaggle/input contents:")
    for entry in sorted(os.listdir(KAGGLE_INPUT)):
        print(f"  {{KAGGLE_INPUT}}/{{entry}}")

print(f"\\n[DONE] Cell 1 complete. GitHub commit pinned: {{GITHUB_COMMIT}}")
'''

CELL2 = '''# =============================================================================
# CELL 2 — Load datasets and verify integrity
# NO GitHub download here — uses loader synced and verified in Cell 1.
# =============================================================================

import json
import os
import sys

import numpy as np

SESSION_META = "/kaggle/working/session_meta.json"
if not os.path.exists(SESSION_META):
    raise RuntimeError("Run Cell 1 first — session_meta.json not found.")

with open(SESSION_META, encoding="utf-8") as f:
    meta = json.load(f)

LOADER_PATH = meta["loader_path"]
print(f"Using loader commit {meta['commit']}: {LOADER_PATH}")

with open(LOADER_PATH, encoding="utf-8") as f:
    compile(f.read(), LOADER_PATH, "exec")

# Purge stale cached modules from previous runs in the same kernel
for _k in list(sys.modules):
    if _k == "src" or _k.startswith("src."):
        del sys.modules[_k]

from src.data.loader import load_tsb_adm, load_all_legacy

print("Loading TSB-AD-M Tuning set...")
tuning_records = load_tsb_adm("data", split="tuning", normalize=True)
print(f"Loaded {len(tuning_records)} series from TSB-AD-M Tuning set.")

print("\\nLoading legacy datasets...")
legacy_data = load_all_legacy("data", normalize=True)
for ds_name, records in legacy_data.items():
    print(f"{ds_name}: {len(records)} series")

all_records = list(tuning_records)
for records in legacy_data.values():
    all_records.extend(records)

print(f"\\nRunning integrity checks on {len(all_records)} series...")
issues = []

for r in all_records:
    if len(r.test) != len(r.labels):
        issues.append(f"{r.name}: test/label length mismatch ({len(r.test)} vs {len(r.labels)})")
    if np.any(~np.isfinite(r.train)):
        issues.append(f"{r.name}: NaN or Inf in training data")
    if np.any(~np.isfinite(r.test)):
        issues.append(f"{r.name}: NaN or Inf in test data")
    unique_labels = np.unique(r.labels)
    if not set(unique_labels).issubset({0, 1}):
        issues.append(f"{r.name}: labels contain values outside {{0,1}}: {unique_labels}")
    ratio = float(r.labels.mean())
    if ratio == 0.0:
        issues.append(f"{r.name}: anomaly ratio is 0% — labels may be missing")
    if ratio > 0.50:
        issues.append(f"{r.name}: anomaly ratio is {ratio:.1%} — suspiciously high")
    if r.name == "SWaT_main":
        if not (400_000 <= len(r.test) <= 500_000):
            issues.append(
                f"SWaT: test length {len(r.test):,} unexpected (benchmark expects ~449,919)"
            )
        if not (0.08 <= ratio <= 0.20):
            issues.append(
                f"SWaT: anomaly ratio {ratio:.1%} outside expected 8-20% for 4-day test period"
            )

if issues:
    print("[INTEGRITY ISSUES FOUND — fix before proceeding:]")
    for issue in issues:
        print(f"  x {issue}")
else:
    print("[OK] All integrity checks passed.")

if tuning_records:
    r = tuning_records[0]
    per_ch_std = r.train.std(axis=0)
    print(f"\\nSample series: {r.name}")
    print(f"  V={r.V} channels | train={r.train.shape} | test={r.test.shape}")
    print(f"  Anomaly ratio: {r.labels.mean():.3f}")
    print(f"  Train mean (per-channel z-score): {r.train.mean():.4f}")
    print(f"  Per-channel train std: mean={per_ch_std.mean():.3f}, min={per_ch_std.min():.3f}")

if "SWaT" in legacy_data and legacy_data["SWaT"]:
    sw = legacy_data["SWaT"][0]
    print(f"\\nSWaT: train={sw.train.shape} test={sw.test.shape} anomaly_ratio={sw.labels.mean():.3f}")
'''

CELL3 = '''# =============================================================================
# CELL 3 — CPU baselines on legacy datasets
# Expected time: ~30-60 minutes (no GPU used)
# =============================================================================

import time

import numpy as np

from src.evaluation.harness import append_result, load_completed_runs, save_scores
from src.evaluation.metrics import compute_all_metrics
from src.models.baselines.cpu_baselines import run_all_cpu_baselines

RESULTS_CSV = "results/main_results.csv"
SCORES_DIR  = "results/scores"

completed = load_completed_runs(RESULTS_CSV)
print(f"Already completed: {len(completed)} runs (will be skipped).")

for ds_name, records in legacy_data.items():
    print(f"\\n{'='*60}")
    print(f"Dataset: {ds_name} ({len(records)} series)")
    print("=" * 60)

    for series in records:
        print(
            f"\\n  Series: {series.name} | V={series.V} | "
            f"train={len(series.train):,} | test={len(series.test):,} | "
            f"anomaly_ratio={series.labels.mean():.3f}"
        )

        t0 = time.time()
        scores_dict = run_all_cpu_baselines(series.train, series.test, seed=42)
        print(f"  All baselines finished in {time.time()-t0:.1f}s")

        for method, scores in scores_dict.items():
            run_key = (method, ds_name, series.name, "42", "")
            if run_key in completed:
                print(f"    [SKIP] {method} already done")
                continue

            if len(scores) != len(series.labels):
                print(
                    f"    [ERROR] {method}: score length {len(scores)} != "
                    f"label length {len(series.labels)}. Skipping."
                )
                continue

            if np.std(scores) < 1e-10:
                print(f"    [WARN] {method}: all scores identical ({scores[0]:.4f}).")

            save_scores(SCORES_DIR, method, ds_name, series.name, 42, scores, series.labels)
            metrics = compute_all_metrics(scores, series.labels)
            row = {
                "method": method,
                "dataset": ds_name,
                "series_name": series.name,
                "seed": "42",
                "ablation": "",
                "n_params": 0,
                "flops_per_window": 0,
                "peak_vram_gb": 0,
                "train_time_s": 0,
                **metrics,
            }
            append_result(RESULTS_CSV, row)
            completed.add(run_key)
            print(
                f"    {method:10s}: VUS-PR={metrics['vus_pr']:.4f}  "
                f"DQE={metrics['dqe']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}"
            )

print("\\n[DONE] CPU baselines complete.")
'''

CELL4 = '''# =============================================================================
# CELL 4 — Validate evaluation harness against known reference ranges
# =============================================================================

import pandas as pd

from src.evaluation.harness import get_summary_table

RESULTS_CSV = "results/main_results.csv"
summary = get_summary_table(RESULTS_CSV, metric="vus_pr")
print("=" * 65)
print("VUS-PR summary (mean across series, seed=42)")
print("=" * 65)
print(summary.to_string(index=False))
print()

pca_smd = summary[(summary["method"] == "PCA") & (summary["dataset"] == "SMD")]
if pca_smd.empty:
    print("[SKIP] PCA/SMD not yet in results — run Cell 3 first.")
else:
    vus = float(pca_smd["vus_pr_mean"].values[0])
    lo, hi = 0.25, 0.40
    status = "OK" if lo <= vus <= hi else "FAIL"
    print(f"[{status}] PCA on SMD: VUS-PR = {vus:.4f}  (expected {lo:.2f}-{hi:.2f})")

random_rows = summary[summary["method"] == "Random"]
if not random_rows.empty:
    avg_random_vus = float(random_rows["vus_pr_mean"].mean())
    print(f"[INFO] Random score avg VUS-PR = {avg_random_vus:.4f}  (should be low, ~0.03-0.10)")

for ds in ["SMD", "PSM", "SWaT"]:
    pca_v = summary[(summary["method"] == "PCA") & (summary["dataset"] == ds)]["vus_pr_mean"]
    ols_v = summary[(summary["method"] == "OLS-RRR") & (summary["dataset"] == ds)]["vus_pr_mean"]
    if pca_v.empty or ols_v.empty:
        continue
    pca_val, ols_val = float(pca_v.values[0]), float(ols_v.values[0])
    status = "OK" if ols_val >= pca_val - 0.02 else "WARN"
    print(
        f"[{status}] {ds}: OLS-RRR={ols_val:.4f} vs PCA={pca_val:.4f} "
        f"({'OLS-RRR leads' if ols_val >= pca_val else 'PCA leads — investigate'})"
    )

print("\\nIf all checks are OK, proceed to Session S2.")
'''

CELL5 = '''# =============================================================================
# CELL 5 — Per-dataset anomaly ratio and score distribution inspection
# =============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np

from src.models.baselines.cpu_baselines import pca_anomaly_score

_SCORE_SEP = "___"
SCORES_DIR = "results/scores"


def plot_series_inspection(series, scores_pca, title=""):
    T = len(series.test)
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(title or series.name, fontsize=11)
    axes[0].plot(series.test[:, 0], linewidth=0.5)
    axes[0].set_ylabel("Channel 0")
    axes[1].plot(scores_pca, linewidth=0.5, color="orange")
    axes[1].set_ylabel("PCA score")
    axes[2].fill_between(range(T), series.labels, alpha=0.7, color="red")
    axes[2].set_ylabel("Anomaly label")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylim(-0.1, 1.3)
    plt.tight_layout()
    plt.show()


for ds_name, records in legacy_data.items():
    if not records:
        continue
    s = records[0]
    key = f"PCA{_SCORE_SEP}{ds_name}{_SCORE_SEP}{s.name}{_SCORE_SEP}seed42"
    scores_path = os.path.join(SCORES_DIR, f"{key}__scores.npy")
    if os.path.exists(scores_path):
        pca_scores = np.load(scores_path)
        print(f"{ds_name}/{s.name}: loaded PCA scores from disk")
    else:
        print(f"  [WARNING] {s.name}: PCA .npy not found — recomputing (run Cell 3 first).")
        pca_scores = pca_anomaly_score(s.train, s.test)
    print(f"  anomaly_ratio={s.labels.mean():.3f} | T_test={len(s.test):,} | V={s.V}")
    plot_series_inspection(s, pca_scores, title=f"{ds_name} — {s.name}")
'''

CELL6 = '''# =============================================================================
# CELL 6 — Save session outputs before the session ends
# =============================================================================

import os

import pandas as pd

RESULTS_CSV = "results/main_results.csv"
if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    print(f"Results CSV: {len(df)} rows | {df['method'].nunique()} methods | {df['dataset'].nunique()} datasets")
    print("\\nVUS-PR summary:")
    print(df.groupby(["method", "dataset"])["vus_pr"].mean().unstack().to_string())
else:
    print("No results yet — run Cells 2-4 first.")

scores_dir = "results/scores"
n_scores = len([f for f in os.listdir(scores_dir) if f.endswith(".npy")]) if os.path.exists(scores_dir) else 0
print(f"\\nScore files saved: {n_scores}")
print("\\n[REMINDER] Click 'Save Version' -> name it 'ds-patchmamba-run1' -> Save.")
'''

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "cells": [
        md(CELL0),
        code(CELL1),
        code(CELL2),
        code(CELL3),
        code(CELL4),
        code(CELL5),
        code(CELL6),
    ],
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
