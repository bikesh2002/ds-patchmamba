"""
Shared evaluation harness — used identically for ALL methods.

Loads saved .npy score/label files, computes all metrics,
and appends results to the main results CSV.

Design principles:
    1. Append-only writes — we never read-then-modify the CSV in place,
       which would corrupt the file if the session dies mid-write.
    2. Filename convention uses a fixed separator that avoids clashes with
       dataset/series names that may themselves contain underscores or hyphens.
    3. All methods share one identical evaluation protocol: no point adjustment
       in the main table, same max_buffer=100, DSPOT threshold when available.
    4. PA-F1 is stored in a separate column (appendix only).
"""

import os
import csv
import numpy as np
import pandas as pd
from typing import Optional, List

from .metrics import compute_all_metrics

# ── Single source of truth for result column order ───────────────────────────
RESULT_COLUMNS: List[str] = [
    "method", "dataset", "series_name", "seed", "ablation",
    "vus_pr", "dqe", "vus_roc", "auc_pr", "auc_roc",
    "standard_f1", "event_f1", "r_based_f1", "affiliation_f1", "pa_f1",
    "n_params", "flops_per_window", "peak_vram_gb", "train_time_s",
]

# Separator used in score file names.
# Must never appear in method / dataset / series_name strings.
_SEP = "___"   # three underscores — uncommon in dataset filenames


def _make_key(method: str, dataset: str, series_name: str, seed: int) -> str:
    """Canonical filename stem for a single run."""
    return f"{method}{_SEP}{dataset}{_SEP}{series_name}{_SEP}seed{seed}"


def _parse_key(key: str):
    """
    Parse the stem back to (method, dataset, series_name, seed).
    Returns None if the stem does not match the expected format.
    """
    parts = key.split(_SEP)
    if len(parts) != 4:
        return None
    method, dataset, series_name, seed_str = parts
    if not seed_str.startswith("seed"):
        return None
    try:
        seed = int(seed_str[4:])
    except ValueError:
        return None
    return method, dataset, series_name, seed


# ─────────────────────────────────────────────────────────────────────────────
# Score file I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_scores(
    scores_dir:  str,
    method:      str,
    dataset:     str,
    series_name: str,
    seed:        int,
    scores:      np.ndarray,
    labels:      np.ndarray,
):
    """Save raw anomaly scores and labels as .npy (Rule 2)."""
    os.makedirs(scores_dir, exist_ok=True)
    key = _make_key(method, dataset, series_name, seed)
    np.save(os.path.join(scores_dir, f"{key}__scores.npy"), scores)
    np.save(os.path.join(scores_dir, f"{key}__labels.npy"), labels)


# ─────────────────────────────────────────────────────────────────────────────
# Results CSV — append-only; never read-modify-write
# ─────────────────────────────────────────────────────────────────────────────

def append_result(results_csv: str, row: dict):
    """
    Append one fully-populated result row to the CSV.
    Creates the file with a header row if it does not yet exist.

    Append-only avoids corrupting the file if the session dies during a write.
    Duplicate rows (if a run is re-evaluated) can be deduplicated offline with
    load_results(...).drop_duplicates().
    """
    parent = os.path.dirname(results_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_completed_runs(results_csv: str) -> set:
    """
    Return a set of (method, dataset, series_name, seed, ablation) tuples
    for all runs already present in the CSV (Rule 4 — skip completed).

    Takes the last occurrence of each (method, dataset, series_name, seed,
    ablation) combination so that a re-evaluated run does not appear twice
    in the skip-set.
    """
    if not os.path.exists(results_csv):
        return set()
    completed = set()
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                row["method"],
                row["dataset"],
                row["series_name"],
                row["seed"],
                row.get("ablation", ""),
            ))
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_saved_scores(
    scores_dir:      str,
    results_csv:     str,
    method:          str,
    dataset:         str,
    series_name:     str,
    seed:            int,
    ablation:        str = "",
    vus_max_buffer:  int = 100,
    dspot_threshold: Optional[float] = None,
    extra_cols:      Optional[dict] = None,
) -> Optional[dict]:
    """
    Load saved .npy scores and labels → compute all metrics → append to CSV.

    All methods pass through this single function, guaranteeing identical
    evaluation conditions (same VUS buffer, same threshold logic, same CSV
    column order).

    Args:
        extra_cols: dict of non-metric columns to include (e.g. n_params,
                    train_time_s). Merged into the result row before writing.

    Returns:
        metrics dict, or None if the score file was not found.
    """
    key         = _make_key(method, dataset, series_name, seed)
    scores_path = os.path.join(scores_dir, f"{key}__scores.npy")
    labels_path = os.path.join(scores_dir, f"{key}__labels.npy")

    if not os.path.exists(scores_path):
        print(f"  [WARN] Scores not found: {scores_path}")
        return None

    scores = np.load(scores_path)
    labels = np.load(labels_path)

    if dspot_threshold is None:
        dspot_threshold = float(np.percentile(scores, 99.0))

    metrics = compute_all_metrics(
        scores, labels,
        threshold=dspot_threshold,
        vus_max_buffer=vus_max_buffer,
    )

    row = {
        "method":      method,
        "dataset":     dataset,
        "series_name": series_name,
        "seed":        str(seed),
        "ablation":    ablation,
        **metrics,
        **(extra_cols or {}),
    }
    append_result(results_csv, row)

    print(
        f"  {method}/{series_name}/seed{seed}: "
        f"VUS-PR={metrics['vus_pr']:.4f}  "
        f"DQE={metrics['dqe']:.4f}  "
        f"AUC-ROC={metrics['auc_roc']:.4f}"
    )
    return metrics


def evaluate_all_saved(
    scores_dir:     str,
    results_csv:    str,
    vus_max_buffer: int = 100,
):
    """
    Re-evaluate every .npy score file in scores_dir.
    Useful for recomputing metrics without retraining (e.g. after a metric fix).
    Duplicate rows can be cleaned up afterwards with load_results().
    """
    if not os.path.exists(scores_dir):
        print(f"[WARN] Scores directory not found: {scores_dir}")
        return

    score_files = sorted(
        f for f in os.listdir(scores_dir) if f.endswith("__scores.npy")
    )
    print(f"Found {len(score_files)} score files to evaluate.")

    for sf in score_files:
        # Strip the trailing __scores.npy to get the key stem
        stem   = sf[: -len("__scores.npy")]
        parsed = _parse_key(stem)
        if parsed is None:
            print(f"  [SKIP] Cannot parse filename stem: {stem!r}")
            continue

        method, dataset, series_name, seed = parsed
        evaluate_saved_scores(
            scores_dir, results_csv,
            method, dataset, series_name, seed,
            vus_max_buffer=vus_max_buffer,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Results loading and aggregation
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_csv: str) -> pd.DataFrame:
    """
    Load the results CSV.
    Deduplicates automatically: if the same (method, dataset, series_name,
    seed, ablation) appears more than once (due to re-evaluation), keeps the
    last row, which has the most recent metric values.
    """
    if not os.path.exists(results_csv):
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.read_csv(results_csv, dtype={"seed": str, "ablation": str})
    df["ablation"] = df["ablation"].fillna("")
    df = df.drop_duplicates(
        subset=["method", "dataset", "series_name", "seed", "ablation"],
        keep="last",
    )
    return df.reset_index(drop=True)


def get_summary_table(
    results_csv: str,
    metric:      str = "vus_pr",
    groupby:     Optional[list] = None,
) -> pd.DataFrame:
    """
    Aggregate by method and dataset, reporting mean ± std across seeds.
    Rows where the chosen metric is NaN (run not yet complete) are excluded
    so that partial results do not inflate the standard deviation.

    Returns the main results table ready for copy-paste into the paper.
    """
    df = load_results(results_csv)
    if df.empty:
        return df

    groupby = groupby or ["method", "dataset"]

    # Drop rows where the metric was not yet computed
    df = df[df[metric].notna()]

    agg = df.groupby(groupby)[metric].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_mean", f"{metric}_std", "n_seeds"]
    return agg.reset_index()
