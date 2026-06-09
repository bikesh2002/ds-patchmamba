"""
Dataset loader for TSB-AD-M (primary) and legacy datasets (SMD, PSM, SWaT).

TSB-AD-M structure:
    TSB-AD-M-Eva.csv    — 85% of series, used for ALL final results
    TSB-AD-M-Tuning.csv — 15% of series, used ONLY for hyperparameter selection

CRITICAL: Never load TSB-AD-M-Eva.csv during HP search. The Tuning set exists
for exactly this purpose. Violating this leaks test information.

Legacy datasets follow a fixed convention:
    {dataset}/train/  — normal training data (no anomalies)
    {dataset}/test/   — test data with anomalies
    {dataset}/labels/ — binary anomaly labels aligned to test
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from .preprocessing import zscore_normalize, train_val_split_normal


@dataclass
class SeriesRecord:
    name: str
    train: np.ndarray       # (T_train, V) — normal data
    test: np.ndarray        # (T_test,  V) — may contain anomalies
    labels: np.ndarray      # (T_test,)    — binary {0,1}
    V: int                  # number of channels
    avg_length: float       # used to decide short-series stride


# ─────────────────────────────────────────────
# TSB-AD-M Loader
# ─────────────────────────────────────────────

def load_tsb_adm(
    tsb_dir: str,
    split: str = "eval",    # "eval" → Eva.csv  |  "tuning" → Tuning.csv
    normalize: bool = True,
) -> List[SeriesRecord]:
    """
    Load TSB-AD-M multivariate series.

    Args:
        tsb_dir: Directory containing TSB-AD-M-Eva.csv and TSB-AD-M-Tuning.csv
        split:   "eval" or "tuning"
        normalize: whether to apply per-channel z-score on training split

    Returns:
        List of SeriesRecord objects
    """
    assert split in ("eval", "tuning"), f"split must be 'eval' or 'tuning', got {split}"

    filename = "TSB-AD-M-Eva.csv" if split == "eval" else "TSB-AD-M-Tuning.csv"
    index_path = os.path.join(tsb_dir, filename)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{index_path} not found.\n"
            "Download with: wget https://www.thedatum.org/datasets/TSB-AD-M.zip && unzip TSB-AD-M.zip"
        )

    index_df = pd.read_csv(index_path)
    records = []

    for _, row in index_df.iterrows():
        series_path = os.path.join(tsb_dir, row["filepath"])
        if not os.path.exists(series_path):
            print(f"  [WARN] Series file not found, skipping: {series_path}")
            continue

        df = pd.read_csv(series_path)

        # Identify label column robustly.
        # TSB-AD-M uses "label" or "Label"; some series use "anomaly".
        # We do NOT fall back to the last column blindly because the last column
        # might be a numeric sensor channel, not a binary label.
        label_candidates = [c for c in df.columns
                            if c.lower() in ("label", "labels", "anomaly", "is_anomaly")]
        if not label_candidates:
            print(f"  [WARN] No label column found in {series_path}. Skipping.")
            continue
        label_col = label_candidates[0]
        data_cols = [c for c in df.columns if c != label_col]

        data   = df[data_cols].values.astype(np.float32)   # (T, V)
        labels = df[label_col].values.astype(np.int32)     # (T,)

        # Determine train / test split.
        # Priority 1: explicit is_train column (preferred — no heuristic needed).
        # Priority 2: split before the first anomaly (semi-supervised convention:
        #             training data is normal-only).
        # Priority 3: 60/40 split (last resort — only when no anomalies exist).
        if "is_train" in df.columns:
            train_mask  = df["is_train"].values.astype(bool)
            train_data  = data[train_mask]
            test_data   = data[~train_mask]
            test_labels = labels[~train_mask]
        else:
            anomaly_positions = np.where(labels > 0)[0]
            if len(anomaly_positions) == 0:
                # No anomalies at all — use 60/40 temporal split.
                # This is a degenerate case; series with no anomalies contribute
                # nothing to anomaly detection metrics but may appear in TSB-AD-M.
                split_idx = int(0.6 * len(data))
            else:
                # Split right before the first anomaly so training data is clean.
                # If the very first timestep is anomalous (split_idx=0), we have
                # no clean training data — fall back to 60/40 with a warning.
                split_idx = int(anomaly_positions[0])
                if split_idx < 10:
                    print(
                        f"  [WARN] {os.path.basename(series_path)}: first anomaly at "
                        f"t={split_idx}. Only {split_idx} clean timesteps available "
                        f"for training. Falling back to 60/40 split."
                    )
                    split_idx = int(0.6 * len(data))

            train_data  = data[:split_idx]
            test_data   = data[split_idx:]
            test_labels = labels[split_idx:]

        # Guard: test data and labels must have the same length.
        if len(test_data) != len(test_labels):
            raise ValueError(
                f"Length mismatch in {series_path}: "
                f"test_data has {len(test_data)} rows but test_labels has "
                f"{len(test_labels)} entries."
            )

        # Guard: training data must have enough timesteps for at least one window.
        if len(train_data) < 10:
            print(f"  [WARN] {os.path.basename(series_path)}: only {len(train_data)} "
                  f"training timesteps. Skipping.")
            continue

        if normalize:
            train_data, test_data = zscore_normalize(train_data, test_data)

        records.append(SeriesRecord(
            name=row.get("name", os.path.splitext(os.path.basename(series_path))[0]),
            train=train_data,
            test=test_data,
            labels=test_labels,
            V=data.shape[1],
            avg_length=len(data),
        ))

    return records


# ─────────────────────────────────────────────
# Legacy Dataset Loaders
# ─────────────────────────────────────────────

LEGACY_DATASETS = {
    "SMD":  {"V": 38, "avg_length": 25466},
    "PSM":  {"V": 25, "avg_length": 217624},
    "SWaT": {"V": 51, "avg_length": 207458},
}


def load_legacy(
    data_dir: str,
    dataset: str,
    normalize: bool = True,
) -> List[SeriesRecord]:
    """
    Load a legacy dataset (SMD, PSM, SWaT).

    Handles three common folder/file conventions found across Kaggle uploads:

    Convention A (SMD original from NetManAIOps/OmniAnomaly):
        {dataset}/train/machine-1-1.txt   — space/comma separated, no header
        {dataset}/test/machine-1-1.txt
        {dataset}/test_label/machine-1-1.txt

    Convention B (PSM and some repackaged SMD uploads):
        {dataset}/train/train.csv         — comma separated, no header
        {dataset}/test/test.csv
        {dataset}/labels/test_label.csv

    Convention C (SWaT from Anomaly Transformer Google Drive):
        {dataset}/normal.csv              — training data, 51 sensor columns, no header
        {dataset}/attack.csv              — test data, 51 sensor columns + label column
                                            label column contains "Normal"/"Attack" strings
                                            OR 0/1 integers

    The loader detects which convention is present automatically.
    """
    assert dataset in LEGACY_DATASETS, f"Unknown legacy dataset: {dataset}"

    base = os.path.join(data_dir, dataset)

    # ── Convention C: SWaT-style flat files (normal.csv + attack.csv) ─────
    normal_path = os.path.join(base, "normal.csv")
    attack_path = os.path.join(base, "attack.csv")
    if os.path.exists(normal_path) and os.path.exists(attack_path):
        return _load_swat_style(base, dataset, normal_path, attack_path, normalize)

    # ── Locate train folder (Convention A and B) ───────────────────────────
    train_dir = os.path.join(base, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"train/ folder not found at: {train_dir}\n"
            f"Also looked for normal.csv + attack.csv at {base} — not found.\n"
            f"On Kaggle: add the '{dataset}' dataset as an input and update "
            f"LEGACY_PATHS in Cell 1 to its mount path."
        )

    # ── Locate test folder ─────────────────────────────────────────────────
    test_dir = os.path.join(base, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test/ folder not found at: {test_dir}")

    # ── Locate labels folder — try multiple common names ───────────────────
    label_dir = None
    for candidate in ("labels", "test_label", "label", "test_labels"):
        p = os.path.join(base, candidate)
        if os.path.exists(p):
            label_dir = p
            break
    if label_dir is None:
        raise FileNotFoundError(
            f"Could not find a labels folder for '{dataset}' under {base}.\n"
            f"Looked for: labels/, test_label/, label/, test_labels/\n"
            f"Check the folder structure of the Kaggle dataset you added."
        )

    # ── Collect files — accepts both .csv and .txt ─────────────────────────
    train_files = sorted(_collect_data_files(train_dir))
    test_files  = sorted(_collect_data_files(test_dir))
    label_files = sorted(_collect_data_files(label_dir))

    if len(train_files) == 0:
        raise FileNotFoundError(
            f"No .csv or .txt files found in {train_dir}."
        )

    if not (len(train_files) == len(test_files) == len(label_files)):
        raise ValueError(
            f"File count mismatch for '{dataset}':\n"
            f"  train/      : {len(train_files)} files\n"
            f"  test/       : {len(test_files)} files\n"
            f"  {os.path.basename(label_dir)}/ : {len(label_files)} files\n"
            f"Each machine/entity must have exactly one file in each subfolder."
        )

    records = []
    for t_f, te_f, l_f in zip(train_files, test_files, label_files):
        train_data  = _read_data_file(t_f).astype(np.float32)
        test_data   = _read_data_file(te_f).astype(np.float32)
        test_labels = _read_data_file(l_f).squeeze().astype(np.int32)

        # Ensure 1D labels — squeeze can produce 0-D on single-value files
        if test_labels.ndim == 0:
            test_labels = test_labels.reshape(1)

        if len(test_data) != len(test_labels):
            raise ValueError(
                f"Length mismatch in {te_f}: "
                f"test has {len(test_data)} rows, labels has {len(test_labels)} entries."
            )

        if normalize:
            train_data, test_data = zscore_normalize(train_data, test_data)

        name = os.path.splitext(os.path.basename(t_f))[0]
        records.append(SeriesRecord(
            name=f"{dataset}_{name}",
            train=train_data,
            test=test_data,
            labels=test_labels,
            V=train_data.shape[1],
            avg_length=(len(train_data) + len(test_data)) / 2,
        ))

    return records


def _load_swat_style(
    base: str,
    dataset: str,
    normal_path: str,
    attack_path: str,
    normalize: bool,
) -> List[SeriesRecord]:
    """
    Load SWaT-style datasets where:
        normal.csv  = training data (all normal, sensor columns only)
        attack.csv  = test data (sensor columns + 1 label column at the end)

    The label column may contain:
        - String values: "Normal" → 0, "Attack" / "attack" → 1
        - Integer values: 0 / 1 directly
    """
    # ── Load training data ─────────────────────────────────────────────────
    train_df = pd.read_csv(normal_path, header=0, low_memory=False)

    # Drop any unnamed index columns that pandas sometimes adds
    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]

    # Drop any non-numeric columns (e.g. timestamp columns)
    train_df = train_df.select_dtypes(include=[np.number])
    train_data = train_df.values.astype(np.float32)

    # ── Load test data + labels ────────────────────────────────────────────
    attack_df = pd.read_csv(attack_path, header=0, low_memory=False)
    attack_df = attack_df.loc[:, ~attack_df.columns.str.startswith("Unnamed")]

    # Identify the label column — last column, or one named Normal/Attack/label
    label_col = None
    for col in attack_df.columns:
        if col.strip().lower() in ("normal/attack", "label", "labels", "attack", "anomaly"):
            label_col = col
            break
    if label_col is None:
        # Fall back to last column if it contains only two unique values
        last_col = attack_df.columns[-1]
        if attack_df[last_col].nunique() <= 2:
            label_col = last_col

    if label_col is None:
        raise ValueError(
            f"Could not identify a label column in {attack_path}.\n"
            f"Columns found: {list(attack_df.columns)}"
        )

    # Convert labels to binary int32
    raw_labels = attack_df[label_col].values
    if raw_labels.dtype == object or raw_labels.dtype.kind in ("U", "S"):
        # String labels — "Normal" → 0, anything else → 1
        test_labels = np.where(
            np.char.lower(raw_labels.astype(str)) == "normal", 0, 1
        ).astype(np.int32)
    else:
        test_labels = (raw_labels != 0).astype(np.int32)

    # Drop label column; keep only numeric sensor columns
    sensor_cols = [c for c in attack_df.columns if c != label_col]
    attack_df   = attack_df[sensor_cols].select_dtypes(include=[np.number])
    test_data   = attack_df.values.astype(np.float32)

    # Guard: column count must match between train and test
    if train_data.shape[1] != test_data.shape[1]:
        # Trim to the smaller of the two (handles minor version differences)
        V = min(train_data.shape[1], test_data.shape[1])
        train_data = train_data[:, :V]
        test_data  = test_data[:, :V]
        print(f"  [WARN] {dataset}: train/test column mismatch — trimmed to {V} columns.")

    if len(test_data) != len(test_labels):
        raise ValueError(
            f"Length mismatch in {attack_path}: "
            f"test has {len(test_data)} rows but labels has {len(test_labels)} entries."
        )

    if normalize:
        train_data, test_data = zscore_normalize(train_data, test_data)

    V = train_data.shape[1]
    return [SeriesRecord(
        name=f"{dataset}_main",
        train=train_data,
        test=test_data,
        labels=test_labels,
        V=V,
        avg_length=(len(train_data) + len(test_data)) / 2,
    )]


def _read_data_file(path: str) -> np.ndarray:
    """
    Read a .csv or .txt data file into a numpy array.
    Handles both comma-separated and space-separated formats.
    Assumes no header row (standard for all SMD/PSM/SWaT files).
    """
    try:
        # Try comma separator first (PSM, SWaT, repackaged SMD)
        arr = pd.read_csv(path, header=None, sep=",").values
        if arr.shape[1] == 1:
            # Single column after comma split — likely space-separated
            raise ValueError("single column — retry with space separator")
        return arr
    except Exception:
        # Fall back to whitespace separator (original SMD .txt files)
        return pd.read_csv(path, header=None, sep=r"\s+").values


def _collect_data_files(directory: str) -> List[str]:
    """Return all .csv and .txt files in a directory, sorted by name."""
    if os.path.isdir(directory):
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".csv") or f.endswith(".txt")
        ]
    return []


# ─────────────────────────────────────────────
# Convenience: load all datasets for an experiment run
# ─────────────────────────────────────────────

def load_all_legacy(data_dir: str, normalize: bool = True) -> Dict[str, List[SeriesRecord]]:
    """Load SMD, PSM, SWaT and return as a dict keyed by dataset name."""
    result = {}
    for ds in LEGACY_DATASETS:
        ds_path = os.path.join(data_dir, ds)
        if os.path.exists(ds_path):
            result[ds] = load_legacy(data_dir, ds, normalize=normalize)
        else:
            print(f"[WARNING] Legacy dataset not found: {ds_path}")
    return result
