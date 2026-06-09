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

    # Official TSB-AD repo uses 'file_name'; some mirrors use 'filepath'.
    if "filepath" in index_df.columns:
        path_col = "filepath"
    elif "file_name" in index_df.columns:
        path_col = "file_name"
    else:
        raise ValueError(
            f"{index_path} must contain a 'file_name' or 'filepath' column. "
            f"Found: {list(index_df.columns)}"
        )

    records = []

    for _, row in index_df.iterrows():
        rel_path    = str(row[path_col]).strip()
        series_path = _resolve_tsb_series_path(tsb_dir, rel_path)
        if series_path is None:
            print(f"  [WARN] Series file not found, skipping: {rel_path}")
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
            name=row.get("name", os.path.splitext(os.path.basename(rel_path))[0]),
            train=train_data,
            test=test_data,
            labels=test_labels,
            V=data.shape[1],
            avg_length=len(data),
        ))

    return records


def _resolve_tsb_series_path(tsb_dir: str, rel_path: str) -> Optional[str]:
    """
    Locate a TSB-AD-M series CSV under tsb_dir.

    The zip from thedatum.org contains only series CSVs (often under TSB-AD-M/).
    Eval/Tuning index files come separately from the TSB-AD GitHub repo and list
    entries by file_name only, e.g. '004_MSL_id_3_Sensor_tr_530_1st_630.csv'.
    """
    rel_path = rel_path.strip().replace("\\", "/")
    basename = os.path.basename(rel_path)

    candidates = [
        os.path.join(tsb_dir, rel_path),
        os.path.join(tsb_dir, "TSB-AD-M", rel_path),
        os.path.join(tsb_dir, "TSB-AD-M", basename),
        os.path.join(tsb_dir, basename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Last resort: walk tsb_dir and match by basename (handles nested unzip layouts).
    for root, _, files in os.walk(tsb_dir):
        if basename in files:
            return os.path.join(root, basename)

    return None


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

    # ── Convention D: PSM-style flat files (train.csv + test.csv + test_label.csv)
    # All three files sit in the same directory with no subfolders.
    train_flat = os.path.join(base, "train.csv")
    test_flat  = os.path.join(base, "test.csv")
    for label_name in ("test_label.csv", "labels.csv", "test_labels.csv"):
        label_flat = os.path.join(base, label_name)
        if os.path.exists(train_flat) and os.path.exists(test_flat) and os.path.exists(label_flat):
            return _load_flat_style(base, dataset, train_flat, test_flat, label_flat, normalize)

    # ── Locate train folder (Convention A and B) ───────────────────────────
    train_dir = os.path.join(base, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Could not load '{dataset}' from {base}.\n"
            f"Tried all supported conventions:\n"
            f"  A: train/, test/, test_label/ subfolders with .txt files (SMD)\n"
            f"  B: train/, test/, labels/ subfolders with .csv files\n"
            f"  C: normal.csv + attack.csv flat files (SWaT)\n"
            f"  D: train.csv + test.csv + test_label.csv flat files (PSM)\n"
            f"Check the folder structure and update LEGACY_PATHS in Cell 1."
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
        test_labels = _read_label_file(l_f)

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


def _load_flat_style(
    base: str,
    dataset: str,
    train_path: str,
    test_path: str,
    label_path: str,
    normalize: bool,
) -> List[SeriesRecord]:
    """
    Load PSM-style datasets where train.csv, test.csv, test_label.csv
    all sit in the same directory with no subfolders.

    PSM structure (Kaggle upload may include headers and a timestamp column):
        data/train.csv        — header row + 25 sensor columns (+ optional timestamp)
        data/test.csv         — same
        data/test_label.csv   — header row + binary label column
    """
    train_data  = _read_data_file(train_path).astype(np.float32)
    test_data   = _read_data_file(test_path).astype(np.float32)
    test_labels = _read_label_file(label_path)

    if len(test_data) != len(test_labels):
        raise ValueError(
            f"Length mismatch in {test_path}: "
            f"test has {len(test_data)} rows, labels has {len(test_labels)} entries."
        )

    if normalize:
        train_data, test_data = zscore_normalize(train_data, test_data)

    return [SeriesRecord(
        name=f"{dataset}_main",
        train=train_data,
        test=test_data,
        labels=test_labels,
        V=train_data.shape[1],
        avg_length=(len(train_data) + len(test_data)) / 2,
    )]


def _read_data_file(path: str, numeric_only: bool = True) -> np.ndarray:
    """
    Read a .csv or .txt data file into a numpy array.

    Handles:
        - Comma- or whitespace-separated files (SMD .txt uses whitespace)
        - Optional header row (PSM Kaggle uploads include 'timestamp_(min)' etc.)
        - Non-numeric columns dropped automatically (timestamps, metadata)
    """
    path = str(path)

    # ── Comma-separated (.csv and some .txt) ──────────────────────────────
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
        arr = _numeric_frame_to_array(df, path, sep=",")
        if arr is not None:
            return arr
        raise ValueError(f"No numeric data found in {path}")

    # ── Whitespace-separated (original SMD .txt) ────────────────────────
    df = pd.read_csv(path, header=None, sep=r"\s+", low_memory=False)
    if numeric_only:
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if num.shape[1] == 0:
            raise ValueError(f"No numeric data found in {path}")
        return num.values
    return df.values


def _numeric_frame_to_array(df: pd.DataFrame, path: str, sep: str) -> Optional[np.ndarray]:
    """
    Extract numeric sensor columns from a DataFrame.
    Retries with header=None if the first row was mis-parsed as header.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        return num.values

    # First row may have been misread as column names (headerless file).
    df2 = pd.read_csv(path, header=None, sep=sep, low_memory=False)
    num2 = df2.apply(pd.to_numeric, errors="coerce")
    num2 = num2.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if num2.shape[1] > 0:
        return num2.values

    return None


def _read_label_file(path: str) -> np.ndarray:
    """
    Read a binary label file (0/1) from legacy datasets.
    Handles optional header row and named label columns.
    """
    df = pd.read_csv(path, low_memory=False)

    # Headerless single-column file: first value became column name '0' or '1'.
    if df.shape[1] == 1 and str(df.columns[0]) in ("0", "1", "0.0", "1.0"):
        df = pd.read_csv(path, header=None, low_memory=False)

    for col in df.columns:
        if str(col).lower() in ("label", "labels", "anomaly", "is_anomaly"):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return vals.astype(np.int32).values.ravel()

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 1:
        vals = pd.to_numeric(num.iloc[:, 0], errors="coerce").fillna(0)
        return vals.astype(np.int32).values.ravel()

    # Last resort: treat as headerless single column
    df = pd.read_csv(path, header=None, low_memory=False)
    vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0)
    return vals.astype(np.int32).values.ravel()


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
