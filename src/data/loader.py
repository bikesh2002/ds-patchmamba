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
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from .preprocessing import zscore_normalize, train_val_split_normal


def _sanitize_sensor_data(arr: np.ndarray) -> np.ndarray:
    """
    Replace Inf with NaN, then forward/backward fill remaining NaNs.
    SWaT and some PSM uploads contain missing sensor readings.
    """
    arr = np.asarray(arr, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    if arr.size == 0:
        return arr.astype(np.float32)
    df = pd.DataFrame(arr)
    df = df.ffill().bfill().fillna(0.0)
    return df.values.astype(np.float32)


def _parse_binary_labels(raw_labels) -> np.ndarray:
    """
    Parse binary anomaly labels from numeric or string columns.
    Handles: 0/1, 0.0/1.0, 'Normal'/'Attack', '0'/'1' strings, True/False.
    """
    raw = np.asarray(raw_labels).ravel()

    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").values
    finite_mask = np.isfinite(numeric)
    if finite_mask.mean() > 0.95:
        finite = numeric[finite_mask]
        u = np.unique(finite)
        if len(u) <= 3:
            if set(u).issubset({0, 1, 0.0, 1.0}):
                vals = np.where(np.isfinite(numeric), numeric, 0.0)
                # Some uploads use 1=Normal (majority); standard SWaT uses 0=Normal
                if (vals == 1).mean() > 0.5:
                    return (vals == 0).astype(np.int32)
                return (vals != 0).astype(np.int32)
            # Generic two-class numeric: minority class = anomaly
            lo = float(np.min(u))
            hi = float(np.max(u))
            filled = np.where(np.isfinite(numeric), numeric, lo)
            labels = (filled == hi).astype(np.int32)
            if labels.mean() > 0.5 and lo != hi:
                labels = (filled == lo).astype(np.int32)
            return labels.astype(np.int32)

    s = pd.Series(raw).astype(str).str.strip().str.lower()

    if (s == "normal").mean() > 0.05:
        return np.where(s == "normal", 0, 1).astype(np.int32)
    if s.isin(["0", "0.0"]).mean() > 0.05:
        return np.where(s.isin(["0", "0.0"]), 0, 1).astype(np.int32)
    if s.isin(["false", "no"]).mean() > 0.05:
        return np.where(s.isin(["false", "no"]), 0, 1).astype(np.int32)
    if (s == "attack").mean() > 0.05:
        return np.where(s == "attack", 1, 0).astype(np.int32)
    if s.isin(["true", "false"]).any():
        return np.where(s == "true", 1, 0).astype(np.int32)

    return np.where(
        s.isin(["normal", "0", "0.0", "false", "no"]),
        0,
        1,
    ).astype(np.int32)


def _is_swat_label_column_name(col: str) -> bool:
    key = str(col).strip().lstrip("\ufeff").lower().replace(" ", "").replace("-", "_")
    if key in (
        "normal/attack", "normal_attack", "label", "labels",
        "anomaly", "anomaly_label", "attack_label", "ground_truth",
    ):
        return True
    return "normal" in key and "attack" in key


def _is_swat_sensor_column_name(col: str) -> bool:
    """Return True for SWaT PLC tag names (P302, LIT101, P1_AIT_001, …)."""
    if _is_swat_label_column_name(col):
        return False
    name = str(col).strip()
    if re.match(r"^P[1-6]_", name, re.I):
        return True
    if re.match(r"^(LIT|FIT|AIT|PIT|MV|UV|DPIT|PMP|VALV|SCADA)", name, re.I):
        return True
    if re.match(r"^P\d{3}$", name, re.I):
        return True
    return False


def _column_has_swat_string_labels(series: pd.Series) -> bool:
    s = series.astype(str).str.strip().str.lower()
    return bool((s == "normal").mean() > 0.05)



def _read_swat_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _find_swat_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Locate the SWaT label column by name or by 'Normal' string content.
    Returns None if no trustworthy label column is found.
    """
    for col in df.columns:
        if _is_swat_label_column_name(col):
            return col

    for col in df.columns:
        if not _is_swat_sensor_column_name(col) and _column_has_swat_string_labels(df[col]):
            return col

    last = df.columns[-1]
    if not _is_swat_sensor_column_name(last):
        numeric = pd.to_numeric(df[last], errors="coerce")
        if numeric.notna().mean() > 0.95:
            u = set(numeric.dropna().unique())
            if u.issubset({0, 1, 0.0, 1.0}):
                return last

    return None


def _read_swat_test_with_labels(
    path: str,
) -> Tuple[pd.DataFrame, str, np.ndarray, float]:
    """Load a SWaT test CSV and return (df, label_col, binary labels, anomaly ratio)."""
    df = _read_swat_csv(path)

    label_col = _find_swat_label_column(df)
    if label_col is None:
        raise ValueError(
            f"No SWaT label column found in {path}. "
            f"Expected 'Normal/Attack' or a column containing 'Normal' strings. "
            f"Last columns: {list(df.columns[-5:])}"
        )
    if _is_swat_sensor_column_name(label_col):
        raise ValueError(
            f"SWaT label column resolved to sensor tag {label_col!r} in {path} — "
            f"refusing to use sensor data as labels."
        )

    test_labels = _parse_binary_labels(df[label_col].values)
    ratio       = float(test_labels.mean())

    if ratio > 0.9:
        flipped = 1 - test_labels
        if 0.001 <= flipped.mean() <= 0.45:
            test_labels = flipped
            ratio       = float(test_labels.mean())

    return df, label_col, test_labels, ratio


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


# Official SWaT 2015 attack-period length (iTrust / Anomaly Transformer benchmark).
SWAT_ATTACK_PERIOD_ROWS = 449_919


def _swat_use_attack_csv_directly(df: pd.DataFrame, label_col: str, ratio: float) -> bool:
    """
    Return True when attack.csv is the full 4-day test file (mixed Normal/Attack).
    Some Kaggle uploads ship a separate attack.csv containing ONLY attack
    timesteps (all labels 'Attack', ~54k rows) — that file is NOT a valid test set.
    """
    if len(df) < 350_000:
        return False
    if not (0.05 <= ratio <= 0.25):
        return False
    return _column_has_swat_string_labels(df[label_col]) or _is_swat_label_column_name(label_col)


def _load_swat_test_period(
    base: str,
    attack_path: str,
) -> Tuple[pd.DataFrame, str, np.ndarray, float, str]:
    """
    Load SWaT test data + labels.

    Standard: attack.csv (~450k rows, ~12% anomalies).
    Kaggle vishala28 quirk: attack.csv is attack-only (~54k rows, 100% Attack);
    full test period = last SWAT_ATTACK_PERIOD_ROWS rows of merged.csv.
    """
    attack_df, label_col, test_labels, ratio = _read_swat_test_with_labels(attack_path)
    attack_rows = len(attack_df)

    if _swat_use_attack_csv_directly(attack_df, label_col, ratio):
        return attack_df, label_col, test_labels, ratio, "attack.csv"

    merged_path = os.path.join(base, "merged.csv")
    if not os.path.exists(merged_path):
        raise ValueError(
            f"{attack_path} is not a valid SWaT test file "
            f"(ratio {ratio:.1%}, {attack_rows:,} rows). "
            f"Expected ~{SWAT_ATTACK_PERIOD_ROWS:,} rows with mixed Normal/Attack labels, "
            f"or a merged.csv to extract the attack period from."
        )

    mdf, mcol, _, _ = _read_swat_test_with_labels(merged_path)
    test_len = min(SWAT_ATTACK_PERIOD_ROWS, len(mdf))
    mdf = mdf.iloc[-test_len:].reset_index(drop=True)
    test_labels = _parse_binary_labels(mdf[mcol].values)
    ratio = float(test_labels.mean())

    if not (0.05 <= ratio <= 0.25):
        raise ValueError(
            f"SWaT test slice from merged.csv has unexpected anomaly ratio {ratio:.1%} "
            f"({len(test_labels):,} rows). Expected ~12%."
        )

    print(
        f"  [INFO] SWaT: attack.csv contains only attack timesteps "
        f"({attack_rows:,} rows, all labeled Attack) — "
        f"using last {test_len:,} rows of merged.csv as the 4-day test period."
    )
    return mdf, mcol, test_labels, ratio, f"merged.csv (last {test_len:,} rows)"
    base: str,
    dataset: str,
    normal_path: str,
    attack_path: str,
    normalize: bool,
) -> List[SeriesRecord]:
    """
    Load SWaT-style datasets where:
        normal.csv  = training data (all normal, sensor columns only)
        attack.csv  = test data OR attack-only snippets (Kaggle vishala28 upload)
        merged.csv  = full timeline; last 449,919 rows used when attack.csv is attack-only

    The label column may contain:
        - String values: "Normal" → 0, "Attack" / attack types → 1
        - Integer values: 0 / 1 directly
    """
    # ── Load training data ─────────────────────────────────────────────────
    train_df = pd.read_csv(normal_path, header=0, low_memory=False)

    # Drop any unnamed index columns that pandas sometimes adds
    train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]

    # Drop any non-numeric columns (e.g. timestamp columns)
    train_df = train_df.select_dtypes(include=[np.number])
    train_data = _sanitize_sensor_data(train_df.values)

    # ── Load test data + labels ────────────────────────────────────────────
    attack_df, label_col, test_labels, ratio, test_source = _load_swat_test_period(
        base, attack_path
    )

    if _is_swat_sensor_column_name(label_col):
        raise ValueError(
            f"{dataset}: label column {label_col!r} is a sensor tag, not a label column."
        )

    print(
        f"  [INFO] {dataset}: test from {test_source}, labels in {label_col!r} "
        f"(anomaly ratio {ratio:.1%}, test rows {len(test_labels):,})."
    )

    # Drop label column; keep only numeric sensor columns
    sensor_cols = [c for c in attack_df.columns if c != label_col]
    attack_df   = attack_df[sensor_cols].select_dtypes(include=[np.number])
    test_data   = _sanitize_sensor_data(attack_df.values)

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
        - Non-numeric / timestamp columns dropped automatically
        - Columns stored as object dtype coerced via pd.to_numeric
    """
    path = str(path)

    if path.lower().endswith(".csv"):
        return _read_csv_numeric(path)

    return _read_txt_numeric(path)


def _read_txt_numeric(path: str) -> np.ndarray:
    """
    Read headerless .txt sensor files (SMD).

    SMD files are comma-separated floats, one row per timestep.
    np.loadtxt is the primary parser; pandas is fallback only.
    """
    path = str(path)

    # Primary: np.loadtxt (fast and reliable for clean numeric SMD files)
    for delimiter in (",", None):   # None = any whitespace
        try:
            arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.size > 0 and arr.shape[1] > 0:
                return _sanitize_sensor_data(arr)
        except (ValueError, OSError):
            continue

    # Fallback: pandas (tolerant of ragged rows)
    for sep in [",", r"\s+"]:
        try:
            df = pd.read_csv(path, header=None, sep=sep, low_memory=False)
            num = df.apply(pd.to_numeric, errors="coerce")
            num = num.dropna(axis=1, how="all").dropna(axis=0, how="any")
            if num.shape[1] > 0 and num.shape[0] > 0:
                return _sanitize_sensor_data(num.values)
        except Exception:
            continue

    raise ValueError(f"No numeric data found in {path}")


def _read_csv_numeric(path: str) -> np.ndarray:
    """Read a CSV sensor file, dropping timestamps and coercing all sensor cols to float."""
    df = pd.read_csv(path, low_memory=False)

    # Drop known non-sensor columns by name (PSM Kaggle upload)
    drop = [
        c for c in df.columns
        if str(c).lower() in ("timestamp_(min)", "timestamp", "time", "date", "index")
    ]
    if drop:
        df = df.drop(columns=drop)

    # Coerce every remaining column to numeric (handles object-dtype sensor cols)
    coerced = df.apply(pd.to_numeric, errors="coerce")
    coerced = coerced.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if coerced.shape[1] > 0:
        return _sanitize_sensor_data(coerced.values)

    # Headerless file: first row may literally be column names like 'timestamp_(min)'
    df2 = pd.read_csv(path, header=None, low_memory=False)
    first = df2.iloc[0, 0]
    if isinstance(first, str) and not _looks_numeric(first):
        df2 = df2.iloc[1:].reset_index(drop=True)

    coerced2 = df2.apply(pd.to_numeric, errors="coerce")
    coerced2 = coerced2.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if coerced2.shape[1] == 0:
        raise ValueError(f"No numeric data found in {path}")
    return _sanitize_sensor_data(coerced2.values)


def _looks_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _read_label_file(path: str) -> np.ndarray:
    """
    Read a binary label file (0/1) from legacy datasets.
    Handles optional header row and named label columns.
    """
    path = str(path)

    # SMD test_label/*.txt — one binary label per line, no header
    if path.lower().endswith(".txt"):
        vals = np.loadtxt(path, dtype=np.float64)
        return vals.astype(np.int32).ravel()

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
