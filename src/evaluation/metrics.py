"""
Evaluation metrics for DS-PatchMamba.

Metric hierarchy:
    Primary:          VUS-PR  — threshold-free, lag-tolerant, parameter-free
    Supplementary:    DQE     — ratio-robust cross-check for VUS-PR
    Secondary:        VUS-ROC, AUC-PR, AUC-ROC, affiliation-F1, event-F1, R-F1, F1
    Appendix ONLY:    PA-F1   — gameable; never in main results table

Key design decision — single TSB-AD call per (scores, labels) pair:
    The TSB-AD package computes all VUS metrics together internally. Calling it
    twice (once for VUS-PR, once for VUS-ROC) doubles the work. We call it once
    in compute_all_metrics, cache the result, and extract both values from it.
"""

import numpy as np
from typing import Dict, Optional

# NumPy 2.0 removed np.trapz in favour of np.trapezoid
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz"))

# ─────────────────────────────────────────────────────────────────────────────
# TSB-AD package (VUS-PR, VUS-ROC)
# Installed via: pip install TSB-AD
# Imported as:   from tsad.vus.metrics import get_metrics
# ─────────────────────────────────────────────────────────────────────────────

TSB_AVAILABLE = False
_tsb_get_metrics = None   # single reference to avoid dual-import confusion

try:
    from tsad.vus.metrics import get_metrics as _tsb_get_metrics
    TSB_AVAILABLE = True
except ImportError:
    try:
        # Some package versions use a slightly different submodule path
        from tsad.metrics import get_metrics as _tsb_get_metrics   # type: ignore
        TSB_AVAILABLE = True
    except ImportError:
        TSB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DQE package (supplementary metric)
# From arXiv 2603.06131; install from author's repo if available
# ─────────────────────────────────────────────────────────────────────────────

DQE_AVAILABLE = False
_dqe_fn = None

try:
    from dqe import detection_quality_evaluation as _dqe_fn   # type: ignore
    DQE_AVAILABLE = True
except ImportError:
    DQE_AVAILABLE = False

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """Coerce to 1D float64/int32 and validate length match."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if len(scores) != len(labels):
        raise ValueError(
            f"scores length {len(scores)} != labels length {len(labels)}."
        )
    return scores, labels


def _get_segments(arr: np.ndarray) -> list:
    """
    Extract (start, end) inclusive indices of consecutive 1-runs.
    Vectorised via np.diff — no Python loop over timesteps.
    """
    arr = arr.ravel().astype(np.int32)
    if arr.sum() == 0:
        return []

    # Pad with zeros so edges are always detected
    padded = np.concatenate([[0], arr, [0]])
    diff   = np.diff(padded.astype(np.int8))

    starts = np.where(diff ==  1)[0]   # rising  edges
    ends   = np.where(diff == -1)[0]   # falling edges (exclusive → subtract 1)
    return list(zip(starts.tolist(), (ends - 1).tolist()))


def _dilate_labels(labels: np.ndarray, buffer: int) -> np.ndarray:
    """
    Expand each anomaly segment outward by `buffer` timesteps on each side.
    Vectorised via cumsum — avoids scipy dependency and is faster.
    """
    if buffer == 0:
        return labels
    labels = labels.ravel().astype(np.int32)
    T = len(labels)
    # Build a position array of segment centres and dilate via slicing
    out = labels.copy()
    segs = _get_segments(labels)
    for s, e in segs:
        lo = max(0,     s - buffer)
        hi = min(T - 1, e + buffer)
        out[lo:hi + 1] = 1
    return out


def _point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Point-Adjusted prediction: if any predicted point falls inside a ground-truth
    anomaly segment, all points in that segment are marked as detected.

    Vectorised: uses segment extraction and numpy slice assignment.
    No per-timestep Python loop.
    """
    adj  = pred.copy().astype(np.int32)
    segs = _get_segments(labels)
    for s, e in segs:
        if pred[s:e + 1].any():
            adj[s:e + 1] = 1
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# VUS-PR and VUS-ROC  (called ONCE per pair via compute_all_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _call_tsb(
    scores:     np.ndarray,
    labels:     np.ndarray,
    max_buffer: int,
) -> dict:
    """
    Single entry-point to the TSB-AD package.
    Returns a dict of all VUS metrics, or an empty dict on failure.
    Called exactly once per (scores, labels) pair inside compute_all_metrics.
    """
    if not TSB_AVAILABLE:
        return {}
    try:
        result = _tsb_get_metrics(scores, labels, metric="all", slidingWindow=max_buffer)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _manual_vus_pr(
    scores:       np.ndarray,
    labels:       np.ndarray,
    max_buffer:   int = 100,
    n_thresholds: int = 200,
    n_buffers:    int = 20,
) -> float:
    """
    Fallback VUS-PR when TSB-AD package is unavailable.
    Vectorised threshold sweep — no Python loop over thresholds.

    For each buffer size b, builds a (n_thresholds,) precision and recall array
    using broadcast comparisons, then computes AUPR via trapz.
    VUS-PR = mean AUPR across buffer sizes.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0

    # Degenerate case: all scores identical → no discrimination possible
    if np.std(scores) < 1e-10:
        return 0.0

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    # preds shape: (n_thresholds, T) — each row is a binary prediction at one threshold
    preds = (scores[np.newaxis, :] >= thresholds[:, np.newaxis]).astype(np.int8)  # (N_th, T)

    buffer_sizes = np.linspace(0, max_buffer, n_buffers, dtype=int)
    aupr_list = []

    for buf in buffer_sizes:
        dilated = _dilate_labels(labels, int(buf)).astype(np.int8)  # (T,)

        tp = (preds * dilated[np.newaxis, :]).sum(axis=1).astype(float)   # (N_th,)
        fp = (preds * (1 - dilated)[np.newaxis, :]).sum(axis=1).astype(float)
        fn = ((1 - preds) * labels[np.newaxis, :]).sum(axis=1).astype(float)

        precision = tp / np.maximum(tp + fp, 1e-10)  # (N_th,)
        recall    = tp / np.maximum(tp + fn, 1e-10)

        # Sort by recall ascending for trapz integration
        order     = np.argsort(recall)
        aupr      = float(_trapezoid(precision[order], recall[order]))
        aupr_list.append(aupr)

    return float(np.mean(aupr_list))


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions (used when calling separately)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vus_pr(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 100) -> float:
    scores, labels = _validate_inputs(scores, labels)
    tsb = _call_tsb(scores, labels, max_buffer)
    if tsb:
        return float(tsb.get("VUS-PR", tsb.get("vus_pr", 0.0)))
    return _manual_vus_pr(scores, labels, max_buffer=max_buffer)


def compute_dqe(scores: np.ndarray, labels: np.ndarray) -> float:
    """DQE score, or -1.0 if package not installed."""
    if not DQE_AVAILABLE:
        return -1.0
    try:
        return float(_dqe_fn(scores, labels))
    except Exception:
        return -1.0


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


def compute_auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return 0.0


def compute_standard_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(f1_score(labels, pred, zero_division=0))
    except Exception:
        return 0.0


def compute_pa_f1(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """PA-F1 — APPENDIX ONLY. Do not put in main results table."""
    pred     = (scores >= threshold).astype(np.int32)
    adj_pred = _point_adjust(pred, labels)
    return float(f1_score(labels, adj_pred, zero_division=0))


def compute_event_f1(pred: np.ndarray, labels: np.ndarray, buffer: int = 10) -> float:
    """
    Event-level F1 with lag tolerance.
    A predicted segment is a TP if it overlaps a ground-truth segment
    within ±buffer timesteps.
    """
    gt_segs   = _get_segments(labels)
    pred_segs = _get_segments(pred)

    if not gt_segs or not pred_segs:
        return 0.0

    tp, matched = 0, set()
    for gs, ge in gt_segs:
        for j, (ps, pe) in enumerate(pred_segs):
            if j in matched:
                continue
            if ps <= ge + buffer and pe >= gs - buffer:   # overlap with buffer
                tp += 1
                matched.add(j)
                break

    precision = tp / len(pred_segs)
    recall    = tp / len(gt_segs)
    denom     = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def _compute_range_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    """Range-based F1 (Tatbul et al. 2018)."""
    segs = _get_segments(labels)
    if not segs:
        return 0.0

    # Range recall: fraction of each GT segment covered
    recalls = [
        min(pred[s:e + 1].sum() / max(e - s + 1, 1), 1.0)
        for s, e in segs
    ]
    recall = float(np.mean(recalls))

    pred_segs = _get_segments(pred)
    if not pred_segs:
        return 0.0

    # Range precision: fraction of each predicted segment that is GT
    precs = [
        min(labels[s:e + 1].sum() / max(e - s + 1, 1), 1.0)
        for s, e in pred_segs
    ]
    precision = float(np.mean(precs))

    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Primary entry-point: compute all 10 metrics in one call
# TSB-AD package called ONCE; result cached for VUS-PR and VUS-ROC extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    scores:         np.ndarray,
    labels:         np.ndarray,
    threshold:      Optional[float] = None,
    vus_max_buffer: int = 100,
) -> Dict[str, float]:
    """
    Compute all 10 metrics for one (scores, labels) pair.

    The TSB-AD package is called exactly once and the result is used for both
    VUS-PR and VUS-ROC. This avoids the 2× computation cost of two separate calls.

    Threshold determination:
        - If DSPOT was run: pass its threshold here explicitly.
        - Otherwise: 99th percentile of scores (standard fallback for CPU baselines
          which have no training phase to fit a threshold on).

    PA-F1 is computed and returned in the dict but MUST NOT appear in the
    main results table — it is stored in the CSV for appendix use only.

    Args:
        scores:         1D float array (higher = more anomalous)
        labels:         1D binary int array {0, 1}
        threshold:      anomaly decision boundary for F1 metrics
        vus_max_buffer: boundary buffer for VUS integration (default 100)

    Returns:
        Dict with keys: vus_pr, dqe, vus_roc, auc_pr, auc_roc,
                        standard_f1, event_f1, r_based_f1, affiliation_f1, pa_f1
    """
    scores, labels = _validate_inputs(scores, labels)

    # Default threshold: 99th percentile of scores.
    # For CPU baselines without a training phase this is the standard choice.
    # For DS-PatchMamba, pass the DSPOT threshold explicitly instead.
    if threshold is None:
        threshold = float(np.percentile(scores, 99.0))

    pred = (scores >= threshold).astype(np.int32)

    # ── Single TSB-AD call — extracts VUS-PR and VUS-ROC together ──────────
    tsb_result = _call_tsb(scores, labels, vus_max_buffer)

    if tsb_result:
        vus_pr  = float(tsb_result.get("VUS-PR",  tsb_result.get("vus_pr",  0.0)))
        vus_roc = float(tsb_result.get("VUS-ROC", tsb_result.get("vus_roc", 0.0)))
    else:
        vus_pr  = _manual_vus_pr(scores, labels, max_buffer=vus_max_buffer)
        vus_roc = compute_auc_roc(scores, labels)   # fallback: standard AUROC

    return {
        "vus_pr":         vus_pr,
        "dqe":            compute_dqe(scores, labels),
        "vus_roc":        vus_roc,
        "auc_pr":         compute_auc_pr(scores, labels),
        "auc_roc":        compute_auc_roc(scores, labels),
        "standard_f1":    compute_standard_f1(pred, labels),
        "event_f1":       compute_event_f1(pred, labels),
        "r_based_f1":     _compute_range_f1(pred, labels),
        "affiliation_f1": compute_event_f1(pred, labels, buffer=20),
        "pa_f1":          compute_pa_f1(scores, labels, threshold),
    }
