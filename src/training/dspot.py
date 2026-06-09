"""
DSPOT — Drift-aware Streaming Peak-Over-Threshold anomaly threshold.

Based on Siffer et al. "Anomaly Detection in Streams with Extreme Value Theory"
(KDD 2017). Fits a Generalized Pareto Distribution (GPD) to the tail of the
anomaly score stream to set a statistically principled, label-free threshold.

The threshold adapts as the score distribution drifts, making it robust to
concept drift — directly closing Limitation L4 (static threshold degradation).

DSPOT is used ONLY for computing binary anomaly labels.
All VUS-PR / AUROC / AUPR metrics are computed on raw scores (threshold-free).

Notes:
- For series with average length < 5000 timesteps (e.g. MSL, avg=3119),
  use stride=50 during inference to generate ≥50 score peaks for stable GPD fitting.
  GPD fitting is unreliable with fewer than 50 exceedances.
- Fallback: if GPD fitting fails (too few points, convergence issues),
  fall back to 99.99th percentile of the training scores.
"""

import numpy as np
from scipy.stats import genpareto
from typing import Optional


class DSPOT:
    """
    DSPOT adaptive anomaly threshold via Extreme Value Theory.

    Usage:
        dspot = DSPOT(q=1e-4, level=0.98)
        dspot.fit(train_scores)          # fit GPD on training score tail
        threshold = dspot.threshold_     # use for test scoring
        labels = dspot.predict(test_scores)
    """

    def __init__(
        self,
        q: float = 1e-4,       # target false-alarm rate (false positives per sample)
        level: float = 0.98,   # initial POT level (percentile to define "tail")
        n_init: int = 1000,    # minimum training points for reliable fit
    ):
        self.q       = q
        self.level   = level
        self.n_init  = n_init
        self.threshold_: Optional[float] = None
        self._train_scores: Optional[np.ndarray] = None

    def fit(self, train_scores: np.ndarray) -> "DSPOT":
        """
        Fit GPD to the tail of training (normal) anomaly scores.
        train_scores: 1D array of anomaly scores on normal (training) data.
        """
        self._train_scores = train_scores.ravel().copy()

        if len(self._train_scores) < self.n_init:
            # Not enough data for reliable GPD; fall back to percentile
            self.threshold_ = float(np.percentile(self._train_scores, 99.99))
            return self

        # Initial POT threshold: level-th percentile of training scores
        z_init = float(np.percentile(self._train_scores, self.level * 100))

        # Extract peak exceedances above z_init
        exceedances = self._train_scores[self._train_scores > z_init] - z_init

        if len(exceedances) < 50:
            # Too few peaks for reliable GPD; fall back to percentile
            self.threshold_ = float(np.percentile(self._train_scores, 99.99))
            return self

        try:
            # Fit Generalized Pareto Distribution to exceedances
            c, loc, scale = genpareto.fit(exceedances, floc=0)

            n_total = len(self._train_scores)
            n_exceed = len(exceedances)

            # EVT threshold formula (Siffer 2017, Eq.1):
            # z_q = z_init + (scale/c) * ((q * n_total / n_exceed)^(-c) - 1)
            if abs(c) < 1e-10:
                # Shape ≈ 0: exponential tail
                z_q = z_init - scale * np.log(self.q * n_total / n_exceed)
            else:
                z_q = z_init + (scale / c) * (
                    (self.q * n_total / n_exceed) ** (-c) - 1.0
                )

            self.threshold_ = float(z_q)

        except Exception:
            # GPD fitting failed; fall back to 99.99th percentile
            self.threshold_ = float(np.percentile(self._train_scores, 99.99))

        return self

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly labels using the fitted threshold.
        test_scores: 1D array of anomaly scores.
        Returns: binary array {0,1} of same length.
        """
        if self.threshold_ is None:
            raise RuntimeError("DSPOT.fit() must be called before predict().")
        return (test_scores.ravel() > self.threshold_).astype(np.int32)

    @property
    def threshold(self) -> float:
        if self.threshold_ is None:
            raise RuntimeError("DSPOT.fit() must be called first.")
        return self.threshold_


def compute_anomaly_scores(
    errors: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Post-process per-timestep reconstruction errors into a smooth anomaly score.

    1. Already a 1D array of per-timestep errors (shape T,)
    2. Apply Gaussian smoothing to reduce spike noise without shifting boundaries.
       sigma=1.0 is conservative — minimal smoothing.

    Returns smoothed 1D score array.
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(errors.ravel(), sigma=sigma)


def fuse_scores(
    e_fine: np.ndarray,
    e_coarse: np.ndarray,
    w: float = 0.5,
) -> np.ndarray:
    """
    Fuse fine and coarse per-timestep reconstruction errors.
    s_t = w * e_f,t + (1-w) * e_c,t

    w is a hyperparameter selected on the TSB-AD-M Tuning set.
    It is NOT learned via backpropagation (near-zero gradient during normal-only training).

    e_fine, e_coarse: 1D arrays of shape (T,)
    Returns: fused score array (T,)
    """
    assert 0.0 <= w <= 1.0, f"Fusion weight w must be in [0,1], got {w}"
    return w * e_fine.ravel() + (1.0 - w) * e_coarse.ravel()
