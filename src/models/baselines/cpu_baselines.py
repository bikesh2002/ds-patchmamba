"""
CPU baselines for DS-PatchMamba comparison.

These run entirely on CPU with no GPU cost — they must be included
in every Kaggle session as they are trivial to run.

Baselines implemented:
    1. Random score                — sanity floor
    2. PCA reconstruction error   — top-performer on TSB-AD-M (VUS-PR ~0.31)
    3. IsolationForest score       — tree-based, fast
    4. OLS-RRR (linear AR)        — MANDATORY per arXiv 2602.00672 (ICLR 2026)
    5. T²-VAR (SPC)               — cross-variable linear baseline for partial-CD argument

Theoretical chain this creates in the paper:
    Random → PCA (marginal) → T²-VAR (linear cross-variable)
    → DS-PatchMamba (non-linear cross-variable)

This chain empirically answers: "Why do you need non-linear cross-variable modeling?"
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
import warnings

# statsmodels for VAR (T²-VAR baseline)
try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ─────────────────────────────────────────────
# Baseline 1: Random score
# ─────────────────────────────────────────────

def random_score(test: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Random anomaly score — sanity check floor.
    Expected VUS-PR ≈ anomaly_ratio (random guess).
    test: (T, V) — shape used only for output size
    """
    rng = np.random.default_rng(seed)
    return rng.random(test.shape[0]).astype(np.float32)


# ─────────────────────────────────────────────
# Baseline 2: PCA Reconstruction
# ─────────────────────────────────────────────

def pca_anomaly_score(
    train: np.ndarray,
    test:  np.ndarray,
    n_components: float = 0.9,   # explained variance ratio
) -> np.ndarray:
    """
    PCA reconstruction-based anomaly score.
    Anomaly score = per-timestep MSE between original and PCA-reconstructed test data.

    Consistently one of the top-3 methods on TSB-AD-M (VUS-PR ~0.31).
    Must be beaten for a credible DS-PatchMamba contribution.

    train: (T_train, V)
    test:  (T_test,  V)
    Returns: (T_test,) anomaly scores
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train)
    test_recon = pca.inverse_transform(pca.transform(test))
    return np.mean((test - test_recon) ** 2, axis=1).astype(np.float32)


# ─────────────────────────────────────────────
# Baseline 3: IsolationForest
# ─────────────────────────────────────────────

def iforest_anomaly_score(
    train: np.ndarray,
    test:  np.ndarray,
    n_estimators: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    IsolationForest anomaly score (negated decision function).
    train: (T_train, V), test: (T_test, V)
    Returns: (T_test,) anomaly scores (higher = more anomalous)
    """
    clf = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(train)
    raw = -clf.decision_function(test)   # negate: higher = more anomalous
    return raw.astype(np.float32)


# ─────────────────────────────────────────────
# Baseline 4: OLS-RRR (Mandatory Linear AR)
# ─────────────────────────────────────────────

def ols_rrr_anomaly_score(
    train:     np.ndarray,
    test:      np.ndarray,
    lag:       int   = 10,
    alpha:     float = 1e-3,
) -> np.ndarray:
    """
    Linear autoregressive anomaly detector using Ridge regression.

    MANDATORY baseline per arXiv 2602.00672 (ICLR 2026 submission): OLS-style
    linear AR consistently outperforms SOTA deep detectors on standard TSAD
    benchmarks. Including it prevents reviewer rejection on the question
    "why not include strong linear baselines?"

    Implementation note: the paper's OLS-RRR uses a single weight matrix
    W = AB^T (reduced rank factorisation). Here we use independent Ridge
    regression per output channel, which is equivalent to full-rank multivariate
    Ridge regression and captures the same cross-variable AR structure.
    This is clearly documented in the paper as "multivariate Ridge-AR" rather
    than strict RRR to remain technically honest.

    How it works:
        Fit (on training data only):
            X[t] = W * [X[t-1], ..., X[t-p]]^T + noise
        Anomaly score (on test data):
            s[t] = mean_v( (X[t,v] - X_hat[t,v])^2 )   — per-timestep MSE

    train: (T_train, V), test: (T_test, V)
    lag:   number of lagged timesteps p
    Returns: (T_test,) float32 anomaly scores
             (first `lag` entries are 0 — no prediction possible before lag steps)
    """
    T_train, V = train.shape

    def _build_lagged(series: np.ndarray, p: int):
        """Build (T-p, V*p) lagged feature matrix and (T-p, V) target matrix."""
        T = series.shape[0]
        if T <= p:
            return None, None
        # Column order: [X[t-1], X[t-2], ..., X[t-p]] for each target t
        X_lag = np.hstack([series[p - i - 1: T - i - 1] for i in range(p)])
        y     = series[p:]
        return X_lag, y

    X_train, y_train = _build_lagged(train, lag)
    if X_train is None:
        return np.zeros(test.shape[0], dtype=np.float32)

    # Fit one Ridge model per output variable on training data
    models = []
    for v in range(V):
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X_train, y_train[:, v])
        models.append(m)

    X_test, y_test = _build_lagged(test, lag)
    if X_test is None:
        return np.zeros(test.shape[0], dtype=np.float32)

    # Vectorised prediction across all channels at once
    W   = np.vstack([m.coef_ for m in models])                   # (V, V*lag)
    b   = np.array([m.intercept_ for m in models])               # (V,)
    y_pred     = X_test @ W.T + b                                 # (T_test-lag, V)
    residuals  = np.mean((y_test - y_pred) ** 2, axis=1)         # (T_test-lag,)

    # Pad first `lag` positions with zeros (no prediction possible)
    return np.concatenate([
        np.zeros(lag, dtype=np.float32),
        residuals.astype(np.float32),
    ])


# ─────────────────────────────────────────────
# Baseline 5: T²-VAR (Hotelling T² on VAR residuals)
# ─────────────────────────────────────────────

def t2_var_anomaly_score(
    train: np.ndarray,
    test:  np.ndarray,
    lag:   int = 5,
) -> np.ndarray:
    """
    Hotelling T² statistic on Vector Autoregressive (VAR) model residuals.

    Origin: Statistical Process Control (quality engineering / manufacturing).
    VAR-residual variant from arXiv 2501.11649 (2025).

    Directly applicable to SMD, SWaT, PSM (all industrial sensor monitoring).
    Captures cross-variable covariance through the Mahalanobis distance:
        T²_t = ε_t^T  Σ^{-1}  ε_t
    where ε_t = X_t − X̂_t^{VAR} and Σ is the training residual covariance.

    Theoretical chain for the paper:
        PCA         → no cross-variable covariance
        T²-VAR      → linear cross-variable covariance (this method)
        DS-PatchMamba → non-linear cross-variable (our model)
    If T²-VAR > PCA: cross-variable covariance carries anomaly signal.
    If DS-PatchMamba > T²-VAR: non-linear modeling adds further value.

    Implementation is fully vectorised — no Python loop over timesteps.
    For PSM (217K timesteps) this runs in seconds rather than hours.

    train: (T_train, V), test: (T_test, V)
    Returns: (T_test,) float32 Hotelling T² scores
             (first `lag` entries are 0 — no prediction before lag steps)
    """
    if not STATSMODELS_AVAILABLE:
        warnings.warn(
            "statsmodels not available. T²-VAR returning zeros. "
            "Install with: pip install statsmodels"
        )
        return np.zeros(test.shape[0], dtype=np.float32)

    T_train, V = train.shape

    # Cap lag to prevent over-parameterisation on small training sets.
    # Rule of thumb: need at least 10*V*lag observations to estimate VAR reliably.
    lag = min(lag, max(1, T_train // (10 * V)))

    # ── Fit VAR model on training data ────────────────────────────────────
    try:
        var_model = VAR(train)
        var_result = var_model.fit(lag, trend='c')
        lag_used  = var_result.k_ar
    except Exception as e:
        warnings.warn(f"VAR(lag={lag}) fitting failed: {e}. Trying lag=1.")
        try:
            var_result = VAR(train).fit(1, trend='c')
            lag_used   = 1
        except Exception as e2:
            warnings.warn(f"VAR(lag=1) also failed: {e2}. Returning zeros.")
            return np.zeros(test.shape[0], dtype=np.float32)

    # ── Covariance of training residuals (for Mahalanobis distance) ───────
    # result.resid shape: (T_train - lag_used, V)
    train_resid = var_result.resid
    # Small ridge regularisation prevents singular covariance on low-data series
    sigma     = np.cov(train_resid.T) + np.eye(V) * 1e-6    # (V, V)
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)

    # ── Build full test prediction in one vectorised pass ─────────────────
    # We need lag_used timesteps of context before each test prediction.
    # For t < lag_used we borrow from the end of the training series.
    # Concatenating [train tail | test] lets us index uniformly.
    context  = train[-lag_used:]                        # (lag_used, V)
    extended = np.vstack([context, test])               # (lag_used + T_test, V)

    # Build lagged feature matrix for the test portion.
    # For target index i in [0, T_test), the features are
    #   extended[i], extended[i+1], ..., extended[i+lag_used-1]
    # i.e. the lag_used timesteps immediately before position lag_used+i.
    T_ext = extended.shape[0]
    # X_lag shape: (T_test, V * lag_used)
    X_lag = np.hstack([
        extended[lag_used - k - 1: T_ext - k - 1]   # lag k+1 column block
        for k in range(lag_used)
    ])   # (T_test, V * lag_used)

    # Extract VAR coefficient matrix from statsmodels result.
    # var_result.params shape: (1 + lag_used * V, V)  — first row is intercept.
    intercept   = var_result.params[0]             # (V,)
    coef_matrix = var_result.params[1:].T          # (V, lag_used * V)

    # Vectorised prediction for all test timesteps at once
    y_pred    = X_lag @ coef_matrix.T + intercept  # (T_test, V)
    residuals = test - y_pred                       # (T_test, V)

    # ── Vectorised Hotelling T²: diag(R @ Σ⁻¹ @ R^T) ─────────────────────
    # Equivalent to: T²[t] = residuals[t] @ sigma_inv @ residuals[t]
    # Efficiently computed as: sum over v of (residuals @ sigma_inv) * residuals
    t2_scores = np.sum((residuals @ sigma_inv) * residuals, axis=1)  # (T_test,)

    # First lag_used timesteps used training context — set to 0
    # (no purely-test prediction available there)
    t2_scores[:lag_used] = 0.0

    return t2_scores.astype(np.float32)


# ─────────────────────────────────────────────
# Convenience: run all CPU baselines
# ─────────────────────────────────────────────

def run_all_cpu_baselines(
    train: np.ndarray,
    test:  np.ndarray,
    seed:  int = 42,
) -> dict:
    """
    Run all CPU baselines and return a dict of {method_name: score_array}.
    Typical runtime: < 5 minutes total on any dataset.
    """
    print("  Running CPU baselines...")
    results = {}

    results["Random"]  = random_score(test, seed)
    print("    [1/5] Random — done")

    results["PCA"]     = pca_anomaly_score(train, test)
    print("    [2/5] PCA — done")

    results["IForest"] = iforest_anomaly_score(train, test, seed=seed)
    print("    [3/5] IForest — done")

    results["OLS-RRR"] = ols_rrr_anomaly_score(train, test)
    print("    [4/5] OLS-RRR — done")

    results["T2-VAR"]  = t2_var_anomaly_score(train, test)
    print("    [5/5] T²-VAR — done")

    return results
