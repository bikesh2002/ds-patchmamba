"""
Preprocessing: per-channel z-score normalisation, sliding windows, channel masking.
All statistics computed on training split ONLY to prevent leakage.
"""

import numpy as np
from typing import Tuple, Optional


def zscore_normalize(
    train: np.ndarray,
    test: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel z-score normalisation using TRAINING statistics only.

    The mean and std are computed from the training split and then applied
    identically to both train and test. Using test statistics here would
    constitute data leakage — the model would implicitly know the test
    distribution before seeing the test data.

    train / test shape: (T, V)
    Returns: normalised (train, test) as float32 arrays.
    """
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(
            f"zscore_normalize expects 2D (T, V) arrays. "
            f"Got train={train.shape}, test={test.shape}."
        )
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Channel count mismatch: train has {train.shape[1]} channels, "
            f"test has {test.shape[1]}."
        )

    mean = train.mean(axis=0, keepdims=True)            # (1, V) — train only
    std  = train.std(axis=0,  keepdims=True) + eps      # (1, V) — train only
    return (
        ((train - mean) / std).astype(np.float32),
        ((test  - mean) / std).astype(np.float32),
    )


def sliding_windows(
    series: np.ndarray,
    window_length: int,
    stride: int,
) -> np.ndarray:
    """
    Extract sliding windows from a time series using vectorised index arithmetic.

    Avoids materialising copies in a Python loop (which is slow for 200K+ timesteps).
    Instead builds an index matrix of shape (N, L) and uses NumPy fancy indexing,
    which is O(1) index construction and a single contiguous copy.

    series shape : (T, V)
    Returns      : (N, L, V) where N = number of windows
    """
    if series.ndim != 2:
        raise ValueError(f"sliding_windows expects a 2D (T, V) array, got shape {series.shape}.")
    T, V = series.shape
    if window_length > T:
        raise ValueError(
            f"window_length={window_length} is larger than series length T={T}."
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    # Build start indices, then a 2D index array (N, L)
    starts  = np.arange(0, T - window_length + 1, stride)   # (N,)
    offsets = np.arange(window_length)                        # (L,)
    indices = starts[:, None] + offsets[None, :]              # (N, L)

    return series[indices]   # (N, L, V) — single vectorised gather


def train_val_split_normal(
    train_series: np.ndarray,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split normal training data into train and validation portions.

    Split is strictly temporal (no shuffling) to preserve time-series structure.
    Shuffling would allow the model to see future normal patterns during training,
    which would be unrealistic at deployment.

    train_series shape : (T, V)
    Returns            : (train_part, val_part)
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    split = int(len(train_series) * (1.0 - val_fraction))
    if split < 2:
        raise ValueError(
            f"Training split has only {split} timestep(s) after reserving "
            f"{val_fraction*100:.0f}% for validation. Use a longer series or "
            f"a smaller val_fraction."
        )
    return train_series[:split], train_series[split:]


def channel_mask_augment(
    windows: np.ndarray,
    mask_frac: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Randomly zero out k channels per window (data augmentation for training).

    Purpose: regularises the cross-channel attention module against overfitting
    to spurious correlations that only appear in the training data.

    k = max(1, floor(V * mask_frac))
    The max(1, ...) floor ensures at least one channel is always masked even
    for low-V datasets (e.g. V=2 medical data: floor(2 * 0.15) = 0 without it).
    k is capped at V-1 so at least one channel always remains visible.

    Each window in the batch gets an independently drawn random mask.
    The vectorised implementation avoids a Python loop over batch items.

    windows shape : (B, L, V)
    Returns       : masked copy of windows, same shape
    """
    if windows.ndim != 3:
        raise ValueError(f"channel_mask_augment expects (B, L, V) array, got {windows.shape}.")

    if rng is None:
        rng = np.random.default_rng()

    B, L, V = windows.shape
    k = max(1, int(np.floor(V * mask_frac)))
    k = min(k, V - 1)

    # Build random channel permutations for all batch items at once — vectorised.
    # rng.permuted shuffles each row of a (B, V) integer matrix independently.
    channel_order = rng.permuted(
        np.tile(np.arange(V, dtype=np.int32), (B, 1)),
        axis=1,
    )   # (B, V) — each row is a random permutation of [0, ..., V-1]

    # The first k columns are the channels to mask per batch item
    mask_channels = channel_order[:, :k]   # (B, k)

    # Convert to a boolean mask of shape (B, V): True = keep, False = zero out
    channel_mask = np.ones((B, V), dtype=np.float32)
    batch_idx    = np.repeat(np.arange(B), k)      # [0,0,...,1,1,...,B-1,...]
    chan_idx      = mask_channels.ravel()            # flattened channel indices
    channel_mask[batch_idx, chan_idx] = 0.0

    # Broadcast (B, V) → (B, L, V) and apply
    return windows * channel_mask[:, np.newaxis, :]


def get_short_series_stride(avg_series_length: float, default_stride: int) -> int:
    """
    Return a reduced stride for short series to ensure enough training windows
    for DSPOT's GPD tail fitting.

    GPD fitting requires at least 50 peak exceedances for reliable estimation.
    MSL in TSB-AD-M has average series length 3,119 timesteps. With default
    stride=150, the 80% training portion yields ~17 windows — far too few.
    Using stride=50 increases this to ~50 windows, which is the minimum viable.

    Series with avg_length < 5000 get stride=50; all others use default_stride.
    """
    if avg_series_length < 5000:
        return 50
    return default_stride
