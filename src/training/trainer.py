"""
Training loop for DS-PatchMamba with full session-death protection.

Implements all 7 mandatory code rules from WORKFLOW.md Section 10:
    Rule 1: Checkpoint after every epoch
    Rule 2: Save anomaly scores as .npy immediately after inference
    Rule 3: Append results to CSV row-by-row
    Rule 4: Skip already-completed runs
    Rule 5: Each cell ≤ 10h of compute
    Rule 6: Persist outputs via Kaggle dataset between sessions (handled externally)
    Rule 7: Wall-clock guard in training loop
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any

from ..models.ds_patchmamba import DSPatchMamba, ModelConfig, AblationConfig
from ..data.preprocessing import sliding_windows, channel_mask_augment, train_val_split_normal, get_short_series_stride
from .losses import DSPatchMambaLoss
from .dspot import DSPOT, compute_anomaly_scores, fuse_scores

# Single source of truth for result helpers lives in harness.py.
# Import from there to avoid two definitions that can diverge.
from ..evaluation.harness import (
    RESULT_COLUMNS,
    load_completed_runs,
    append_result,
    save_scores,
)


def save_checkpoint(ckpt_dir: str, model: nn.Module, optimizer, epoch: int,
                    val_loss: float, best_val_loss: float,
                    dataset: str, series_name: str, seed: int):
    """Save training checkpoint (Rule 1)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    key = f"{dataset}__{series_name}__seed{seed}"
    path = os.path.join(ckpt_dir, f"{key}__latest.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(ckpt_dir: str, model: nn.Module, optimizer,
                    dataset: str, series_name: str, seed: int) -> tuple:
    """Load checkpoint if it exists; return (start_epoch, best_val_loss)."""
    key = f"{dataset}__{series_name}__seed{seed}"
    path = os.path.join(ckpt_dir, f"{key}__latest.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"] + 1, ckpt["best_val_loss"]
    return 0, float("inf")


# ─────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Trains DS-PatchMamba on a single SeriesRecord and produces anomaly scores.

    Session-death-proof: checkpoints every epoch, saves scores immediately after
    inference, appends results row-by-row. Resumes from checkpoint if one exists.
    Wall-clock guard prevents exceeding 10h per cell.

    Args:
        cfg:            ModelConfig
        abl:            AblationConfig (for ablation variants)
        results_csv:    path to results CSV (row-by-row append)
        scores_dir:     directory to save .npy score arrays
        ckpt_dir:       directory to save .pt checkpoints
        max_epochs:     training epoch limit
        patience:       early stopping patience on val MSE
        batch_size:     training batch size
        lr:             learning rate
        alpha, beta:    loss weights
        max_train_seconds: wall-clock guard (Rule 7)
        w:              fusion weight (hyperparameter, not learned)
        device:         torch device
    """

    def __init__(
        self,
        cfg:                ModelConfig   = None,
        abl:                AblationConfig = None,
        results_csv:        str           = "results/main_results.csv",
        scores_dir:         str           = "results/scores",
        ckpt_dir:           str           = "checkpoints",
        max_epochs:         int           = 50,
        patience:           int           = 7,
        batch_size:         int           = 64,
        lr:                 float         = 1e-3,
        weight_decay:       float         = 1e-4,
        alpha:              float         = 0.5,
        beta:               float         = 0.01,
        max_train_seconds:  int           = 36000,   # 10 hours (Rule 7)
        w:                  float         = 0.5,
        window_length:      int           = 300,
        stride:             int           = 150,
        device:             str           = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.cfg               = cfg or ModelConfig()
        self.abl               = abl or AblationConfig()
        self.results_csv       = results_csv
        self.scores_dir        = scores_dir
        self.ckpt_dir          = ckpt_dir
        self.max_epochs        = max_epochs
        self.patience          = patience
        self.batch_size        = batch_size
        self.lr                = lr
        self.weight_decay      = weight_decay
        self.alpha             = alpha
        self.beta              = beta
        self.max_train_seconds = max_train_seconds
        self.w                 = w
        self.window_length     = window_length
        self.stride            = stride
        self.device            = torch.device(device)

    def run(
        self,
        series,                     # SeriesRecord
        method_name: str = "DS-PatchMamba",
        dataset_name: str = "",
        seed: int = 42,
        ablation_id: str = "",
        completed_runs: set = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Train on series.train, score series.test, save everything.
        Returns result dict or None if skipped (already completed).
        """
        # Rule 4: Skip if already done
        if completed_runs is None:
            completed_runs = load_completed_runs(self.results_csv)

        run_key = (method_name, dataset_name, series.name, str(seed), ablation_id)
        if run_key in completed_runs:
            print(f"  [SKIP] {run_key} already in results CSV")
            return None

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Data preparation ───────────────────────────────────────────
        stride = get_short_series_stride(series.avg_length, self.stride)

        train_full, val_data = train_val_split_normal(series.train, val_fraction=0.2)

        # Sliding windows
        train_wins = sliding_windows(train_full, self.window_length, stride)   # (N, L, V)
        val_wins   = sliding_windows(val_data,   self.window_length, stride)

        train_tensor = torch.from_numpy(train_wins).float()
        val_tensor   = torch.from_numpy(val_wins).float()

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=self.batch_size, shuffle=False,
        )

        # ── Model, optimizer, loss ─────────────────────────────────────
        model = DSPatchMamba(V=series.V, cfg=self.cfg, abl=self.abl).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs,
        )
        criterion = DSPatchMambaLoss(alpha=self.alpha, beta=self.beta)

        # Linear warmup for first 5 epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
        )

        # ── Resume from checkpoint ─────────────────────────────────────
        start_epoch, best_val_loss = load_checkpoint(
            self.ckpt_dir, model, optimizer, dataset_name, series.name, seed
        )
        patience_counter = 0
        best_model_state = None

        print(f"\n  Training: {method_name} | {dataset_name}/{series.name} | seed={seed}")
        print(f"  Model: {model.n_parameters:,} params | V={series.V} | "
              f"train_wins={len(train_wins)} | val_wins={len(val_wins)}")

        # ── Training loop ──────────────────────────────────────────────
        train_start = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            # Rule 7: Wall-clock guard
            if time.time() - train_start > self.max_train_seconds:
                print(f"  [WALL-CLOCK] Reached {self.max_train_seconds/3600:.1f}h limit at epoch {epoch}. Saving checkpoint.")
                save_checkpoint(
                    self.ckpt_dir, model, optimizer, epoch - 1,
                    best_val_loss, best_val_loss, dataset_name, series.name, seed
                )
                return None

            # Training epoch
            model.train()
            rng = np.random.default_rng(seed + epoch)
            train_loss = 0.0

            for (batch,) in train_loader:
                batch_np = batch.numpy()
                batch_np = channel_mask_augment(batch_np, mask_frac=0.15, rng=rng)
                batch = torch.from_numpy(batch_np).float().to(self.device)

                optimizer.zero_grad()
                outputs = model(batch)
                x_fine   = outputs.get("x_hat_fine",   batch)
                x_coarse = outputs.get("x_hat_coarse", batch)
                loss, _ = criterion(batch, x_fine, x_coarse)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.step_gumbel_temperature()
                train_loss += loss.item()

            # Validation on held-out normal data (val MSE — label-free)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    x_fine   = outputs.get("x_hat_fine",   batch)
                    x_coarse = outputs.get("x_hat_coarse", batch)
                    loss, _ = criterion(batch, x_fine, x_coarse)
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)

            # LR schedule
            if epoch < 5:
                warmup_scheduler.step()
            else:
                scheduler.step()

            # Rule 1: Checkpoint after every epoch
            save_checkpoint(
                self.ckpt_dir, model, optimizer, epoch,
                val_loss, best_val_loss, dataset_name, series.name, seed
            )

            # Early stopping on val MSE
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                elapsed = (time.time() - train_start) / 60
                print(f"  Epoch {epoch+1:3d}/{self.max_epochs} | "
                      f"train_loss={train_loss/len(train_loader):.4f} | "
                      f"val_loss={val_loss:.4f} | patience={patience_counter} | "
                      f"elapsed={elapsed:.1f}m")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        train_time = time.time() - train_start

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # ── Inference on test set ──────────────────────────────────────
        test_scores = self._score_series(model, series)

        # Rule 2: Save scores immediately
        save_scores(
            self.scores_dir, method_name, dataset_name,
            series.name, seed, test_scores, series.labels
        )

        # ── DSPOT threshold ────────────────────────────────────────────
        # Fit on training scores (inference on training windows)
        train_scores_full = self._score_series_array(model, train_wins)
        dspot = DSPOT(q=1e-4, level=0.98, n_init=1000)
        dspot.fit(train_scores_full)

        # ── Rule 3: Append results row-by-row ─────────────────────────
        result = {
            "method": method_name,
            "dataset": dataset_name,
            "series_name": series.name,
            "seed": str(seed),
            "ablation": ablation_id,
            "n_params": model.n_parameters,
            "train_time_s": round(train_time, 1),
            # Metrics are computed by the evaluation harness (harness.py)
            # and merged back into the CSV — placeholder here
            "vus_pr": None, "dqe": None, "vus_roc": None,
            "auc_pr": None, "auc_roc": None, "standard_f1": None,
            "event_f1": None, "r_based_f1": None,
            "affiliation_f1": None, "pa_f1": None,
            "flops_per_window": None, "peak_vram_gb": None,
        }

        append_result(self.results_csv, result)
        print(f"  Done. Scores saved. Train time: {train_time/60:.1f}m")
        return result

    def _score_series(self, model: DSPatchMamba, series) -> np.ndarray:
        """
        Score the full test series using overlap-add inference.
        Returns 1D anomaly score array aligned to test series length.
        """
        stride = get_short_series_stride(series.avg_length, self.stride)
        test_wins = sliding_windows(series.test, self.window_length, stride)
        raw_scores = self._score_series_array(model, test_wins)

        # Project windowed scores back to full series via overlap-add
        return self._overlap_add(raw_scores, series.test.shape[0], stride)

    def _score_series_array(self, model: DSPatchMamba, windows: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for an array of windows (N, L, V).
        Returns (N, L) score array.
        """
        model.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch = torch.from_numpy(
                    windows[i:i + self.batch_size]
                ).float().to(self.device)
                outputs = model(batch)
                scores  = model.compute_anomaly_scores(batch, outputs, w=self.w)
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)  # (N, L)

    def _overlap_add(
        self,
        windowed_scores: np.ndarray,  # (N, L)
        T: int,
        stride: int,
    ) -> np.ndarray:
        """
        Reconstruct per-timestep scores from overlapping windows via averaging.
        """
        N, L = windowed_scores.shape
        score_sum   = np.zeros(T, dtype=np.float64)
        score_count = np.zeros(T, dtype=np.float64)

        for i, start in enumerate(range(0, T - L + 1, stride)):
            if i >= N:
                break
            end = min(start + L, T)
            score_sum[start:end]   += windowed_scores[i, :end - start]
            score_count[start:end] += 1.0

        score_count = np.maximum(score_count, 1.0)
        raw = (score_sum / score_count).astype(np.float32)
        return compute_anomaly_scores(raw, sigma=1.0)
