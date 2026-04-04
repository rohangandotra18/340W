"""
Traffic dataset loader for METR-LA, PEMS-BAY, PEMS04, PEMS08.

Expects a preprocessed .npz file with:
    data:   (T, N, C)  float32   — T timesteps, N nodes, C features
    adj_mx: (N, N)     float32   — adjacency matrix (optional, can supply separately)

Standard LibCity preprocessed files match this layout exactly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    Sliding-window dataset that generates (X, Y) pairs.

    X: (in_horizon, N, C)  — historical window
    Y: (out_horizon, N)    — future speed/flow (feature index 0)

    Train / val / test split is done by time (70 / 10 / 20 % default).
    Statistics are computed on the training set only to avoid leakage.

    Args:
        data_path:    path to .npz traffic data file
        adj_path:     optional path to a separate adjacency .npz
        in_horizon:   input time steps (12 = 1 hour at 5-min resolution)
        out_horizon:  output time steps to predict
        split:        'train' | 'val' | 'test'
        train_ratio:  fraction of time steps for training
        val_ratio:    fraction for validation
        normalize:    z-score normalise the data (recommended)
    """

    def __init__(
        self,
        data_path: str,
        adj_path: Optional[str] = None,
        in_horizon: int = 12,
        out_horizon: int = 12,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_horizon = in_horizon
        self.out_horizon = out_horizon

        # ── load raw data ────────────────────────────────────────────────
        data_npz = np.load(data_path, allow_pickle=True)
        raw: np.ndarray = data_npz["data"].astype(np.float32)     # (T, N, C)
        T, N, C = raw.shape
        self.n_nodes = N

        # ── adjacency matrix ─────────────────────────────────────────────
        if adj_path is not None:
            adj = np.load(adj_path, allow_pickle=True)["adj_mx"].astype(np.float32)
        elif "adj_mx" in data_npz:
            adj = data_npz["adj_mx"].astype(np.float32)
        else:
            # Fall back to identity (no topology information)
            adj = np.eye(N, dtype=np.float32)

        # Row-normalise
        row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
        self.adj: torch.Tensor = torch.from_numpy(adj / row_sum)

        # ── time split ───────────────────────────────────────────────────
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        slices = {"train": slice(0, train_end),
                  "val": slice(train_end, val_end),
                  "test": slice(val_end, None)}
        assert split in slices, f"split must be 'train', 'val', or 'test'; got '{split}'"

        # ── normalisation (fit on train, apply everywhere) ───────────────
        if normalize:
            train_data = raw[:train_end]
            self.mean: np.ndarray = train_data.mean(axis=(0, 1), keepdims=True)   # (1,1,C)
            self.std: np.ndarray = train_data.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
            raw = (raw - self.mean) / self.std
        else:
            self.mean = np.zeros((1, 1, C), dtype=np.float32)
            self.std = np.ones((1, 1, C), dtype=np.float32)

        self._mean_t = torch.from_numpy(self.mean)
        self._std_t = torch.from_numpy(self.std)

        # ── build sliding windows ─────────────────────────────────────────
        data_split = raw[slices[split]]
        window = in_horizon + out_horizon
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        for i in range(len(data_split) - window + 1):
            self._X.append(data_split[i: i + in_horizon])                         # (in_T, N, C)
            self._Y.append(data_split[i + in_horizon: i + window, :, 0])         # (out_T, N) speed

        # Pre-stack for faster __getitem__
        self._X_arr = np.stack(self._X, axis=0)   # (S, in_T, N, C)
        self._Y_arr = np.stack(self._Y, axis=0)   # (S, out_T, N)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._X_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._X_arr[idx])   # (in_T, N, C)
        y = torch.from_numpy(self._Y_arr[idx])   # (out_T, N)
        return x, y

    # ------------------------------------------------------------------
    def denormalize_speed(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert normalised speed predictions back to original units.
        y: (..., N) or (B, T', N)
        """
        mean = self._mean_t[..., 0].to(y.device)   # (1,1) or (1,)
        std = self._std_t[..., 0].to(y.device)
        return y * std + mean
