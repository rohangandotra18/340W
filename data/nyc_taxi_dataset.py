"""
NYC Taxi dataset loader for integrated traffic prediction and routing.

Unlike METR-LA / PEMS-BAY (which lack OD demand information), NYC Taxi
trip records provide:
    - Zone-to-zone OD demand matrices
    - Trip durations (ground-truth travel times)
    - Temporal patterns at 5-minute or 15-minute resolution

This loader converts raw TLC trip records into the spatiotemporal tensor
format expected by PDFormer++ and generates the adjacency graph from
observed trip connections between zones.

Data source:
    NYC TLC Trip Record Data — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NYCTaxiDataset(Dataset):
    """
    Spatiotemporal traffic dataset from NYC Taxi trip records.

    Expects a preprocessed .npz file with:
        speed_data:   (T, N, C)   float32 — zone-level aggregated traffic features
                                            C typically = 3 (speed, flow, demand)
        adj_mx:       (N, N)      float32 — zone adjacency matrix
        od_demand:    (T, N, N)   float32 — OD demand matrices (optional)
        zone_ids:     (N,)        int     — taxi zone IDs (optional)

    Preprocessing script (not included) should:
        1. Download TLC yellow/green taxi parquet files
        2. Aggregate trips by 5-min intervals and pickup/dropoff zone
        3. Compute zone-level speed = distance / duration
        4. Build adjacency from zone connectivity (shared borders or trip frequency)

    Args:
        data_path:    path to preprocessed .npz file
        in_horizon:   input time steps
        out_horizon:  prediction horizon
        split:        'train' | 'val' | 'test'
        train_ratio:  fraction for training
        val_ratio:    fraction for validation
        normalize:    z-score normalisation
        use_od:       load OD demand matrices as additional feature
    """

    def __init__(
        self,
        data_path: str,
        in_horizon: int = 12,
        out_horizon: int = 12,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        normalize: bool = True,
        use_od: bool = False,
    ):
        super().__init__()
        self.in_horizon = in_horizon
        self.out_horizon = out_horizon
        self.use_od = use_od

        # ── load data ─────────────────────────────────────────────────
        data_npz = np.load(data_path, allow_pickle=True)

        # Speed/flow tensor
        if "speed_data" in data_npz:
            raw = data_npz["speed_data"].astype(np.float32)
        elif "data" in data_npz:
            raw = data_npz["data"].astype(np.float32)
        else:
            raise KeyError("Expected 'speed_data' or 'data' key in .npz file")

        T, N, C = raw.shape
        self.n_nodes = N
        self.n_zones = N

        # ── adjacency ─────────────────────────────────────────────────
        if "adj_mx" in data_npz:
            adj = data_npz["adj_mx"].astype(np.float32)
        else:
            # Build adjacency from OD demand if available
            if "od_demand" in data_npz:
                od = data_npz["od_demand"].astype(np.float32)
                adj = (od.sum(axis=0) > 0).astype(np.float32)
                np.fill_diagonal(adj, 1.0)
            else:
                adj = np.eye(N, dtype=np.float32)

        row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
        self.adj = torch.from_numpy(adj / row_sum)

        # ── OD demand matrices ────────────────────────────────────────
        self.od_demand = None
        if use_od and "od_demand" in data_npz:
            self.od_demand = data_npz["od_demand"].astype(np.float32)

        # ── zone metadata ─────────────────────────────────────────────
        self.zone_ids = None
        if "zone_ids" in data_npz:
            self.zone_ids = data_npz["zone_ids"]

        # ── time split ────────────────────────────────────────────────
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        slices = {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, None),
        }
        assert split in slices, f"split must be 'train', 'val', or 'test'; got '{split}'"

        # ── normalisation ─────────────────────────────────────────────
        if normalize:
            train_data = raw[:train_end]
            self.mean = train_data.mean(axis=(0, 1), keepdims=True)
            self.std = train_data.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
            raw = (raw - self.mean) / self.std
        else:
            self.mean = np.zeros((1, 1, C), dtype=np.float32)
            self.std = np.ones((1, 1, C), dtype=np.float32)

        self._mean_t = torch.from_numpy(self.mean)
        self._std_t = torch.from_numpy(self.std)

        # ── sliding windows ───────────────────────────────────────────
        data_split = raw[slices[split]]
        window = in_horizon + out_horizon
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        self._OD: list[np.ndarray] = []

        od_split = None
        if self.od_demand is not None:
            od_split = self.od_demand[slices[split]]

        for i in range(len(data_split) - window + 1):
            self._X.append(data_split[i : i + in_horizon])
            self._Y.append(data_split[i + in_horizon : i + window, :, 0])  # speed
            if od_split is not None:
                # Mean OD demand over prediction horizon
                self._OD.append(od_split[i + in_horizon : i + window].mean(axis=0))

        self._X_arr = np.stack(self._X, axis=0)
        self._Y_arr = np.stack(self._Y, axis=0)
        self._OD_arr = np.stack(self._OD, axis=0) if self._OD else None

    def __len__(self) -> int:
        return len(self._X_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = torch.from_numpy(self._X_arr[idx])
        y = torch.from_numpy(self._Y_arr[idx])
        if self._OD_arr is not None:
            od = torch.from_numpy(self._OD_arr[idx])
            return x, y, od
        return x, y

    def denormalize_speed(self, y: torch.Tensor) -> torch.Tensor:
        """Convert normalised speed predictions back to original units."""
        mean = self._mean_t[..., 0].to(y.device)
        std = self._std_t[..., 0].to(y.device)
        return y * std + mean

    def get_od_pairs(self, threshold: float = 0.0) -> list[Tuple[int, int]]:
        """
        Extract frequently-served OD pairs from the demand data.

        Args:
            threshold: minimum mean demand to include a pair

        Returns:
            List of (origin_zone_idx, destination_zone_idx) tuples
        """
        if self.od_demand is None:
            return []
        mean_od = self.od_demand.mean(axis=0)  # (N, N)
        pairs = []
        for i in range(self.n_zones):
            for j in range(self.n_zones):
                if i != j and mean_od[i, j] > threshold:
                    pairs.append((i, j))
        return pairs
