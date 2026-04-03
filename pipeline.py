"""
Full predict-then-route pipeline.

Steps:
  1. PDFormer++ predicts future speeds/flows for all N nodes
  2. Speed-to-travel-time conversion (speed_to_travel_time or BPR)
  3. Time-dependent Dijkstra finds the minimum-cost path for an OD pair

Usage:
    pipeline = TrafficTransformerPipeline(
        checkpoint_path="checkpoints/best_model.pt",
        edges=edge_list,
        link_lengths_km=lengths,
    )
    result = pipeline.route(x_tensor, adj_tensor, origin=0, destination=100)
    print(result["path"], result["travel_time_min"])
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Optional, Tuple

import numpy as np
import torch

from model.pdformer_plus import PDFormerPlusPlus
from routing.bpr_converter import speed_to_travel_time
from routing.dijkstra_router import TimeDependentRouter


class TrafficTransformerPipeline:
    """
    Args:
        checkpoint_path:    path to .pt checkpoint saved by train.py
        edges:              list of (u, v) directed edge tuples
        link_lengths_km:    (n_edges,) array of link lengths in km
        link_capacities:    (n_edges,) array of link capacities (veh/hr) — optional,
                            used only if you switch to BPR conversion
        ffs:                free-flow speed in km/h (for congestion label decoding)
        device:             'cuda' | 'cpu' | 'auto'
    """

    CONGESTION_LABELS = ["free-flow", "slow", "congested"]

    def __init__(
        self,
        checkpoint_path: str,
        edges: List[Tuple[int, int]],
        link_lengths_km: np.ndarray,
        link_capacities: Optional[np.ndarray] = None,
        ffs: float = 65.0,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.edges = edges
        self.link_lengths = torch.from_numpy(np.asarray(link_lengths_km, dtype=np.float32))
        self.link_capacities = (
            torch.from_numpy(np.asarray(link_capacities, dtype=np.float32))
            if link_capacities is not None
            else None
        )
        self.ffs = ffs

        # ── load model ────────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        a = ckpt["args"]
        self.model = PDFormerPlusPlus(
            n_nodes=a["n_nodes"],
            in_channels=a["in_channels"],
            d_model=a["d_model"],
            out_horizon=a["out_horizon"],
            n_temporal_layers=a["n_temporal_layers"],
            n_spatial_layers=a["n_spatial_layers"],
            n_heads=a["n_heads"],
            d_state=a["d_state"],
            n_classes=a["n_classes"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.out_horizon: int = a["out_horizon"]

        # ── router ────────────────────────────────────────────────────────
        self.router = TimeDependentRouter(edges)
        print(f"Pipeline ready on {self.device}. Val MAE: {ckpt['val_mae']:.4f}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> dict:
        """
        Run the prediction model.

        Args:
            x:   (T, N, C) or (1, T, N, C)
            adj: (N, N) adjacency matrix

        Returns:
            speed_pred:  (N, T') numpy array of predicted speeds
            congestion:  (N,)   numpy array of class indices
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x   = x.to(self.device)
        adj = adj.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            out = self.model(x, adj)

        return {
            "speed_pred": out["speed_pred"].squeeze(0).cpu().numpy(),         # (N, T')
            "congestion": out["congestion"].squeeze(0).argmax(-1).cpu().numpy(),  # (N,)
        }

    # ------------------------------------------------------------------
    def _speeds_to_edge_times(self, speeds: np.ndarray) -> np.ndarray:
        """
        Convert predicted node speeds → per-edge travel times (minutes).

        Averages the speeds of the two endpoint nodes for each edge,
        then applies: t = (length_km / speed_km_h) × 60.

        Args:
            speeds: (N, T') numpy array

        Returns:
            (T', n_edges) numpy float32 array of travel times in minutes
        """
        T_prime = speeds.shape[1]
        n_edges = len(self.edges)
        travel_times = np.zeros((T_prime, n_edges), dtype=np.float32)

        speeds_t = torch.from_numpy(speeds)  # (N, T')

        for i, (u, v) in enumerate(self.edges):
            edge_speed = (speeds_t[u] + speeds_t[v]) / 2.0   # (T',)
            edge_speed = edge_speed.clamp(min=1.0)
            tt = speed_to_travel_time(edge_speed, self.link_lengths[i])  # (T',)
            travel_times[:, i] = tt.numpy()

        return travel_times

    # ------------------------------------------------------------------
    def route(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        origin: int,
        destination: int,
        time_dependent: bool = True,
    ) -> dict:
        """
        Full pipeline: predict → convert → route.

        Args:
            x:               (T, N, C) or (1, T, N, C) historical traffic
            adj:             (N, N) adjacency
            origin:          source node index
            destination:     target node index
            time_dependent:  use TD-Dijkstra (True) or single-snapshot (False)

        Returns dict:
            path:              list of node indices
            travel_time_min:   estimated total travel time in minutes
            congestion_labels: congestion class string per node on path
            speed_pred:        (N, T') raw speed predictions
        """
        pred = self.predict(x, adj)
        travel_times = self._speeds_to_edge_times(pred["speed_pred"])   # (T', n_edges)

        if time_dependent:
            path, cost = self.router.time_dependent_route(origin, destination, travel_times)
        else:
            mean_times = travel_times.mean(axis=0)
            self.router.build_graph(mean_times)
            path, cost = self.router.route(origin, destination)

        path_congestion = [
            self.CONGESTION_LABELS[int(pred["congestion"][n])]
            for n in path
        ]

        return {
            "path":              path,
            "travel_time_min":   cost,
            "congestion_labels": path_congestion,
            "speed_pred":        pred["speed_pred"],
        }
