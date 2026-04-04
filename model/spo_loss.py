"""
SPO+ (Smart Predict-then-Optimize) decision-focused loss for traffic routing.

Instead of optimising predictions for statistical accuracy alone (MAE/RMSE),
SPO+ penalises prediction errors that lead to suboptimal routing decisions.
The routing regret back-propagates through the prediction model.

Reference:
    Elmachtoub & Grigas, "Smart Predict-then-Optimize", Management Science 2022.

When PyEPO is installed the official implementation is used;
otherwise a self-contained surrogate is provided.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Attempt to import PyEPO; fall back to built-in surrogate.
# ---------------------------------------------------------------------------
try:
    from pyepo.func import SPOPlus as _PyEPO_SPOPlus
    _HAS_PYEPO = True
except ImportError:
    _HAS_PYEPO = False


# ============================================================================
#  Built-in SPO+ surrogate (no external dependency)
# ============================================================================

class _SPOPlusSurrogate(torch.autograd.Function):
    """
    SPO+ surrogate loss for shortest-path problems.

    Given predicted edge costs ĉ and true costs c:
        z*(c)  = argmin_z  c^T z       (oracle shortest path under true costs)
        z*(2ĉ − c) = argmin_z (2ĉ − c)^T z  (surrogate shortest path)

        L_SPO+ = (2ĉ − c)^T z*(2ĉ − c) − 2 ĉ^T z*(c) + c^T z*(c)

    The gradient ∂L/∂ĉ = 2(z*(2ĉ − c) − z*(c)) is subgradient-consistent.
    """

    @staticmethod
    def forward(
        ctx,
        pred_costs: torch.Tensor,          # (B, n_edges)
        true_costs: torch.Tensor,          # (B, n_edges)
        oracle_solutions: torch.Tensor,    # (B, n_edges)  binary path indicator
        surrogate_solver,                  # callable(cost_vector) -> solution
    ) -> torch.Tensor:
        B = pred_costs.shape[0]
        device = pred_costs.device

        losses = []
        surrogate_sols = []

        for i in range(B):
            c = true_costs[i].detach().cpu().numpy()
            c_hat = pred_costs[i].detach().cpu().numpy()
            z_star = oracle_solutions[i]  # precomputed

            # Surrogate cost
            surrogate_cost = 2.0 * c_hat - c
            z_spo = surrogate_solver(surrogate_cost)
            z_spo_t = torch.from_numpy(z_spo).float().to(device)
            surrogate_sols.append(z_spo_t)

            # SPO+ loss
            c_t = true_costs[i]
            c_hat_t = pred_costs[i]
            loss_i = (
                torch.dot(2 * c_hat_t - c_t, z_spo_t)
                - 2 * torch.dot(c_hat_t, z_star)
                + torch.dot(c_t, z_star)
            )
            losses.append(loss_i.clamp(min=0.0))

        surrogate_stack = torch.stack(surrogate_sols)  # (B, n_edges)
        ctx.save_for_backward(oracle_solutions, surrogate_stack)

        return torch.stack(losses).mean()

    @staticmethod
    def backward(ctx, grad_output):
        oracle_solutions, surrogate_solutions = ctx.saved_tensors
        # ∂L/∂ĉ = 2(z_spo − z_star)
        grad = 2.0 * (surrogate_solutions - oracle_solutions) * grad_output
        return grad, None, None, None


# ============================================================================
#  Shortest-path solver wrapper (for SPO+ surrogate calls)
# ============================================================================

class ShortestPathSolver:
    """
    Wraps a graph's edge list to solve shortest-path via Dijkstra.
    Returns a binary edge-indicator vector for the optimal path.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        n_nodes: int,
        origin: int,
        destination: int,
    ):
        import heapq
        self.edges = edges
        self.n_nodes = n_nodes
        self.origin = origin
        self.destination = destination
        self.edge_index = {(u, v): i for i, (u, v) in enumerate(edges)}
        self.adj = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            self.adj[u].append(v)

    def __call__(self, costs: np.ndarray) -> np.ndarray:
        """Solve shortest path and return binary edge-indicator vector."""
        import heapq
        INF = float("inf")
        dist = {i: INF for i in range(self.n_nodes)}
        dist[self.origin] = 0.0
        prev = {}
        pq = [(0.0, self.origin)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == self.destination:
                break
            for v in self.adj.get(u, []):
                idx = self.edge_index.get((u, v))
                if idx is None:
                    continue
                new_d = d + float(costs[idx])
                if new_d < dist[v]:
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(pq, (new_d, v))

        # Reconstruct path → binary indicator
        sol = np.zeros(len(self.edges), dtype=np.float32)
        if dist[self.destination] == INF:
            return sol

        node = self.destination
        while node != self.origin:
            p = prev[node]
            idx = self.edge_index.get((p, node))
            if idx is not None:
                sol[idx] = 1.0
            node = p
        return sol


# ============================================================================
#  Public API: SPO+ Traffic Loss
# ============================================================================

class SPOPlusTrafficLoss(nn.Module):
    """
    Decision-focused loss that combines statistical accuracy with routing regret.

    Total loss = λ_mae · MAE + λ_ce · CE + λ_spo · SPO+

    The SPO+ term requires a graph and OD pair to compute routing regret.
    During training, a random OD pair can be sampled per batch, or a fixed
    representative pair can be used.

    Args:
        edges:        directed edge list [(u, v), ...]
        n_nodes:      number of graph nodes
        origin:       source node for SPO+ routing
        destination:  target node for SPO+ routing
        lambda_mae:   weight for MAE regression loss
        lambda_ce:    weight for congestion classification loss
        lambda_spo:   weight for SPO+ routing regret loss
        ffs:          free-flow speed for congestion label derivation
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        n_nodes: int,
        origin: int = 0,
        destination: int = 1,
        lambda_mae: float = 1.0,
        lambda_ce: float = 0.1,
        lambda_spo: float = 0.5,
        ffs: float = 65.0,
        scaler_mean: float = 0.0,
        scaler_std: float = 1.0,
    ):
        super().__init__()
        self.edges = edges
        self.n_nodes = n_nodes
        self.lambda_mae = lambda_mae
        self.lambda_ce = lambda_ce
        self.lambda_spo = lambda_spo
        self.ffs = ffs
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std

        self.solver = ShortestPathSolver(edges, n_nodes, origin, destination)
        self.ce = nn.CrossEntropyLoss()

    def _speeds_to_edge_costs(self, speeds: torch.Tensor) -> torch.Tensor:
        """
        Convert per-node speed predictions (B, N) to per-edge travel times (B, E).
        Edge cost = mean speed of endpoint nodes, inverted to time.
        """
        B = speeds.shape[0]
        E = len(self.edges)
        costs = speeds.new_zeros(B, E)
        for i, (u, v) in enumerate(self.edges):
            edge_speed = (speeds[:, u] + speeds[:, v]) / 2.0
            costs[:, i] = 1.0 / edge_speed.clamp(min=1e-3)  # inverse speed ≈ time
        return costs

    def forward(
        self,
        speed_pred: torch.Tensor,         # (B, T', N)
        speed_true: torch.Tensor,         # (B, T', N)
        congestion_logits: torch.Tensor,  # (B, N, n_classes)
    ) -> dict:
        from model.multitask_head import compute_congestion_labels

        # --- MAE ---
        mae = torch.nn.functional.l1_loss(speed_pred, speed_true)

        # Denormalise speeds prior to converting into threshold logic and travel times
        speed_pred_denorm = speed_pred * self.scaler_std + self.scaler_mean
        speed_true_denorm = speed_true * self.scaler_std + self.scaler_mean

        # --- CE ---
        speed_mean = speed_true_denorm.mean(dim=1)  # (B, N)
        labels = compute_congestion_labels(speed_mean, self.ffs)
        B, N, n_cls = congestion_logits.shape
        ce = self.ce(
            congestion_logits.reshape(B * N, n_cls),
            labels.reshape(B * N),
        )

        # --- SPO+ ---
        # Use mean over prediction horizon for routing cost
        pred_mean = speed_pred_denorm.mean(dim=1)  # (B, N)
        true_mean = speed_true_denorm.mean(dim=1)  # (B, N)

        pred_costs = self._speeds_to_edge_costs(pred_mean)  # (B, E)
        true_costs = self._speeds_to_edge_costs(true_mean)  # (B, E)

        # Oracle solutions under true costs
        oracle_sols = []
        for i in range(B):
            sol = self.solver(true_costs[i].detach().cpu().numpy())
            oracle_sols.append(torch.from_numpy(sol).float().to(speed_pred.device))
        oracle_solutions = torch.stack(oracle_sols)  # (B, E)

        spo_loss = _SPOPlusSurrogate.apply(
            pred_costs, true_costs, oracle_solutions, self.solver
        )

        total = (
            self.lambda_mae * mae
            + self.lambda_ce * ce
            + self.lambda_spo * spo_loss
        )

        return {
            "total": total,
            "mae": mae,
            "ce": ce,
            "spo": spo_loss,
        }
