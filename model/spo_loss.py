"""
SPO+ (Smart Predict-then-Optimize) decision-focused loss for traffic routing.

Instead of optimising predictions for statistical accuracy alone (MAE/RMSE),
SPO+ penalises prediction errors that lead to suboptimal routing decisions.
The routing regret back-propagates through the prediction model.

Reference:
    Elmachtoub & Grigas, "Smart Predict-then-Optimize", Management Science 2022.
"""

from __future__ import annotations

from typing import List, Tuple
import gc

import numpy as np
import torch
import torch.nn as nn


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
        self.edges = edges
        self.n_nodes = n_nodes
        self.origin = origin
        self.destination = destination
        self.edge_index = {(u, v): i for i, (u, v) in enumerate(edges)}
        self.adj = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            self.adj[u].append(v)

    def __call__(self, costs: np.ndarray) -> np.ndarray:
        """Solve shortest path and return binary edge-indicator vector.

        Costs MUST be non-negative for Dijkstra correctness. Callers should
        clamp before calling. As a safety net, negative costs are clamped
        to zero here and an iteration limit prevents infinite loops.
        """
        import heapq

        # Safety: Dijkstra requires non-negative costs
        costs = np.maximum(costs, 0.0)

        INF = float("inf")
        dist = {i: INF for i in range(self.n_nodes)}
        dist[self.origin] = 0.0
        prev = {}
        pq = [(0.0, self.origin)]

        # Iteration limit: worst-case Dijkstra pops O(E log V) entries
        max_iters = len(self.edges) * 20 + 1000
        iters = 0

        while pq:
            iters += 1
            if iters > max_iters:
                break  # safety: prevent infinite loops
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
        path_len = 0
        while node != self.origin:
            path_len += 1
            if path_len > self.n_nodes:  # cycle guard
                return np.zeros(len(self.edges), dtype=np.float32)
            p = prev.get(node)
            if p is None:  # broken path
                return np.zeros(len(self.edges), dtype=np.float32)
            idx = self.edge_index.get((p, node))
            if idx is not None:
                sol[idx] = 1.0
            node = p
        return sol


# ============================================================================
#  Public API: SPO+ Traffic Loss (memory-efficient, no custom autograd)
# ============================================================================


class SPOPlusTrafficLoss(nn.Module):
    """
    Decision-focused loss that combines statistical accuracy with routing regret.

    Total loss = λ_mae · MAE + λ_ce · CE + λ_spo · SPO+

    This implementation avoids torch.autograd.Function to prevent OOM.
    The SPO+ gradient ∂L/∂ĉ = 2(z_spo − z_star) is computed analytically
    in numpy and injected via a dot-product trick:
        spo_loss = pred_costs · gradient.detach()
    This gives identical gradients to the full SPO+ formulation but uses
    constant memory regardless of batch count.

    Robustness: If the edge list is empty or the destination is unreachable,
    the SPO+ term gracefully falls back to zero (prediction-only training).
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
        self._spo_disabled = False  # set True if graph is unusable
        self._first_call = True

        from model.multitask_head import FocalLoss

        self.ce = FocalLoss(gamma=2.0)

        # Guard: empty edge list means no routing is possible
        if len(edges) == 0:
            print("  ⚠ SPO+ WARNING: edge list is EMPTY (adj matrix has no "
                  "off-diagonal entries). SPO+ loss will be disabled — "
                  "training with MAE + CE only.", flush=True)
            self._spo_disabled = True
            self.solver = None
            self.register_buffer('_src_idx', torch.tensor([], dtype=torch.long))
            self.register_buffer('_dst_idx', torch.tensor([], dtype=torch.long))
            return

        self.solver = ShortestPathSolver(edges, n_nodes, origin, destination)

        # Connectivity check: verify origin→destination is reachable
        test_costs = np.ones(len(edges), dtype=np.float32)
        test_sol = self.solver(test_costs)
        if test_sol.sum() == 0:
            print(f"  ⚠ SPO+ WARNING: no path from origin={origin} to "
                  f"destination={destination} in the graph ({len(edges)} edges, "
                  f"{n_nodes} nodes). SPO+ loss will be disabled.", flush=True)
            self._spo_disabled = True

        # Pre-compute edge endpoint indices for vectorised cost computation
        src_idx = [u for u, v in edges]
        dst_idx = [v for u, v in edges]
        self.register_buffer('_src_idx', torch.tensor(src_idx, dtype=torch.long))
        self.register_buffer('_dst_idx', torch.tensor(dst_idx, dtype=torch.long))

    def _speeds_to_edge_costs(self, speeds: torch.Tensor) -> torch.Tensor:
        """
        Convert per-node speed predictions (B, N) to per-edge travel times (B, E).
        Vectorised — no Python for-loop over edges.
        """
        src_speeds = speeds[:, self._src_idx]  # (B, E)
        dst_speeds = speeds[:, self._dst_idx]  # (B, E)
        edge_speed = (src_speeds + dst_speeds) / 2.0
        return 1.0 / edge_speed.clamp(min=1.0)

    def _compute_spo_loss(self, pred_costs: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient SPO+ loss computation.

        Instead of a custom autograd.Function (which keeps the full computation
        graph alive and leaks memory), we compute the SPO+ subgradient
        analytically in numpy and inject it back via:

            loss = (pred_costs * spo_grad.detach()).mean()

        This ensures ∂loss/∂pred_costs = spo_grad, which is exactly the
        SPO+ subgradient: 2(z_spo − z_star).
        """
        # Guard: if edges are empty or SPO is disabled, return zero
        if self._spo_disabled or pred_costs.shape[-1] == 0:
            return pred_costs.new_tensor(0.0)

        B = pred_costs.shape[0]

        # All Dijkstra solving happens in pure numpy — no torch graphs
        pred_np = pred_costs.detach().cpu().numpy()
        true_np = true_costs.detach().cpu().numpy()

        # Check for NaN/Inf in inputs (can come from denormalized speeds)
        if not np.isfinite(pred_np).all() or not np.isfinite(true_np).all():
            return pred_costs.new_tensor(0.0)

        spo_grads = np.zeros_like(pred_np)  # (B, E)

        for i in range(B):
            # Oracle: shortest path under true costs
            z_star = self.solver(true_np[i])

            # Surrogate: shortest path under 2*pred - true
            # Clamp to non-negative: Dijkstra requires non-negative costs
            surrogate_cost = np.maximum(2.0 * pred_np[i] - true_np[i], 1e-8)
            z_spo = self.solver(surrogate_cost)

            # SPO+ subgradient: 2(z_spo - z_star)
            spo_grads[i] = 2.0 * (z_spo - z_star)

        # Convert gradient to torch (detached — no graph needed)
        spo_grad_t = torch.from_numpy(spo_grads).float().to(pred_costs.device)

        # Dot-product trick: creates a scalar loss whose gradient w.r.t.
        # pred_costs equals spo_grad_t (the analytical SPO+ subgradient)
        spo_loss = (pred_costs * spo_grad_t).mean()

        # Safety: replace NaN/Inf with zero
        if torch.isnan(spo_loss) or torch.isinf(spo_loss):
            return pred_costs.new_tensor(0.0)

        # Clean up
        del pred_np, true_np, spo_grads, spo_grad_t
        gc.collect()

        return spo_loss.clamp(min=0.0)

    def forward(
        self,
        speed_pred: torch.Tensor,  # (B, T', N)
        speed_true: torch.Tensor,  # (B, T', N)
        congestion_logits: torch.Tensor,  # (B, N, n_classes)
    ) -> dict:
        from model.multitask_head import compute_congestion_labels

        # Safety: if model produced NaN/Inf outputs, return a safe fallback
        # (the training loop SHOULD catch this upstream, but defense in depth)
        if (torch.isnan(speed_pred).any() or torch.isinf(speed_pred).any()
                or torch.isnan(speed_true).any() or torch.isinf(speed_true).any()):
            zero = speed_pred.new_tensor(0.0)
            return {"total": speed_pred.new_tensor(float('nan')),
                    "mae": zero, "ce": zero, "spo": zero}

        # --- Smooth L1 ---
        mae = torch.nn.functional.smooth_l1_loss(speed_pred, speed_true, beta=1.0)

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

        # --- SPO+ (memory-efficient) ---
        if self._spo_disabled:
            spo_loss = speed_pred.new_tensor(0.0)
        else:
            pred_mean = speed_pred_denorm.mean(dim=1).clamp(min=0.1)  # (B, N)
            true_mean = speed_true_denorm.mean(dim=1).clamp(min=0.1)  # (B, N)

            pred_costs = self._speeds_to_edge_costs(pred_mean)  # (B, E)
            true_costs = self._speeds_to_edge_costs(true_mean)  # (B, E)

            spo_loss = self._compute_spo_loss(pred_costs, true_costs)

        # Diagnostic: log once on first call
        if self._first_call:
            self._first_call = False
            n_edges = len(self.edges)
            print(f"  SPO+ first-call diagnostic:", flush=True)
            print(f"    Edges: {n_edges}  |  SPO disabled: {self._spo_disabled}", flush=True)
            print(f"    MAE: {mae.item():.4f}  |  CE: {ce.item():.4f}  |  SPO: {spo_loss.item():.4f}", flush=True)

        total = self.lambda_mae * mae + self.lambda_ce * ce + self.lambda_spo * spo_loss

        return {
            "total": total,
            "mae": mae,
            "ce": ce,
            "spo": spo_loss,
        }
