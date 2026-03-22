"""
Learned Fundamental Diagram — neural-network-based volume-delay function.

A small MLP that maps traffic state features (speed, flow, density) to
per-edge travel times.  This is a flexible, data-driven alternative to the
fixed-parameter BPR function.

Inspired by:
    PIDL+FDL (Physics-Informed Deep Learning + Fundamental Diagram Learning)
    — IEEE Transactions on Intelligent Transportation Systems, 2022.

Key design choices:
    • Monotonicity regularisation: travel time should increase with volume
    • Physics warm-start: initialised to approximate BPR(α=0.15, β=4)
    • Can be jointly trained with PDFormer++ or pretrained separately
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedFundamentalDiagram(nn.Module):
    """
    Neural volume-delay function:  f(speed, flow, density) → travel_time.

    Args:
        in_features:    input dimension (default 3: speed, flow, density)
        hidden_dim:     hidden layer size
        n_layers:       number of hidden layers
        residual_bpr:   if True, output = BPR_init + NN_correction
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 3,
        residual_bpr: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.residual_bpr = residual_bpr

        layers = []
        dim = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Learnable BPR parameters (for residual mode)
        if residual_bpr:
            self.alpha = nn.Parameter(torch.tensor(0.15))
            self.beta = nn.Parameter(torch.tensor(4.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        free_flow_time: torch.Tensor | None = None,
        capacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (..., in_features) traffic state features
            free_flow_time:  (...,) optional free-flow travel times for BPR residual
            capacity:        (...,) optional link capacities for BPR residual

        Returns:
            travel_time:     (..., 1) predicted travel times
        """
        correction = self.net(x)  # (..., 1)

        if self.residual_bpr and free_flow_time is not None and capacity is not None:
            # BPR base: t = t_ff * (1 + α * (V/C)^β)
            # Use first feature as flow proxy
            flow = x[..., 1:2] if x.shape[-1] > 1 else x[..., 0:1]
            ratio = flow / capacity.unsqueeze(-1).clamp(min=1.0)
            bpr_base = free_flow_time.unsqueeze(-1) * (
                1.0 + self.alpha * ratio.pow(self.beta.clamp(min=1.0))
            )
            return F.softplus(bpr_base + correction)  # ensure positive
        else:
            return F.softplus(correction)  # ensure positive

    def monotonicity_loss(self, x: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
        """
        Physics-informed regularisation: ∂t/∂flow ≥ 0 (travel time should
        increase with flow).

        Computes a penalty when the gradient of travel time with respect to
        the flow/volume input (index 1) is negative.
        """
        x_reg = x.detach().requires_grad_(True)
        t = self.forward(x_reg)
        grad = torch.autograd.grad(
            t.sum(), x_reg, create_graph=True
        )[0]

        # Gradient w.r.t. flow (feature index 1, or 0 if only 1 feature)
        flow_idx = min(1, x.shape[-1] - 1)
        flow_grad = grad[..., flow_idx]

        # Penalise negative gradients (violations of monotonicity)
        violation = F.relu(-flow_grad + eps)
        return violation.mean()


class FundamentalDiagramConverter:
    """
    Drop-in replacement for BPR conversion in the pipeline.
    Wraps a trained LearnedFundamentalDiagram model to convert
    predicted traffic states to travel times.
    """

    def __init__(self, model: LearnedFundamentalDiagram, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def convert(
        self,
        speeds: torch.Tensor,
        flows: torch.Tensor | None = None,
        densities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert traffic state predictions to edge travel times.

        Args:
            speeds:    (...,) predicted speeds
            flows:     (...,) predicted flows (optional)
            densities: (...,) predicted densities (optional)

        Returns:
            travel_times: (..., 1)
        """
        parts = [speeds.unsqueeze(-1)]
        if flows is not None:
            parts.append(flows.unsqueeze(-1))
        if densities is not None:
            parts.append(densities.unsqueeze(-1))

        # Pad to in_features if needed
        while len(parts) < self.model.in_features:
            parts.append(torch.zeros_like(parts[0]))

        x = torch.cat(parts, dim=-1).to(self.device)
        return self.model(x)
