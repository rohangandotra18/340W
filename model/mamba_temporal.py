"""
Mamba temporal backbone — pure PyTorch selective SSM (S6).
No custom CUDA kernels required; works on any device.
For maximum throughput on A100, install mamba-ssm and swap SelectiveSSM
with the official MambaBlock from that package.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — core Mamba recurrence.
    Pure-PyTorch implementation (sequential scan).

    Args:
        d_model:  input/output dimension
        d_state:  SSM latent state dimension (N in the paper)
        expand:   inner expansion factor (default 2 → d_inner = 2·d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # Pre-norm + gated input projection
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depth-wise causal conv for local context (kernel=3, causal via padding)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner,
            bias=True,
        )

        # Selective parameters: Δ, B, C projected from x
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj bias so Δ ~ Uniform(0.001, 0.1) after softplus
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A (log-parameterised to keep eigenvalues negative)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float)
            .unsqueeze(0)
            .expand(self.d_inner, -1)
        )
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip / residual weight
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def _ssm_scan(
        self,
        x: torch.Tensor,  # (B, L, d_inner)
        delta: torch.Tensor,  # (B, L, d_inner)
        A: torch.Tensor,  # (d_inner, d_state)
        B: torch.Tensor,  # (B, L, d_state)
        C: torch.Tensor,  # (B, L, d_state)
    ) -> torch.Tensor:
        """Selective recurrence scan. Returns (B, L, d_inner)."""
        B_sz, L, d_inner = x.shape
        d_state = A.shape[-1]

        delta = F.softplus(delta).clamp(max=10.0)  # (B, L, d_inner) Prevent delta explosion
        # Cast A to match input dtype so dA/dB stay in float16 under AMP.
        # exp(delta*A) is safe in float16: A<0 and delta>0 so the product is
        # negative and exp maps it to (0,1], well within float16 range.
        A = A.to(x.dtype)
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        h = x.new_zeros(B_sz, d_inner, d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # (B, d_inner, d_state)
            h = h.clamp(-1e4, 1e4)  # Prevent recurrent state explosion causing NaNs
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))  # (B, d_inner)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        # Gated projection
        xz = self.in_proj(x)  # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        # Causal conv  (trim right padding to preserve causality)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, : x_in.shape[1]].transpose(
            1, 2
        )
        x_conv = F.silu(x_conv)

        # Selective parameters
        x_ssm = self.x_proj(x_conv)  # (B, L, dt_rank + 2·d_state)
        delta, B_mat, C_mat = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(delta)  # (B, L, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        y = self._ssm_scan(x_conv, delta, A, B_mat, C_mat)
        y = y + x_conv * self.D.to(y.dtype)  # skip connection (keep float16 under AMP)
        y = y * F.silu(z)  # gating

        return self.out_proj(y) + residual


class MambaTemporalBlock(nn.Module):
    """
    Stack of SelectiveSSM layers for processing per-node time series.

    Args:
        d_model:    feature dimension
        n_layers:   number of stacked SSM layers
        d_state:    SSM state dimension
        expand:     inner dimension expansion factor
    """

    def __init__(
        self, d_model: int, n_layers: int = 2, d_state: int = 16, expand: int = 2
    ):
        super().__init__()
        
        if HAS_MAMBA_SSM:
            self.layers = nn.ModuleList(
                [
                    OfficialMamba(d_model=d_model, d_state=d_state, expand=expand)
                    for _ in range(n_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    SelectiveSSM(d_model, d_state=d_state, expand=expand)
                    for _ in range(n_layers)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return x
