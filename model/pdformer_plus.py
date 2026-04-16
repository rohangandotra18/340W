"""
PDFormer++ — spatiotemporal traffic prediction model.

Architecture (three interleaved encode-decode stages):
  Input (B, T, N, C)
    → Input projection
    → Adaptive spatial embedding (added)
    → [Mamba temporal block × N_temporal + Spatial transformer × N_spatial] × N_layers
    → Output regression head  → speed_pred  (B, N, T')
    → Multi-task head         → congestion  (B, N, n_classes)

References:
  PDFormer    — BUAABIGSCity/PDFormer (LibCity)
  Mamba (S6)  — Gu & Dao, NeurIPS 2023
  SPO+        — Elmachtoub & Grigas, Management Science 2022
"""

import torch
import torch.nn as nn

from .mamba_temporal import MambaTemporalBlock
from .spatial_attention import AdaptiveSpatialEmbedding, SpatialTransformerLayer
from .multitask_head import MultiTaskHead


class PDFormerPlusPlus(nn.Module):
    """
    Args:
        n_nodes:            number of sensor/road nodes (N)
        in_channels:        input feature channels per node per timestep (C)
        d_model:            internal feature dimension
        out_horizon:        number of future timesteps to predict (T')
        n_temporal_layers:  Mamba SSM layers per interleaved stage
        n_spatial_layers:   number of interleaved temporal+spatial stages
        n_heads:            attention heads in spatial transformer
        d_state:            Mamba SSM state dimension
        n_classes:          congestion classes (3 = free/slow/congested)
        dropout:            dropout rate in spatial layers
    """

    def __init__(
        self,
        n_nodes: int,
        in_channels: int,
        d_model: int = 64,
        out_horizon: int = 12,
        n_temporal_layers: int = 2,
        n_spatial_layers: int = 2,
        n_heads: int = 8,
        d_state: int = 16,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.out_horizon = out_horizon
        self.n_spatial_layers = n_spatial_layers

        # --- Input embedding ---
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # --- Spatial node embeddings ---
        self.spatial_embed = AdaptiveSpatialEmbedding(n_nodes, d_model)

        # --- Interleaved temporal (Mamba) + spatial (PD-Attention) blocks ---
        # Each "stage" = one Mamba block + one spatial transformer layer
        self.temporal_blocks = nn.ModuleList(
            [
                MambaTemporalBlock(d_model, n_layers=n_temporal_layers, d_state=d_state)
                for _ in range(n_spatial_layers)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerLayer(d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_spatial_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # --- Regression decoder: d_model → T' (applied to time-pooled embedding) ---
        self.temporal_out = nn.Linear(d_model, out_horizon)

        # --- Multi-task classification head ---
        self.head = MultiTaskHead(d_model, n_classes=n_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_bias(adj: torch.Tensor) -> torch.Tensor:
        """
        Propagation-delay bias matrix from the normalised adjacency.
        −log(adj) makes strong connections (high weight) → small bias,
        matching the intuition that nearby nodes propagate congestion faster.
        """
        bias = -torch.log(adj.clamp(min=1e-6))  # (N, N)
        # Cap the bias to prevent extreme values from dominating attention
        return bias.clamp(max=20.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> dict:
        """
        Args:
            x:   (B, T, N, C)  historical traffic tensor
            adj: (N, N)         row-normalised adjacency matrix (on same device as x)

        Returns dict:
            speed_pred:  (B, N, T')        predicted speed/flow per node per future step
            congestion:  (B, N, n_classes) congestion class logits per node
            embed:       (B, N, d_model)   final encoder embedding (for downstream use)
        """
        B, T, N, C = x.shape

        # Input projection: (B, T, N, C) → (B, T, N, d_model)
        h = self.input_proj(x)

        # Add topology-aware spatial embeddings (broadcast over B, T)
        node_ids = torch.arange(N, device=x.device)
        h = h + self.spatial_embed(node_ids, adj)  # (N, d_model) → broadcast

        # Precompute spatial bias
        sp_bias = self._spatial_bias(adj)  # (N, N)

        # Interleaved temporal ↔ spatial encoding
        for temp_block, spat_block in zip(self.temporal_blocks, self.spatial_blocks):
            # --- Temporal: per-node across time ---
            # Reshape (B, T, N, d) → (B·N, T, d), run Mamba, reshape back
            h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
            h_t = temp_block(h_t)
            h = h_t.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d)

            # --- Spatial: per-timestep across nodes ---
            # Reshape (B, T, N, d) → (B·T, N, d), run attention, reshape back
            h_s = h.reshape(B * T, N, self.d_model)
            h_s = spat_block(h_s, sp_bias)
            h = h_s.reshape(B, T, N, self.d_model)

        h = self.final_norm(h)  # (B, T, N, d_model)

        # Time-pool: average over input horizon → (B, N, d_model)
        h_pool = h.mean(dim=1)

        # Regression: (B, N, d_model) → (B, N, T')
        speed_pred = self.temporal_out(h_pool)

        # Multi-task head
        head_out = self.head(h_pool)

        return {
            "speed_pred": speed_pred,  # (B, N, T')
            "congestion": head_out["congestion"],  # (B, N, n_classes)
            "embed": h_pool,  # (B, N, d_model)
        }
