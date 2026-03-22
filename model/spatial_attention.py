"""
Propagation-Delay aware spatial attention and adaptive node embeddings.
Implements the spatial encoding from PDFormer (BUAABIGSCity/PDFormer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpatialEmbedding(nn.Module):
    """
    Learnable per-node embeddings that are topology-aware.
    One message-passing step over the adjacency matrix makes each
    embedding aware of its neighbourhood (spatial context).
    """

    def __init__(self, n_nodes: int, embed_dim: int):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, embed_dim)
        self.neighbor_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.neighbor_proj.weight)

    def forward(self, node_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        node_ids: (N,)     — integer node indices
        adj:      (N, N)   — row-normalised adjacency
        Returns:  (N, embed_dim)
        """
        e = self.node_embed(node_ids)                   # (N, d)
        e_nbr = torch.mm(adj, self.neighbor_proj(e))    # (N, d)  topology-propagated
        return self.norm(e + e_nbr)


class PropagationDelayAttention(nn.Module):
    """
    Multi-head graph attention with a learnable propagation-delay bias.

    The bias shifts attention scores based on graph distance, letting the
    model weight how much future congestion at a distant node will affect
    a query node within the prediction horizon.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Per-head learnable delay bias (broadcast over batch and nodes)
        self.delay_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, spatial_bias: torch.Tensor) -> torch.Tensor:
        """
        x:            (B, N, d_model)
        spatial_bias: (N, N)  — graph-derived distance/delay matrix
        Returns:      (B, N, d_model)
        """
        B, N, _ = x.shape
        residual = x
        x = self.norm(x)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.q_proj(x))     # (B, H, N, d_head)
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Attention with propagation-delay bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        scores = scores + spatial_bias.unsqueeze(0) + self.delay_bias  # broadcast

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                    # (B, H, N, d_head)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(out) + residual


class SpatialTransformerLayer(nn.Module):
    """Spatial attention + position-wise FFN with pre-norm on the FFN."""

    def __init__(self, d_model: int, n_heads: int = 8, ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.attn = PropagationDelayAttention(d_model, n_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, spatial_bias: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)  →  (B, N, d_model)"""
        x = self.attn(x, spatial_bias)
        x = x + self.ffn(self.ffn_norm(x))
        return x
