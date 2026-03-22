"""
Volume-delay functions for converting predicted flows/speeds to per-edge travel times.

BPR (Bureau of Public Roads):
    t = t_ff × (1 + α × (V/C)^β)
    α = 0.15, β = 4  (standard values from HCM)

Speed → travel time:
    t = (length / speed) × 60   [minutes]
"""
import torch


def bpr_travel_time(
    flow: torch.Tensor,
    capacity: torch.Tensor,
    t_ff: torch.Tensor,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Bureau of Public Roads volume-delay function.

    Args:
        flow:     predicted volume (vehicles/hour), any broadcastable shape
        capacity: link capacity (vehicles/hour)
        t_ff:     free-flow travel time (minutes)
        alpha:    BPR α coefficient (default 0.15)
        beta:     BPR β exponent    (default 4.0)

    Returns:
        travel time in the same units as t_ff
    """
    ratio = flow / capacity.clamp(min=1.0)
    return t_ff * (1.0 + alpha * ratio.pow(beta))


def speed_to_travel_time(speed: torch.Tensor, length_km: torch.Tensor) -> torch.Tensor:
    """
    Convert link speed (km/h) and length (km) to travel time (minutes).
    Speed is clamped to ≥ 1 km/h to avoid division by zero.
    """
    return (length_km / speed.clamp(min=1.0)) * 60.0
