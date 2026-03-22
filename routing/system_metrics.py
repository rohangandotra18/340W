"""
System-level traffic evaluation metrics.

These go beyond per-prediction accuracy (MAE/RMSE) to measure
how well the predict-then-route system performs at a network level.

Metrics follow the evaluation framework described in:
    - Highway Capacity Manual (HCM 7th ed.)
    - Transportation Research Part C standards
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_vht(
    travel_times: np.ndarray,
    flows: np.ndarray,
) -> float:
    """
    Vehicle Hours Traveled (VHT) — total time spent by all vehicles.

    Args:
        travel_times: (n_edges,) travel time per edge in minutes
        flows:        (n_edges,) vehicle count per edge

    Returns:
        VHT in vehicle-hours
    """
    return float((travel_times * flows).sum() / 60.0)


def compute_avg_network_speed(
    travel_times: np.ndarray,
    lengths_km: np.ndarray,
    flows: Optional[np.ndarray] = None,
) -> float:
    """
    Flow-weighted average network speed (km/h).

    If flows are not provided, uses unweighted average.

    Args:
        travel_times: (n_edges,) travel time per edge in minutes
        lengths_km:   (n_edges,) edge length in km
        flows:        (n_edges,) optional vehicle counts for weighting
    """
    speeds = lengths_km / np.maximum(travel_times / 60.0, 1e-6)  # km/h
    if flows is not None and flows.sum() > 0:
        return float(np.average(speeds, weights=flows))
    return float(speeds.mean())


def compute_throughput(
    flows: np.ndarray,
    time_period_hours: float = 1.0,
) -> float:
    """
    Network throughput: total vehicles served per hour.

    Args:
        flows:              (n_edges,) vehicles traversing each edge
        time_period_hours:  duration of the measurement period
    """
    return float(flows.sum() / max(time_period_hours, 1e-6))


def compute_travel_time_savings(
    predicted_cost: float,
    freeflow_cost: float,
    baseline_cost: float,
) -> Dict[str, float]:
    """
    Percentage travel time savings relative to free-flow and baseline routing.

    Args:
        predicted_cost: travel time (min) of the route found by our system
        freeflow_cost:  travel time (min) under free-flow (theoretical minimum)
        baseline_cost:  travel time (min) from a baseline router (e.g. static Dijkstra)

    Returns dict with:
        savings_vs_baseline: % reduction relative to baseline
        delay_index:         predicted / free-flow  (1.0 = ideal)
    """
    savings = max(0.0, (baseline_cost - predicted_cost) / max(baseline_cost, 1e-6)) * 100.0
    delay = predicted_cost / max(freeflow_cost, 1e-6)
    return {
        "savings_vs_baseline_pct": round(savings, 2),
        "delay_index": round(delay, 4),
    }


def compute_congestion_encounter_rate(
    path: List[int],
    congestion_labels: np.ndarray,
    congested_class: int = 2,
) -> float:
    """
    Fraction of nodes on the route that are in a congested state.

    Args:
        path:              list of node indices
        congestion_labels: (N,) integer class per node
        congested_class:   class index indicating congestion (default 2)
    """
    if len(path) == 0:
        return 0.0
    n_congested = sum(1 for n in path if int(congestion_labels[n]) == congested_class)
    return n_congested / len(path)


def evaluate_system(
    paths: List[List[int]],
    costs: List[float],
    travel_times_edges: np.ndarray,
    lengths_km: np.ndarray,
    flows: Optional[np.ndarray] = None,
    congestion_labels: Optional[np.ndarray] = None,
    freeflow_times: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive system-level evaluation over a batch of routed OD pairs.

    Returns a dictionary of aggregate metrics.
    """
    results: Dict[str, float] = {}

    # --- Route-level aggregates ---
    if costs:
        results["mean_travel_time_min"] = float(np.mean(costs))
        results["median_travel_time_min"] = float(np.median(costs))
        results["p95_travel_time_min"] = float(np.percentile(costs, 95))

    # --- Mean path length ---
    if paths:
        results["mean_path_hops"] = float(np.mean([len(p) for p in paths]))

    # --- Network-level ---
    if flows is not None:
        results["vht"] = compute_vht(travel_times_edges, flows)
        results["throughput_veh_per_hr"] = compute_throughput(flows)

    results["avg_network_speed_kmh"] = compute_avg_network_speed(
        travel_times_edges, lengths_km, flows
    )

    # --- Congestion encounter ---
    if congestion_labels is not None and paths:
        rates = [compute_congestion_encounter_rate(p, congestion_labels) for p in paths]
        results["mean_congestion_encounter_rate"] = float(np.mean(rates))

    # --- Delay index vs. free-flow ---
    if freeflow_times is not None:
        ff_speed = compute_avg_network_speed(freeflow_times, lengths_km)
        if ff_speed > 0:
            results["network_delay_index"] = results["avg_network_speed_kmh"] / ff_speed

    return results
