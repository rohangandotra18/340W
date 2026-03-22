"""
Time-dependent Dijkstra routing on predicted traffic graph.

Complexity: O((|V| + |E|) log |V|) per query.
Sub-millisecond on CPU for demonstration-scale networks (200–900 nodes).

Two modes:
  route()                — standard Dijkstra on a single travel-time snapshot
  time_dependent_route() — TD-Dijkstra where edge weights vary per future timestep
"""
import heapq
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class TimeDependentRouter:
    """
    Builds and maintains a directed graph for routing.
    Edge weights are updated from PDFormer++ predicted travel times.

    Args:
        edges:        list of (u, v) directed edge tuples
        n_nodes:      total node count (optional; inferred from edges if None)
    """

    def __init__(self, edges: List[Tuple[int, int]], n_nodes: Optional[int] = None):
        self.edges = edges
        self.n_nodes = n_nodes or (max(max(u, v) for u, v in edges) + 1)
        self.edge_index: Dict[Tuple[int, int], int] = {(u, v): i for i, (u, v) in enumerate(edges)}

        # Adjacency list for TD-Dijkstra
        self.adj: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        for u, v in edges:
            self.adj[u].append(v)

        self.G: Optional[nx.DiGraph] = None

    # ------------------------------------------------------------------
    def build_graph(self, travel_times: np.ndarray) -> None:
        """
        (Re)build the NetworkX graph with new edge weights.

        Args:
            travel_times: (n_edges,) float array of travel times in minutes
        """
        self.G = nx.DiGraph()
        for (u, v), t in zip(self.edges, travel_times):
            self.G.add_edge(u, v, weight=float(t))

    # ------------------------------------------------------------------
    def route(self, origin: int, destination: int) -> Tuple[List[int], float]:
        """
        Standard Dijkstra shortest path (requires build_graph() called first).

        Returns:
            (path, total_travel_time_minutes)
            path is [] and cost is inf if no path exists.
        """
        if self.G is None:
            raise RuntimeError("Call build_graph() before route().")
        try:
            path = nx.dijkstra_path(self.G, origin, destination, weight="weight")
            cost = nx.dijkstra_path_length(self.G, origin, destination, weight="weight")
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    # ------------------------------------------------------------------
    def batch_route(self, od_pairs: List[Tuple[int, int]]) -> List[Tuple[List[int], float]]:
        """Route multiple OD pairs. Requires build_graph() first."""
        return [self.route(o, d) for o, d in od_pairs]

    # ------------------------------------------------------------------
    def time_dependent_route(
        self,
        origin: int,
        destination: int,
        travel_times_per_horizon: np.ndarray,   # (T', n_edges)
        departure_step: int = 0,
        step_minutes: float = 5.0,
    ) -> Tuple[List[int], float]:
        """
        Time-dependent Dijkstra (Cooke & Halsey, 1966).
        Edge weights change as the vehicle traverses the network: when a
        vehicle arrives at a node at time t, it uses the travel time
        predicted for timestep t on each outgoing edge.

        Args:
            travel_times_per_horizon: (T', n_edges) predicted times per future step
            departure_step:           horizon index corresponding to departure time
            step_minutes:             real-world minutes per prediction step

        Returns:
            (path, total_travel_time_minutes)
        """
        T_prime = travel_times_per_horizon.shape[0]
        INF = float("inf")

        dist: Dict[int, float] = {i: INF for i in range(self.n_nodes)}
        dist[origin] = 0.0
        prev: Dict[int, int] = {}

        # Priority queue entries: (cost, horizon_step, node)
        pq: List[Tuple[float, int, int]] = [(0.0, departure_step, origin)]

        while pq:
            cost, t_step, u = heapq.heappop(pq)
            if cost > dist[u]:
                continue
            if u == destination:
                break

            for v in self.adj.get(u, []):
                edge_idx = self.edge_index.get((u, v))
                if edge_idx is None:
                    continue

                t_idx = min(t_step, T_prime - 1)
                edge_time = float(travel_times_per_horizon[t_idx, edge_idx])
                new_cost = cost + edge_time

                if new_cost < dist[v]:
                    dist[v] = new_cost
                    prev[v] = u
                    next_step = t_step + max(1, round(edge_time / step_minutes))
                    heapq.heappush(pq, (new_cost, next_step, v))

        if dist[destination] == INF:
            return [], INF

        # Reconstruct path
        path: List[int] = []
        node = destination
        while node != origin:
            path.append(node)
            node = prev[node]
        path.append(origin)
        path.reverse()

        return path, dist[destination]

    # ------------------------------------------------------------------
    def route_quality_metrics(
        self,
        pred_path: List[int],
        pred_cost: float,
        true_travel_times: np.ndarray,   # (n_edges,) ground-truth
        oracle_cost: float,
    ) -> dict:
        """
        Compute individual route quality metrics (from the paper's eval section).

        Metrics:
            - path_optimality_ratio: actual_cost / oracle_cost  (1.0 = perfect)
            - congestion_encounter_rate: fraction of path edges in congested state
        """
        if not pred_path or oracle_cost == 0.0:
            return {"path_optimality_ratio": float("inf"), "congestion_encounter_rate": 0.0}

        # Actual travel time on predicted path using true weights
        actual_cost = 0.0
        n_congested = 0
        for u, v in zip(pred_path[:-1], pred_path[1:]):
            idx = self.edge_index.get((u, v))
            if idx is not None:
                t = float(true_travel_times[idx])
                actual_cost += t
                # Heuristic: edge is "congested" if its true time > 1.5× minimum in graph
                if t > 1.5 * true_travel_times.min():
                    n_congested += 1

        n_edges_path = max(len(pred_path) - 1, 1)
        return {
            "path_optimality_ratio": actual_cost / oracle_cost,
            "congestion_encounter_rate": n_congested / n_edges_path,
        }
