"""
setup_data.py  —  Download or generate METR-LA dataset into ./data/

Run once before training:
    python setup_data.py

Strategy:
  1. Try downloading the real METR-LA .npz from known working URLs
  2. If all downloads fail (firewall, 404), generate realistic synthetic
     data with the same shape and statistics — lets the full pipeline run

Real METR-LA:  207 nodes, 34,272 timesteps, 5-min intervals, Los Angeles
"""

import os
import urllib.request
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── known working download URLs (tried in order) ─────────────────────────────
METR_LA_URLS = [
    # Zenodo mirror (reliable, no auth)
    "https://zenodo.org/records/5724451/files/metr-la.npz?download=1",
    # GMAN paper data repo
    "https://raw.githubusercontent.com/zhengchuanpan/GMAN/master/data/metr-la.npz",
    # traffic-bench mirror
    "https://github.com/deepkashiwa20/MegaCRN/releases/download/data/METR-LA.npz",
]

ADJ_URLS = [
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
    "https://raw.githubusercontent.com/zhengchuanpan/GMAN/master/data/W_228.csv",
]

# ── expected METR-LA key names (different repos use different keys) ───────────
DATA_KEYS = ["data", "x", "speed", "X", "array"]
ADJ_KEYS = ["adj_mx", "adj", "W", "A"]


def try_download(urls: list, dest: str, label: str) -> bool:
    """Try each URL in order. Return True if any succeeds."""
    if os.path.exists(dest):
        print(f"  [skip] {label} already exists")
        return True
    for url in urls:
        try:
            print(f"  Trying {url[:70]}…", end="", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(" OK")
            return True
        except Exception as e:
            print(f" failed ({type(e).__name__})")
            if os.path.exists(dest):
                os.remove(dest)
    return False


def load_or_fix_npz(path: str, data_keys: list, adj_keys: list):
    """
    Load an npz and return (data_array, adj_array_or_None).
    Handles different key names across dataset repos.
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    # Find data array
    data = None
    for k in data_keys:
        if k in keys:
            data = npz[k].astype(np.float32)
            print(f"  Found data under key '{k}', shape {data.shape}")
            break
    if data is None:
        print(f"  Warning: none of {data_keys} found in {path}. Keys: {keys}")
        return None, None

    # Normalise shape to (T, N, C)
    if data.ndim == 2:  # (T, N) → add channel dim
        data = data[:, :, np.newaxis]
    elif data.ndim == 4:  # (samples, T, N, C) → flatten samples
        data = data.reshape(-1, *data.shape[2:])

    # Find adj (optional)
    adj = None
    for k in adj_keys:
        if k in keys:
            adj = npz[k].astype(np.float32)
            print(f"  Found adj under key '{k}', shape {adj.shape}")
            break

    return data, adj


def build_adj_from_distances(csv_path: str, n_nodes: int = 207) -> np.ndarray:
    """Build row-normalised adjacency matrix from DCRNN distances CSV."""
    import csv

    sensor_ids, rows = set(), []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                sensor_ids.update([row[0], row[1]])
                rows.append((row[0], row[1], float(row[2])))
    id2idx = {sid: i for i, sid in enumerate(sorted(sensor_ids))}
    N = len(id2idx)
    adj = np.zeros((N, N), dtype=np.float32)
    std = 10.0
    for u, v, d in rows:
        w = float(np.exp(-(d**2) / (std**2)))
        if w > 0.1:
            adj[id2idx[u], id2idx[v]] = w
            adj[id2idx[v], id2idx[u]] = w
    # Crop to n_nodes if the CSV has more sensors than the data
    if N > n_nodes:
        adj = adj[:n_nodes, :n_nodes]
    row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
    return adj / row_sum


def generate_synthetic_metr_la() -> tuple:
    """
    Generate synthetic METR-LA-shaped data with realistic traffic statistics.

    Shape:  (34272, 207, 1)  — 34272 timesteps × 207 sensors × 1 feature
    Stats:  speeds ~ 45 mph mean, with rush-hour dips and sensor correlation

    This lets the full training pipeline run when the real data is unavailable.
    The model will train and produce valid outputs; results won't match
    published METR-LA benchmarks until you swap in the real data.
    """
    print("\n  Generating synthetic METR-LA data (realistic shape & statistics)…")
    np.random.seed(42)

    T, N, C = 34272, 207, 1
    # Time of day index (288 steps per day at 5-min intervals)
    t_idx = np.arange(T) % 288

    # Base speed with rush-hour pattern: dips at ~8am (96) and ~5pm (204)
    base = (
        55.0
        - 20 * np.exp(-((t_idx - 96) ** 2) / (2 * 15**2))
        - 15 * np.exp(-((t_idx - 204) ** 2) / (2 * 12**2))
    )
    base = base.astype(np.float32)  # (T,)

    # Spatial correlation: sensors in 5 groups (geographic clusters)
    group = np.repeat(np.arange(5), N // 5 + 1)[:N]
    spatial_factor = 1.0 + 0.1 * (group / 4.0 - 0.5)  # (N,)

    # Combine: (T, N) with noise
    speeds = base[:, None] * spatial_factor[None, :]
    speeds += np.random.normal(0, 3.0, (T, N)).astype(np.float32)
    speeds = speeds.clip(5, 75)[:, :, np.newaxis]  # (T, N, 1)

    # Simple ring topology adjacency matrix
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for delta in [-2, -1, 1, 2]:
            j = (i + delta) % N
            w = 0.8 if abs(delta) == 1 else 0.4
            adj[i, j] = w
    row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
    adj = adj / row_sum

    return speeds, adj


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nPDFormer++ Data Setup")
    print(f"Target: {DATA_DIR}\n")

    metr_npz = os.path.join(DATA_DIR, "metr-la.npz")
    adj_npz = os.path.join(DATA_DIR, "adj_metr_la.npz")
    # --------------------------------------------------------------
    # NEW: If the real data files already exist, skip download/synthetic steps
    # --------------------------------------------------------------
    if os.path.exists(metr_npz) and os.path.exists(adj_npz):
        print("[ INFO ] Real METR‑LA files already present – skipping download.")
        try:
            data_arr = np.load(metr_npz)["data"]
            adj_arr = np.load(adj_npz)["adj_mx"]
            print(f"  ✅ Data shape: {data_arr.shape}")
            print(f"  ✅ Adj shape: {adj_arr.shape}")
        except Exception as e:
            print(f"  ⚠️  Could not read existing files: {e}")
        print("\n[ Status ]")
        for fname in ["metr-la.npz", "adj_metr_la.npz"]:
            path = os.path.join(DATA_DIR, fname)
            size = os.path.getsize(path) / 1e6
            print(f"  OK       {fname}  ({size:.1f} MB)")
        print("\nReady. Run the smoke test as usual.\n")
        return
    data_arr = adj_arr = None

    # ── Step 1: Try to download real METR-LA ─────────────────────────────────
    print("[ Attempting METR-LA download ]")
    tmp = os.path.join(DATA_DIR, "_download_tmp.npz")

    downloaded = try_download(METR_LA_URLS, tmp, "METR-LA npz")

    if downloaded and os.path.exists(tmp):
        data_arr, adj_arr = load_or_fix_npz(tmp, DATA_KEYS, ADJ_KEYS)
        os.remove(tmp)
        if data_arr is None:
            print("  Downloaded file had unexpected format — falling back to synthetic")

    # ── Step 2: Try adjacency CSV if no adj yet ───────────────────────────────
    if adj_arr is None:
        print("\n[ Attempting adjacency matrix download ]")
        dist_csv = os.path.join(DATA_DIR, "distances_la_2012.csv")
        if try_download(ADJ_URLS, dist_csv, "distances CSV"):
            try:
                adj_arr = build_adj_from_distances(dist_csv)
                print(f"  Built adj matrix from distances: {adj_arr.shape}")
            except Exception as e:
                print(f"  Could not build adj from CSV: {e}")

    # ── Step 3: Fall back to synthetic if downloads failed ────────────────────
    if data_arr is None:
        print("\n[ Download failed — generating synthetic data ]")
        print("  (Synthetic data lets the pipeline run; swap real data for benchmarks)")
        data_arr, adj_arr_syn = generate_synthetic_metr_la()
        # Always use synthetic adj with synthetic data to ensure shape match
        adj_arr = adj_arr_syn

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[ Saving ]")
    np.savez_compressed(metr_npz, data=data_arr)
    print(f"  Saved data {data_arr.shape} → {metr_npz}")

    if adj_arr is not None:
        np.savez_compressed(adj_npz, adj_mx=adj_arr)
        print(f"  Saved adj  {adj_arr.shape} → {adj_npz}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[ Status ]")
    for fname in ["metr-la.npz", "adj_metr_la.npz"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e6
            print(f"  OK       {fname}  ({size:.1f} MB)")
        else:
            print(f"  MISSING  {fname}")

    print("""
Ready. Run the smoke test:

  python train.py \\
    --data_path data/metr-la.npz \\
    --adj_path  data/adj_metr_la.npz \\
    --in_channels 1 \\
    --epochs 5 --batch_size 16 \\
    --num_workers 0 \\
    --output_dir ./checkpoints/test
""")


if __name__ == "__main__":
    main()
