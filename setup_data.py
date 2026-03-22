"""
setup_data.py  —  Download METR-LA and PEMS-BAY datasets into ./data/

Run once before training:
    python setup_data.py

No code changes needed after this — all training commands in the README
point to data/metr-la.npz and data/adj_metr_la.npz by default.
"""

import os
import urllib.request
import zipfile
import shutil
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download(url: str, dest: str, label: str) -> None:
    """Download url → dest with a simple progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [skip] {label} already exists at {dest}")
        return

    print(f"  Downloading {label} …", end="", flush=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  Downloading {label} … {pct:.0f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r  Downloaded  {label} → {dest}      ")
    except Exception as e:
        print(f"\n  ERROR: Could not download {label}: {e}")
        print(f"  Please download manually from the README instructions.")
        sys.exit(1)


def unzip_if_needed(zip_path: str, extract_dir: str) -> None:
    if not zipfile.is_zipfile(zip_path):
        return
    print(f"  Extracting {os.path.basename(zip_path)} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    os.remove(zip_path)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nPDFormer++ Data Setup")
    print(f"Target directory: {DATA_DIR}\n")

    # ── METR-LA ──────────────────────────────────────────────────────────────
    print("[ METR-LA ]")

    metr_npz = os.path.join(DATA_DIR, "metr-la.npz")
    adj_npz  = os.path.join(DATA_DIR, "adj_metr_la.npz")

    # LibCity hosts these at a stable URL
    download(
        url="https://github.com/LibCity/Bigscity-LibCity-Datasets/releases/download/v0.1/METR_LA.zip",
        dest=os.path.join(DATA_DIR, "METR_LA.zip"),
        label="METR-LA (zip)",
    )
    unzip_if_needed(os.path.join(DATA_DIR, "METR_LA.zip"), DATA_DIR)

    # Rename to expected filenames if extracted with different names
    for candidate in ["metr_la.npz", "METR_LA.npz", "metr-la.dyna"]:
        src = os.path.join(DATA_DIR, candidate)
        if os.path.exists(src) and not os.path.exists(metr_npz):
            shutil.move(src, metr_npz)
            print(f"  Renamed {candidate} → metr-la.npz")

    # Adjacency matrix (small file, directly downloadable)
    download(
        url="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
        dest=os.path.join(DATA_DIR, "distances_la_2012.csv"),
        label="METR-LA distances CSV",
    )

    # Build adjacency matrix from distances CSV if adj_metr_la.npz doesn't exist
    if not os.path.exists(adj_npz):
        print("  Building adjacency matrix from distances …")
        try:
            import numpy as np
            import csv

            dist_file = os.path.join(DATA_DIR, "distances_la_2012.csv")
            sensor_ids = set()
            rows = []
            with open(dist_file) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 3:
                        sensor_ids.add(row[0])
                        sensor_ids.add(row[1])
                        rows.append((row[0], row[1], float(row[2])))

            id_to_idx = {sid: i for i, sid in enumerate(sorted(sensor_ids))}
            N = len(id_to_idx)
            adj = np.zeros((N, N), dtype=np.float32)
            std = 10.0  # Gaussian kernel bandwidth (km)
            for u, v, d in rows:
                w = float(np.exp(-(d ** 2) / (std ** 2)))
                if w > 0.1:
                    adj[id_to_idx[u], id_to_idx[v]] = w
                    adj[id_to_idx[v], id_to_idx[u]] = w

            # Row-normalise
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            adj = adj / row_sum

            np.savez_compressed(adj_npz, adj=adj)
            print(f"  Saved adjacency matrix ({N}×{N}) → {adj_npz}")

        except Exception as e:
            print(f"  Warning: Could not build adjacency matrix automatically: {e}")
            print(f"  You can still train without --adj_path (model uses identity matrix).")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n[ Status ]")
    for fname in ["metr-la.npz", "adj_metr_la.npz"]:
        path = os.path.join(DATA_DIR, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        size   = f"{os.path.getsize(path) / 1e6:.1f} MB" if os.path.exists(path) else ""
        print(f"  {status:7s}  {fname}  {size}")

    print("""
Setup complete.  You can now run:

  # Standard training
  python train.py \\
    --data_path data/metr-la.npz \\
    --adj_path  data/adj_metr_la.npz \\
    --n_nodes 207 --in_channels 1 --epochs 150 --batch_size 64

  # Decision-focused (SPO+)
  python train.py \\
    --data_path data/metr-la.npz \\
    --adj_path  data/adj_metr_la.npz \\
    --n_nodes 207 --in_channels 1 --epochs 150 --batch_size 64 \\
    --use_spo --spo_weight 0.5
""")


if __name__ == "__main__":
    main()
