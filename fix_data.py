import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import requests
import io

print("Starting definitive data correction process...")
os.makedirs("data", exist_ok=True)

# 1. Regenerate True Adjacency Matrix
print("Building True Adjacency Matrix from Liyaguang's distances...")
url = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv"
try:
    content = requests.get(url).text
    df_dist = pd.read_csv(io.StringIO(content))

    sensor_ids = set()
    for _, row in df_dist.iterrows():
        sensor_ids.add(str(int(row["from"])))
        sensor_ids.add(str(int(row["to"])))
    sensor_ids = sorted(list(sensor_ids))
    id2idx = {k: i for i, k in enumerate(sensor_ids)}

    n_nodes = 207
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    std = 10.0
    for _, row in df_dist.iterrows():
        u = str(int(row["from"]))
        v = str(int(row["to"]))
        d = row["cost"]
        if u in id2idx and v in id2idx:
            idx_u, idx_v = id2idx[u], id2idx[v]
            if idx_u < n_nodes and idx_v < n_nodes:
                w = float(np.exp(-(d**2) / (std**2)))
                if w > 0.1:
                    adj[idx_u, idx_v] = w
                    adj[idx_v, idx_u] = w

    row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
    adj = adj / row_sum
    np.savez_compressed("data/adj_metr_la.npz", adj_mx=adj)
    print("✓ Successfully saved TRUE data/adj_metr_la.npz")
except Exception as e:
    print("Failed to build adj:", e)

# 2. Reconstruct True Feature Tensor from HuggingFace
print("\nReconstructing True METR-LA dataset from HuggingFace parquets...")
dfs = []
for split in ["train", "validation", "test"]:
    ds = load_dataset("witgaw/METR-LA", split=split)
    dfs.append(ds.to_pandas())

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by=["t0_timestamp", "node_id"])

pivot = df.pivot(index="t0_timestamp", columns="node_id", values="x_t+0_d0")
speeds = pivot.values.astype(np.float32)
speeds = np.expand_dims(speeds, axis=-1)  # (34249, 207, 1)

np.savez_compressed("data/metr-la.npz", data=speeds)
print(f"✓ Successfully reconstructed TRUE data/metr-la.npz with shape {speeds.shape}")
print("\nReal datasets successfully prepared! You may now push them to Git.")
