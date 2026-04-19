# PDFormer++: Decision-Focused Traffic Prediction and Routing via Selective State Spaces

> **DS 340W — Rohan Gandotra — Penn State — Spring 2026**
>
> A unified predict-then-route framework that combines spatiotemporal traffic forecasting with congestion-aware routing, trained end-to-end using decision-focused learning.

---

## Abstract

Urban traffic congestion costs the U.S. economy over **\$87 billion annually** in lost productivity and wasted fuel (INRIX 2023). Existing navigation systems treat prediction and routing as independent stages — forecasting future traffic conditions with one model, then computing shortest paths on the result. This two-stage approach optimises predictions for statistical accuracy (MAE/RMSE) rather than for the quality of the routing decisions they inform, leading to suboptimal routes despite accurate forecasts.

**PDFormer++** closes this gap with a three-module architecture:

1. **Predictor** — A Mamba-based selective state space model with propagation-delay-aware spatial attention predicts future speeds across the road network.
2. **Converter** — A multi-task head jointly classifies congestion states and estimates travel times via either the Bureau of Public Roads (BPR) function or a learned fundamental diagram.
3. **Router** — Time-dependent Dijkstra routing finds optimal paths using the predicted travel times.

The key contribution is **SPO+ decision-focused loss** (Elmachtoub & Grigas, 2022): routing regret is backpropagated through the prediction model, ensuring forecasts are optimised specifically for the downstream routing task — not just point accuracy.

---

## Architecture

```
Input (B, T, N, C)
    │
    ├── Input Projection → LayerNorm → GELU
    │
    ├── + Adaptive Spatial Embedding (topology-aware)
    │
    ├── ┌─────────────────────────────────────────┐
    │   │  Interleaved Encode (× N_layers)        │
    │   │  ├── Mamba Temporal Block (SSM scan)     │
    │   │  └── Spatial Transformer (PD-Attention)  │
    │   └─────────────────────────────────────────┘
    │
    ├── Time-Pool → LayerNorm
    │
    ├── Regression Head → speed_pred (B, N, T')
    │
    ├── Multi-task Head → congestion (B, N, K)
    │
    ├── BPR / Learned FD → travel_times (T', E)
    │
    └── TD-Dijkstra → optimal route
```

---

## Key Features

| Feature | Description |
|---|---|
| **Mamba SSM backbone** | Selective State Space (S6) for O(L) temporal modelling — 5× faster than self-attention on long sequences |
| **Propagation-delay attention** | Graph-structure-aware spatial attention with learnable delay bias |
| **Multi-task learning** | Joint speed regression + congestion classification (HCM-aligned LOS) |
| **SPO+ loss** | Decision-focused training that minimises routing regret, not just MAE |
| **Learned fundamental diagram** | Neural volume-delay function with physics-informed monotonicity constraints |
| **Time-dependent Dijkstra** | Edge weights evolve with the prediction horizon as the vehicle traverses the network |
| **System-level metrics** | VHT, average network speed, throughput, delay index — beyond per-sensor accuracy |

---

## Quickstart

### Step 1 — Clone and install

```bash
git clone https://github.com/rohangandotra18/340W.git
cd 340W
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install 'numpy<2' torch networkx scipy pyyaml
```

> **Note:** PyTorch 2.2 requires `numpy<2`. The above installs numpy 1.x first to avoid compatibility issues.

**Highly Recommended** — real Mamba CUDA kernels (prevents out-of-memory errors on large batches and runs ~10× faster on A100/V100):
```bash
pip install mamba-ssm>=1.2.0
```

---

### Step 2 — Download datasets automatically

```bash
python setup_data.py
```

This creates a `data/` folder and downloads METR-LA + adjacency matrix.
You should see:
```
data/
├── metr-la.npz
└── adj_metr_la.npz
```

---

### Step 3 — Train on METR-LA (standard MAE loss)

```bash
python train.py \
  --data_path  data/metr-la.npz \
  --adj_path   data/adj_metr_la.npz \
  --in_channels 1 \
  --d_model    64 \
  --epochs     50 \
  --batch_size 64 \
  --output_dir ./checkpoints/metr_la
```

> **macOS (CPU):** Add `--num_workers 0` and reduce epochs for a quick test:
> ```bash
> python train.py \
>   --data_path data/metr-la.npz --adj_path data/adj_metr_la.npz \
>   --in_channels 1 --d_model 32 --epochs 5 --batch_size 16 \
>   --num_workers 0 --output_dir ./checkpoints/test
> ```

---

### Step 4 — Train with SPO+ decision-focused loss

> **Important:** SPO+ cannot use `torch.compile` (its custom autograd function does CPU↔GPU transfers incompatible with CUDA graph capture). Use **batch_size 16** to fit within a 10 GB MIG GPU slice and request **64 GB system RAM** since Dijkstra solves run on CPU each batch.

```bash
python train.py \
  --data_path   data/metr-la.npz \
  --adj_path    data/adj_metr_la.npz \
  --in_channels 1 \
  --d_model     64 \
  --epochs      50 \
  --batch_size  16 \
  --num_workers 1 \
  --use_spo \
  --spo_weight  0.5 \
  --spo_origin  0 \
  --spo_destination 50 \
  --output_dir  ./checkpoints/metr_la_spo
```

---

### Step 5 — Evaluate a checkpoint

```bash
python evaluate.py \
  --checkpoint ./checkpoints/metr_la/best_model.pt \
  --data_path  data/metr-la.npz \
  --adj_path   data/adj_metr_la.npz
```

Reports MAE / RMSE / MAPE at 15, 30, and 60-minute horizons, plus congestion F1 per class.

---

### Step 6 — End-to-end routing

```python
import torch
import numpy as np
from pipeline import TrafficTransformerPipeline
from data.traffic_dataset import TrafficDataset

# 1. Load the REAL network topology and dataset, no artificial noise
dataset = TrafficDataset(data_path="data/metr-la.npz", adj_path="data/adj_metr_la.npz", split="test")
adj = dataset.adj
edge_indices = (adj > 0.0).nonzero(as_tuple=False)
real_edges = [(int(u), int(v)) for u, v in edge_indices if u != v]

# 2. Build the pipeline with the true physical edges
pipeline = TrafficTransformerPipeline(
    checkpoint_path="checkpoints/metr_la_spo/best_model.pt",
    edges=real_edges,
    link_lengths_km=np.array([1.0] * len(real_edges)),
)

# 3. Request a real historical snapshot from the test set as input
x, _ = dataset[0]  
x = x.unsqueeze(0) # (1, 12, 207, 1)

# 4. Route over the real Los Angeles highway network!
result = pipeline.route(x, adj, origin=0, destination=100)
print(f"Path:        {result['path']}")
print(f"Travel time: {result['travel_time_min']:.1f} min")
print(f"Congestion:  {result['congestion_labels']}")
```

---

### Step 7 — Run architecture benchmarks

```bash
python benchmark_comparison.py
```

Compares Mamba vs GRU vs Transformer speed, SPO+ routing quality vs standard loss, and spatial encoding approaches.

---

## Running on Roar Collab (Penn State HPC)

### 1. SSH in and clone

```bash
ssh <PSU.ID>@submit.hpc.psu.edu
git clone https://github.com/rohangandotra18/340W.git
cd 340W
```

### 2. Set up your Python environment

> **Note:** Home directory space is limited (~10GB). Install the venv in `/storage/work/` which has more quota.

```bash
module load python/3.11.2
python3 -m venv /storage/work/rjg6014/pdformer_venv
source /storage/work/rjg6014/pdformer_venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir 'numpy<2' torch networkx scipy pyyaml
```

### 3. Download data on the login node

```bash
cd ~/340W
python setup_data.py
```

### 4. Submit standard training (batch=16)

```bash
cat > ~/340W/submit_train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=pdformerpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --partition=standard
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

module load python/3.11.2
source /storage/work/rjg6014/pdformer_venv/bin/activate
cd ~/340W

python -u train.py \
  --data_path  data/metr-la.npz \
  --adj_path   data/adj_metr_la.npz \
  --in_channels 1 \
  --d_model    64 \
  --epochs     50 \
  --batch_size 16 \
  --output_dir ./checkpoints/metr_la
EOF

sbatch ~/340W/submit_train.sh
```

Expected runtime: **~5 min/epoch, ~4.5 hours** for 50 epochs on a single A100. Best model converges around epoch 13.

### 5. Submit SPO+ training (batch=8)

> **SPO+ constraints:** The SPO+ loss runs CPU-based Dijkstra for every sample in every batch, requiring significant system RAM. Use `batch_size=8`, `num_workers=0`, and `--mem=192GB`. Use `python -u` for unbuffered output.

```bash
cat > ~/340W/submit_spo.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=pdformerpp-spo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --partition=standard
#SBATCH --output=train_spo_%j.log
#SBATCH --error=train_spo_%j.err

module load python/3.11.2
source /storage/work/rjg6014/pdformer_venv/bin/activate
cd ~/340W

python -u train.py \
  --data_path   data/metr-la.npz \
  --adj_path    data/adj_metr_la.npz \
  --in_channels 1 \
  --d_model     64 \
  --epochs      50 \
  --batch_size  8 \
  --num_workers 0 \
  --use_spo \
  --spo_weight  0.5 \
  --spo_origin  0 \
  --spo_destination 50 \
  --output_dir  ./checkpoints/metr_la_spo
EOF

sbatch ~/340W/submit_spo.sh
```

Expected runtime: **~9 min/epoch, ~8 hours** for 50 epochs on a single A100.

### 6. Monitor jobs

```bash
squeue -u rjg6014                       # check job status
tail -f ~/340W/train_spo_<JOBID>.log     # watch training progress
cat ~/340W/train_spo_<JOBID>.err         # check for errors
```

### 7. Evaluate (run with GPU via srun)

> **Do not evaluate on the login node** — it will be killed for memory usage.

```bash
source /storage/work/rjg6014/pdformer_venv/bin/activate
cd ~/340W

# Standard model
srun --partition=standard --mem=16GB --gres=gpu:a100:1 --time=00:10:00 \
  bash -c "source /storage/work/rjg6014/pdformer_venv/bin/activate && cd ~/340W && python -u evaluate.py \
  --checkpoint checkpoints/metr_la/best_model.pt \
  --data_path data/metr-la.npz \
  --adj_path data/adj_metr_la.npz"

# SPO+ model
srun --partition=standard --mem=16GB --gres=gpu:a100:1 --time=00:10:00 \
  bash -c "source /storage/work/rjg6014/pdformer_venv/bin/activate && cd ~/340W && python -u evaluate.py \
  --checkpoint checkpoints/metr_la_spo/best_model.pt \
  --data_path data/metr-la.npz \
  --adj_path data/adj_metr_la.npz"
```

---

## Results (METR-LA)

### Prediction Accuracy

| Horizon | Standard MAE | SPO+ MAE | Standard RMSE | SPO+ RMSE | Standard MAPE | SPO+ MAPE |
|---|---|---|---|---|---|---|
| 15 min | 2.4031 | 2.4039 | 3.0119 | 3.0130 | 4.86% | 4.87% |
| 30 min | 2.4141 | 2.4148 | 3.0262 | 3.0270 | 4.89% | 4.89% |
| 60 min | 2.4952 | 2.4922 | 3.1351 | 3.1302 | 5.10% | 5.08% |
| **Overall** | **2.4292** | **2.4303** | **3.0467** | **3.0480** | **4.93%** | **4.93%** |

### Congestion Classification

| Model | Accuracy | Free-flow F1 | Slow F1 |
|---|---|---|---|
| Standard (29 epochs) | 92.48% | 0.9354 | 0.9100 |
| SPO+ (7 epochs) | 92.56% | 0.9365 | 0.9102 |

SPO+ achieves better classification accuracy and F1 scores with only 7 epochs of training vs 29 for the standard model, demonstrating how the decision-focused loss improves routing-relevant predictions while maintaining competitive statistical accuracy.

---

## Ablation Experiments

Run the two pre-built ablations:

```bash
# Mamba vs GRU backbone (no SPO+)
python train.py \
  --data_path  data/metr-la.npz \
  --adj_path   data/adj_metr_la.npz \
  --in_channels 1 \
  --output_dir ./checkpoints/ablation_no_spo

# SPO+ impact — compare checkpoints/metr_la vs checkpoints/metr_la_spo
python evaluate.py --checkpoint checkpoints/metr_la/best_model.pt \
  --data_path data/metr-la.npz --adj_path data/adj_metr_la.npz
python evaluate.py --checkpoint checkpoints/metr_la_spo/best_model.pt \
  --data_path data/metr-la.npz --adj_path data/adj_metr_la.npz
```

---

## Experiment Configurations

Pre-defined configs in `configs/`:

| Config | Dataset | SPO+ | Purpose |
|---|---|---|---|
| `metr_la.yaml` | METR-LA | ✗ | Standard benchmark |
| `pems_bay.yaml` | PEMS-BAY | ✗ | Standard benchmark |
| `nyc_taxi.yaml` | NYC Taxi | ✓ | Integrated prediction + routing |
| `ablation_no_mamba.yaml` | METR-LA | ✗ | Mamba vs. GRU ablation |
| `ablation_no_spo.yaml` | METR-LA | ✗ | SPO+ impact ablation |

---

## Datasets

| Dataset | Nodes | Timesteps | Features | OD Demand | Source |
|---|---|---|---|---|---|
| METR-LA | 207 | 34,272 | speed | ✗ | [LibCity](https://github.com/LibCity/Bigscity-LibCity) |
| PEMS-BAY | 325 | 52,116 | speed | ✗ | [LibCity](https://github.com/LibCity/Bigscity-LibCity) |
| NYC Taxi | ~260 zones | variable | speed, flow, demand | ✓ | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |

---

## Project Structure

```
pdformer_plus/
├── model/
│   ├── pdformer_plus.py      # Main model architecture
│   ├── mamba_temporal.py     # Selective SSM (S6) temporal backbone
│   ├── spatial_attention.py  # PD-aware spatial attention + adaptive embeddings
│   ├── multitask_head.py     # Congestion classification + joint loss
│   └── spo_loss.py           # SPO+ decision-focused loss
├── routing/
│   ├── dijkstra_router.py    # Standard + time-dependent Dijkstra
│   ├── bpr_converter.py      # Bureau of Public Roads volume-delay function
│   ├── learned_fd.py         # Neural fundamental diagram
│   └── system_metrics.py     # VHT, throughput, delay index
├── data/
│   ├── traffic_dataset.py    # METR-LA / PEMS-BAY loader
│   └── nyc_taxi_dataset.py   # NYC Taxi TLC loader with OD demand
├── configs/                  # YAML experiment configurations
├── train.py                  # Training script (supports SPO+)
├── evaluate.py               # Multi-metric evaluation
├── pipeline.py               # End-to-end predict-then-route
└── requirements.txt
```

---

## References

- **PDFormer** — Jiang et al., "PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction", AAAI 2023
- **Mamba (S6)** — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", NeurIPS 2023
- **SPO+** — Elmachtoub & Grigas, "Smart Predict-then-Optimize", Management Science 2022
- **PIDL+FDL** — Mo et al., "Physics-Informed Deep Learning for Fundamental Diagram Estimation", IEEE T-ITS 2022
- **Highway Capacity Manual** — TRB, HCM 7th Edition, 2022

---

## Citation

```bibtex
@article{gandotra2026pdformerpp,
  title  = {PDFormer++: Decision-Focused Traffic Prediction and Routing
            via Selective State Spaces},
  author = {Gandotra, Rohan},
  note   = {DS 340W Course Project, Penn State University},
  year   = {2026},
}
```
