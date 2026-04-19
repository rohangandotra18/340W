"""PDFormer++ training script module docstring."""

from __future__ import annotations
from model.spo_loss import SPOPlusTrafficLoss
from model.multitask_head import TrafficLoss
from model.pdformer_plus import PDFormerPlusPlus
from data.traffic_dataset import TrafficDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path
import time
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
Training script for PDFormer++.

Example (METR-LA):
    python train.py \
        --data_path  /path/to/metr-la.npz \
        --adj_path   /path/to/adj_metr_la.npz \
        --n_nodes    207 \
        --in_channels 3 \
        --d_model 64 \
        --epochs 150 \
        --batch_size 64 \
        --output_dir ./checkpoints

On Roar Collab A100 this trains in < 1 hour for 150 epochs on METR-LA.
"""


# ── helpers ──────────────────────────────────────────────────────────────────


def _has_nan_params(model: nn.Module) -> bool:
    """Check if any model parameter or gradient contains NaN/Inf."""
    for p in model.parameters():
        if p.data is not None and (torch.isnan(p.data).any() or torch.isinf(p.data).any()):
            return True
    return False


def _has_nan_grads(model: nn.Module) -> bool:
    """Check if any computed gradient contains NaN/Inf."""
    for p in model.parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            return True
    return False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TrafficLoss,
    adj: torch.Tensor,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss = total_mae = n = 0
    nan_batches = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)  # (B, in_T, N, C)
        y = y.to(device, non_blocking=True)  # (B, out_T, N)

        optimizer.zero_grad(set_to_none=True)

        # AMP is unconditionally disabled: the pure-PyTorch Mamba SSM scan
        # accumulates recurrent states that overflow float16 range.
        with torch.amp.autocast("cuda", enabled=False):
            out = model(x, adj)

        # Check for NaN in model output BEFORE computing loss
        speed_pred = out["speed_pred"].permute(0, 2, 1).float()  # (B, T', N)
        if torch.isnan(speed_pred).any() or torch.isnan(out["congestion"]).any():
            nan_batches += 1
            if nan_batches <= 3:
                print(f"    [Batch {i:3d}] WARNING: NaN in model output — skipping batch", flush=True)
            continue

        # Criterion runs outside autocast: SPO+ does CPU↔GPU transfers (numpy
        # Dijkstra) that crash if the inductor has CUDA graph capture active.
        losses = criterion(speed_pred, y.float(), out["congestion"].float())

        # Skip batches that produce NaN loss (prevents corrupting model weights)
        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            nan_batches += 1
            if nan_batches <= 3:
                print(f"    [Batch {i:3d}] WARNING: NaN/Inf loss — skipping batch", flush=True)
            continue

        losses["total"].backward()

        # CRITICAL: check gradients for NaN AFTER backward(). This is the
        # main source of silent weight corruption — a finite loss can still
        # produce NaN gradients through divisions, exps, and log in the
        # backward graph. Applying NaN gradients via optimizer.step()
        # permanently corrupts model weights.
        if _has_nan_grads(model):
            nan_batches += 1
            if nan_batches <= 3:
                print(f"    [Batch {i:3d}] WARNING: NaN gradients after backward — skipping update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_mae += losses["mae"].item()
        n += 1

        # Periodic progress update to SLURM output
        if i % 100 == 0 and i > 0:
            print(
                f"    [Batch {i:3d}/{len(loader)}] Loss: {losses['total'].item():.4f}",
                flush=True,
            )

    if nan_batches > 0:
        print(f"    ⚠ {nan_batches}/{len(loader)} batches skipped due to NaN", flush=True)

    return (total_loss / max(n, 1), total_mae / max(n, 1))


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    dataset: TrafficDataset,
) -> tuple[float, float]:
    model.eval()
    total_mae = total_rmse = n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Ensure AMP is disabled here too to prevent NaN in pure PyTorch Mamba
        with torch.amp.autocast("cuda", enabled=False):
            out = model(x, adj)
        speed_pred = out["speed_pred"].permute(0, 2, 1)  # (B, T', N)

        # Denormalise before computing metrics
        speed_pred = dataset.denormalize_speed(speed_pred)
        y_real = dataset.denormalize_speed(y)

        total_mae += (speed_pred - y_real).abs().mean().item()
        total_rmse += ((speed_pred - y_real) ** 2).mean().sqrt().item()
        n += 1

    return total_mae / max(n, 1), total_rmse / max(n, 1)


# ── main ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ── data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    common = dict(
        data_path=args.data_path,
        adj_path=args.adj_path,
        in_horizon=args.in_horizon,
        out_horizon=args.out_horizon,
    )
    train_ds = TrafficDataset(**common, split="train")
    val_ds = TrafficDataset(**common, split="val")

    _loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        _loader_kw.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **_loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_kw,
    )

    adj = train_ds.adj.to(device)
    n_nodes = train_ds.n_nodes
    print(
        f"  Nodes: {n_nodes}  |  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}"
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = PDFormerPlusPlus(
        n_nodes=n_nodes,
        in_channels=args.in_channels,
        d_model=args.d_model,
        out_horizon=args.out_horizon,
        n_temporal_layers=args.n_temporal_layers,
        n_spatial_layers=args.n_spatial_layers,
        n_heads=args.n_heads,
        d_state=args.d_state,
        n_classes=args.n_classes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Compile model for massive speedups if using PyTorch 2.0+ on CUDA.
    # Skip when SPO+ is active: its custom autograd function does CPU↔GPU
    # transfers inside forward() that are incompatible with CUDA graph tracing.
    if device.type == "cuda" and hasattr(torch, "compile") and not args.use_spo:
        print("  Skipping torch.compile() due to A100 MIG crashes (CUDA misaligned address).")
        # try:
        #     model = torch.compile(model)
        # except Exception as e:
        #     print(f"  Warning: torch.compile() failed ({e}), continuing without it.")

    # ── optimisation ─────────────────────────────────────────────────────
    if args.use_spo:
        # Extract edges from the adjacency matrix (exclude self-loops)
        edge_indices = (adj > 0.0).nonzero(as_tuple=False)
        edges = [(int(u), int(v)) for u, v in edge_indices if u != v]
        print(f"  SPO+ edge extraction: {len(edges)} edges from adj > 0 "
              f"(total non-zero: {(adj > 0).sum().item()}, "
              f"self-loops: {(adj > 0).sum().item() - len(edges)})")

        # Fallback: if adj is identity-like (no off-diagonal edges), build a
        # k-nearest-neighbor graph from row-normalised adjacency weights.
        # This happens when the adj file only contains self-loops.
        if len(edges) == 0:
            print("  ⚠ No off-diagonal edges in adj — building kNN graph "
                  "(k=4) as SPO fallback …", flush=True)
            k = min(4, n_nodes - 1)
            # Use the raw adj; even if all zeros off-diag, build a simple
            # ring/chain topology so SPO+ has something to route on
            adj_cpu = adj.cpu()
            # Zero out diagonal so it doesn't dominate topk
            adj_no_diag = adj_cpu.clone()
            adj_no_diag.fill_diagonal_(0.0)

            if adj_no_diag.sum() > 0:
                # Use actual weights to pick neighbours
                _, topk_idx = adj_no_diag.topk(k, dim=1)
            else:
                # Adj is truly identity — build a simple chain graph
                # Connect each node i → i±1, i±2 (wrap around)
                print("    → adj is pure identity; creating chain topology", flush=True)
                topk_idx = torch.zeros(n_nodes, k, dtype=torch.long)
                for i in range(n_nodes):
                    neighbours = [(i + d) % n_nodes for d in range(1, k + 1)]
                    topk_idx[i] = torch.tensor(neighbours)

            edge_set = set()
            for i in range(n_nodes):
                for j in topk_idx[i].tolist():
                    if i != j:
                        edge_set.add((i, j))
                        edge_set.add((j, i))  # make undirected
            edges = sorted(edge_set)
            print(f"    → Built kNN graph with {len(edges)} edges", flush=True)

        # Verify destination is reachable from origin; auto-select if not
        from collections import deque as _deque
        _origin = args.spo_origin
        _dest = min(args.spo_destination, n_nodes - 1)
        _adj_spo = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            _adj_spo[u].append(v)
        _visited = {}
        _q = _deque([_origin])
        _visited[_origin] = 0
        while _q:
            _nd = _q.popleft()
            for _nb in _adj_spo[_nd]:
                if _nb not in _visited:
                    _visited[_nb] = _visited[_nd] + 1
                    _q.append(_nb)
        if _dest not in _visited:
            # Pick the farthest reachable node as destination
            if len(_visited) > 1:
                _dest_new = max(_visited, key=_visited.get)
                print(f"  ⚠ SPO+ destination {_dest} unreachable from origin {_origin}. "
                      f"Auto-selecting node {_dest_new} (farthest reachable, "
                      f"{_visited[_dest_new]} hops).", flush=True)
                _dest = _dest_new
            else:
                print(f"  ⚠ SPO+ origin {_origin} has no outgoing edges!", flush=True)
        else:
            print(f"  SPO+ routing: origin={_origin} → destination={_dest} "
                  f"({_visited[_dest]} hops)", flush=True)

        criterion = SPOPlusTrafficLoss(
            edges=edges,
            n_nodes=n_nodes,
            origin=_origin,
            destination=_dest,
            lambda_mae=args.lambda1,
            lambda_ce=args.lambda2,
            lambda_spo=args.spo_weight,
            ffs=args.ffs,
            scaler_mean=float(train_ds.mean[0, 0, 0]),
            scaler_std=float(train_ds.std[0, 0, 0]),
        )
        print(f"  Using SPO+ decision-focused loss (λ_spo={args.spo_weight})")
    else:
        criterion = TrafficLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            ffs=args.ffs,
            scaler_mean=float(train_ds.mean[0, 0, 0]),
            scaler_std=float(train_ds.std[0, 0, 0]),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler = None  # Disabled AMP because it causes NaN overflow in pure PyTorch Mamba

    # ── training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    nan_epoch_count = 0
    print(f"\nTraining for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Check for corrupted weights at epoch start
        if _has_nan_params(model):
            nan_epoch_count += 1
            print(f"Ep {epoch:3d}/{args.epochs} | ⚠ Model weights contain NaN!", flush=True)
            best_ckpt = out_dir / "best_model.pt"
            if best_ckpt.exists():
                print(f"  → Rolling back to last best checkpoint …", flush=True)
                ckpt = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Lower LR after rollback to prevent re-divergence
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                print(f"  → Rolled back. LR halved to {optimizer.param_groups[0]['lr']:.2e}", flush=True)
            else:
                print(f"  → No checkpoint to roll back to — reinitialising weights", flush=True)
                model._init_weights()
            if nan_epoch_count >= 5:
                print(f"  ✗ Too many NaN rollbacks ({nan_epoch_count}) — stopping early.", flush=True)
                break
            continue

        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, adj, device, scaler
        )
        val_mae, val_rmse = eval_epoch(model, val_loader, adj, device, val_ds)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | Train MAE {train_mae:.4f} | "
            f"Val MAE {val_mae:.4f} | Val RMSE {val_rmse:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Only save if val_mae is a real number (not NaN/inf)
        if not (val_mae != val_mae) and val_mae < best_val_mae:  # NaN != NaN is True
            best_val_mae = val_mae
            nan_epoch_count = 0  # Reset NaN counter on good save
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "args": {
                        **vars(args),
                        "n_nodes": n_nodes,
                        "scaler_mean": float(train_ds.mean[0, 0, 0]),
                        "scaler_std": float(train_ds.std[0, 0, 0]),
                    },
                },
                out_dir / "best_model.pt",
            )
            print(f"  ✓ Saved best (Val MAE {best_val_mae:.4f})")

    print(f"\nDone. Best Val MAE: {best_val_mae:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PDFormer++")

    # Data
    p.add_argument("--data_path", required=True)
    p.add_argument("--adj_path", default=None)
    p.add_argument(
        "--in_channels", type=int, default=3, help="Feature channels per node"
    )
    p.add_argument(
        "--in_horizon", type=int, default=12, help="Input timesteps (12 = 1hr @ 5min)"
    )
    p.add_argument("--out_horizon", type=int, default=12, help="Prediction horizon")

    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_temporal_layers", type=int, default=2)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_state", type=int, default=16, help="Mamba SSM state dim")
    p.add_argument("--n_classes", type=int, default=3, help="Congestion classes")
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda1", type=float, default=1.0, help="Regression loss weight")
    p.add_argument(
        "--lambda2", type=float, default=0.1, help="Classification loss weight"
    )
    p.add_argument(
        "--ffs", type=float, default=65.0, help="Free-flow speed (km/h or mph)"
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="./checkpoints")

    # SPO+ decision-focused loss
    p.add_argument(
        "--use_spo", action="store_true", help="Enable SPO+ decision-focused loss"
    )
    p.add_argument(
        "--spo_weight", type=float, default=0.5, help="Weight for SPO+ routing regret"
    )
    p.add_argument(
        "--spo_origin", type=int, default=0, help="Source node for SPO+ routing"
    )
    p.add_argument(
        "--spo_destination",
        type=int,
        default=50,
        help="Destination node for SPO+ routing",
    )

    main(p.parse_args())
