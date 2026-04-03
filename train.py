"""PDFormer++ training script module docstring."""
from __future__ import annotations
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
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.traffic_dataset import TrafficDataset
from model.pdformer_plus import PDFormerPlusPlus
from model.multitask_head import TrafficLoss
from model.spo_loss import SPOPlusTrafficLoss


# ── helpers ──────────────────────────────────────────────────────────────────

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

    for x, y in loader:
        x = x.to(device, non_blocking=True)   # (B, in_T, N, C)
        y = y.to(device, non_blocking=True)   # (B, out_T, N)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x, adj)
            speed_pred = out["speed_pred"].permute(0, 2, 1)   # (B, T', N)
            losses = criterion(speed_pred, y, out["congestion"])

        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += losses["total"].item()
        total_mae  += losses["mae"].item()
        n += 1

    return total_loss / n, total_mae / n


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

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x, adj)
        speed_pred = out["speed_pred"].permute(0, 2, 1)   # (B, T', N)

        # Denormalise before computing metrics
        speed_pred = dataset.denormalize_speed(speed_pred)
        y_real     = dataset.denormalize_speed(y)

        total_mae  += (speed_pred - y_real).abs().mean().item()
        total_rmse += ((speed_pred - y_real) ** 2).mean().sqrt().item()
        n += 1

    return total_mae / n, total_rmse / n


# ── main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    common = dict(
        data_path=args.data_path,
        adj_path=args.adj_path,
        in_horizon=args.in_horizon,
        out_horizon=args.out_horizon,
    )
    train_ds = TrafficDataset(**common, split="train")
    val_ds   = TrafficDataset(**common, split="val")

    _loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        _loader_kw.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **_loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **_loader_kw,
    )

    adj = train_ds.adj.to(device)
    n_nodes = train_ds.n_nodes
    print(f"  Nodes: {n_nodes}  |  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

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

    # Compile model for massive speedups if using PyTorch 2.0+ on CUDA
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile() for speed...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  Warning: torch.compile() failed ({e}), continuing without it.")

    # ── optimisation ─────────────────────────────────────────────────────
    if args.use_spo:
        # Extract REAL edges from the adjacency matrix (no artificial data)
        edge_indices = (adj > 0.0).nonzero(as_tuple=False)
        edges = [(int(u), int(v)) for u, v in edge_indices if u != v]
        criterion = SPOPlusTrafficLoss(
            edges=edges,
            n_nodes=n_nodes,
            origin=args.spo_origin,
            destination=min(args.spo_destination, n_nodes - 1),
            lambda_mae=args.lambda1,
            lambda_ce=args.lambda2,
            lambda_spo=args.spo_weight,
            ffs=args.ffs,
        )
        print(f"  Using SPO+ decision-focused loss (λ_spo={args.spo_weight})")
    else:
        criterion = TrafficLoss(lambda1=args.lambda1, lambda2=args.lambda2, ffs=args.ffs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    print(f"\nTraining for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, adj, device, scaler)
        val_mae, val_rmse     = eval_epoch(model, val_loader, adj, device, val_ds)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | Train MAE {train_mae:.4f} | "
            f"Val MAE {val_mae:.4f} | Val RMSE {val_rmse:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "args": {**vars(args), "n_nodes": n_nodes},
                },
                out_dir / "best_model.pt",
            )
            print(f"  ✓ Saved best (Val MAE {best_val_mae:.4f})")

    print(f"\nDone. Best Val MAE: {best_val_mae:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PDFormer++")

    # Data
    p.add_argument("--data_path",   required=True)
    p.add_argument("--adj_path",    default=None)
    p.add_argument("--in_channels", type=int, default=3,  help="Feature channels per node")
    p.add_argument("--in_horizon",  type=int, default=12, help="Input timesteps (12 = 1hr @ 5min)")
    p.add_argument("--out_horizon", type=int, default=12, help="Prediction horizon")

    # Model
    p.add_argument("--d_model",            type=int,   default=64)
    p.add_argument("--n_temporal_layers",  type=int,   default=2)
    p.add_argument("--n_spatial_layers",   type=int,   default=2)
    p.add_argument("--n_heads",            type=int,   default=8)
    p.add_argument("--d_state",            type=int,   default=16,  help="Mamba SSM state dim")
    p.add_argument("--n_classes",          type=int,   default=3,   help="Congestion classes")
    p.add_argument("--dropout",            type=float, default=0.1)

    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--lambda1",       type=float, default=1.0,  help="Regression loss weight")
    p.add_argument("--lambda2",       type=float, default=0.1,  help="Classification loss weight")
    p.add_argument("--ffs",           type=float, default=65.0, help="Free-flow speed (km/h or mph)")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    default="./checkpoints")

    # SPO+ decision-focused loss
    p.add_argument("--use_spo",        action="store_true",     help="Enable SPO+ decision-focused loss")
    p.add_argument("--spo_weight",     type=float, default=0.5, help="Weight for SPO+ routing regret")
    p.add_argument("--spo_origin",     type=int,   default=0,   help="Source node for SPO+ routing")
    p.add_argument("--spo_destination",type=int,   default=50,  help="Destination node for SPO+ routing")

    main(p.parse_args())
