"""
Evaluation script for PDFormer++.

Reports:
  1. Per-horizon MAE / RMSE / MAPE at 15-min, 30-min, and 60-min horizons
     following the standard LibCity / LargeST evaluation protocol.
  2. Congestion classification accuracy (per-class precision, recall, F1).
  3. System-level metrics (VHT, avg network speed, throughput) when edge
     information is available.

Usage:
    python evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data_path  /path/to/metr-la.npz \
        --adj_path   /path/to/adj_metr_la.npz
"""

from model.multitask_head import compute_congestion_labels
from model.pdformer_plus import PDFormerPlusPlus
from data.traffic_dataset import TrafficDataset
from torch.utils.data import DataLoader
import torch
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


HORIZONS_STEPS = [3, 6, 12]  # 5-min steps → 15 / 30 / 60 min


# ── Prediction metrics ───────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_prediction(
    model: PDFormerPlusPlus,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    dataset: TrafficDataset,
    ffs: float = 65.0,
) -> None:
    model.eval()

    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    all_cong_pred: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x, adj)
        speed_pred = out["speed_pred"].permute(0, 2, 1).cpu()  # (B, T', N)
        cong_logits = out["congestion"].cpu()  # (B, N, n_cls)

        all_pred.append(speed_pred)
        all_true.append(y)

        # Congestion predictions and labels
        cong_pred = cong_logits.argmax(dim=-1)  # (B, N)
        all_cong_pred.append(cong_pred)

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)

    # Denormalise
    pred = dataset.denormalize_speed(pred)
    true = dataset.denormalize_speed(true)

    # ── Per-horizon metrics ───────────────────────────────────
    print(f"\n{'Horizon':<12} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>12}")
    print("─" * 48)

    for h in HORIZONS_STEPS:
        if h > pred.shape[1]:
            continue
        p = pred[:, h - 1, :]
        t = true[:, h - 1, :]
        mae = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        mape = ((p - t).abs() / t.clamp(min=1e-6)).mean().item() * 100.0
        label = f"{h * 5} min"
        print(f"{label:<12} {mae:>10.4f} {rmse:>10.4f} {mape:>11.2f}%")

    # Overall
    mae = (pred - true).abs().mean().item()
    rmse = ((pred - true) ** 2).mean().sqrt().item()
    mape = ((pred - true).abs() / true.clamp(min=1e-6)).mean().item() * 100.0
    print("─" * 48)
    print(f"{'Overall':<12} {mae:>10.4f} {rmse:>10.4f} {mape:>11.2f}%\n")

    # ── Congestion classification metrics ─────────────────────
    cong_pred_all = torch.cat(all_cong_pred, dim=0)  # (total, N)
    # Derive true labels from mean speed over horizon
    true_mean = true.mean(dim=1)  # (total, N)
    cong_true_all = compute_congestion_labels(true_mean, ffs)

    n_classes = 3
    class_names = ["Free-flow", "Slow", "Congested"]
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("─" * 58)

    total_tp = total_fp = total_fn = 0
    for c in range(n_classes):
        tp = ((cong_pred_all == c) & (cong_true_all == c)).sum().item()
        fp = ((cong_pred_all == c) & (cong_true_all != c)).sum().item()
        fn = ((cong_pred_all != c) & (cong_true_all == c)).sum().item()
        support = (cong_true_all == c).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(
            f"{class_names[c]:<14} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}"
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall accuracy
    acc = (cong_pred_all == cong_true_all).float().mean().item() * 100
    print("─" * 58)
    print(f"{'Accuracy':<14} {acc:>30.2f}%\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt["args"]

    test_ds = TrafficDataset(
        data_path=args.data_path,
        adj_path=args.adj_path,
        in_horizon=a["in_horizon"],
        out_horizon=a["out_horizon"],
        split="test",
    )
    loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )
    adj = test_ds.adj.to(device)

    model = PDFormerPlusPlus(
        n_nodes=a["n_nodes"],
        in_channels=a["in_channels"],
        d_model=a["d_model"],
        out_horizon=a["out_horizon"],
        n_temporal_layers=a["n_temporal_layers"],
        n_spatial_layers=a["n_spatial_layers"],
        n_heads=a["n_heads"],
        d_state=a["d_state"],
        n_classes=a["n_classes"],
    ).to(device)

    # Handle models trained with torch.compile() by stripping the _orig_mod. prefix
    state_dict = ckpt["model_state_dict"]
    uncompiled_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(uncompiled_state_dict)
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']}  (Val MAE {ckpt['val_mae']:.4f})"
    )

    ffs = a.get("ffs", args.ffs)
    evaluate_prediction(model, loader, adj, device, test_ds, ffs=ffs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--adj_path", default=None)
    p.add_argument(
        "--ffs", type=float, default=65.0, help="Free-flow speed for congestion labels"
    )
    main(p.parse_args())
