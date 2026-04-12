"""
Multi-task head: speed/flow regression branch + congestion classification branch.
Total loss = λ₁ · MAE_regression + λ₂ · CrossEntropy_classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_congestion_labels(speeds: torch.Tensor, ffs: float = 65.0) -> torch.Tensor:
    """
    Convert speed values to 3-class congestion labels via speed/FFS ratio.

    Thresholds follow China's national standard generalised to any road type:
        ≥ 0.8 · FFS  →  0  (free-flow)
        ≥ 0.4 · FFS  →  1  (slow / basically smooth)
        <  0.4 · FFS →  2  (congested)

    Args:
        speeds: arbitrary-shape float tensor (mph or km/h)
        ffs:    free-flow speed in the same unit
    Returns:
        long tensor of same leading shape with class indices
    """
    ratio = speeds / max(ffs, 1e-6)
    labels = torch.zeros_like(speeds, dtype=torch.long)
    labels[ratio < 0.8] = 1
    labels[ratio < 0.4] = 2
    return labels


class FocalLoss(nn.Module):
    """
    Focal Loss dynamically down-weights easy, common examples (like free-flow)
    and aggressively penalizes the model for missing rare minority classes (congestion).
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiTaskHead(nn.Module):
    """
    Branches from the shared encoder embedding:
      - Classification: Linear → GELU → Linear → n_classes logits
    The regression output lives in PDFormerPlusPlus.temporal_out (shared decoder).
    """

    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, embed: torch.Tensor) -> dict:
        """
        embed: (B, N, d_model)
        Returns: {'congestion': (B, N, n_classes)}
        """
        return {"congestion": self.classifier(embed)}


class TrafficLoss(nn.Module):
    """
    Combined regression + classification loss.

    Loss = λ₁ · MAE(speed_pred, speed_true) + λ₂ · CE(congestion_logits, labels)

    Congestion labels are derived on-the-fly from the true speed values
    so no separate label preprocessing step is needed.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.1,
        ffs: float = 65.0,
        scaler_mean: float = 0.0,
        scaler_std: float = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ffs = ffs
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.ce = FocalLoss(gamma=2.0)

    def forward(
        self,
        speed_pred: torch.Tensor,  # (B, T', N)
        speed_true: torch.Tensor,  # (B, T', N)
        congestion_logits: torch.Tensor,  # (B, N, n_classes)
    ) -> dict:
        mae = F.smooth_l1_loss(speed_pred, speed_true, beta=1.0)

        # Denormalise speeds to compute physical congestion thresholds
        speed_true_denorm = speed_true * self.scaler_std + self.scaler_mean

        # Derive congestion labels from mean speed over prediction horizon
        speed_mean = speed_true_denorm.mean(dim=1)  # (B, N)
        labels = compute_congestion_labels(speed_mean, self.ffs)  # (B, N)

        B, N, n_cls = congestion_logits.shape
        ce = self.ce(congestion_logits.reshape(B * N, n_cls), labels.reshape(B * N))

        total = self.lambda1 * mae + self.lambda2 * ce
        return {"total": total, "mae": mae, "ce": ce}
