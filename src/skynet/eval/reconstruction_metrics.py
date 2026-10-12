"""Reconstruction metrics: Chamfer Distance and FID."""

from __future__ import annotations

import torch


def _check_cloud(name: str, points: torch.Tensor) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(points.shape)}")
    if len(points) == 0:
        raise ValueError(f"{name} is empty")


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Symmetric Chamfer Distance between two point clouds of shape (N, 3).

    Mean of nearest-neighbour Euclidean distances in both directions, summed
    (unsquared; some papers use squared distances, so state which one is used).
    Units are those of the input coordinates. Memory is O(N * M), so subsample large
    LiDAR clouds to a fixed size before calling this.
    """
    _check_cloud("pred", pred)
    _check_cloud("target", target)
    # The default compute mode uses matrix multiplication, which is slightly inexact
    # (identical clouds would not give exactly 0); the exact mode is deterministic.
    dists = torch.cdist(pred, target, compute_mode="donot_use_mm_for_euclid_dist")
    return float(dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean())


@torch.no_grad()
def fid_score(
    real: torch.Tensor,
    fake: torch.Tensor,
    feature: int = 2048,
    batch_size: int = 64,
    device: str = "cpu",
) -> float:
    """FID between two image batches of shape (B, 3, H, W), float in [0, 1].

    Needs well over 2048 samples per set for a reliable value with feature=2048;
    with fewer, the result is only indicative.
    """
    # Imported here so that Chamfer-only code does not pay for the heavy dependency.
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
    for i in range(0, len(real), batch_size):
        metric.update(real[i : i + batch_size].to(device), real=True)
    for i in range(0, len(fake), batch_size):
        metric.update(fake[i : i + batch_size].to(device), real=False)
    return float(metric.compute())
