"""Planning metrics: L2 error and collision rate at 1s, 2s, 3s (open-loop)"""

from __future__ import annotations

import math

import torch

# nuScenes keyframes are 2 Hz, so a horizon in seconds is 2 * seconds steps.
HORIZON_STEPS = {"1s": 2, "2s": 4, "3s": 6}

# Ego vehicle footprint in metres (nuScenes Renault Zoe, as used by UniAD/VAD).
EGO_LENGTH_M = 4.084
EGO_WIDTH_M = 1.85


def _with_avg(values: dict[str, float]) -> dict[str, float]:
    """Add the 'avg' entry: the plain mean of the per-horizon values."""
    out = dict(values)
    out["avg"] = sum(values.values()) / len(values)
    return out


def l2_error(pred: torch.Tensor, gt: torch.Tensor, cumulative: bool = False) -> dict[str, float]:
    """L2 distance between predicted and ground-truth ego positions, in metres.

    Parameters
    ----------
    pred, gt : torch.Tensor
        Ego x/y positions of shape (T, 2) for one scene or (B, T, 2) for a batch,
        with T >= 6 (2 Hz, so 6 steps = 3 s).
    cumulative : bool
        False: the value at horizon h is the error at step 2*h only.
        True: the value at horizon h is the mean error over all steps up to 2*h.
        With a batch, errors are averaged over scenes.

    Returns
    -------
    dict
        Keys "1s", "2s", "3s" and "avg" (mean of the three horizons).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.ndim not in (2, 3) or pred.shape[-1] != 2:
        raise ValueError(f"expected (T, 2) or (B, T, 2), got {tuple(pred.shape)}")
    if pred.shape[-2] < max(HORIZON_STEPS.values()):
        raise ValueError(f"need at least {max(HORIZON_STEPS.values())} steps")

    dist = torch.linalg.norm(pred - gt, dim=-1)  # (T,) or (B, T)
    per_step = dist.reshape(-1, dist.shape[-1]).mean(dim=0)  # mean over scenes
    values = {}
    for name, n in HORIZON_STEPS.items():
        values[name] = float(per_step[:n].mean() if cumulative else per_step[n - 1])
    return _with_avg(values)


def _corners(cx: float, cy: float, length: float, width: float, yaw: float):
    """Corners of a rotated rectangle, as four (x, y) tuples in order."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2, width / 2
    return [
        (cx + c * dx - s * dy, cy + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def _rects_overlap(a, b) -> bool:
    """Separating-axis test for two rectangles; touching counts as overlap."""
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = -(y2 - y1), x2 - x1  # normal to this edge
            proj_a = [nx * x + ny * y for x, y in a]
            proj_b = [nx * x + ny * y for x, y in b]
            if max(proj_a) < min(proj_b) or max(proj_b) < min(proj_a):
                return False  # found a separating axis
    return True


def collision_rate(
    pred_traj: torch.Tensor,
    gt_boxes: list[list[torch.Tensor]],
    ego_size: tuple[float, float] = (EGO_LENGTH_M, EGO_WIDTH_M),
    cumulative: bool = False,
) -> dict[str, float]:
    """Fraction of scenes where the predicted ego footprint hits a ground-truth box.

    Parameters
    ----------
    pred_traj : torch.Tensor
        Predicted ego poses (x, y, yaw) of shape (B, T, 3), T >= 6.
    gt_boxes : list of list of torch.Tensor
        ``gt_boxes[b][t]`` is a (K, 5) tensor of objects at step t of scene b, each
        row (cx, cy, length, width, yaw); K may be 0.
    ego_size : tuple
        Ego (length, width) in metres.
    cumulative : bool
        False: a scene collides at horizon h if the ego hits something at step 2*h.
        True: if it hits something at any step up to 2*h.

    Returns
    -------
    dict
        Keys "1s", "2s", "3s" and "avg"; values are fractions in [0, 1].
    """
    if pred_traj.ndim != 3 or pred_traj.shape[-1] != 3:
        raise ValueError(f"expected (B, T, 3), got {tuple(pred_traj.shape)}")
    batch, steps = pred_traj.shape[0], max(HORIZON_STEPS.values())
    if pred_traj.shape[1] < steps:
        raise ValueError(f"need at least {steps} steps")
    if len(gt_boxes) != batch:
        raise ValueError("gt_boxes must have one entry per scene")

    hits = [[False] * steps for _ in range(batch)]
    for b in range(batch):
        for t in range(steps):
            x, y, yaw = pred_traj[b, t].tolist()
            ego = _corners(x, y, ego_size[0], ego_size[1], yaw)
            hits[b][t] = any(_rects_overlap(ego, _corners(*box)) for box in gt_boxes[b][t].tolist())

    values = {}
    for name, n in HORIZON_STEPS.items():
        flags = [any(h[:n]) if cumulative else h[n - 1] for h in hits]
        values[name] = sum(flags) / batch
    return _with_avg(values)
