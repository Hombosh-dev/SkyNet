"""Forecasting metrics for the object-centric representation: ADE, FDE, miss rate.

These are the standard agent-forecasting metrics; they complement the shared metrics
of the benchmark (L2 / collision rate for the ego, Chamfer Distance on point clouds).
Horizons follow the planning metrics: nuScenes keyframes are 2 Hz, so 1 s, 2 s and
3 s are steps 2, 4 and 6.
"""

from __future__ import annotations

import torch

HORIZON_STEPS = {"1s": 2, "2s": 4, "3s": 6}


def displacement_errors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor | None = None,
    miss_threshold_m: float = 2.0,
) -> dict[str, dict[str, float]]:
    """ADE, FDE and miss rate of agent trajectories, per horizon.

    Parameters
    ----------
    pred, gt : torch.Tensor
        Agent positions of shape (A, T, 2), T >= 6, in metres (any common frame).
    valid : torch.Tensor, optional
        Bool (A, T): False where the agent is not annotated at that step. Masked
        steps are ignored; a horizon with no valid step gives NaN.
    miss_threshold_m : float
        An agent is missed at horizon h if its error at step 2h exceeds this value.

    Returns
    -------
    dict
        ``{"ade": {...}, "fde": {...}, "miss_rate": {...}}`` with keys "1s", "2s",
        "3s". ADE averages the error over all valid (agent, step) pairs up to the
        horizon; FDE and miss rate use the agents valid at the horizon step.
    """
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(
            f"expected matching (A, T, 2) tensors, got {tuple(pred.shape)} / {tuple(gt.shape)}"
        )
    if pred.shape[1] < max(HORIZON_STEPS.values()):
        raise ValueError(f"need at least {max(HORIZON_STEPS.values())} steps")
    if valid is None:
        valid = torch.ones(pred.shape[:2], dtype=torch.bool)
    if tuple(valid.shape) != tuple(pred.shape[:2]):
        raise ValueError(f"valid must have shape {tuple(pred.shape[:2])}, got {tuple(valid.shape)}")

    dist = torch.linalg.norm(pred.double() - gt.double(), dim=-1)
    out: dict[str, dict[str, float]] = {"ade": {}, "fde": {}, "miss_rate": {}}
    nan = float("nan")
    for name, n in HORIZON_STEPS.items():
        upto = valid[:, :n]
        at = valid[:, n - 1]
        out["ade"][name] = float(dist[:, :n][upto].mean()) if upto.any() else nan
        out["fde"][name] = float(dist[:, n - 1][at].mean()) if at.any() else nan
        missed = (dist[:, n - 1][at] > miss_threshold_m).double()
        out["miss_rate"][name] = float(missed.mean()) if at.any() else nan
    return out
