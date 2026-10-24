"""Ego-trajectory baselines that read only the past.

* ``zero_motion``: the ego stays at the origin of its current frame.
* ``constant_velocity``: the ego keeps the velocity it had between the previous
  keyframe and now.

The constant-velocity baseline of the midterm notebook took its velocity from the
first *future* waypoint (``gt[0] / 0.5``), i.e. from the ground truth it is scored
against; this version is the replacement used for reported numbers.
"""

from __future__ import annotations

import numpy as np
import torch

from skynet.data.nuscenes import NuScenesTables


def ego_windows(tables: NuScenesTables, scene_name: str, num_future: int = 6):
    """(sample_token, prev_xy (2,), dt_prev, future_xy (T, 2), future_dt (T,)) for
    every keyframe with a previous keyframe and ``num_future`` successors.

    Positions are in the ego frame of the current keyframe (x forward, y left).
    """
    tokens = tables.scene_sample_tokens(scene_name)
    out = []
    for i in range(1, len(tokens) - num_future):
        rot, trans = tables.ego_pose(tokens[i])

        def local(token):
            return (rot.T @ (tables.ego_pose(token)[1] - trans))[:2]

        now = tables.timestamp(tokens[i])
        future = tokens[i + 1 : i + 1 + num_future]
        out.append(
            (
                tokens[i],
                local(tokens[i - 1]),
                now - tables.timestamp(tokens[i - 1]),
                np.stack([local(t) for t in future]),
                np.array([tables.timestamp(t) - now for t in future]),
            )
        )
    return out


def zero_motion(prev_xy: np.ndarray, dt_prev: float, future_dt: np.ndarray) -> np.ndarray:
    return np.zeros((len(future_dt), 2))


def constant_velocity(prev_xy: np.ndarray, dt_prev: float, future_dt: np.ndarray) -> np.ndarray:
    velocity = -np.asarray(prev_xy) / dt_prev  # previous keyframe -> now
    return future_dt[:, None] * velocity[None, :]


EGO_BASELINES = {"zero_motion": zero_motion, "constant_velocity": constant_velocity}


def evaluate_ego_baselines(tables: NuScenesTables, scene_names: list[str], num_future: int = 6):
    """L2 (planning_metrics.l2_error) of each ego baseline over all windows of the scenes."""
    from skynet.eval.planning_metrics import l2_error

    windows = [w for name in scene_names for w in ego_windows(tables, name, num_future)]
    gt = torch.tensor(np.stack([w[3] for w in windows]))
    out = {"num_windows": len(windows)}
    for name, fn in EGO_BASELINES.items():
        pred = torch.tensor(np.stack([fn(w[1], w[2], w[4]) for w in windows]))
        out[name] = l2_error(pred, gt)
    return out
