"""Baselines on the object-centric representation and their evaluation windows.

* ``persistence``: every agent stays where it was last seen.
* ``constant_velocity``: every agent keeps its last (backward-difference) velocity.

Both use only the current keyframe, so they are the minimum any learned
object-centric world model has to beat.
"""

from __future__ import annotations

import torch

from skynet.models.object_centric.scene_repr import FEATURES, ObjectScene

X, Y, VX, VY = (FEATURES.index(k) for k in ("x", "y", "vx", "vy"))


def forecast_windows(scene: ObjectScene, num_future: int = 6):
    """Yield (t0, positions_t0 (A, 2), velocities_t0 (A, 2), future_gt (A, T, 2),
    future_valid (A, T), dt (T,)) for every keyframe with ``num_future`` successors.

    ``scene`` must use ``frame="global"`` so that positions of different keyframes
    are comparable. Agents are those annotated at t0.
    """
    if scene.frame != "global":
        raise ValueError("forecast windows need an ObjectScene built with frame='global'")
    feats, mask = scene.features, scene.mask
    for t0 in range(feats.shape[0] - num_future):
        agents = mask[t0]
        if not agents.any():
            continue
        steps = slice(t0 + 1, t0 + 1 + num_future)
        yield (
            t0,
            feats[t0, agents][:, [X, Y]].double(),
            feats[t0, agents][:, [VX, VY]].double(),
            feats[steps][:, agents][..., [X, Y]].transpose(0, 1).double(),
            mask[steps][:, agents].transpose(0, 1),
            scene.timestamps[steps] - scene.timestamps[t0],
        )


def persistence(pos: torch.Tensor, vel: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """(A, T, 2): the t0 position repeated."""
    return pos[:, None, :].expand(-1, len(dt), -1).clone()


def constant_velocity(pos: torch.Tensor, vel: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """(A, T, 2): straight-line motion with the t0 velocity."""
    return pos[:, None, :] + dt[None, :, None] * vel[:, None, :]


BASELINES = {"persistence": persistence, "constant_velocity": constant_velocity}
