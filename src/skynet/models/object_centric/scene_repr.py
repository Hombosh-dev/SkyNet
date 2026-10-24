"""Object-centric scene representation: every annotated agent as a feature vector.

One nuScenes scene becomes a dense tensor ``features`` of shape ``(T, N, F)``:
``T`` keyframes, ``N`` agents (one per nuScenes instance, i.e. the tracking id is the
index on the agent axis) and ``F = len(FEATURES)`` features. ``mask[t, n]`` is True
where agent ``n`` is annotated at keyframe ``t``; elsewhere the row is zero. For one
frame, :meth:`ObjectScene.frame` returns the ``(N_agents, F)`` matrix of the agents
present. Every annotation of the scene is kept (no category, distance or LiDAR-point
filter), so ``mask.sum()`` equals the scene's number of annotations.

Velocity is a *backward* difference to the instance's previous annotation, so the
representation of frame t uses no information from later frames and is safe as a
world-model input. Agents seen for the first time have zero velocity and
``velocity_valid`` False. (nuscenes-devkit's ``box_velocity`` uses a centred
difference, which reads the next frame.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from skynet.data.nuscenes import NuScenesTables, quaternion_to_matrix, yaw_of

FEATURES = (
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw_sin",
    "yaw_cos",
    "vx",
    "vy",
    "class_id",
)
CLASSES = ("vehicle", "pedestrian", "two_wheeler", "obstacle", "animal")
FRAMES = ("ego", "global")


def class_id(category: str) -> int:
    """Coarse class of a nuScenes category name, as an index into ``CLASSES``."""
    if category.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return CLASSES.index("two_wheeler")
    if category.startswith("vehicle."):
        return CLASSES.index("vehicle")
    if category.startswith("human.pedestrian"):
        return CLASSES.index("pedestrian")
    if category.startswith("animal"):
        return CLASSES.index("animal")
    return CLASSES.index("obstacle")  # movable_object.*, static_object.*


@dataclass
class ObjectScene:
    """Object-centric tensors of one scene (see the module docstring)."""

    scene_name: str
    frame: str  # "ego": each keyframe in its own ego frame; "global": world frame
    sample_tokens: list[str]  # (T,) chronological keyframes
    timestamps: torch.Tensor  # (T,) seconds since the first keyframe
    track_ids: list[str]  # (N,) nuScenes instance tokens; index = tracking id
    categories: list[str]  # (N,) nuScenes category names
    features: torch.Tensor  # (T, N, F) float32
    mask: torch.Tensor  # (T, N) bool
    velocity_valid: torch.Tensor  # (T, N) bool

    @property
    def num_annotations(self) -> int:
        return int(self.mask.sum())

    def frame_features(self, t: int) -> torch.Tensor:
        """``(N_agents, F)`` features of the agents present at keyframe ``t``."""
        return self.features[t][self.mask[t]]

    def frame_track_ids(self, t: int) -> list[str]:
        """Instance tokens matching the rows of :meth:`frame_features`."""
        return [self.track_ids[n] for n in torch.nonzero(self.mask[t]).flatten().tolist()]


def build_object_scene(tables: NuScenesTables, scene_name: str, frame: str = "ego") -> ObjectScene:
    """Convert all annotations of one scene into an :class:`ObjectScene`.

    Parameters
    ----------
    tables : NuScenesTables
    scene_name : str
        e.g. ``"scene-0061"``.
    frame : str
        ``"ego"`` (default): positions, yaw and velocity of keyframe t are expressed
        in the ego frame of keyframe t (x forward, y left). ``"global"``: world frame,
        convenient for forecasting metrics across frames.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    tokens = tables.scene_sample_tokens(scene_name)
    stamps = [tables.timestamp(t) for t in tokens]

    track_ids: list[str] = []
    index: dict[str, int] = {}
    for token in tokens:  # agents ordered by first appearance, then by instance token
        for inst in sorted(a["instance_token"] for a in tables.annotations_of.get(token, [])):
            if inst not in index:
                index[inst] = len(track_ids)
                track_ids.append(inst)

    T, N = len(tokens), len(track_ids)
    features = np.zeros((T, N, len(FEATURES)), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)
    vel_valid = np.zeros((T, N), dtype=bool)

    for t, token in enumerate(tokens):
        if frame == "ego":
            rot, trans = tables.ego_pose(token)
        else:
            rot, trans = np.eye(3), np.zeros(3)
        for ann in tables.annotations_of.get(token, []):
            n = index[ann["instance_token"]]
            if mask[t, n]:
                raise ValueError(f"instance {ann['instance_token']} annotated twice at {token}")
            pos = rot.T @ (np.asarray(ann["translation"], float) - trans)
            yaw = yaw_of(rot.T @ quaternion_to_matrix(ann["rotation"]))
            vel = np.zeros(2)
            prev = tables.annotation.get(ann["prev"]) if ann["prev"] else None
            if prev is not None:
                dt = tables.timestamp(ann["sample_token"]) - tables.timestamp(prev["sample_token"])
                v_global = (np.asarray(ann["translation"]) - np.asarray(prev["translation"])) / dt
                vel = (rot.T @ v_global)[:2]
                vel_valid[t, n] = True
            width, length, height = ann["size"]  # nuScenes order is [w, l, h]
            features[t, n] = [
                *pos,
                length,
                width,
                height,
                math.sin(yaw),
                math.cos(yaw),
                *vel,
                class_id(tables.category_of_instance[ann["instance_token"]]),
            ]
            mask[t, n] = True

    return ObjectScene(
        scene_name=scene_name,
        frame=frame,
        sample_tokens=tokens,
        timestamps=torch.tensor([s - stamps[0] for s in stamps], dtype=torch.float64),
        track_ids=track_ids,
        categories=[tables.category_of_instance[i] for i in track_ids],
        features=torch.from_numpy(features),
        mask=torch.from_numpy(mask),
        velocity_valid=torch.from_numpy(vel_valid),
    )
