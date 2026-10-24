"""Minimal, dependency-free reader of the nuScenes JSON tables (no nuscenes-devkit).

Only keyframes are indexed: ``sample_data.json`` also contains sweeps (LiDAR at 20 Hz,
cameras at 12 Hz) that carry a keyframe's ``sample_token``, so sensor records are
filtered on ``is_key_frame``. Scene frames are ordered by following
``scene.first_sample_token -> next``, not by the order of ``sample.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Official mini splits (nuscenes-devkit, nuscenes/utils/splits.py).
MINI_SPLITS = {
    "mini_train": [
        "scene-0061",
        "scene-0553",
        "scene-0655",
        "scene-0757",
        "scene-0796",
        "scene-1077",
        "scene-1094",
        "scene-1100",
    ],
    "mini_val": ["scene-0103", "scene-0916"],
}


def quaternion_to_matrix(q) -> np.ndarray:
    """3x3 rotation matrix of a quaternion ``[w, x, y, z]``."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_of(rotation: np.ndarray) -> float:
    """Heading (rotation about z) of a 3x3 rotation matrix, in radians."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


class NuScenesTables:
    """Indexed nuScenes metadata.

    Parameters
    ----------
    dataroot : str or Path
        Folder that contains the ``<version>/`` folder with the JSON tables.
    version : str
        ``v1.0-mini`` or ``v1.0-trainval``.
    """

    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini"):
        meta = Path(dataroot) / version
        if not (meta / "sample.json").exists():
            raise FileNotFoundError(f"nuScenes tables not found in {meta}")

        def load(name: str) -> list[dict]:
            with (meta / f"{name}.json").open(encoding="utf-8") as f:
                return json.load(f)

        self.scenes = load("scene")
        self.scene_by_name = {s["name"]: s for s in self.scenes}
        self.sample = {r["token"]: r for r in load("sample")}
        self.ego_pose_records = {r["token"]: r for r in load("ego_pose")}
        calib = {r["token"]: r for r in load("calibrated_sensor")}
        channel_of = {r["token"]: r["channel"] for r in load("sensor")}
        categories = {r["token"]: r["name"] for r in load("category")}
        self.category_of_instance = {
            r["token"]: categories[r["category_token"]] for r in load("instance")
        }
        self.keyframe: dict[tuple[str, str], dict] = {}
        for rec in load("sample_data"):
            if rec["is_key_frame"]:
                channel = channel_of[calib[rec["calibrated_sensor_token"]]["sensor_token"]]
                self.keyframe[(rec["sample_token"], channel)] = rec
        self.annotation: dict[str, dict] = {}
        self.annotations_of: dict[str, list[dict]] = {}
        for ann in load("sample_annotation"):
            self.annotation[ann["token"]] = ann
            self.annotations_of.setdefault(ann["sample_token"], []).append(ann)

    def split_scene_names(self, split: str) -> list[str]:
        """Scene names of ``mini_train``, ``mini_val`` or ``all``."""
        if split == "all":
            return [s["name"] for s in self.scenes]
        if split not in MINI_SPLITS:
            raise ValueError(f"unknown split {split!r}; use mini_train, mini_val or all")
        return list(MINI_SPLITS[split])

    def scene_sample_tokens(self, scene_name: str) -> list[str]:
        """Keyframe sample tokens of a scene in chronological order."""
        if scene_name not in self.scene_by_name:
            raise KeyError(f"scene {scene_name!r} is not in this nuScenes version")
        tokens, token = [], self.scene_by_name[scene_name]["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.sample[token]["next"]
        return tokens

    def timestamp(self, sample_token: str) -> float:
        """Sample timestamp in seconds."""
        return self.sample[sample_token]["timestamp"] * 1e-6

    def ego_pose(self, sample_token: str) -> tuple[np.ndarray, np.ndarray]:
        """Ego -> global rotation (3, 3) and translation (3,) at the LiDAR keyframe."""
        rec = self.keyframe[(sample_token, "LIDAR_TOP")]
        pose = self.ego_pose_records[rec["ego_pose_token"]]
        return quaternion_to_matrix(pose["rotation"]), np.asarray(pose["translation"], float)
