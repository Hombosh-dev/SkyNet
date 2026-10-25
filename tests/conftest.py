import json
import math

import pytest

DT_US = 500_000


def _yaw_q(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _annotation(token, sample, inst, xyz, size, yaw):
    return {
        "token": token,
        "sample_token": sample,
        "instance_token": inst,
        "translation": list(xyz),
        "size": list(size),
        "rotation": _yaw_q(yaw),
        "prev": "",
        "next": "",
    }


@pytest.fixture
def make_nuscenes(tmp_path):
    """Tiny nuScenes tables: ego drives along global +y at 10 m/s (yaw 90 deg).

    A car drives alongside, 5 m to the ego's right and 20 m ahead (all 8 frames); a
    pedestrian stands still (frames 2-5); a barrier is visible in frames 0-1. Decoy
    sweep records carry a far-away ego pose, and sample.json is written reversed.
    """

    def _make(num_samples=8, scene_name="scene-0103"):
        meta = tmp_path / "nuscenes" / "v1.0-mini"
        meta.mkdir(parents=True)
        tokens = [f"s{k}" for k in range(num_samples)]
        samples, sample_data, ego_poses, anns = [], [], [], []
        for k, tok in enumerate(tokens):
            t = 0.5 * k
            samples.append(
                {
                    "token": tok,
                    "timestamp": 10**15 + k * DT_US,
                    "scene_token": "scene",
                    "prev": tokens[k - 1] if k else "",
                    "next": tokens[k + 1] if k + 1 < num_samples else "",
                }
            )
            ego_poses += [
                {
                    "token": f"{tok}-ego",
                    "translation": [0.0, 10.0 * t, 0.0],
                    "rotation": _yaw_q(math.pi / 2),
                },
                {
                    "token": f"{tok}-sweep-ego",
                    "translation": [900.0, 0.0, 0.0],
                    "rotation": _yaw_q(0.0),
                },
            ]
            sample_data += [
                {
                    "token": f"{tok}-lidar",
                    "sample_token": tok,
                    "ego_pose_token": f"{tok}-ego",
                    "calibrated_sensor_token": "calib-lidar",
                    "is_key_frame": True,
                },
                {
                    "token": f"{tok}-sweep",
                    "sample_token": tok,
                    "ego_pose_token": f"{tok}-sweep-ego",
                    "calibrated_sensor_token": "calib-lidar",
                    "is_key_frame": False,
                },
            ]
            anns.append(
                _annotation(
                    f"{tok}-car",
                    tok,
                    "inst-car",
                    (5.0, 10.0 * t + 20.0, 0.8),
                    (2.0, 4.5, 1.6),
                    math.pi / 2,
                )
            )
            if 2 <= k <= 5:
                anns.append(
                    _annotation(
                        f"{tok}-ped", tok, "inst-ped", (-3.0, 30.0, 0.9), (0.7, 0.7, 1.8), 0.0
                    )
                )
            if k <= 1:
                anns.append(
                    _annotation(
                        f"{tok}-bar", tok, "inst-bar", (2.0, 8.0, 0.5), (0.5, 2.0, 1.0), 0.0
                    )
                )
        by_inst = {}
        for a in anns:
            by_inst.setdefault(a["instance_token"], []).append(a)
        for chain in by_inst.values():
            for prev, nxt in zip(chain, chain[1:]):
                prev["next"], nxt["prev"] = nxt["token"], prev["token"]

        tables = {
            "scene": [
                {
                    "token": "scene",
                    "name": scene_name,
                    "first_sample_token": tokens[0],
                    "last_sample_token": tokens[-1],
                    "nbr_samples": num_samples,
                }
            ],
            "sample": list(reversed(samples)),
            "sample_data": sample_data,
            "ego_pose": ego_poses,
            "calibrated_sensor": [{"token": "calib-lidar", "sensor_token": "sensor-lidar"}],
            "sensor": [{"token": "sensor-lidar", "channel": "LIDAR_TOP"}],
            "category": [
                {"token": "cat-car", "name": "vehicle.car"},
                {"token": "cat-ped", "name": "human.pedestrian.adult"},
                {"token": "cat-bar", "name": "movable_object.barrier"},
            ],
            "instance": [
                {"token": "inst-car", "category_token": "cat-car"},
                {"token": "inst-ped", "category_token": "cat-ped"},
                {"token": "inst-bar", "category_token": "cat-bar"},
            ],
            "sample_annotation": anns,
        }
        for name, rows in tables.items():
            (meta / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        return tmp_path / "nuscenes"

    return _make


@pytest.fixture
def nuscenes_root(make_nuscenes):
    return make_nuscenes()


@pytest.fixture
def tables(nuscenes_root):
    from skynet.data.nuscenes import NuScenesTables

    return NuScenesTables(nuscenes_root)
