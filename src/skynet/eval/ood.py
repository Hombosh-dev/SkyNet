"""OOD lateral ego-shift transforms and evaluation harness (0.5m ego shift)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from skynet.data.nuscenes import NuScenesTables, quaternion_to_matrix, yaw_of
from skynet.eval.planning_metrics import HORIZON_STEPS
from skynet.eval.reconstruction_metrics import chamfer_distance, fid_score
from skynet.models.object_centric.scene_repr import FEATURES, class_id

Y_INDEX = FEATURES.index("y")
CLASS_TO_CHANNEL = (0, 1, 1, 2, 2)  # vehicle -> 0, ped/2wheeler -> 1, obstacle/animal -> 2


@dataclass(frozen=True)
class BEVSpec:
    """Metric bounds and resolution of the shared 3-channel BEV grid."""

    x_range: tuple[float, float] = (-50.0, 50.0)
    y_range: tuple[float, float] = (-50.0, 50.0)
    resolution_m: float = 1.0

    @property
    def height(self) -> int:
        return int(round((self.x_range[1] - self.x_range[0]) / self.resolution_m))

    @property
    def width(self) -> int:
        return int(round((self.y_range[1] - self.y_range[0]) / self.resolution_m))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (3, self.height, self.width)


@dataclass
class SceneWindow:
    """Single P-to-N evaluation window anchored at the ``t0`` ego frame."""

    scene_name: str
    sample_token: str
    past_agents: torch.Tensor  # (A, P, 9): x, y, l, w, sin_yaw, cos_yaw, vx, vy, class_id
    past_valid: torch.Tensor  # (A, P) bool
    past_ego: torch.Tensor  # (P, 3): x, y, yaw in t0 ego frame
    ego_action: torch.Tensor  # (N, 3): future ego dx, dy, dyaw in t0 ego frame
    future_dt: torch.Tensor  # (N,) seconds after t0
    past_bev: torch.Tensor  # (P, 3, H, W) float32 in [0, 1]
    gt_bev: torch.Tensor  # (N, 3, H, W) float32 in [0, 1]
    gt_points: list[torch.Tensor]  # length N, each (M_t, 3) float32
    future_agent_xy: torch.Tensor  # (A, N, 2) float32
    future_agent_valid: torch.Tensor  # (A, N) bool


class OODPredictor(Protocol):
    """Protocol implemented by predictors evaluated under OOD ego shifts."""

    def predict_window(self, window: SceneWindow) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return ``(pred_points_per_step, pred_bev)`` for ``N`` future frames."""


def shift_nuscenes_ego(
    tables: NuScenesTables,
    shift_m: float,
    scene_names: list[str] | None = None,
) -> NuScenesTables:
    """Return a non-mutating copy of ``tables`` with ego poses shifted laterally (+y left)."""
    target_scenes = list(tables.scene_by_name) if scene_names is None else list(scene_names)
    target_samples = {tok for name in target_scenes for tok in tables.scene_sample_tokens(name)}

    pose_tokens: set[str] = set()
    for (sample_tok, _channel), rec in tables.keyframe.items():
        if sample_tok in target_samples:
            pose_tokens.add(rec["ego_pose_token"])

    cloned = copy.copy(tables)
    cloned.ego_pose_records = dict(tables.ego_pose_records)
    delta_local = float(shift_m)
    for pose_token in pose_tokens:
        pose = dict(cloned.ego_pose_records[pose_token])
        rot = quaternion_to_matrix(pose["rotation"])
        trans = np.asarray(pose["translation"], dtype=np.float64)
        pose["translation"] = (trans + delta_local * rot[:, 1]).tolist()
        cloned.ego_pose_records[pose_token] = pose
    return cloned


def shift_point_cloud(points: torch.Tensor, shift_m: float) -> torch.Tensor:
    """Shift a point cloud in ego frame (+x forward, +y left) under a lateral ego shift."""
    if points.ndim < 2 or points.shape[-1] < 3:
        raise ValueError(f"expected shape (..., N, >=3), got {tuple(points.shape)}")
    shifted = points.clone()
    shifted[..., 1] = shifted[..., 1] - float(shift_m)
    return shifted


def shift_object_features(
    features: torch.Tensor,
    mask: torch.Tensor | None,
    shift_m: float,
) -> torch.Tensor:
    """Shift ego-frame object-centric feature tensor along lateral ``y``."""
    if features.shape[-1] != len(FEATURES):
        raise ValueError(f"expected last dimension {len(FEATURES)}, got {tuple(features.shape)}")
    shifted = features.clone()
    if mask is None:
        shifted[..., Y_INDEX] = shifted[..., Y_INDEX] - float(shift_m)
        return shifted
    if tuple(mask.shape) != tuple(features.shape[:-1]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match features {tuple(features.shape)}"
        )
    shifted[..., Y_INDEX] = torch.where(
        mask, shifted[..., Y_INDEX] - float(shift_m), shifted[..., Y_INDEX]
    )
    return shifted


def shift_bev_grid(
    grid: torch.Tensor,
    shift_m: float,
    resolution_m: float,
) -> torch.Tensor:
    """Shift a ``(C, H, W)`` or ``(B, C, H, W)`` BEV tensor along lateral ``W`` (y axis)."""
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    if grid.ndim not in (3, 4):
        raise ValueError(f"expected (C, H, W) or (B, C, H, W), got {tuple(grid.shape)}")

    squeeze = grid.ndim == 3
    batched = grid.unsqueeze(0) if squeeze else grid
    batch, _, height, width = batched.shape

    cell_shift = float(shift_m) / float(resolution_m)
    base_y = (torch.arange(width, device=batched.device, dtype=batched.dtype) + 0.5) * (
        2.0 / width
    ) - 1.0
    base_x = (torch.arange(height, device=batched.device, dtype=batched.dtype) + 0.5) * (
        2.0 / height
    ) - 1.0
    grid_x, grid_y = torch.meshgrid(base_x, base_y + 2.0 * cell_shift / width, indexing="ij")
    sample_grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    shifted = F.grid_sample(
        batched,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return shifted.squeeze(0) if squeeze else shifted


def _grid_centers(spec: BEVSpec) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(
        spec.x_range[0] + 0.5 * spec.resolution_m,
        spec.x_range[1] - 0.5 * spec.resolution_m,
        spec.height,
        dtype=torch.float32,
    )
    ys = torch.linspace(
        spec.y_range[0] + 0.5 * spec.resolution_m,
        spec.y_range[1] - 0.5 * spec.resolution_m,
        spec.width,
        dtype=torch.float32,
    )
    return torch.meshgrid(xs, ys, indexing="ij")


def boxes_to_bev(boxes: torch.Tensor, spec: BEVSpec = BEVSpec()) -> torch.Tensor:
    """Rasterize ``(K, 6)`` boxes ``(x, y, length, width, yaw, class_id)`` into ``(3, H, W)``."""
    bev = torch.zeros(spec.shape, dtype=torch.float32)
    if len(boxes) == 0:
        return bev

    gx, gy = _grid_centers(spec)
    soft_scale = 0.35 * spec.resolution_m
    for row in boxes:
        bx, by, length, width, yaw, cls_idx = row.tolist()
        if not (
            spec.x_range[0] - length <= bx <= spec.x_range[1] + length
            and spec.y_range[0] - width <= by <= spec.y_range[1] + width
        ):
            continue
        dx = gx - bx
        dy = gy - by
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        lx = c_yaw * dx + s_yaw * dy
        ly = -s_yaw * dx + c_yaw * dy
        margin = torch.maximum(lx.abs() - 0.5 * length, ly.abs() - 0.5 * width)
        occ = torch.sigmoid(-margin / soft_scale)
        occ = torch.where(margin <= 3.0 * soft_scale, occ, torch.zeros_like(occ))
        ch = CLASS_TO_CHANNEL[int(cls_idx)]
        bev[ch] = torch.maximum(bev[ch], occ)
    return bev.clamp(0.0, 1.0)


def boxes_to_point_cloud(boxes: torch.Tensor, spec: BEVSpec = BEVSpec()) -> torch.Tensor:
    """Sample deterministic footprint points ``(M, 3)`` for boxes inside ``spec`` bounds."""
    if len(boxes) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    pts: list[list[float]] = []
    local_corners = ((0.0, 0.0), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5))
    for row in boxes.tolist():
        bx, by, length, width, yaw, _ = row
        if not (
            spec.x_range[0] <= bx <= spec.x_range[1] and spec.y_range[0] <= by <= spec.y_range[1]
        ):
            continue
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        for ux, uy in local_corners:
            px = bx + c_yaw * (ux * length) - s_yaw * (uy * width)
            py = by + s_yaw * (ux * length) + c_yaw * (uy * width)
            pts.append([px, py, 0.0])
    if not pts:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.tensor(pts, dtype=torch.float32)


def bev_to_point_cloud(
    bev: torch.Tensor,
    spec: BEVSpec = BEVSpec(),
    threshold: float = 0.3,
    fallback_topk: int = 5,
) -> torch.Tensor:
    """Extract occupied ``(x, y, 0.0)`` points from a ``(3, H, W)`` BEV probability map."""
    if bev.ndim != 3 or bev.shape[0] != 3:
        raise ValueError(f"expected BEV shape (3, H, W), got {tuple(bev.shape)}")
    occ = bev.max(dim=0).values
    coords = torch.nonzero(occ >= threshold, as_tuple=False)
    if len(coords) == 0:
        k = min(fallback_topk, occ.numel())
        flat_idx = torch.topk(occ.flatten(), k=k).indices
        coords = torch.stack((flat_idx // occ.shape[1], flat_idx % occ.shape[1]), dim=-1)

    xs = spec.x_range[0] + (coords[:, 0].float() + 0.5) * spec.resolution_m
    ys = spec.y_range[0] + (coords[:, 1].float() + 0.5) * spec.resolution_m
    zs = torch.zeros_like(xs)
    return torch.stack((xs, ys, zs), dim=-1)


def _annotation_in_frame(
    tables: NuScenesTables,
    ann: dict[str, Any],
    rot0: np.ndarray,
    trans0: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, tuple[float, float], int]:
    pos = rot0.T @ (np.asarray(ann["translation"], dtype=np.float64) - trans0)
    yaw = yaw_of(rot0.T @ quaternion_to_matrix(ann["rotation"]))
    vel = np.zeros(2, dtype=np.float64)
    prev = tables.annotation.get(ann["prev"]) if ann["prev"] else None
    if prev is not None:
        dt = tables.timestamp(ann["sample_token"]) - tables.timestamp(prev["sample_token"])
        if dt > 0:
            v_global = (
                np.asarray(ann["translation"], dtype=np.float64)
                - np.asarray(prev["translation"], dtype=np.float64)
            ) / dt
            vel = (rot0.T @ v_global)[:2]
    width, length, _height = ann["size"]
    cls_idx = class_id(tables.category_of_instance[ann["instance_token"]])
    return pos[:2], yaw, vel, (float(length), float(width)), cls_idx


def _boxes_at_sample(
    tables: NuScenesTables,
    sample_token: str,
    rot0: np.ndarray,
    trans0: np.ndarray,
) -> tuple[torch.Tensor, dict[str, tuple[np.ndarray, float, np.ndarray, tuple[float, float], int]]]:
    by_inst: dict[str, tuple[np.ndarray, float, np.ndarray, tuple[float, float], int]] = {}
    rows: list[list[float]] = []
    for ann in tables.annotations_of.get(sample_token, []):
        parsed = _annotation_in_frame(tables, ann, rot0, trans0)
        pos, yaw, _vel, (length, width), cls_idx = parsed
        by_inst[ann["instance_token"]] = parsed
        rows.append([float(pos[0]), float(pos[1]), length, width, float(yaw), float(cls_idx)])
    tensor = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 6))
    return tensor, by_inst


def extract_scene_windows(
    tables: NuScenesTables,
    scene_names: list[str],
    past_frames: int = 3,
    future_frames: int = 6,
    spec: BEVSpec = BEVSpec(),
) -> list[SceneWindow]:
    """Build ``P -> N`` evaluation windows in each keyframe's ``t0`` ego coordinates."""
    windows: list[SceneWindow] = []
    for scene_name in scene_names:
        tokens = tables.scene_sample_tokens(scene_name)
        stamps = [tables.timestamp(tok) for tok in tokens]
        for t0 in range(past_frames - 1, len(tokens) - future_frames):
            rot0, trans0 = tables.ego_pose(tokens[t0])
            yaw0 = yaw_of(rot0)

            past_tokens = tokens[t0 - past_frames + 1 : t0 + 1]
            future_tokens = tokens[t0 + 1 : t0 + 1 + future_frames]

            past_bevs: list[torch.Tensor] = []
            past_maps = []
            for tok in past_tokens:
                boxes, inst_map = _boxes_at_sample(tables, tok, rot0, trans0)
                past_bevs.append(boxes_to_bev(boxes, spec))
                past_maps.append(inst_map)

            future_bevs: list[torch.Tensor] = []
            future_pts: list[torch.Tensor] = []
            future_maps = []
            for tok in future_tokens:
                boxes, inst_map = _boxes_at_sample(tables, tok, rot0, trans0)
                future_bevs.append(boxes_to_bev(boxes, spec))
                future_pts.append(boxes_to_point_cloud(boxes, spec))
                future_maps.append(inst_map)

            anchor_ids = sorted(past_maps[-1])
            num_agents = len(anchor_ids)
            past_agents = torch.zeros((num_agents, past_frames, 9), dtype=torch.float32)
            past_valid = torch.zeros((num_agents, past_frames), dtype=torch.bool)
            future_xy = torch.zeros((num_agents, future_frames, 2), dtype=torch.float32)
            future_valid = torch.zeros((num_agents, future_frames), dtype=torch.bool)

            for a_idx, inst_id in enumerate(anchor_ids):
                for p_idx, p_map in enumerate(past_maps):
                    if inst_id in p_map:
                        pos, yaw, vel, (length, width), cls_idx = p_map[inst_id]
                        past_agents[a_idx, p_idx] = torch.tensor(
                            [
                                float(pos[0]),
                                float(pos[1]),
                                length,
                                width,
                                math.sin(yaw),
                                math.cos(yaw),
                                float(vel[0]),
                                float(vel[1]),
                                float(cls_idx),
                            ],
                            dtype=torch.float32,
                        )
                        past_valid[a_idx, p_idx] = True
                for f_idx, f_map in enumerate(future_maps):
                    if inst_id in f_map:
                        pos = f_map[inst_id][0]
                        future_xy[a_idx, f_idx] = torch.tensor(
                            [float(pos[0]), float(pos[1])], dtype=torch.float32
                        )
                        future_valid[a_idx, f_idx] = True

            def _ego_rel(tok: str) -> list[float]:
                rot, trans = tables.ego_pose(tok)
                d_xy = (rot0.T @ (trans - trans0))[:2]
                d_yaw = (yaw_of(rot) - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
                return [float(d_xy[0]), float(d_xy[1]), float(d_yaw)]

            past_ego = torch.tensor([_ego_rel(tok) for tok in past_tokens], dtype=torch.float32)
            ego_action = torch.tensor([_ego_rel(tok) for tok in future_tokens], dtype=torch.float32)
            future_dt = torch.tensor(
                [stamps[t0 + 1 + k] - stamps[t0] for k in range(future_frames)],
                dtype=torch.float32,
            )

            windows.append(
                SceneWindow(
                    scene_name=scene_name,
                    sample_token=tokens[t0],
                    past_agents=past_agents,
                    past_valid=past_valid,
                    past_ego=past_ego,
                    ego_action=ego_action,
                    future_dt=future_dt,
                    past_bev=torch.stack(past_bevs, dim=0),
                    gt_bev=torch.stack(future_bevs, dim=0),
                    gt_points=future_pts,
                    future_agent_xy=future_xy,
                    future_agent_valid=future_valid,
                )
            )
    return windows


def relative_drop_pct(baseline: float, shifted: float) -> float:
    """Percentage metric degradation under OOD shift: ``100 * (shifted - baseline) / baseline``."""
    if baseline == 0.0:
        return 0.0 if shifted == 0.0 else float("inf")
    return 100.0 * (shifted - baseline) / baseline


def evaluate_predictor_windows(
    predictor: OODPredictor,
    windows: list[SceneWindow],
    fid_feature: int | None = 64,
    fid_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> dict[str, Any]:
    """Compute per-horizon Chamfer Distance and BEV FID across ``windows``."""
    if not windows:
        raise ValueError("windows must not be empty")
    num_steps = len(windows[0].gt_points)
    step_cds: list[list[float]] = [[] for _ in range(num_steps)]
    real_bevs: list[torch.Tensor] = []
    fake_bevs: list[torch.Tensor] = []

    for win in windows:
        pred_pts, pred_bev = predictor.predict_window(win)
        real_bevs.append(win.gt_bev.clamp(0.0, 1.0))
        fake_bevs.append(pred_bev.clamp(0.0, 1.0))
        for step in range(num_steps):
            gt_pts = win.gt_points[step]
            if len(gt_pts) == 0:
                continue
            p_pts = pred_pts[step] if len(pred_pts[step]) > 0 else torch.zeros((1, 3))
            step_cds[step].append(chamfer_distance(p_pts, gt_pts))

    cd_by_horizon: dict[str, float] = {}
    for name, horizon_step in HORIZON_STEPS.items():
        vals = step_cds[horizon_step - 1]
        cd_by_horizon[name] = sum(vals) / len(vals) if vals else float("nan")
    cd_by_horizon["avg"] = sum(cd_by_horizon[k] for k in HORIZON_STEPS) / len(HORIZON_STEPS)
    all_vals = [v for vals in step_cds for v in vals]
    cd_by_horizon["all_steps_mean"] = sum(all_vals) / len(all_vals) if all_vals else float("nan")

    real_batch = torch.cat(real_bevs, dim=0)
    fake_batch = torch.cat(fake_bevs, dim=0)
    if fid_fn is not None:
        fid_val: float | None = float(fid_fn(real_batch, fake_batch))
    elif fid_feature is not None:
        fid_val = float(fid_score(real_batch, fake_batch, feature=fid_feature))
    else:
        fid_val = None

    return {"cd": cd_by_horizon, "fid": fid_val, "num_windows": len(windows)}


def evaluate_ood_shift(
    predictors: dict[str, OODPredictor],
    tables: NuScenesTables,
    scene_names: list[str],
    shift_m: float = 0.5,
    past_frames: int = 3,
    future_frames: int = 6,
    spec: BEVSpec = BEVSpec(),
    fid_feature: int | None = 64,
    fid_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all predictors on unshifted vs ``shift_m`` laterally shifted scenes."""
    baseline_windows = extract_scene_windows(tables, scene_names, past_frames, future_frames, spec)
    shifted_tables = shift_nuscenes_ego(tables, shift_m, scene_names)
    shifted_windows = extract_scene_windows(
        shifted_tables, scene_names, past_frames, future_frames, spec
    )

    results: dict[str, dict[str, Any]] = {}
    for name, predictor in predictors.items():
        base = evaluate_predictor_windows(
            predictor, baseline_windows, fid_feature=fid_feature, fid_fn=fid_fn
        )
        shifted = evaluate_predictor_windows(
            predictor, shifted_windows, fid_feature=fid_feature, fid_fn=fid_fn
        )
        horizon_drop = {
            k: relative_drop_pct(base["cd"][k], shifted["cd"][k]) for k in ("1s", "2s", "3s", "avg")
        }
        fid_drop = (
            relative_drop_pct(base["fid"], shifted["fid"])
            if base["fid"] is not None and shifted["fid"] is not None
            else None
        )
        results[name] = {
            "baseline": base,
            "shifted": shifted,
            "cd_drop_pct": horizon_drop["avg"],
            "cd_horizon_drop_pct": horizon_drop,
            "fid_drop_pct": fid_drop,
        }
    return results


def format_ood_markdown_table(
    results: dict[str, dict[str, Any]],
    shift_m: float = 0.5,
) -> str:
    """Format a Markdown table comparing unshifted baseline vs ``shift_m`` OOD shift."""
    labels = {
        "object_centric": "Object-centric",
        "bev": "BEV",
        "latent_bev": "Latent BEV",
    }
    lines = [
        f"| Representation | CD 0m (m) ↓ | CD {shift_m:g}m (m) ↓ | % CD Drop (Δ%) "
        f"| FID 0m ↓ | FID {shift_m:g}m ↓ | % FID Drop (Δ%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for key, entry in results.items():
        label = labels.get(key, key)
        cd0 = entry["baseline"]["cd"]["avg"]
        cd1 = entry["shifted"]["cd"]["avg"]
        cd_drop = entry["cd_drop_pct"]
        fid0 = entry["baseline"]["fid"]
        fid1 = entry["shifted"]["fid"]
        fid_drop = entry["fid_drop_pct"]
        fid0_s = f"{fid0:.2f}" if fid0 is not None else "n/a"
        fid1_s = f"{fid1:.2f}" if fid1 is not None else "n/a"
        fid_drop_s = f"{fid_drop:+.2f}%" if fid_drop is not None else "n/a"
        lines.append(
            f"| {label} | {cd0:.3f} | {cd1:.3f} | {cd_drop:+.2f}% "
            f"| {fid0_s} | {fid1_s} | {fid_drop_s} |"
        )
    return "\n".join(lines)
