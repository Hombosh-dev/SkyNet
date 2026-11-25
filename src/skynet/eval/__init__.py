"""Evaluation metrics and OOD stress-test utilities."""

from skynet.eval.object_metrics import displacement_errors
from skynet.eval.ood import (
    BEVSpec,
    OODPredictor,
    SceneWindow,
    bev_to_point_cloud,
    boxes_to_bev,
    boxes_to_point_cloud,
    evaluate_ood_shift,
    evaluate_predictor_windows,
    extract_scene_windows,
    format_ood_markdown_table,
    relative_drop_pct,
    shift_bev_grid,
    shift_nuscenes_ego,
    shift_object_features,
    shift_point_cloud,
)
from skynet.eval.planning_metrics import collision_rate, l2_error
from skynet.eval.reconstruction_metrics import chamfer_distance, fid_score

__all__ = [
    "BEVSpec",
    "OODPredictor",
    "SceneWindow",
    "bev_to_point_cloud",
    "boxes_to_bev",
    "boxes_to_point_cloud",
    "chamfer_distance",
    "collision_rate",
    "displacement_errors",
    "evaluate_ood_shift",
    "evaluate_predictor_windows",
    "extract_scene_windows",
    "fid_score",
    "format_ood_markdown_table",
    "l2_error",
    "relative_drop_pct",
    "shift_bev_grid",
    "shift_nuscenes_ego",
    "shift_object_features",
    "shift_point_cloud",
]
