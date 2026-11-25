"""Offline tests for OOD lateral ego-shift transforms and stress-test evaluation."""

import math

import pytest
import torch

from skynet.eval.ood import (
    BEVSpec,
    boxes_to_bev,
    evaluate_ood_shift,
    extract_scene_windows,
    format_ood_markdown_table,
    relative_drop_pct,
    shift_bev_grid,
    shift_nuscenes_ego,
    shift_object_features,
    shift_point_cloud,
    train_reference_models,
)
from skynet.models.object_centric.scene_repr import FEATURES, build_object_scene


def test_shift_point_cloud_shifts_only_lateral_y():
    pts = torch.tensor([[10.0, 2.5, 0.8], [-3.0, 0.0, 1.2]])
    shifted = shift_point_cloud(pts, shift_m=0.5)
    assert torch.allclose(shifted[:, 0], pts[:, 0])
    assert torch.allclose(shifted[:, 1], pts[:, 1] - 0.5)
    assert torch.allclose(shifted[:, 2], pts[:, 2])


def test_shift_point_cloud_rejects_bad_shape():
    with pytest.raises(ValueError):
        shift_point_cloud(torch.zeros((3,)), shift_m=0.5)
    with pytest.raises(ValueError):
        shift_point_cloud(torch.zeros((4, 2)), shift_m=0.5)


def test_shift_object_features_preserves_masked_rows_and_velocities():
    features = torch.zeros((2, 3, len(FEATURES)))
    mask = torch.tensor([[True, False, True], [False, True, False]])
    y_idx = FEATURES.index("y")
    vx_idx = FEATURES.index("vx")
    features[0, 0, y_idx] = 5.0
    features[0, 2, y_idx] = -2.0
    features[1, 1, y_idx] = 1.5
    features[0, 0, vx_idx] = 3.0

    shifted = shift_object_features(features, mask, shift_m=0.5)
    assert shifted[0, 0, y_idx] == pytest.approx(4.5)
    assert shifted[0, 1, y_idx] == 0.0  # masked out
    assert shifted[0, 2, y_idx] == pytest.approx(-2.5)
    assert shifted[1, 1, y_idx] == pytest.approx(1.0)
    assert shifted[0, 0, vx_idx] == pytest.approx(3.0)


def test_shift_nuscenes_ego_is_non_mutating_and_consistent_with_object_scene(tables):
    orig_rot, orig_trans = tables.ego_pose("s0")
    shifted_tables = shift_nuscenes_ego(tables, shift_m=0.5, scene_names=["scene-0103"])

    # Original tables are untouched
    check_rot, check_trans = tables.ego_pose("s0")
    assert check_trans.tolist() == pytest.approx(orig_trans.tolist())
    assert check_rot.flatten().tolist() == pytest.approx(orig_rot.flatten().tolist())

    # Ego drives along +y (yaw=pi/2), so ego +y (left) points along global -x
    _, new_trans = shifted_tables.ego_pose("s0")
    assert new_trans.tolist() == pytest.approx((orig_trans + [-0.5, 0.0, 0.0]).tolist(), abs=1e-6)

    base_ego = build_object_scene(tables, "scene-0103", frame="ego")
    shifted_ego = build_object_scene(shifted_tables, "scene-0103", frame="ego")
    expected_ego = shift_object_features(base_ego.features, base_ego.mask, shift_m=0.5)
    assert torch.allclose(shifted_ego.features, expected_ego, atol=1e-5)

    base_global = build_object_scene(tables, "scene-0103", frame="global")
    shifted_global = build_object_scene(shifted_tables, "scene-0103", frame="global")
    assert torch.allclose(shifted_global.features, base_global.features, atol=1e-5)


def test_shift_bev_grid_matches_shifted_boxes():
    spec = BEVSpec(x_range=(-20.0, 20.0), y_range=(-20.0, 20.0), resolution_m=1.0)
    box_at_4 = torch.tensor([[5.0, 4.0, 4.0, 2.0, 0.0, 0.0]])
    box_at_2 = torch.tensor([[5.0, 2.0, 4.0, 2.0, 0.0, 0.0]])

    bev_4 = boxes_to_bev(box_at_4, spec)
    bev_2 = boxes_to_bev(box_at_2, spec)
    shifted_bev = shift_bev_grid(bev_4, shift_m=2.0, resolution_m=spec.resolution_m)
    assert torch.allclose(shifted_bev, bev_2, atol=1e-4)

    # Sub-pixel 0.5m shift moves lateral center of mass by -0.5m
    ys = torch.linspace(
        spec.y_range[0] + 0.5 * spec.resolution_m,
        spec.y_range[1] - 0.5 * spec.resolution_m,
        spec.width,
    )
    half_shifted = shift_bev_grid(bev_4, shift_m=0.5, resolution_m=spec.resolution_m)
    mass_before = bev_4[0].sum()
    mass_after = half_shifted[0].sum()
    com_before = float((bev_4[0] * ys[None, :]).sum() / mass_before)
    com_after = float((half_shifted[0] * ys[None, :]).sum() / mass_after)
    assert com_after - com_before == pytest.approx(-0.5, abs=0.05)


def test_relative_drop_pct():
    assert relative_drop_pct(2.0, 2.5) == pytest.approx(25.0)
    assert relative_drop_pct(2.0, 1.5) == pytest.approx(-25.0)
    assert relative_drop_pct(2.0, 2.0) == pytest.approx(0.0)
    assert relative_drop_pct(0.0, 0.0) == 0.0
    assert math.isinf(relative_drop_pct(0.0, 1.0))


def test_end_to_end_ood_evaluation_and_markdown_table(make_nuscenes):
    root = make_nuscenes(num_samples=12, scene_name="scene-0103")
    from skynet.data.nuscenes import NuScenesTables

    synth_tables = NuScenesTables(root)
    spec = BEVSpec(x_range=(-20.0, 60.0), y_range=(-20.0, 20.0), resolution_m=1.0)
    windows = extract_scene_windows(
        synth_tables, ["scene-0103"], past_frames=3, future_frames=6, spec=spec
    )
    assert len(windows) == 4

    predictors = train_reference_models(
        windows,
        past_frames=3,
        future_frames=6,
        spec=spec,
        seed=42,
        epochs=2,
        diffusion_steps=3,
    )

    def _fast_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
        return float(((real.mean(dim=0) - fake.mean(dim=0)) ** 2).sum().item() * 100.0 + 1.0)

    results = evaluate_ood_shift(
        predictors,
        synth_tables,
        ["scene-0103"],
        shift_m=0.5,
        past_frames=3,
        future_frames=6,
        spec=spec,
        fid_fn=_fast_fid,
    )

    assert set(results) == {"object_centric", "bev", "latent_bev"}
    for entry in results.values():
        assert math.isfinite(entry["baseline"]["cd"]["avg"])
        assert math.isfinite(entry["shifted"]["cd"]["avg"])
        assert math.isfinite(entry["cd_drop_pct"])
        assert math.isfinite(entry["fid_drop_pct"])

    table = format_ood_markdown_table(results, shift_m=0.5)
    assert "Object-centric" in table
    assert "BEV" in table
    assert "Latent BEV" in table
    assert "% CD Drop" in table
    assert "% FID Drop" in table
