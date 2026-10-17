"""Tests for L2 error and collision rate."""

import pytest
import torch

from skynet.eval.planning_metrics import collision_rate, l2_error

T = 6  # 3 s at 2 Hz


def _traj(offsets):
    """Ground truth on the x axis, prediction shifted sideways by ``offsets``."""
    gt = torch.stack([torch.arange(T, dtype=torch.float32), torch.zeros(T)], dim=-1)
    pred = gt + torch.stack([torch.zeros(T), torch.as_tensor(offsets)], dim=-1)
    return pred, gt


def test_l2_identical_is_zero():
    pred, gt = _traj([0.0] * T)
    assert l2_error(pred, gt) == {"1s": 0.0, "2s": 0.0, "3s": 0.0, "avg": 0.0}


def test_l2_constant_offset():
    pred, gt = _traj([1.0] * T)
    out = l2_error(pred, gt)
    assert out == pytest.approx({"1s": 1.0, "2s": 1.0, "3s": 1.0, "avg": 1.0})


def test_l2_reads_the_step_at_each_horizon():
    """Errors 0.5, 1.0, ..., 3.0: 1s is step 2, 2s is step 4, 3s is step 6."""
    pred, gt = _traj([0.5 * (t + 1) for t in range(T)])
    out = l2_error(pred, gt)
    assert out == pytest.approx({"1s": 1.0, "2s": 2.0, "3s": 3.0, "avg": 2.0})


def test_l2_cumulative_averages_steps_up_to_horizon():
    pred, gt = _traj([0.5 * (t + 1) for t in range(T)])
    out = l2_error(pred, gt, cumulative=True)
    assert out == pytest.approx({"1s": 0.75, "2s": 1.25, "3s": 1.75, "avg": 1.25})


def test_l2_batch_is_mean_over_scenes():
    p1, gt = _traj([1.0] * T)
    p3, _ = _traj([3.0] * T)
    out = l2_error(torch.stack([p1, p3]), torch.stack([gt, gt]))
    assert out["avg"] == pytest.approx(2.0)


def test_l2_rejects_bad_inputs():
    pred, gt = _traj([0.0] * T)
    with pytest.raises(ValueError):
        l2_error(pred, gt[:-1])  # shape mismatch
    with pytest.raises(ValueError):
        l2_error(torch.zeros(T, 3), torch.zeros(T, 3))  # not x/y
    with pytest.raises(ValueError):
        l2_error(pred[:4], gt[:4])  # shorter than 3 s


# --- collision rate -------------------------------------------------------------
# Ego stays at the origin heading along +x. Its footprint is 4.084 x 1.85 m, so it
# spans x in [-2.042, 2.042] and y in [-0.925, 0.925].


def _ego(batch=1):
    return torch.zeros(batch, T, 3)


def _empty():
    return torch.zeros(0, 5)


def _scene(box_at=None, box=None):
    """Boxes for one scene: ``box`` (cx, cy, length, width, yaw) at the given steps."""
    box_at = box_at or []
    return [torch.tensor([box]) if t in box_at else _empty() for t in range(T)]


def test_collision_no_objects():
    out = collision_rate(_ego(), [_scene()])
    assert out == {"1s": 0.0, "2s": 0.0, "3s": 0.0, "avg": 0.0}


def test_collision_box_on_top_of_ego():
    scene = _scene(range(T), (0.0, 0.0, 2.0, 2.0, 0.0))
    assert collision_rate(_ego(), [scene])["avg"] == 1.0


def test_collision_overlap_vs_near_miss():
    hit = _scene(range(T), (3.0, 0.0, 2.0, 2.0, 0.0))  # spans x 2.0..4.0, ego ends 2.042
    miss = _scene(range(T), (3.2, 0.0, 2.0, 2.0, 0.0))  # spans x 2.2..4.2
    assert collision_rate(_ego(), [hit])["avg"] == 1.0
    assert collision_rate(_ego(), [miss])["avg"] == 0.0


def test_collision_rotated_box_crossing_ego():
    scene = _scene(range(T), (0.0, 3.0, 6.0, 0.5, 3.14159265 / 2))  # vertical bar
    assert collision_rate(_ego(), [scene])["avg"] == 1.0


def test_collision_uses_real_shapes_not_bounding_boxes():
    """A 45-degree diamond whose bounding box overlaps the ego corner but does not
    touch it: only a proper rotated-rectangle test gets this right."""
    scene = _scene(range(T), (2.7, 1.5, 1.0, 1.0, 3.14159265 / 4))
    assert collision_rate(_ego(), [scene])["avg"] == 0.0


def test_collision_pointwise_vs_cumulative():
    """Hit only at step 3 (index 2): invisible pointwise, seen cumulatively from 2s."""
    scene = _scene([2], (0.0, 0.0, 2.0, 2.0, 0.0))
    pointwise = collision_rate(_ego(), [scene])
    cumulative = collision_rate(_ego(), [scene], cumulative=True)
    assert pointwise == {"1s": 0.0, "2s": 0.0, "3s": 0.0, "avg": 0.0}
    assert cumulative == pytest.approx({"1s": 0.0, "2s": 1.0, "3s": 1.0, "avg": 2 / 3})


def test_collision_rate_is_fraction_of_scenes():
    hit = _scene(range(T), (0.0, 0.0, 2.0, 2.0, 0.0))
    out = collision_rate(_ego(2), [hit, _scene()])
    assert out == pytest.approx({"1s": 0.5, "2s": 0.5, "3s": 0.5, "avg": 0.5})


def test_collision_rejects_bad_inputs():
    with pytest.raises(ValueError):
        collision_rate(torch.zeros(1, T, 2), [_scene()])  # missing yaw
    with pytest.raises(ValueError):
        collision_rate(_ego(), [])  # one scene, no boxes entry
