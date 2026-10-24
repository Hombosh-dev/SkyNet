import pytest
import torch

from skynet.baselines.object_centric import constant_velocity, forecast_windows, persistence
from skynet.eval.object_metrics import displacement_errors
from skynet.models.object_centric.scene_repr import build_object_scene


@pytest.fixture
def windows(tables):
    return list(forecast_windows(build_object_scene(tables, "scene-0103", frame="global")))


def test_windows_start_where_six_future_frames_exist(windows):
    assert [w[0] for w in windows] == [0, 1]


def test_window_shapes(windows):
    _, pos, vel, gt, valid, dt = windows[1]

    assert (
        tuple(pos.shape),
        tuple(vel.shape),
        tuple(gt.shape),
        tuple(valid.shape),
        dt.tolist(),
    ) == (
        (2, 2),
        (2, 2),
        (2, 6, 2),
        (2, 6),
        pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    )


def test_constant_velocity_is_exact_for_constant_speed_car(windows):
    _, pos, vel, gt, valid, dt = windows[1]

    err = displacement_errors(constant_velocity(pos, vel, dt)[1:], gt[1:], valid[1:])

    assert err["fde"]["3s"] == pytest.approx(0.0, abs=1e-6)


def test_persistence_error_equals_distance_travelled(windows):
    _, pos, vel, gt, valid, dt = windows[1]

    err = displacement_errors(persistence(pos, vel, dt)[1:], gt[1:], valid[1:])

    assert err["fde"] == pytest.approx({"1s": 10.0, "2s": 20.0, "3s": 30.0})


def test_persistence_repeats_position():
    pos = torch.tensor([[1.0, 2.0]])

    assert persistence(pos, torch.zeros(1, 2), torch.ones(3)).tolist() == [[[1.0, 2.0]] * 3]


def test_windows_require_global_frame(tables):
    with pytest.raises(ValueError, match="global"):
        list(forecast_windows(build_object_scene(tables, "scene-0103")))
