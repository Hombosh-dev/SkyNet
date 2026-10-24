import numpy as np
import pytest

from skynet.baselines.ego import constant_velocity, ego_windows, evaluate_ego_baselines, zero_motion


@pytest.fixture
def windows(tables):
    return ego_windows(tables, "scene-0103")


def test_windows_need_a_previous_and_six_future_keyframes(windows):
    assert [w[0] for w in windows] == ["s1"]


def test_window_positions_are_in_current_ego_frame(windows):
    _, prev_xy, dt_prev, future_xy, future_dt = windows[0]

    assert (prev_xy.tolist(), dt_prev, future_xy[:, 0].tolist(), future_dt.tolist()) == (
        pytest.approx([-5.0, 0.0], abs=1e-9),
        pytest.approx(0.5),
        pytest.approx([5.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
        pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    )


def test_constant_velocity_uses_previous_keyframe_only():
    pred = constant_velocity(np.array([-4.0, 1.0]), 0.5, np.array([0.5, 1.0]))

    assert pred.tolist() == [[4.0, -1.0], [8.0, -2.0]]


def test_zero_motion_stays_at_origin():
    assert zero_motion(np.array([-4.0, 0.0]), 0.5, np.ones(3)).tolist() == [[0.0, 0.0]] * 3


def test_evaluate_ego_baselines_on_constant_speed_ego(tables):
    out = evaluate_ego_baselines(tables, ["scene-0103"])

    assert (out["num_windows"], out["zero_motion"], out["constant_velocity"]) == (
        1,
        pytest.approx({"1s": 10.0, "2s": 20.0, "3s": 30.0, "avg": 20.0}),
        pytest.approx({"1s": 0.0, "2s": 0.0, "3s": 0.0, "avg": 0.0}, abs=1e-9),
    )
