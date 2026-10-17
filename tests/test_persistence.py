import torch

from skynet.baselines.persistence import evaluate_persistence, persistence_forecast


def _cloud(x):
    return torch.tensor([[x, 0.0, 0.0]])


def test_forecast_repeats_last_frame():
    past = [_cloud(0.0), _cloud(1.0), _cloud(2.0)]
    forecast = persistence_forecast(past, n_future=6)
    assert len(forecast) == 6
    assert all(torch.equal(f, past[-1]) for f in forecast)


def test_forecast_ignores_older_frames():
    a = persistence_forecast([_cloud(0.0), _cloud(2.0)], n_future=3)
    b = persistence_forecast([_cloud(9.0), _cloud(2.0)], n_future=3)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_static_scene_has_zero_error():
    past = [_cloud(1.0)] * 3
    future = [_cloud(1.0)] * 6
    out = evaluate_persistence([(past, future)])
    assert out == {"1s": 0.0, "3s": 0.0}


def test_moving_scene_error_grows_with_horizon():
    past = [_cloud(0.0)] * 3
    future = [_cloud(0.5 * (i + 1)) for i in range(6)]
    out = evaluate_persistence([(past, future)])
    assert out == {"1s": 2.0, "3s": 6.0}
