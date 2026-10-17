from skynet.eval.reconstruction_metrics import chamfer_distance

HORIZON_INDEX = {"1s": 1, "3s": 5}


def persistence_forecast(past, n_future=6):
    """Predict that nothing changes: the future is the last observed frame, repeated.

    past: sequence of P frames (each a (N, 3) point cloud). Returns a list of
    n_future frames.
    """
    return [past[-1].clone() for _ in range(n_future)]


def evaluate_persistence(samples):
    """Mean Chamfer Distance of the persistence forecast at 1s and 3s.

    samples: iterable of (past, future); both are lists of (N, 3) point clouds.
    """
    values = {name: [] for name in HORIZON_INDEX}
    for past, future in samples:
        forecast = persistence_forecast(past, n_future=len(future))
        for name, i in HORIZON_INDEX.items():
            values[name].append(chamfer_distance(forecast[i], future[i]))
    return {name: sum(v) / len(v) for name, v in values.items()}
