import math

import pytest
import torch

from skynet.eval.object_metrics import displacement_errors


def _offset(values):
    gt = torch.zeros(1, 6, 2)
    pred = gt.clone()
    pred[0, :, 0] = torch.tensor(values)
    return pred, gt


def test_constant_offset_gives_same_ade_and_fde():
    pred, gt = _offset([1.0] * 6)

    out = displacement_errors(pred, gt)

    assert out == {
        "ade": pytest.approx({"1s": 1.0, "2s": 1.0, "3s": 1.0}),
        "fde": pytest.approx({"1s": 1.0, "2s": 1.0, "3s": 1.0}),
        "miss_rate": pytest.approx({"1s": 0.0, "2s": 0.0, "3s": 0.0}),
    }


def test_growing_offset_separates_ade_fde_and_misses():
    pred, gt = _offset([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    out = displacement_errors(pred, gt)

    assert out == {
        "ade": pytest.approx({"1s": 0.75, "2s": 1.25, "3s": 1.75}),
        "fde": pytest.approx({"1s": 1.0, "2s": 2.0, "3s": 3.0}),
        "miss_rate": pytest.approx({"1s": 0.0, "2s": 0.0, "3s": 1.0}),
    }


def test_masked_steps_are_ignored():
    pred = torch.zeros(2, 6, 2)
    gt = torch.zeros(2, 6, 2)
    gt[1, 3:, 0] = 100.0
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, 3:] = False

    out = displacement_errors(pred, gt, valid)

    assert (out["fde"]["3s"], out["ade"]["3s"]) == (0.0, 0.0)


def test_horizon_without_valid_step_is_nan():
    pred, gt = _offset([1.0] * 6)
    valid = torch.tensor([[True] * 5 + [False]])

    assert math.isnan(displacement_errors(pred, gt, valid)["fde"]["3s"])


@pytest.mark.parametrize(
    "pred, gt",
    [
        (torch.zeros(1, 6, 2), torch.zeros(1, 5, 2)),
        (torch.zeros(6, 2), torch.zeros(6, 2)),
        (torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)),
    ],
)
def test_rejects_bad_shapes(pred, gt):
    with pytest.raises(ValueError):
        displacement_errors(pred, gt)


def test_rejects_mask_of_wrong_shape():
    with pytest.raises(ValueError, match="valid"):
        displacement_errors(
            torch.zeros(1, 6, 2), torch.zeros(1, 6, 2), torch.ones(1, 5, dtype=torch.bool)
        )
