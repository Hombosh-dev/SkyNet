"""Tests for Chamfer Distance and FID."""

import pytest
import torch

from skynet.eval.reconstruction_metrics import chamfer_distance, fid_score


def _chamfer_loops(a, b):
    """Slow reference implementation: same algorithm, no vectorization."""

    def one_way(x, y):
        return sum(min(float(torch.dist(p, q)) for q in y) for p in x) / len(x)

    return one_way(a, b) + one_way(b, a)


def test_chamfer_identical_clouds_is_zero():
    """Exact equality is intended: cdist runs in exact (non-matmul) mode.

    With the default mode, rounding error would give a tiny non-zero value.
    """
    pts = torch.randn(100, 3)
    assert chamfer_distance(pts, pts) == 0.0


def test_chamfer_known_value():
    """Hand-computed case: two points 5 apart, counted once per direction."""
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[3.0, 4.0, 0.0]])
    assert chamfer_distance(a, b) == pytest.approx(10.0)  # 5 + 5


def test_chamfer_is_symmetric():
    torch.manual_seed(0)
    a, b = torch.randn(50, 3), torch.randn(70, 3)
    assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a))


def test_chamfer_matches_loop_reference():
    """Guards the vectorized version against the plain double-loop definition."""
    torch.manual_seed(0)
    a, b = torch.randn(20, 3), torch.randn(30, 3)
    assert chamfer_distance(a, b) == pytest.approx(_chamfer_loops(a, b), abs=1e-5)


def test_chamfer_is_deterministic():
    torch.manual_seed(0)
    a, b = torch.randn(200, 3), torch.randn(200, 3)
    assert chamfer_distance(a, b) == chamfer_distance(a, b)


def test_chamfer_rejects_bad_shape():
    with pytest.raises(ValueError):
        chamfer_distance(torch.randn(3, 100), torch.randn(100, 3))


def test_chamfer_rejects_empty_cloud():
    with pytest.raises(ValueError):
        chamfer_distance(torch.empty(0, 3), torch.randn(10, 3))


def test_fid_orders_distributions_and_is_deterministic():
    """Sanity check only: feature=64 is not a real FID, small samples are noisy.

    We check the ordering (same distribution < different) and repeatability,
    not absolute values."""
    pytest.importorskip("torch_fidelity")  # skipped if the FID extra is missing
    torch.manual_seed(0)
    a = torch.rand(64, 3, 64, 64)
    b = torch.rand(64, 3, 64, 64)  # same distribution as a
    c = 0.9 + 0.05 * torch.rand(64, 3, 64, 64)  # clearly different
    same = fid_score(a, b, feature=64)
    different = fid_score(a, c, feature=64)
    assert same < different
    assert fid_score(a, c, feature=64) == pytest.approx(different)


def test_fid_identical_batches():
    """Sanity check: FID between identical batches should be near zero."""
    pytest.importorskip("torch_fidelity")
    torch.manual_seed(0)
    a = torch.rand(10, 3, 32, 32)
    score = fid_score(a, a, feature=64)
    assert score < 1e-4


def test_fid_rejects_bad_shape():
    """Ensure fid_score catches wrong dimensions or channel counts."""
    valid = torch.rand(2, 3, 32, 32)

    with pytest.raises(ValueError, match="dimensions"):
        fid_score(torch.rand(3, 32, 32), valid)

    with pytest.raises(ValueError, match="channels"):
        fid_score(torch.rand(2, 1, 32, 32), valid)

    with pytest.raises(ValueError, match="channels"):
        fid_score(torch.rand(2, 32, 32, 3), valid)


def test_fid_rejects_bad_range():
    """Ensure fid_score catches unnormalized data."""
    valid = torch.rand(2, 3, 32, 32)

    with pytest.raises(ValueError, match="range"):
        fid_score(torch.rand(2, 3, 32, 32) * 255.0, valid)  # [0, 255]

    with pytest.raises(ValueError, match="range"):
        fid_score(torch.rand(2, 3, 32, 32) - 0.5, valid)  # Negative samples
