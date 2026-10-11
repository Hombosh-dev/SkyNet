
"""Tests for config loading, overrides and validation."""

import pytest

from skynet.config import ConfigError, load_config

PATH = "config/default.yaml"


@pytest.mark.parametrize("name", ["object_centric", "bev", "latent_bev"])
def test_all_representations_load(name):
    cfg = load_config(PATH, overrides=[f"scene_repr={name}"])
    assert cfg["scene_repr"] == name


def test_invalid_scene_repr_raises():
    with pytest.raises(ConfigError):
        load_config(PATH, overrides=["scene_repr=foo"])


def test_frame_counts_fixed():
    cfg = load_config(PATH)
    assert cfg["data"]["past_frames"] == 3
    assert cfg["data"]["future_frames"] == 6


def test_override_nested_key():
    cfg = load_config(PATH, overrides=["eval.ood_shift_m=1.0"])
    assert cfg["eval"]["ood_shift_m"] == 1.0


def test_override_unknown_key_raises():
    with pytest.raises(ConfigError):
        load_config(PATH, overrides=["data.nope=1"])


def test_override_without_equals_raises():
    with pytest.raises(ConfigError):
        load_config(PATH, overrides=["scene_repr"])


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
