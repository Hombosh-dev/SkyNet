import numpy as np
import pytest

from skynet.data.nuscenes import MINI_SPLITS, NuScenesTables


def test_scene_samples_follow_next_links_not_file_order(tables):
    assert tables.scene_sample_tokens("scene-0103") == [f"s{k}" for k in range(8)]


def test_ego_pose_ignores_sweeps(tables):
    _, translation = tables.ego_pose("s2")

    assert translation.tolist() == [0.0, 10.0, 0.0]


def test_unknown_scene_raises(tables):
    with pytest.raises(KeyError):
        tables.scene_sample_tokens("scene-9999")


@pytest.mark.parametrize("split", ["mini_train", "mini_val"])
def test_split_scene_names_returns_official_mini_split(tables, split):
    assert tables.split_scene_names(split) == MINI_SPLITS[split]


def test_split_all_lists_every_scene(tables):
    assert tables.split_scene_names("all") == ["scene-0103"]


def test_unknown_split_raises(tables):
    with pytest.raises(ValueError, match="unknown split"):
        tables.split_scene_names("val")


def test_missing_tables_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        NuScenesTables(tmp_path)


def test_timestamp_is_in_seconds(tables):
    assert np.isclose(tables.timestamp("s1") - tables.timestamp("s0"), 0.5)
