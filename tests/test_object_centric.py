import math

import pytest
import torch

from skynet.models.object_centric.scene_repr import CLASSES, FEATURES, build_object_scene, class_id


@pytest.fixture
def scene(tables):
    return build_object_scene(tables, "scene-0103")


@pytest.fixture
def global_scene(tables):
    return build_object_scene(tables, "scene-0103", frame="global")


def test_tensor_has_frames_agents_features_shape(scene):
    assert (tuple(scene.features.shape), tuple(scene.mask.shape), scene.features.dtype) == (
        (8, 3, len(FEATURES)),
        (8, 3),
        torch.float32,
    )


def test_every_annotation_is_kept(scene):
    assert scene.num_annotations == 8 + 4 + 2


def test_track_ids_follow_first_appearance(scene):
    assert (scene.track_ids, scene.categories) == (
        ["inst-bar", "inst-car", "inst-ped"],
        ["movable_object.barrier", "vehicle.car", "human.pedestrian.adult"],
    )


def test_presence_mask_matches_annotations(scene):
    assert scene.mask.T.tolist() == [
        [True, True] + [False] * 6,
        [True] * 8,
        [False, False, True, True, True, True, False, False],
    ]


def test_car_features_in_ego_frame(scene):
    assert scene.features[3, 1].tolist() == pytest.approx(
        [20.0, -5.0, 0.8, 4.5, 2.0, 1.6, 0.0, 1.0, 10.0, 0.0, CLASSES.index("vehicle")], abs=1e-5
    )


def test_car_features_in_global_frame(global_scene):
    assert global_scene.features[3, 1].tolist() == pytest.approx(
        [5.0, 35.0, 0.8, 4.5, 2.0, 1.6, 1.0, 0.0, 0.0, 10.0, CLASSES.index("vehicle")], abs=1e-5
    )


def test_velocity_is_invalid_and_zero_at_first_appearance(scene):
    assert (
        scene.velocity_valid[:, 2].tolist(),
        scene.features[2, 2, [FEATURES.index("vx"), FEATURES.index("vy")]].tolist(),
    ) == ([False, False, False, True, True, True, False, False], [0.0, 0.0])


def test_absent_agents_have_zero_rows(scene):
    assert scene.features[5, 0].abs().sum().item() == 0.0


def test_frame_features_returns_present_agents_only(scene):
    assert (tuple(scene.frame_features(2).shape), scene.frame_track_ids(2)) == (
        (2, len(FEATURES)),
        ["inst-car", "inst-ped"],
    )


def test_timestamps_start_at_zero(scene):
    assert scene.timestamps.tolist() == pytest.approx([0.5 * k for k in range(8)])


def test_unknown_frame_raises(tables):
    with pytest.raises(ValueError, match="frame"):
        build_object_scene(tables, "scene-0103", frame="lidar")


@pytest.mark.parametrize(
    "category, expected",
    [
        ("vehicle.car", "vehicle"),
        ("vehicle.bus.rigid", "vehicle"),
        ("vehicle.bicycle", "two_wheeler"),
        ("vehicle.motorcycle", "two_wheeler"),
        ("human.pedestrian.child", "pedestrian"),
        ("animal", "animal"),
        ("movable_object.trafficcone", "obstacle"),
        ("static_object.bicycle_rack", "obstacle"),
    ],
)
def test_class_id_maps_categories_to_coarse_classes(category, expected):
    assert CLASSES[class_id(category)] == expected


def test_yaw_is_encoded_as_unit_vector(scene):
    s, c = FEATURES.index("yaw_sin"), FEATURES.index("yaw_cos")

    norms = torch.hypot(scene.features[..., s], scene.features[..., c])[scene.mask]

    assert torch.allclose(norms, torch.ones_like(norms))


def test_ego_and_global_frames_agree_on_distances(scene, global_scene):
    x, y = FEATURES.index("x"), FEATURES.index("y")
    car, ped = scene.features[3, 1, [x, y]], scene.features[3, 2, [x, y]]
    gcar, gped = global_scene.features[3, 1, [x, y]], global_scene.features[3, 2, [x, y]]

    assert math.isclose(torch.dist(car, ped).item(), torch.dist(gcar, gped).item(), rel_tol=1e-6)
