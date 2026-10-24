import json
from pathlib import Path

import pytest
import torch

from skynet.baselines.ego import evaluate_ego_baselines
from skynet.baselines.object_centric import BASELINES, forecast_windows
from skynet.eval.object_metrics import displacement_errors
from skynet.models.object_centric.scene_repr import FEATURES, build_object_scene

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "ObjectCentric_Representation.ipynb"
PRELUDE = """
import io, json, math, os, shutil, tarfile, urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
"""


def _sources():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]


@pytest.fixture(scope="module")
def nb():
    namespace = {}
    exec(PRELUDE, namespace)
    for source in _sources():
        if source.startswith("# [library]"):
            exec(source, namespace)
    return namespace


@pytest.fixture
def nb_tables(nb, nuscenes_root):
    return nb["NuScenesTables"](nuscenes_root)


def test_notebook_needs_nothing_from_the_repository():
    code = "".join(_sources())

    assert ("from skynet" in code, "import skynet" in code, "git clone" in code) == (
        False,
        False,
        False,
    )


def test_notebook_features_match_package(nb, nb_tables, tables):
    ours = nb["build_object_scene"](nb_tables, "scene-0103")
    pkg = build_object_scene(tables, "scene-0103")

    assert (
        tuple(nb["FEATURES"]) == FEATURES,
        torch.equal(ours.features, pkg.features),
        torch.equal(ours.mask, pkg.mask),
        ours.track_ids == pkg.track_ids,
    ) == (True, True, True, True)


def test_notebook_object_baselines_and_metrics_match_package(nb, nb_tables, tables):
    ours = list(
        nb["forecast_windows"](nb["build_object_scene"](nb_tables, "scene-0103", frame="global"))
    )
    pkg = list(forecast_windows(build_object_scene(tables, "scene-0103", frame="global")))

    for (_, pos, vel, gt, valid, dt), (_, ppos, pvel, pgt, pvalid, pdt) in zip(ours, pkg):
        for key in BASELINES:
            assert nb["displacement_errors"](
                nb["BASELINES"][key](pos, vel, dt), gt, valid
            ) == displacement_errors(BASELINES[key](ppos, pvel, pdt), pgt, pvalid)


def test_notebook_ego_baselines_match_package(nb, nb_tables, tables):
    assert nb["evaluate_ego_baselines"](nb_tables, ["scene-0103"]) == evaluate_ego_baselines(
        tables, ["scene-0103"]
    )
