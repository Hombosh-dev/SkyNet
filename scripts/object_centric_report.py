"""Convert nuScenes scenes to the object-centric representation and evaluate baselines.

Usage::

    python scripts/object_centric_report.py

nuScenes is looked up in ``--data-root``, ``$NUSCENES_ROOT``, ``paths.data_root`` and
the usual Colab / Kaggle folders; if it is nowhere, the JSON tables of v1.0-mini are
downloaded into ``paths.data_root`` (the 4 GB archive is streamed, ~40 MB is kept).

Checks that every annotation of every scene lands in the tensor (no object loss),
evaluates the persistence and constant-velocity baselines with ADE / FDE / miss rate
at 1, 2 and 3 s and the past-only ego baselines with L2, writes
``results/object_centric_metrics.json`` and prints Markdown tables for the README.
Baselines are not trained, so evaluating on all mini scenes (the default) involves
no train/test leakage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from skynet.baselines.ego import evaluate_ego_baselines
from skynet.baselines.object_centric import BASELINES, forecast_windows
from skynet.config import load_config
from skynet.data.download import find_or_download
from skynet.data.nuscenes import NuScenesTables
from skynet.eval.object_metrics import displacement_errors
from skynet.models.object_centric.scene_repr import FEATURES, build_object_scene


def scene_stats(tables: NuScenesTables, name: str) -> dict:
    scene = build_object_scene(tables, name, frame="ego")
    expected = sum(len(tables.annotations_of.get(t, [])) for t in scene.sample_tokens)
    per_frame = scene.mask.sum(dim=1).double()
    return {
        "keyframes": scene.features.shape[0],
        "agents": scene.features.shape[1],
        "annotations": expected,
        "in_tensor": scene.num_annotations,
        "no_object_loss": scene.num_annotations == expected,
        "agents_per_frame_mean": round(float(per_frame.mean()), 2),
        "agents_per_frame_max": int(per_frame.max()),
        "tensor_shape": list(scene.features.shape),
    }


def evaluate(tables: NuScenesTables, names: list[str], num_future: int) -> dict:
    preds = {k: [] for k in BASELINES}
    gts, valids = [], []
    for name in names:
        scene = build_object_scene(tables, name, frame="global")
        for _, pos, vel, gt, valid, dt in forecast_windows(scene, num_future):
            for key, fn in BASELINES.items():
                preds[key].append(fn(pos, vel, dt))
            gts.append(gt)
            valids.append(valid)
    gt, valid = torch.cat(gts), torch.cat(valids)
    out = {key: displacement_errors(torch.cat(p), gt, valid) for key, p in preds.items()}
    out["num_agent_windows"] = int(gt.shape[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-root", default=None, help="defaults to paths.data_root")
    parser.add_argument("--split", default="all", help="mini_train, mini_val or all")
    parser.add_argument("--output", default="results/object_centric_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    version = cfg["data"]["version"]
    root = find_or_download(
        args.data_root or cfg["paths"]["data_root"], cfg["paths"]["data_root"], version
    )
    print(f"nuScenes tables: {root / version}")
    tables = NuScenesTables(root, version)
    names = tables.split_scene_names(args.split)
    stats = {name: scene_stats(tables, name) for name in names}
    metrics = evaluate(tables, names, cfg["data"]["future_frames"])
    ego = {args.split: evaluate_ego_baselines(tables, names)}
    if "scene-0061" in tables.scene_by_name:  # the scene the BEV notebook benchmarks on
        ego["scene-0061 (as in the BEV notebook)"] = evaluate_ego_baselines(tables, ["scene-0061"])

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": f"nuScenes {cfg['data']['version']}",
        "split": args.split,
        "features": list(FEATURES),
        "all_scenes_lossless": all(s["no_object_loss"] for s in stats.values()),
        "scenes": stats,
        "baselines": metrics,
        "ego_baselines_l2_m": ego,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    agents = sum(s["agents"] for s in stats.values())
    anns = sum(s["annotations"] for s in stats.values())
    lossless = report["all_scenes_lossless"]
    print(f"{len(names)} scenes, {agents} agents, {anns} annotations, lossless: {lossless}")
    print(f"{metrics['num_agent_windows']} agent windows; written to {args.output}\n")
    print("| Baseline | ADE @ 1s | ADE @ 2s | ADE @ 3s | FDE @ 3s | Miss rate @ 3s |")
    print("|---|:---:|:---:|:---:|:---:|:---:|")
    for key in BASELINES:
        m = metrics[key]
        print(
            f"| {key.replace('_', ' ').title()} | {m['ade']['1s']:.2f} | {m['ade']['2s']:.2f} "
            f"| {m['ade']['3s']:.2f} | {m['fde']['3s']:.2f} | {100 * m['miss_rate']['3s']:.1f}% |"
        )

    for where, res in ego.items():
        print(f"\nEgo baselines, past only, {where}: {res['num_windows']} windows")
        print("| Method / Model | L2 @ 1s (m) | L2 @ 2s (m) | L2 @ 3s (m) | L2 Avg (m) |")
        print("|---|:---:|:---:|:---:|:---:|")
        for key, label in (
            ("zero_motion", "Zero-Motion"),
            ("constant_velocity", "Constant Velocity"),
        ):
            r = res[key]
            print(f"| {label} | {r['1s']:.2f} | {r['2s']:.2f} | {r['3s']:.2f} | {r['avg']:.2f} |")


if __name__ == "__main__":
    main()
