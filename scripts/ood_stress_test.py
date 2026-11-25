"""Run the 0.5m lateral ego-shift OOD stress test across all three scene representations.

Usage::

    python scripts/ood_stress_test.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from skynet.config import SCENE_REPRS, load_config
from skynet.data.download import find_or_download
from skynet.data.nuscenes import NuScenesTables
from skynet.eval.ood import (
    BEVSpec,
    evaluate_ood_shift,
    extract_scene_windows,
    format_ood_markdown_table,
    train_reference_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-root", default=None, help="defaults to paths.data_root")
    parser.add_argument("--train-split", default="mini_train")
    parser.add_argument("--val-split", default="mini_val")
    parser.add_argument("--shift-m", type=float, default=None, help="defaults to eval.ood_shift_m")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fid-feature", type=int, default=64)
    parser.add_argument("--output", default="results/ood_stress_test.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    version = cfg["data"]["version"]
    past_frames = int(cfg["data"]["past_frames"])
    future_frames = int(cfg["data"]["future_frames"])
    shift_m = float(cfg["eval"]["ood_shift_m"] if args.shift_m is None else args.shift_m)
    diffusion_steps = int(cfg.get("diffusion", {}).get("steps", 10))
    seed = int(cfg["seed"])

    root = find_or_download(
        args.data_root or cfg["paths"]["data_root"], cfg["paths"]["data_root"], version
    )
    print(f"nuScenes tables: {root / version}")
    tables = NuScenesTables(root, version)
    train_scenes = tables.split_scene_names(args.train_split)
    val_scenes = tables.split_scene_names(args.val_split)

    spec = BEVSpec()
    train_windows = extract_scene_windows(tables, train_scenes, past_frames, future_frames, spec)
    print(
        f"Training {len(SCENE_REPRS)} reference predictors on {len(train_windows)} windows "
        f"({args.train_split}, epochs={args.epochs}, seed={seed})..."
    )
    predictors = train_reference_models(
        train_windows,
        past_frames=past_frames,
        future_frames=future_frames,
        spec=spec,
        seed=seed,
        epochs=args.epochs,
        diffusion_steps=min(diffusion_steps, 10),
    )

    results = evaluate_ood_shift(
        predictors,
        tables,
        val_scenes,
        shift_m=shift_m,
        past_frames=past_frames,
        future_frames=future_frames,
        spec=spec,
        fid_feature=args.fid_feature,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": f"nuScenes {version}",
        "train_split": args.train_split,
        "val_split": args.val_split,
        "val_scenes": val_scenes,
        "seed": seed,
        "epochs": args.epochs,
        "past_frames": past_frames,
        "future_frames": future_frames,
        "shift_m": shift_m,
        "fid_feature": args.fid_feature,
        "bev_spec": {
            "x_range": list(spec.x_range),
            "y_range": list(spec.y_range),
            "resolution_m": spec.resolution_m,
        },
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nOOD stress test ({shift_m:g}m lateral ego shift) on {args.val_split}:")
    print(format_ood_markdown_table(results, shift_m=shift_m))
    print("\nPer-horizon Chamfer Distance (m):")
    print("| Representation | Split | CD @ 1s | CD @ 2s | CD @ 3s | CD Avg |")
    print("|---|---|:---:|:---:|:---:|:---:|")
    for name, entry in results.items():
        for tag, key in (("0m (baseline)", "baseline"), (f"{shift_m:g}m (shifted)", "shifted")):
            cd = entry[key]["cd"]
            print(
                f"| {name} | {tag} | {cd['1s']:.3f} | {cd['2s']:.3f} "
                f"| {cd['3s']:.3f} | {cd['avg']:.3f} |"
            )
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
