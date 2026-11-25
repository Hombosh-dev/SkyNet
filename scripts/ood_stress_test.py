"""Prepare and run the 0.5m lateral ego-shift OOD stress test.

Usage::

    python scripts/ood_stress_test.py

Loads ``config/default.yaml``, builds unshifted and laterally shifted ``mini_val``
evaluation windows, and runs ``evaluate_ood_shift`` on any registered
``OODPredictor`` implementations (to be plugged in from Issues #11, #13, #14-#15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skynet.config import load_config
from skynet.data.download import find_or_download
from skynet.data.nuscenes import NuScenesTables
from skynet.eval.ood import (
    BEVSpec,
    OODPredictor,
    evaluate_ood_shift,
    extract_scene_windows,
    format_ood_markdown_table,
    shift_nuscenes_ego,
)


def run_ood_stress_test(
    predictors: dict[str, OODPredictor],
    config_path: str | Path = "config/default.yaml",
    data_root: str | Path | None = None,
    split: str = "mini_val",
    shift_m: float | None = None,
    fid_feature: int = 64,
    output: str | Path | None = None,
) -> dict:
    """Run ``evaluate_ood_shift`` for ``predictors`` using ``config/default.yaml``."""
    cfg = load_config(config_path)
    version = cfg["data"]["version"]
    past_frames = int(cfg["data"]["past_frames"])
    future_frames = int(cfg["data"]["future_frames"])
    effective_shift = float(cfg["eval"]["ood_shift_m"] if shift_m is None else shift_m)

    root = find_or_download(
        data_root or cfg["paths"]["data_root"], cfg["paths"]["data_root"], version
    )
    tables = NuScenesTables(root, version)
    scene_names = tables.split_scene_names(split)
    spec = BEVSpec()

    if not predictors:
        base_windows = extract_scene_windows(tables, scene_names, past_frames, future_frames, spec)
        shifted_tables = shift_nuscenes_ego(tables, effective_shift, scene_names)
        shifted_windows = extract_scene_windows(
            shifted_tables, scene_names, past_frames, future_frames, spec
        )
        print(
            f"Prepared {len(base_windows)} baseline and {len(shifted_windows)} shifted "
            f"({effective_shift:g}m) windows for {split} ({scene_names})."
        )
        print("No model predictors registered yet (waiting on Issues #11, #13, #14-#15).")
        return {}

    results = evaluate_ood_shift(
        predictors,
        tables,
        scene_names,
        shift_m=effective_shift,
        past_frames=past_frames,
        future_frames=future_frames,
        spec=spec,
        fid_feature=fid_feature,
    )
    print(format_ood_markdown_table(results, shift_m=effective_shift))
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-root", default=None, help="defaults to paths.data_root")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--shift-m", type=float, default=None, help="defaults to eval.ood_shift_m")
    parser.add_argument("--fid-feature", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_ood_stress_test(
        predictors={},
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,
        shift_m=args.shift_m,
        fid_feature=args.fid_feature,
        output=args.output,
    )


if __name__ == "__main__":
    main()
