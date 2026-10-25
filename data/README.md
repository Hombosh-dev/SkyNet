# data/ — data card

## Name and source

**nuScenes** (Caesar et al., 2019), a public autonomous-driving dataset by Motional:
<https://www.nuscenes.org/nuscenes>. 1,000 scenes of ~20 s recorded in Boston and
Singapore, with 6 cameras, 1 LiDAR, 5 radars, HD maps, ego poses and 3D box
annotations of every agent, tracked across frames (instances).

## License and terms of use

CC BY-NC-SA 4.0 plus Motional's additional terms: free for non-commercial use, no
redistribution of the data in this repository. Cite the dataset as:

> H. Caesar et al. *nuScenes: A multimodal dataset for autonomous driving.* CVPR 2020
> (arXiv:1903.11027).

## Versions and splits

| Version | Scenes | Use in this project |
|---|---|---|
| `v1.0-mini` | 10 (`mini_train` 8, `mini_val` 2) | Development, notebook, all current results |
| `v1.0-trainval` | 850 (700 train, 150 val) | Final experiments |

The test split (150 scenes) has no public annotations and is not used. Keyframes
(annotated samples) come at 2 Hz, so the 1/2/3 s horizons are 2/4/6 frames; the
official mini splits are listed in `src/skynet/data/nuscenes.py`.

## How each representation uses the data

- **Object-centric:** `sample_annotation` boxes, linked into tracks by `instance`,
  converted by `src/skynet/models/object_centric/scene_repr.py`.
- **BEV / latent BEV:** the 6 keyframe camera images and the LIDAR_TOP point cloud
  (`.pcd.bin`, 5 floats per point), fused by the notebook's BEVFusion encoder.
- **All:** ego poses for trajectories and for moving data between ego and world frames.

Only keyframe records (`is_key_frame = true`) are used; `sample_data.json` also
contains non-annotated sweeps between keyframes.

## How to get it

Nothing to do by hand for mini: both notebooks download it themselves, and
`scripts/object_centric_report.py` looks for it in `--data-root`, `$NUSCENES_ROOT`,
`paths.data_root` (`data/nuscenes`, gitignored) and the usual Colab / Kaggle folders,
and otherwise streams the archive and keeps only the JSON tables (~40 MB). For the
full dataset, register at nuscenes.org, extract the archives into one folder with
`samples/`, `sweeps/`, `maps/` and `v1.0-*/`, and point `$NUSCENES_ROOT` to it. Raw
archives are never committed; their checksums go to `data/raw/CHECKSUMS.txt`.

## Personal data

Camera images show people and licence plates in public spaces; nuScenes blurs faces
and plates. We use images only as model inputs and do not publish them beyond
illustrative figures.

## Known limitations and biases

- Mini has 10 scenes: every number on it is a pipeline check, not a scientific result.
- Two cities only, left- and right-hand traffic; most scenes are routine driving,
  which makes simple baselines (e.g. constant velocity) strong — open-loop results
  must always be compared against them.
- Annotations are 2 Hz keyframes; motion between keyframes is not observed.
