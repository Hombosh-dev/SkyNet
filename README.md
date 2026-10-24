# 🚗 SkyNet: Which Scene Representation Is Best for a Driving World Model?

A reproducible benchmark of world models for autonomous driving on nuScenes. We compare three ways to represent the same road scene — **object-centric**, **Bird's-Eye View (BEV)** and **latent BEV** — and measure which one predicts future scene states most accurately and stably under a single shared protocol (3 past frames → 6 future frames, 1/2/3 s horizons).

![lint](https://github.com/Hombosh-dev/SkyNet/actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

| Representation | What the model sees | Status |
|---|---|---|
| **Object-centric** | Every agent as a feature vector (box, heading, velocity, class) | Scene tensor + baselines, `ObjectCentric_Representation.ipynb` |
| **BEV** | Camera + LiDAR fused into a top-down grid | BEVFusion encoder + EgoPlanner in the notebook |
| **Latent BEV** | The BEV grid compressed by an autoencoder | Planned |

---

## Object-centric representation

`build_object_scene` turns one nuScenes scene into a tensor `[T, N_agents, 11]` plus a presence mask `[T, N_agents]`. The agent axis is the tracking id (one nuScenes instance per index), and every annotation is kept (no object loss).
- **Features:** position `x, y, z`, size `length, width, height`, heading `sin/cos yaw`, velocity `vx, vy`, coarse class (vehicle, pedestrian, two-wheeler, obstacle, animal).
- **Frames:** each keyframe in its own ego frame (`frame="ego"`) or in world coordinates (`frame="global"`).
- **Velocity:** backward difference to the previous annotation, so frame *t* never uses information from later frames.
- **Metrics:** ADE / FDE / miss rate (> 2 m) at 1, 2 and 3 s, next to the shared L2 and collision rate.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hombosh-dev/SkyNet/blob/feat/object-centric-repr/notebooks/ObjectCentric_Representation.ipynb)

* **Notebook:** [`notebooks/ObjectCentric_Representation.ipynb`](notebooks/ObjectCentric_Representation.ipynb), self-contained: fetches the nuScenes mini tables by itself, checks every scene for object loss, plots the agents, their forecasts and an animated scene, and prints the tables below.
* **Script:** `python scripts/object_centric_report.py` finds nuScenes (or downloads the mini tables) and writes `results/object_centric_metrics.json`.

| Baseline (nuScenes mini, all agents) | ADE @ 1s (m) ↓ | ADE @ 2s (m) ↓ | ADE @ 3s (m) ↓ | FDE @ 3s (m) ↓ | Miss rate @ 3s ↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| Persistence | — | — | — | — | — |
| Constant Velocity | — | — | — | — | — |

*Fill the rows with the table printed by the notebook or the script.*

---

## BEV representation

BEVFusion fuses **6 synchronized surround-view camera feeds** (`[B, 6, 3, 128, 352]`) with **3D LiDAR point clouds** (`[N, 4]`) into a shared metric BEV coordinate space:
- **Camera BEV Branch**: ResNet-18 stem + learned depth distribution projection (`Lift-Splat-Pool`) into vehicle ego coordinates.
- **LiDAR BEV Branch**: PointPillars feature network + 2D CNN encoder in ego coordinates.
- **Multimodal Fusion**: Cross-modal channel concatenation + residual convolution blocks producing a unified feature representation (`[B, 256, 200, 200]`).
- **Downstream EgoPlanner**: Trajectory planning head predicting 3.0s future vehicle waypoints (+0.5s to +3.0s) evaluated against human driver ground truth.

### Midterm Benchmark Results

Evaluation of planning displacement error ($L_2$ error in meters, lower is better) on nuScenes driving sequences:

| Method / Model | L2 @ 1s (m) ↓ | L2 @ 2s (m) ↓ | L2 @ 3s (m) ↓ | L2 Avg (m) ↓ | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Zero-Motion Baseline** | 4.82 | 9.94 | 15.31 | 10.02 | Stationary vehicle assumption ($x=0, y=0$) |
| **Constant Velocity** | — | — | — | — | Linear extrapolation of the last observed velocity (past only) |
| **SkyNet (BEVFusion + EgoPlanner)** | **0.38** | **0.89** | **1.52** | **0.93** | **Multimodal BEV representation + planner** |

The constant-velocity row is being recomputed: the notebook version took its velocity from the first future waypoint. The past-only replacement (`src/skynet/baselines/ego.py`) is printed by the object-centric notebook and script.

### Interactive Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hombosh-dev/SkyNet/blob/feat/bevfusion-encoder/notebooks/BEVFusion_Encoder.ipynb)

* **Location:** [`notebooks/BEVFusion_Encoder.ipynb`](notebooks/BEVFusion_Encoder.ipynb)
* **Zero-Touch Execution:** Automatically downloads and unpacks nuScenes mini if not present, runs multimodal forward pass, verifies acceptance criteria, benchmarks baselines, and generates animated driving simulations.

---

## Evaluation protocol

All representations use the same data split, horizons (1/2/3 s = 2/4/6 keyframes at 2 Hz) and baselines (persistence, constant velocity). Shared metrics: ego **L2** and **collision rate** (`src/skynet/eval/planning_metrics.py`), **Chamfer Distance** on point clouds and **FID** on images (`reconstruction_metrics.py`). Representation-specific metrics (ADE / FDE for object-centric) are reported alongside, with their limitations documented.

---

## Repository Structure

```
.
├── src/skynet/           # Package: config, nuScenes reader/download, object-centric representation, baselines, metrics
├── notebooks/            # BEVFusion_Encoder.ipynb (BEV), ObjectCentric_Representation.ipynb (object-centric)
├── tests/                # pytest suite for config, metrics, nuScenes reader, object-centric representation
├── config/default.yaml   # Representation switch, frame counts, horizons, paths
├── data/                 # Data card and raw archive checksums
├── models/               # Model checkpoints and weights documentation
├── results/              # metrics.json, object_centric_metrics.json, experiments.md
├── scripts/              # object_centric_report.py, slide generator
├── app/                  # Interactive demo (planned)
├── docs/                 # Proposal, slides, and project documentation
└── .github/workflows/    # CI lint workflow
```

---

## Quick Start & Development

```bash
# Clone the repository
git clone git@github.com:Hombosh-dev/SkyNet.git && cd SkyNet

# Setup environment and install dependencies
make setup

# Run the test suite
pytest tests/

# Code quality and style checks
flake8 src/ scripts/ tests/ && black --check src/ scripts/ tests/ && isort --check src/ scripts/ tests/
```

---

## License

Code is licensed under the [MIT License](LICENSE).
Built from [`ucu-ai-course/project-template`](https://github.com/ucu-ai-course/project-template).
