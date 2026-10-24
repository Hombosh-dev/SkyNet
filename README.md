# 🚗 SkyNet: Multimodal BEV Perception & Planning

End-to-end multimodal Bird's-Eye-View (BEV) perception and trajectory planning architecture for autonomous vehicles on the nuScenes dataset.

![lint](https://github.com/Hombosh-dev/SkyNet/actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Overview

SkyNet fuses **6 synchronized surround-view camera feeds** (`[B, 6, 3, 128, 352]`) with **3D LiDAR point clouds** (`[N, 4]`) into a shared metric Bird's-Eye-View (BEV) coordinate space:
- **Camera BEV Branch**: ResNet-18 stem + learned depth distribution projection (`Lift-Splat-Pool`) into vehicle ego coordinates.
- **LiDAR BEV Branch**: PointPillars feature network + 2D CNN encoder in ego coordinates.
- **Multimodal Fusion**: Cross-modal channel concatenation + residual convolution blocks producing a unified feature representation (`[B, 256, 200, 200]`).
- **Downstream EgoPlanner**: Trajectory planning head predicting 3.0s future vehicle waypoints (+0.5s to +3.0s) evaluated against human driver ground truth.

---

## Midterm Benchmark Results

Evaluation of planning displacement error ($L_2$ error in meters, lower is better) on nuScenes driving sequences:

| Method / Model | L2 @ 1s (m) ↓ | L2 @ 2s (m) ↓ | L2 @ 3s (m) ↓ | L2 Avg (m) ↓ | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Zero-Motion Baseline** | 4.82 | 9.94 | 15.31 | 10.02 | Stationary vehicle assumption ($x=0, y=0$) |
| **Constant Velocity** | 1.15 | 2.68 | 4.42 | 2.75 | Linear extrapolation of initial velocity |
| **SkyNet (BEVFusion + EgoPlanner)** | **0.38** | **0.89** | **1.52** | **0.93** | **Multimodal BEV representation + planner** |

---

## Interactive Notebook

The full perception and trajectory planning pipeline is available in an end-to-end Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hombosh-dev/SkyNet/blob/feat/bevfusion-encoder/notebooks/BEVFusion_Encoder.ipynb)

* **Location:** [`notebooks/BEVFusion_Encoder.ipynb`](notebooks/BEVFusion_Encoder.ipynb)
* **Zero-Touch Execution:** Automatically downloads and unpacks nuScenes mini if not present, runs multimodal forward pass, verifies acceptance criteria, benchmarks baselines, and generates animated driving simulations.

---

## Repository Structure

```
.
├── src/skynet/           # Core package: config schema, planning & reconstruction metrics
├── notebooks/            # BEVFusion_Encoder.ipynb (Colab-ready multimodal pipeline)
├── tests/                # test_config.py, test_planning_metrics.py, test_reconstruction_metrics.py
├── config/default.yaml   # Hyperparameters, sensor representations, and paths
├── data/                 # Data card and raw archive checksums
├── models/               # Model checkpoints and weights documentation
├── results/              # metrics.json, experiments.md, and benchmark figures
├── scripts/              # Standalone data & experiment scripts
├── app/                  # Application demo entry point
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
flake8 src/ tests/ && black --check src/ tests/ && isort --check src/ tests/
```

---

## License

Code is licensed under the [MIT License](LICENSE).
Built from [`ucu-ai-course/project-template`](https://github.com/ucu-ai-course/project-template).
