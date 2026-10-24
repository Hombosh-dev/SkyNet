# Experiment Log: SkyNet

Results of the three scene representations (object-centric, BEV, latent BEV) on nuScenes mini.

## BEV representation

### Baselines
- **Zero-Motion (Static Baseline)**: Assumes the ego vehicle remains stationary at (0, 0).
- **Constant Velocity (Linear Extrapolation)**: Extrapolates the velocity between the previous keyframe and now (past only, `src/skynet/baselines/ego.py`). The first version read the velocity from the first future waypoint; its numbers were removed and are being recomputed.

### Proposed Model: BEVFusion + EgoPlanner
End-to-end multimodal perception using 6 surround cameras (Lift-Splat ResNet) + 3D LiDAR (PointPillars) fused into a unified BEV spatial representation `[B, 256, 200, 200]` and decoded by a lightweight trajectory planning head.

| Method / Model | L2 @ 1s (m) ↓ | L2 @ 2s (m) ↓ | L2 @ 3s (m) ↓ | L2 Avg (m) ↓ |
|---|:---:|:---:|:---:|:---:|
| Zero-Motion Baseline | 4.82 | 9.94 | 15.31 | 10.02 |
| Constant Velocity Baseline | — | — | — | — |
| **BEVFusion + EgoPlanner (Ours)** | **0.38** | **0.89** | **1.52** | **0.93** |

## Object-centric representation

Scene tensor `[T, N_agents, 11]` from nuScenes annotations
(`src/skynet/models/object_centric/scene_repr.py`); every annotation kept.

### Baselines
- **Persistence**: every agent stays where it was last seen.
- **Constant Velocity**: every agent keeps its last observed velocity (backward difference).

Metrics: ADE / FDE / miss rate (> 2 m) per horizon, all agents of all mini scenes
(`notebooks/ObjectCentric_Representation.ipynb` or `python scripts/object_centric_report.py`).

| Baseline | ADE @ 1s (m) ↓ | ADE @ 2s (m) ↓ | ADE @ 3s (m) ↓ | FDE @ 3s (m) ↓ | Miss rate @ 3s ↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| Persistence | — | — | — | — | — |
| Constant Velocity | — | — | — | — | — |

## Latent BEV representation

Planned.
