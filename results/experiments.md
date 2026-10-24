# Experiment Log: SkyNet Trajectory Planning

Benchmarking results across the nuScenes mini driving sequences.

## Baselines
- **Zero-Motion (Static Baseline)**: Assumes the ego vehicle remains stationary at (0, 0).
- **Constant Velocity (Linear Extrapolation)**: Extrapolates current vehicle velocity vector linearly forward in time.

## Proposed Model: BEVFusion + EgoPlanner
End-to-end multimodal perception using 6 surround cameras (Lift-Splat ResNet) + 3D LiDAR (PointPillars) fused into a unified BEV spatial representation `[B, 256, 200, 200]` and decoded by a lightweight trajectory planning head.

| Method / Model | L2 @ 1s (m) ↓ | L2 @ 2s (m) ↓ | L2 @ 3s (m) ↓ | L2 Avg (m) ↓ |
|---|:---:|:---:|:---:|:---:|
| Zero-Motion Baseline | 4.82 | 9.94 | 15.31 | 10.02 |
| Constant Velocity Baseline | 1.15 | 2.68 | 4.42 | 2.75 |
| **BEVFusion + EgoPlanner (Ours)** | **0.38** | **0.89** | **1.52** | **0.93** |
