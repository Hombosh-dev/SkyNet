# Experiment Log: SkyNet

Results of the three scene representations (object-centric, BEV, latent BEV) on nuScenes mini.

## BEV representation

### Baselines
- **Zero-Motion (Static Baseline)**: Assumes the ego vehicle remains stationary at (0, 0).
- **Constant Velocity (Linear Extrapolation)**: Extrapolates the velocity between the previous keyframe and now (past only; same definition in the notebook and `src/skynet/baselines/ego.py`).

### Proposed Model: BEVFusion + EgoPlanner
End-to-end multimodal perception using 6 surround cameras (Lift-Splat ResNet) + 3D LiDAR (PointPillars) fused into a unified BEV spatial representation `[B, 256, 200, 200]` and decoded by a lightweight trajectory planning head that also gets the past ego velocity. The encoder has no pretrained weights and stays frozen; the head is trained on `mini_train`.

Protocol: evaluation on `mini_val` (scene-0103, scene-0916); the EgoPlanner is trained on
`mini_train` with the encoder frozen and receives the ego velocity from the previous keyframe.

| Method / Model | L2 @ 1s (m) ↓ | L2 @ 2s (m) ↓ | L2 @ 3s (m) ↓ | L2 Avg (m) ↓ |
|---|:---:|:---:|:---:|:---:|
| Zero-Motion Baseline | 5.23 | 10.43 | 15.55 | 10.40 |
| Constant Velocity Baseline | **0.69** | **2.20** | **4.46** | **2.45** |
| BEVFusion + EgoPlanner (Ours) | 0.80 | 2.56 | 5.06 | 2.81 |

Run: Colab T4, seed 42, 267 training / 67 evaluation keyframes, 100 epochs (final train L1
0.111 m).

**Findings.**
- The planner is 0.1–0.6 m worse than constant velocity at every horizon. Its train L1
  (0.11 m) is far below its validation error, i.e. it overfits 267 keyframes; with a frozen,
  randomly initialised encoder the BEV features add noise rather than scene information.
  Next steps: train (or pretrain) the encoder, use `v1.0-trainval`, and keep constant
  velocity as the bar.
- Error growth (stability): constant velocity 0.69 → 4.46 m from 1 s to 3 s (×6.5), the
  planner 0.80 → 5.06 m (×6.3): errors compound with the horizon for both.
- The 4.3 s "inference latency" printed by the acceptance check includes CUDA
  initialisation on the first call and is not a latency measurement.

**Cross-check with the object-centric notebook** (`evaluate_ego_baselines`, same 67
`mini_val` windows): zero motion is identical (5.23 / 10.43 / 15.55 m); constant velocity
gives 0.65 / 2.12 / 4.35 m. The 0.04–0.11 m gap comes from the time axis: the package
extrapolates to the actual keyframe timestamps, the notebook to nominal 0.5 s steps. On
all 10 mini scenes (334 windows) the ego baselines are 5.55 / 11.13 / 16.75 m (zero motion)
and 0.38 / 1.19 / 2.35 m (constant velocity): the two `mini_val` scenes are harder than
average, so results on `mini_val` and on all scenes must not be mixed.

### Midterm run (withdrawn)

Computed on scene-0061 only (a training scene), with the constant velocity taken from the
first future waypoint and the planner output replaced by ground truth + Gaussian noise
(σ = 0.25 m), so these numbers do not measure the model:

| Method / Model | L2 @ 1s (m) | L2 @ 2s (m) | L2 @ 3s (m) | L2 Avg (m) |
|---|:---:|:---:|:---:|:---:|
| Zero-Motion Baseline | 4.82 | 9.94 | 15.31 | 10.02 |
| Constant Velocity Baseline | 1.15 | 2.68 | 4.42 | 2.75 |
| BEVFusion + EgoPlanner | 0.38 | 0.89 | 1.52 | 0.93 |

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
| Persistence | 1.07 | 1.72 | 2.33 | 3.97 | 33.2% |
| Constant Velocity | **0.18** | **0.33** | **0.50** | **0.98** | **10.7%** |

Run: Kaggle (CPU), 10 scenes, 16,295 agent windows (agents present at t0 with 6 future
keyframes).

**Findings.**
- Acceptance criterion met: every scene converts without object loss — 18,538 annotations,
  18,538 tensor entries; tensors range from `[41, 29, 11]` (scene-0757) to `[39, 227, 11]`
  (scene-0061, up to 156 agents in one keyframe).
- Constant velocity reduces the 3 s endpoint error 4× (0.98 vs 3.97 m) and the miss rate 3×
  (10.7% vs 33.2%), so it is the baseline a learned object-centric world model must beat.
- The averages include static agents (barriers, cones, parked cars) for which persistence is
  exact; a moving-agents-only breakdown would show larger errors and is a useful next
  metric. Object ADE / FDE measure other agents and are not comparable with ego L2.

## Latent BEV representation

Planned.

## OOD stress test (0.5 m lateral ego shift)

Evaluates how each scene representation (`object_centric`, `bev`, `latent_bev`) degrades when the
ego pose in held-out test scenes (`mini_val`: `scene-0103`, `scene-0916`; 65 windows) is
shifted laterally by `+0.5 m` along the ego `+y` (left) axis using `shift_nuscenes_ego`.
Reference predictors (`src/skynet/eval/ood.py`) are trained on `mini_train` (259 windows, `P=3`
past frames → `N=6` future frames, `seed=42`, `epochs=25`, `100 × 100` BEV grid over
`[-50, 50] m` at `1.0 m` resolution) and scored via `python scripts/ood_stress_test.py`
(`results/ood_stress_test.json`). FID uses `feature=64` over the 390 `mini_val` future frames
(`% Drop = 100 * (shifted - baseline) / baseline`; positive `%` indicates accuracy degradation).

| Representation | CD 0m (m) ↓ | CD 0.5m (m) ↓ | % CD Drop (Δ%) | FID 0m ↓ | FID 0.5m ↓ | % FID Drop (Δ%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Object-centric | 1.510 | 1.523 | +0.86% | 0.00 | 0.00 | +2.42% |
| BEV | 2.987 | 2.997 | +0.33% | 2.05 | 2.07 | +1.08% |
| Latent BEV | 8.820 | 8.794 | -0.29% | 2.97 | 3.15 | +6.10% |

Per-horizon Chamfer Distance (m):

| Representation | Split | CD @ 1s | CD @ 2s | CD @ 3s | CD Avg |
|---|---|:---:|:---:|:---:|:---:|
| Object-centric | 0m (baseline) | 0.776 | 1.517 | 2.236 | 1.510 |
| Object-centric | 0.5m (shifted) | 0.776 | 1.521 | 2.271 | 1.523 |
| BEV | 0m (baseline) | 2.384 | 3.016 | 3.561 | 2.987 |
| BEV | 0.5m (shifted) | 2.394 | 3.019 | 3.579 | 2.997 |
| Latent BEV | 0m (baseline) | 8.637 | 8.866 | 8.957 | 8.820 |
| Latent BEV | 0.5m (shifted) | 8.555 | 8.900 | 8.927 | 8.794 |

**Findings.**
- **Object-centric** achieves the lowest baseline error (1.510 m CD, 0.0032 FID) because
  continuous coordinates avoid grid quantization; under the 0.5 m ego shift its CD increases by
  `+0.86%` (mainly at 3 s: 2.236 → 2.271 m) and FID rises `+2.42%` (0.00317 → 0.00325).
- **BEV** (`BEVWorldPredictor` U-Net) is translation-equivariant up to the 1.0 m grid
  discretization, showing the smallest FID increase (`+1.08%`) and a `+0.33%` CD drop.
- **Latent BEV** (`LatentBEVPredictor` 64-d autoencoder + latent denoiser) loses spatial
  precision in the bottleneck (8.820 m baseline CD, so the 0.5 m shift is within noise at
  `-0.29%` CD) while showing the largest perceptual distribution shift (`+6.10%` FID).
- These numbers come from the self-contained reference predictors in `src/skynet/eval/ood.py`;
  standalone checkpoints from Issues #11, #13, and #15 can be evaluated with the same
  `OODPredictor` interface via `evaluate_ood_shift`.
