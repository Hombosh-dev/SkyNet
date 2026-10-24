# models/

Checkpoints of the world models (object-centric, BEV, latent BEV) go here locally.
They are **not committed**: the pre-commit `check-added-large-files` hook stops files
over 1000 KB, GitHub rejects files over 100 MB, and a file removed later still stays
in git history.

| Representation | Code | Checkpoint |
|---|---|---|
| Object-centric | `src/skynet/models/object_centric/` | none yet (baselines need no training) |
| BEV | `notebooks/BEVFusion_Encoder.ipynb` | none committed |
| Latent BEV | planned | — |

To share weights, upload them to the Hugging Face Hub or a GitHub Release and record
the link, the commit and the config they were trained with here. A model card should
state the training split, the metrics from `results/` and the known limitations.
