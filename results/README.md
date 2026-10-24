# results/

| File | Written by | Contains |
|---|---|---|
| `metrics.json` | BEV notebook, benchmark cell (`notebooks/BEVFusion_Encoder.ipynb`) | Ego L2 at 1/2/3 s of the BEV planner and the planning baselines |
| `object_centric_metrics.json` | `python scripts/object_centric_report.py` | Object-centric tensor statistics per scene, the no-object-loss check, ADE / FDE / miss rate of the object baselines |
| `experiments.md` | by hand | Experiment log for all representations |
| `figures/`, `demo/` | notebooks | Plots and animations used in the README and slides |

Result files are committed so the numbers in the README can be checked without
rerunning anything. Every number must come from a command or notebook named here, with
the seed in `config/default.yaml`; regenerate the file after changing code that
affects it.
