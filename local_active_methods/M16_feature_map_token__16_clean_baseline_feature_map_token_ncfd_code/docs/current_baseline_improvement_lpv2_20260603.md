# BloodMNIST Current Baseline Improvement Plan

Date: 2026-06-03

## Purpose

Run a focused Local Patch v2 sweep against the current BloodMNIST baseline.

Current baseline target:

- Dataset: BloodMNIST
- IPC: 10
- Seed: 0 first
- Global NCFM T: 512
- Sampling net: enabled
- Local patch baseline comparison: project_6_2 three-seed baseline showed T512 is the stable baseline.

## Why This Matrix

Historical results point to local patch as the only consistently useful add-on.

- Old BloodMNIST Local Patch v1 favored `grid=4`.
- Old BloodMNIST phase2 favored `lambda=0.2` more than larger lambdas.
- Current LPv2 encoder ablation showed `encoder_blocks=1` beats `encoder_blocks=2`.
- Ensemble teacher was slower and did not clearly improve seed0.
- Batch benchmark showed `batch_real=2330` is the maximum effective BloodMNIST per-class setting; larger values do not add real samples.

## Groups

All groups use:

- `num_freqs=512`
- `local_patch_grid=4`
- `local_patch_encoder_blocks=1`
- `local_patch_encoder_source=premodel_trained`
- `local_patch_premodel_index=0`
- `batch_real=2330`
- `niter=20000`
- `eval_epochs=2000`

| Group | lambda | localT |
|---|---:|---:|
| `LPv2_T512_lam02_g4_lf256_p0_b1` | 0.2 | 256 |
| `LPv2_T512_lam02_g4_lf512_p0_b1` | 0.2 | 512 |
| `LPv2_T512_lam03_g4_lf256_p0_b1` | 0.3 | 256 |
| `LPv2_T512_lam03_g4_lf512_p0_b1` | 0.3 | 512 |
| `LPv2_T512_lam05_g4_lf256_p0_b1` | 0.5 | 256 |
| `LPv2_T512_lam05_g4_lf512_p0_b1` | 0.5 | 512 |

## Runner

Script:

`scripts/run_blood_lpv2_current_improvement_20260603.py`

This runner uses the formal pipeline but calls `run_condense_eval`, not CAM, to save time.

## Remote Output

Planned remote root:

`/data/zengqiang/project_6_2/experiments/current_baseline_improvement_blood_T512_lpv2_seed0_br2330_20260603`

Important outputs:

- `configs/bloodmnist/ipc10_<GROUP>.yaml`
- `runs/bloodmnist/ipc10/<GROUP>/metrics.json`
- `runs/bloodmnist/ipc10/<GROUP>/condensed_path.txt`
- `results/condense/.../distilled_data/data_20000.pt`
- `checkpoints/synthetic_train/bloodmnist/..._best.pth.tar`
- `reports/bloodmnist_lpv2_current_improvement_seed0_*.csv`

