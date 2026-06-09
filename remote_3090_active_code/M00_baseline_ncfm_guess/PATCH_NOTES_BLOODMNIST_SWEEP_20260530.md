# BloodMNIST Seed0 Sweep Patch Notes

Date: 2026-05-30

## Purpose

Prepare an automated single-seed BloodMNIST sweep for:

- clean NCFM baseline
- Local Patch NCFD
- DAM Attention
- SSIM Regularized NCFM

## Code Changes

- Added `condenser/ssim_regularization.py` from the SSIM snapshot.
- Updated `condenser/compute_loss.py` so `use_ssim_regularization=true` adds:
  - `loss_total = loss_ncfm + ssim_weight * loss_ssim`
- Updated `condenser/Condenser.py` to log SSIM loss components during condense.
- Updated `scripts/run_medmnist_formal_pipeline.py` for the L20 Windows runtime:
  - `gloo` backend
  - `USE_LIBUV=0`
  - `TORCH_USE_LIBUV=0`
  - Windows file-store rendezvous: `file:///...store?rank=0&world_size=1`
  - direct script launch on Windows instead of `torch.distributed.run`
  - config passthrough for SSIM keys
- Added `scripts/run_bloodmnist_seed0_sweep.py`.
  - Expands 28 BloodMNIST seed0 groups.
  - Supports two-worker chunking with `--chunk_id` and `--num_chunks`.
- Added `scripts/collect_bloodmnist_seed0_reports.py`.
  - Scans `runs/bloodmnist/ipc10/*/metrics.json`.
  - Writes method-specific and merged CSV/Markdown reports.
- Added per-group `artifact_manifest.json` from `run_medmnist_formal_pipeline.py`.
  - Records config, final synthetic `.pt`, eval checkpoint, metrics, commands, and CAM summary.
- Added `scripts/launch_bloodmnist_seed0_sweep.ps1`.
  - Prepares BloodMNIST data/pretrain assets.
  - Starts two hidden workers, one per L20 GPU.
  - Starts a background collector.
- Added `scripts/show_bloodmnist_seed0_status.ps1`.
  - Prints worker status, GPU utilization, completed run count, and report files.

## Experiment Matrix

- Baseline: `B_T128`, `B_T256`, `B_T512`, `B_T1024`
- Local Patch NCFD: `lambda={0.3,0.5,0.8}` x `grid={2,4,7}`, `local_patch_num_freqs=512`
- DAM Attention: `dam_attention_weight={10,50,100}` x `layers={[0,1],[1,2],[0,1,2]}`
- SSIM Regularized: six compact settings over `ssim_weight` and `ssim_grids`

## Main Outputs

- Per-run metrics: `runs/bloodmnist/ipc10/<GROUP>/metrics.json`
- Final synthetic data: `results/condense/**/distilled_data/data_20000.pt`
- Eval checkpoint: `checkpoints/synthetic_train/bloodmnist/ipc10_<GROUP>_best.pth.tar`
- Final reports: `merged_reports/*.csv` and `merged_reports/*.md`

## Runtime Notes

- The official run uses the short experiment directory `experiments/blood_seed0_0530`.
- This avoids Windows `MAX_PATH` failures when matplotlib saves loss curves under nested condense directories.
