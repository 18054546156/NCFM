# Lab 3090 Multiseed Patch Notes

Date: 2026-06-01

## Purpose

Prepare the unified NCFM MedMNIST code for rerunning the 28-group BloodMNIST and PneumoniaMNIST sweeps on the lab 2x RTX 3090 server with seeds 0, 1, and 2.

## Code Changes

- `scripts/run_medmnist_formal_pipeline.py`
  - Added `--seed` support and passed it into generated YAML configs.
  - Added per-run timing fields:
    - `condense_seconds`
    - `eval_seconds`
    - `cam_seconds`
    - `total_seconds`
  - Added timing fields to `metrics.json`, `artifact_manifest.json`, CSV summaries, and Markdown summaries.

- `scripts/run_lab3090_multiseed_queue.py`
  - New lock-based queue runner for `dataset x seed x group` tasks.
  - Uses shared `data/` and `checkpoints/pretrain/` assets through symlinks under each experiment root.
  - Allows multiple workers to run concurrently without claiming the same task.
  - Intended for launching multiple workers across the two RTX 3090 GPUs.
  - Added a guard for concurrent shared-asset symlink creation when several workers start at the same time.

- `scripts/collect_lab3090_multiseed_reports.py`
  - New collector that scans completed `metrics.json` files across datasets and seeds.
  - Produces per-run and mean/std summary tables.
  - Includes timing statistics so the cost of condense and evaluation can be analyzed.

## Experiment Matrix

- Datasets: `bloodmnist`, `pneumoniamnist`
- Seeds: `0,1,2`
- Groups per dataset/seed: 28
- Total planned runs: `2 x 3 x 28 = 168`

## Notes

- Loss implementations were not changed.
- Existing seed0 experiment outputs on the L20 server were not modified.
- The new queue runner is additive and does not replace the older seed0 runner.
