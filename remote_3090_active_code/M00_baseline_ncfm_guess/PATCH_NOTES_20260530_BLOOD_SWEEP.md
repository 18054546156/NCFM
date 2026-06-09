# BloodMNIST Seed0 Sweep Patch Notes

## Purpose
Prepare the unified L20 code package to run the BloodMNIST seed0 method sweep across baseline NCFM, Local Patch NCFD, DAM Attention, and SSIM Regularized variants.

## Changes

1. Added SSIM regularization support to the unified T512 code package.
   - Added `condenser/ssim_regularization.py` from the clean SSIM branch.
   - Wired SSIM config keys into `scripts/run_medmnist_formal_pipeline.py`.
   - Wired SSIM loss into `condenser/compute_loss.py`.
   - Added optional SSIM component logging in `condenser/Condenser.py`.

2. Added BloodMNIST seed0 sweep automation.
   - Added `scripts/run_bloodmnist_seed0_sweep.py` for the 28-run plan:
     - baseline `T=128/256/512/1024`
     - local patch `lambda=0.3/0.5/0.8 × grid=2/4/7`
     - DAM `weight=10/50/100 × layers=L01/L12/L012`
     - SSIM compact sweep.
   - Added launcher/status/collector scripts for two-GPU L20 execution.

3. Added artifact manifest generation.
   - Each completed run records config, final synthetic `.pt`, eval checkpoint, metrics, eval best metrics, and command files in `artifact_manifest.json`.

4. Patched Windows long-path failure in loss curve saving.
   - `utils/experiment_tracker.py` now saves the loss/accuracy plot as `loss_acc.png` instead of a long filename derived from dataset/config fields.
   - This avoids Windows `MAX_PATH` failures during condense.

## Notes
- The experiment reuses existing BloodMNIST data and 20 pretrained premodels from `ncfm_t512_main_20260528`.
- Official full runs use `niter=20000`, so the final synthetic data is `data_20000.pt`.
- Short smoke/debug runs with `niter=20` save `data_20.pt` as their final synthetic artifact.
