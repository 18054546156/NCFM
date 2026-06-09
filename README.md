# NCFM Code Snapshot: Baseline / M12 / M16 / M19

Created: 2026-06-10 02:23:55 +08:00
Workspace: D:\Project\NCFM_Medmnist

This snapshot preserves the code used for the current NCFM method lines:

- M00 Baseline NCFM
- M12 Local Patch Rand20 Step
- M16 Feature-Map Token NCFD
- M19 Patch + Feature-Token Fusion

## Layout

- local_active_methods/: copied from local Windows active method folders.
- remote_3090_active_code/: copied from 3090 lab server active_code folders.
- SHA256SUMS.txt: hash of every saved file.
- SOURCE_MAP.md: source path mapping.

## Restore

This directory is also a standalone git repository. A cloneable bundle is stored next to it:

`
git clone baseline_m12_m16_m19_*.bundle restored_code
`

## Notes

Large experiment artifacts were intentionally excluded: results, runs, checkpoints, logs, images, .pt/.pth/.pth.tar files, caches, and temporary distributed run folders.
