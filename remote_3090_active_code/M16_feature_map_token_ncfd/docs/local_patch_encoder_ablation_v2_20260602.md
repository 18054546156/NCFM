# Local Patch Encoder Ablation v2 - 2026-06-02

This code snapshot was copied from:

```text
active_methods/05_clean_baseline_dam_no_cam_code
```

It adds Local Patch feature NCFD as an optional loss while preserving clean
baseline behavior when `use_local_patch_feature_ncfd: false`.

## Loss

Baseline:

```text
L_total = L_NCFM
```

Local Patch:

```text
L_total = L_NCFM + lambda_local_patch_ncfd * L_local_patch
```

For each class batch, images are split into `local_patch_grid x local_patch_grid`
non-overlapping patches. Each patch is encoded by one or more frozen ConvNet
premodel encoders, then NCFD is computed at each patch location and averaged
over patch locations.

## Modified Files

```text
condenser/local_patch_ncfd.py
condenser/compute_loss.py
condense/condense_script.py
utils/utils.py
```

## New Encoder Options

Default behavior reproduces v1:

```yaml
use_local_patch_feature_ncfd: true
local_patch_encoder_source: premodel0_trained
local_patch_encoder_blocks: 2
```

Supported encoder sources:

| Value | Meaning |
|---|---|
| `premodel0_trained` | v1 default: use `premodel0_trained.pth.tar` |
| `premodel0_init` | use `premodel0_init.pth.tar` |
| `premodel_trained` | use one trained premodel selected by `local_patch_premodel_index` |
| `premodel_init` | use one init premodel selected by `local_patch_premodel_index` |
| `random_trained` | choose one trained premodel using `local_patch_encoder_seed` |
| `random_init` | choose one init premodel using `local_patch_encoder_seed` |
| `ensemble_trained` | average/concat multiple trained premodel encoders |
| `ensemble_init` | average/concat multiple init premodel encoders |

Additional fields:

```yaml
local_patch_premodel_index: 0
local_patch_premodel_indices: [0, 1, 2, 3]
local_patch_model_num: 20
local_patch_ensemble_size: 4
local_patch_ensemble_random: false
local_patch_ensemble_aggregate: mean  # mean or concat
local_patch_encoder_seed: 0
local_patch_feature_dim: 128          # set 0 to skip dimension check
local_patch_loss_scale: 300.0
```

## Example Configs

Single teacher, block ablation:

```yaml
use_local_patch_feature_ncfd: true
num_freqs: 512
local_patch_grid: 4
local_patch_num_freqs: 512
lambda_local_patch_ncfd: 0.1
local_patch_encoder_source: premodel_trained
local_patch_premodel_index: 0
local_patch_encoder_blocks: 1
local_patch_feature_dim: 128
```

Random teacher:

```yaml
use_local_patch_feature_ncfd: true
num_freqs: 512
local_patch_grid: 4
local_patch_num_freqs: 512
lambda_local_patch_ncfd: 0.1
local_patch_encoder_source: random_trained
local_patch_encoder_seed: 0
local_patch_model_num: 20
local_patch_encoder_blocks: 1
local_patch_feature_dim: 128
```

Teacher ensemble:

```yaml
use_local_patch_feature_ncfd: true
num_freqs: 512
local_patch_grid: 4
local_patch_num_freqs: 512
lambda_local_patch_ncfd: 0.1
local_patch_encoder_source: ensemble_trained
local_patch_premodel_indices: [0, 1, 2, 3]
local_patch_ensemble_aggregate: mean
local_patch_encoder_blocks: 1
local_patch_feature_dim: 128
```

## Recommended BloodMNIST First Sweep

Use `T=512` as the baseline anchor.

First smoke:

| Group | lambda | grid | localT | source | blocks |
|---|---:|---:|---:|---|---:|
| `LPv2_lam01_g4_lf512_p0_b1` | 0.1 | 4 | 512 | `premodel_trained`, index 0 | 1 |
| `LPv2_lam01_g4_lf512_p0_b2` | 0.1 | 4 | 512 | `premodel_trained`, index 0 | 2 |
| `LPv2_lam01_g4_lf512_ens0123_b1` | 0.1 | 4 | 512 | `ensemble_trained`, [0,1,2,3] | 1 |

Then sweep:

```text
lambda: 0.02, 0.05, 0.1, 0.2, 0.3
grid: 2, 4, 7
local_patch_encoder_blocks: 1, 2
encoder_source: premodel_trained first; ensemble_trained for finalists
```

## Notes

- `premodel0_trained + blocks=2` reproduces Local Patch v1.
- `blocks=1` is likely safer for small `7x7` or `4x4` patches because two
  pooling blocks can over-compress tiny patches.
- `ensemble_aggregate: mean` keeps the feature dimension unchanged. Use
  `concat` only as an explicit feature-dimension ablation and set
  `local_patch_feature_dim` accordingly or to `0`.
