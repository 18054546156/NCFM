## Diff From Upstream NCFM

This document summarizes the code-level differences between this repository and the upstream NCFM repository:

- Upstream: `https://github.com/gszfwsb/NCFM.git`
- Compared branch: `upstream/main`
- Local branch snapshot: current checked-in contents of this repository

This summary intentionally excludes experiment artifacts and runtime byproducts, including:

- `.git/`
- `__pycache__/`
- `logs/`
- `ncfm_logs/`
- `results/`
- `frequency_analysis/`
- `pretrained_models/`
- `imbalance_analysis/`

It is meant to answer one narrow question:

> What code and configuration changes exist here relative to upstream NCFM?

### 1. New files added locally

These files exist in this repository but not in upstream NCFM.

#### MedMNIST support

- `data/medmnist_dataset.py`
- `config/ipc1/pathmnist.yaml`
- `config/ipc1/pathmnist_test.yaml`
- `config/ipc1/bloodmnist.yaml`
- `config/ipc1/dermamnist.yaml`
- `config/ipc10/pathmnist.yaml`
- `config/ipc10/bloodmnist.yaml`
- `config/ipc10/dermamnist.yaml`
- `config/ipc50/pathmnist.yaml`
- `config/ipc50/bloodmnist.yaml`
- `config/ipc50/dermamnist.yaml`

These additions adapt NCFM to MedMNIST-style datasets, especially `PathMNIST`, `BloodMNIST`, and `DermaMNIST`.

#### Documentation and helper scripts

- `MEDMNIST_ADAPTER.md`
- `run_medmnist_condense.sh`
- `test_medmnist_loader.py`
- `launch_all_experiments.sh`
- `launch_all_experiments_fixed.sh`
- `launch_dermamnist_gpu1.sh`
- `monitor_experiments.sh`
- `run_all_experiments.sh`
- `run_all_ncfm_experiments.sh`
- `run_dermamnist_gpu1.sh`
- `run_full_pipeline.sh`
- `run_ncfm_pipeline.sh`
- `run_single.sh`

These are local workflow helpers for MedMNIST experiments and multi-run orchestration. They are not part of upstream NCFM.

#### Miscellaneous local file

- `${log_file}`

This appears to be an accidental local shell-expanded artifact rather than a source file.

### 2. Existing upstream files modified locally

The following upstream files were changed in this repository.

#### `models/convnet.py`

Local change:

- added special handling for 28x28 inputs:

```python
if im_size[0] == 28:
    im_size = (32, 32)
```

Effect:

- MedMNIST images are 28x28, while the original ConvNet path in NCFM assumes 32x32-style spatial bookkeeping.
- This local change adjusts feature-shape computation so the ConvNet can be used with 28x28 inputs without shape mismatch.

Note:

- This is shape bookkeeping in the model definition, not interpolation logic by itself.

#### `data/dataset_statistics.py`

Local change:

- added mean/std entries for MedMNIST datasets, including examples such as:
  - `pathmnist`
  - `bloodmnist`
  - `dermamnist`
  - `retinamnist`
  - `tissuemnist`
  - others

Effect:

- allows normalization code to work for MedMNIST datasets through the same statistics table used by the original project.

#### `utils/utils.py`

Local changes include:

- support for loading MedMNIST datasets through `load_resized_data(...)`
- import path adjustment for local usage
- config-path-related behavior changes for pretrained model directory handling

Most important functional difference:

- this repository added a MedMNIST branch that imports `data.medmnist_dataset` and loads datasets such as:
  - `pathmnist`
  - `dermamnist`
  - `bloodmnist`
  - `retinamnist`
  - related MedMNIST variants

This is one of the core adaptation points from upstream CIFAR/ImageNet-oriented NCFM to MedMNIST.

#### `condenser/Condenser.py`

Observed local difference:

- scheduler initialization includes explicit `verbose=False` in `ReduceLROnPlateau(...)`

Effect:

- mostly compatibility / logging-behavior adjustment rather than a new algorithmic change.

This file may also contain additional local edits outside the minimal diff snippet, but the direct upstream comparison clearly shows at least this scheduler behavior change.

### 3. Upstream config files modified locally

The following existing upstream YAMLs differ from upstream:

- `config/ipc1/imagefruit.yaml`
- `config/ipc1/imagemeow.yaml`
- `config/ipc1/imagenette.yaml`
- `config/ipc1/imagesquawk.yaml`
- `config/ipc1/imagewoof.yaml`
- `config/ipc1/imageyellow.yaml`
- `config/ipc10/imagefruit.yaml`
- `config/ipc10/imagemeow.yaml`
- `config/ipc10/imagenette.yaml`
- `config/ipc10/imagesquawk.yaml`
- `config/ipc10/imagewoof.yaml`
- `config/ipc10/imageyellow.yaml`

The main pattern in these config edits is:

- `pretrain_dir` was changed from dataset-specific subdirectories like:

```yaml
pretrain_dir: '../pretrained_models/imagefruit'
```

to a more generic path:

```yaml
pretrain_dir: '../pretrained_models'
```

Effect:

- local pretrained model path layout differs from upstream's expected directory structure.

### 4. What did not change at the code-architecture level

Relative to upstream NCFM, this repository is still fundamentally:

- the same condensation framework
- the same major directory layout
- the same main training entrypoints
- the same NCFM core method

The local fork mainly changes:

1. dataset support
2. config coverage
3. 28x28 ConvNet handling
4. experiment-running helper scripts

It is best understood as:

> Upstream NCFM adapted for MedMNIST datasets, plus local experiment orchestration helpers.

### 5. Important note about files not covered here

This document does **not** describe local experiment outputs or downstream analysis utilities outside the filtered comparison scope.

In particular, it does not summarize:

- generated checkpoints
- evaluation logs
- frequency-analysis outputs
- external project-level wrappers outside this repository root

Those are project artifacts layered on top of the code fork, not part of the upstream-code comparison itself.
