# `training_dynamics.json` Field Reference

This file documents the structure of the `training_dynamics.json` log written by
`condenser/dynamics_monitor.py`.

Default logging intervals in the current MedMNIST configs:

| Record group | Default interval |
| --- | --- |
| `loss` | every 100 steps |
| `per_class_loss` | every 500 steps |
| `gradient_frequency` | every 1000 steps |
| `feature_metrics` | every 2000 steps |
| `intermediate_eval` | every 2000 steps |

## Top-level metadata

| Field | Meaning | How it is obtained |
| --- | --- | --- |
| `dataset` | Dataset name | From runtime args |
| `ipc` | Images per class | From runtime args |
| `nclass` | Number of classes | From runtime args |
| `config_path` | YAML config path | From runtime args |
| `real_probe_counts` | Real-image probe subset size per class | Balanced subset sampled from the train set |
| `real_feature_counts` | Real-image feature subset size per class | Balanced subset sampled from the train set |
| `test_set_size` | Validation/test set size used in intermediate eval | `len(val_dataset)` |

## `loss`

| Field | Meaning | Formula / source |
| --- | --- | --- |
| `step` | Condensation step index | Current iteration index |
| `match_loss_total` | Sum of matching losses across classes | `sum_c match_loss(real_c, syn_c)` |
| `calib_loss_total` | Sum of calibration losses across classes and calib repeats | `sum_i sum_c calib_weight * CE(model_final(syn_c), label_c)` |
| `total_loss` | Combined internal objective | `match_loss_total + calib_loss_total` |
| `match_loss_per_class` | Average matching loss per class | `match_loss_total / nclass` |

Notes:
- `match_loss_total` comes from `condenser/compute_loss.py::compute_match_loss`.
- `calib_loss_total` comes from `condenser/compute_loss.py::compute_calib_loss`.

## `per_class_loss`

Each record:

| Field | Meaning | Formula / source |
| --- | --- | --- |
| `step` | Condensation step index | Current iteration index |
| `values` | Per-class matching loss list | For each class `c`, `match_loss(real_c, syn_c)` on probe subsets |

Notes:
- `values[c]` can be `null` if that class has no usable real or synthetic sample at the probe step.

## `gradient_frequency`

Each record:

| Field | Meaning | Formula / source |
| --- | --- | --- |
| `step` | Condensation step index | Current iteration index |
| `probe_loss` | Probe matching loss used to generate gradients | `sum_c match_loss(real_c, syn_c)` on probe subsets |
| `top20` | Cumulative gradient energy covered by top 20 DCT frequencies | See below |
| `top50` | Cumulative gradient energy covered by top 50 DCT frequencies | See below |
| `top100` | Cumulative gradient energy covered by top 100 DCT frequencies | See below |
| `top200` | Cumulative gradient energy covered by top 200 DCT frequencies | See below |
| `top400` | Cumulative gradient energy covered by top 400 DCT frequencies | See below |

Computation:

1. Clone current synthetic images `S` and enable gradients.
2. Compute probe loss:
   `L_probe = sum_c match_loss(real_c, syn_c)`.
3. Compute image gradient:
   `G = dL_probe / dS`.
4. Apply 2D DCT to each image/channel gradient map.
5. Define frequency energy:
   `E(u, v) = mean(DCT(G)^2)`.
6. Flatten and sort all frequency locations by energy, then compute cumulative ratios:
   `topk = sum_{j <= k} E_j / sum_j E_j`.

Interpretation:
- Larger `top20` or `top50` means gradients are more concentrated in a small set of frequency locations.

## `feature_metrics`

Each record:

| Field | Meaning | Formula / source |
| --- | --- | --- |
| `step` | Condensation step index | Current iteration index |
| `centroid_drift_per_class` | Distance between real and synthetic class centroids | For each class `c`, `||mu_syn_c - mu_real_c||_2` |
| `centroid_drift_mean` | Mean centroid drift across valid classes | Mean of valid per-class drifts |
| `centroid_drift_max` | Max centroid drift across valid classes | Max of valid per-class drifts |
| `fisher_ratio_real` | Class separability of real features | `trace(S_b) / trace(S_w)` on real features |
| `fisher_ratio_syn` | Class separability of synthetic features | `trace(S_b) / trace(S_w)` on synthetic features |
| `fisher_retention` | Synthetic separability relative to real separability | `fisher_ratio_syn / fisher_ratio_real` |
| `mmd` | Distribution gap between synthetic and real features | Gaussian-kernel MMD |
| `silhouette_real` | Real-feature cluster separation score | `silhouette_score(real_features, real_labels)` |
| `silhouette_syn` | Synthetic-feature cluster separation score | `silhouette_score(syn_features, syn_labels)` |
| `davies_bouldin_real` | Real-feature cluster compactness/separation score | `davies_bouldin_score(real_features, real_labels)` |
| `davies_bouldin_syn` | Synthetic-feature cluster compactness/separation score | `davies_bouldin_score(syn_features, syn_labels)` |
| `linear_probe_bacc` | How well synthetic features train a simple classifier that transfers to real features | Logistic regression trained on synthetic features, evaluated on real features with balanced accuracy |

Feature extraction:
- Features are extracted from the current model on:
  - a balanced real-image subset, and
  - the full current synthetic image set.

Interpretation:
- Lower `centroid_drift_*` is better.
- Higher `fisher_retention` is better.
- Lower `mmd` is better.
- Higher `silhouette_*` is usually better.
- Lower `davies_bouldin_*` is usually better.
- Higher `linear_probe_bacc` is better.

## `intermediate_eval`

Each record:

| Field | Meaning | Formula / source |
| --- | --- | --- |
| `step` | Condensation step index | Current iteration index |
| `epochs` | Number of epochs used in the probe training run | `eval_during_condense_epochs` |
| `acc` | Standard accuracy on validation/test set | `accuracy_score(labels, preds)` |
| `balanced_acc` | Balanced accuracy on validation/test set | `balanced_accuracy_score(labels, preds)` |
| `macro_f1` | Macro-averaged F1 | `f1_score(..., average="macro")` |
| `weighted_f1` | Weighted F1 | `f1_score(..., average="weighted")` |
| `per_class_f1` | F1 for each class | `f1_score(..., average=None)` |
| `pred_distribution` | Predicted class histogram | `bincount(preds, minlength=nclass)` |
| `label_distribution` | Ground-truth class histogram | `bincount(labels, minlength=nclass)` |
| `macro_auc` | Multi-class or binary ROC-AUC | `roc_auc_score(...)`, or `null` if undefined |

Procedure:

1. Build a fresh model.
2. Train it only on the current synthetic set for `epochs` epochs.
3. Evaluate it on the validation/test loader.
4. Compute the classification metrics above.

Interpretation:
- `balanced_acc` is the most important downstream metric when classes are imbalanced.
- `pred_distribution` is useful for spotting class collapse.

## Source map

| File | Role |
| --- | --- |
| `condenser/dynamics_monitor.py` | Computes and writes all dynamics fields |
| `condenser/compute_loss.py` | Produces `match_loss_total` and `calib_loss_total` |
| `NCFM/NCFM.py` | Defines `match_loss` and `cailb_loss` |
| `plot_training_dynamics.py` | Reads `training_dynamics.json` and generates summary plots |
