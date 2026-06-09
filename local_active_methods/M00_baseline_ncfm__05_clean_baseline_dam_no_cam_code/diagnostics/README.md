# Diagnostics

This folder contains optional post-hoc analysis tools.

The main baseline and DAM training/evaluation pipeline does not import or run these tools.

- `cam/`: optional Grad-CAM utilities for inspecting trained evaluator checkpoints after experiments.
- `dam_attention/`: optional M05/DataDAM spatial energy visualizer. It saves
  original images, DAM energy maps, overlays, and 100 paired `original | energy`
  images. These maps use the same `sum_c |F|^p` ConvNet feature-map energy used by
  the DAM attention loss; they are not Grad-CAM and not semantic self-attention.

Recommended usage:

1. Run the clean main pipeline first: pretrain -> condense -> evaluation.
2. Select a small number of checkpoints/results worth inspecting.
3. Run CAM diagnostics manually from `diagnostics/cam/` if needed for qualitative analysis.
