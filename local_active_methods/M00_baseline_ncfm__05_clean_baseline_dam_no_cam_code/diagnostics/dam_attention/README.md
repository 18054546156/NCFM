# DAM Attention Energy Maps

`run_dam_energy_maps.py` visualizes the ConvNet spatial energy used by M05/DataDAM:

`sum_c abs(F)^p`

It saves:

- `images/original_*.png`
- `heatmaps/dam_energy_*.png`
- `overlays/overlay_*.png`
- `pairs/pair_*.png`, a side-by-side `original | DAM energy` view
- `summary.csv`

BloodMNIST seed-0 best layer setting:

```bash
python diagnostics/dam_attention/run_dam_energy_maps.py \
  --dataset bloodmnist \
  --data_dir /path/to/data \
  --checkpoint /path/to/evaluator_checkpoint.pth \
  --save_dir /path/to/reports/dam_attention/bloodmnist_DAM_w100_L01 \
  --layers 0,1 \
  --num_samples 100 \
  --panel_size 224
```

PneumoniaMNIST seed-0 best layer setting:

```bash
python diagnostics/dam_attention/run_dam_energy_maps.py \
  --dataset pneumoniamnist \
  --data_dir /path/to/data \
  --checkpoint /path/to/evaluator_checkpoint.pth \
  --save_dir /path/to/reports/dam_attention/pneumoniamnist_DAM_w10_L012 \
  --layers 0,1,2 \
  --num_samples 100 \
  --panel_size 224
```

Use `--overlay` if the right panel should be the heatmap blended onto the image instead of the heatmap alone.
