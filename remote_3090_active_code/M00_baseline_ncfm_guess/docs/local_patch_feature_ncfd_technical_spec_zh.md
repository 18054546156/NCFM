# Local Patch-feature NCFD v1 Technical Spec

**日期**: 2026-05-20  
**实现分支**: `local-patch-feature-ncfd-v1`  
**基线方法**: `B_minmax_ncfm_psi`  
**目标方法**: `B + 4x4 local patch-feature NCFD`  

本文档是实现规格书。它把设计稿中的方法选择落到代码接口、配置字段、运行脚本和实验产物上。

---

## 1. v1 目标

v1 只验证一件事：

> 在 B 的 global min-max NCFM 上，加入固定 patch encoder 的 local patch-feature NCFD，是否能改善 MedMNIST 的局部结构保真。

v1 保持最小改动：

- 保留 global NCFD。
- 保留 global sampling net `psi`。
- 加入 local patch-feature NCFD。
- 不加 local sampling net。
- 不加 SSIM。
- 不引入 raw pixel patch NCFD。
- 不复用历史 `use_local_patch_ncfm` / `lambda_local` 字段。

---

## 2. 新增配置字段

建议字段放在 YAML 的 `condense` 段。

```yaml
use_local_patch_feature_ncfd: true
local_patch_grid: 4
lambda_local_patch_ncfd: 0.3
local_patch_feature_dim: 128
local_patch_encoder_blocks: 2
local_patch_num_freqs: 256
local_patch_encoder_source: premodel0_trained
local_patch_encoder_frozen: true
use_local_patch_sampling_net: false
```

字段含义：

| 字段 | v1 默认 | 含义 |
|---|---:|---|
| `use_local_patch_feature_ncfd` | `false` | 是否开启 local patch-feature NCFD |
| `local_patch_grid` | `4` | 原图切成 `grid x grid` |
| `lambda_local_patch_ncfd` | `0.3` | local loss 权重 |
| `local_patch_feature_dim` | `128` | patch feature 维度 |
| `local_patch_encoder_blocks` | `2` | 从 ConvNet 复制前几个 block |
| `local_patch_num_freqs` | `256` | local NCFD 固定频率数 |
| `local_patch_encoder_source` | `premodel0_trained` | patch encoder 权重来源 |
| `local_patch_encoder_frozen` | `true` | v1 必须冻结 |
| `use_local_patch_sampling_net` | `false` | v1 不启用 local psi |

---

## 3. Patchify 接口

函数：

```python
patches = patchify_images(images, grid)
```

输入：

```text
images: [B, C, H, W]
grid: int
```

约束：

```text
H % grid == 0
W % grid == 0
```

输出：

```text
patches: [B, K, C, patch_h, patch_w]
K = grid * grid
patch_h = H // grid
patch_w = W // grid
```

v1 主尺度：

```text
[B, C, 28, 28]
-> grid = 4
-> [B, 16, C, 7, 7]
```

---

## 4. Patch encoder

v1 使用冻结 patch encoder。

实现方式：

1. 构建一个与当前 Condense backbone 同结构的 ConvNet。
2. 从 `premodel0_trained.pth.tar` 加载权重。
3. 复制前 `local_patch_encoder_blocks=2` 个 block。
4. 后接 `AdaptiveAvgPool2d(1)`。
5. flatten 得到 `[B*K, 128]`。
6. 所有参数 `requires_grad=False`。

形状：

```text
[B*K, C, patch_h, patch_w]
-> conv block 1
-> conv block 2
-> AdaptiveAvgPool2d(1)
-> [B*K, 128]
```

为什么用 2 个 block：

- `7x7 patch` 经过两次 pool 后仍可得到有效局部 feature。
- 三个 block 对 7x7 patch 会过度压缩。
- 两个 block 与当前 ConvNet width=128 对齐。

---

## 5. Local NCFD 函数签名

建议实现：

```python
def local_patch_feature_ncfd_loss(
    img_real,
    img_syn,
    patch_encoder,
    cf_loss_func,
    args,
):
    ...
```

输入：

```text
img_real: [Br, C, 28, 28]
img_syn:  [Bs, C, 28, 28]
patch_encoder: frozen nn.Module
cf_loss_func: CFLossFunc
args: config namespace
```

输出：

```text
loss_local: scalar tensor
```

内部流程：

```text
real patches = patchify(img_real, grid=4)
syn patches  = patchify(img_syn,  grid=4)

real patches: [Br, 16, C, 7, 7] -> [Br*16, C, 7, 7]
syn patches:  [Bs, 16, C, 7, 7] -> [Bs*16, C, 7, 7]

patch_encoder(real) -> [Br*16, 128] -> [Br, 16, 128]
patch_encoder(syn)  -> [Bs*16, 128] -> [Bs, 16, 128]

for k in 1..16:
    loss_k = NCFD(real[:, k, :], syn[:, k, :])

loss_local = mean_k loss_k
```

local 频率：

```text
t_local: [local_patch_num_freqs, 128]
```

v1 默认：

```text
local_patch_num_freqs = 256
```

---

## 6. Total loss 接入

在 `condenser/compute_loss.py` 中：

```python
loss_global = inner_loss_fn(...)

if args.use_local_patch_feature_ncfd:
    loss_local = local_patch_feature_ncfd_loss(...)
    loss = loss_global + args.lambda_local_patch_ncfd * loss_local
else:
    loss = loss_global
```

如果 global sampling net 开启：

```python
loss.backward(retain_graph=True)
optim_img.step()

(-loss_global).backward()
optim_sampling_net.step()
```

注意：

- local loss 不反向更新 global sampling net。
- v1 不存在 local sampling net。
- image update 使用 `global + lambda * local`。
- sampling net update 只最大化 global NCFD。

---

## 7. 需要改的文件

v1 允许改：

```text
condenser/local_patch_ncfd.py        # 新增
condenser/compute_loss.py            # 叠加 local loss
condense/condense_script.py          # 构建 frozen patch encoder
scripts/run_medmnist_formal_pipeline.py  # 增加 local groups / group selector
docs/local_patch_feature_ncfd_technical_spec_zh.md
```

尽量不改：

```text
NCFM/NCFM.py
NCFM/SampleNet.py
condenser/Condenser.py
models/convnet.py
```

---

## 8. 第一版实验组

新增两个组：

```text
D_local_patch_feature_lam03
D_local_patch_feature_lam06
```

参数：

| Group | Base | Grid | Patch | Dim | Lambda | Local psi | SSIM |
|---|---|---:|---:|---:|---:|---|---|
| `B_minmax_ncfm_psi` | B | - | - | - | 0 | no | no |
| `D_local_patch_feature_lam03` | B | 4x4 | 7x7 | 128 | 0.3 | no | no |
| `D_local_patch_feature_lam06` | B | 4x4 | 7x7 | 128 | 0.6 | no | no |

---

## 9. 第一轮运行计划

Smoke test：

```text
dataset: pathmnist
ipc: 1
niter: small
group: D_local_patch_feature_lam03
purpose: shape / loss / grad / checkpoint / CAM path sanity
```

正式第一轮：

```text
datasets:
  - pathmnist
  - bloodmnist
  - pneumoniamnist
ipc: 10
groups:
  - D_local_patch_feature_lam03
  - D_local_patch_feature_lam06
```

其中：

- PathMNIST / BloodMNIST 是主验证。
- PneumoniaMNIST 是全局结构型对照。
- B baseline 使用已存在 formal 结果；如预算允许可重跑 B。

---

## 10. 输出文件

每个 run 至少输出：

```text
runs/<dataset>/ipc10/<group>/condense_stdout.log
runs/<dataset>/ipc10/<group>/condense_stderr.log
runs/<dataset>/ipc10/<group>/eval_stdout.log
runs/<dataset>/ipc10/<group>/eval_stderr.log
runs/<dataset>/ipc10/<group>/metrics.json
runs/<dataset>/ipc10/<group>/eval_metrics.jsonl
runs/<dataset>/ipc10/<group>/eval_metrics_best.json
runs/<dataset>/ipc10/<group>/cam_stdout.log
runs/<dataset>/ipc10/<group>/cam_stderr.log
results/cam/<dataset>/ipc10_<group>/summary.csv
reports/local_patch_feature_ncfd_v1_summary.csv
reports/local_patch_feature_ncfd_v1_summary.md
```

如果后处理 notebook/脚本可用，继续输出：

```text
reports/cam/cam_summary_grouped.csv
reports/cam/cam_similarity_to_real_grouped.csv
reports/cam/cam_spatial_bias_grouped.csv
```

---

## 11. 成功判据

强成功：

```text
分类指标不下降或提升
CAM similarity to real 提高
edge/corner bias 下降
occlusion hot-drop 更接近 real
low-similarity high-confidence case 比例下降
```

中等成功：

```text
ACC/AUC 持平
解释性指标明显改善
```

失败：

```text
分类指标下降
CAM 更偏
edge/corner bias 更强
合成图像视觉质量变差
```

