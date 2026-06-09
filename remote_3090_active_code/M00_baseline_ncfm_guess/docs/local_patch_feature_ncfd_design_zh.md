# 原图切块版 Multi-scale NCFD 设计稿

**日期**: 2026-05-20  
**基线仓库**: `NCFM_medmnist_clean`  
**基线分支**: `clean-medmnist`  
**基线提交**: `138e182`  
**基线 tag**: `baseline-medmnist-clean-cam-20260520`  

本文档只记录下一阶段方法设计和实验计划，不改动代码。

---

## 1. 当前研究判断

当前最合理的研究主线不是继续比较 A/B/C 哪个配置高 1 个点，而是：

> 以 B 组作为当前 NCFM / SOTA 基线，研究 MedMNIST 医学小图像中哪些局部结构对判别重要，以及 NCFM 的全局 NCFD 是否没有充分保住这些局部结构。

现有 CAM、real-vs-synthetic 相似性、spatial bias、病例可视化和遮挡分析，已经支持一个比较明确的问题诊断：

1. B 基线能分类，但它的 CAM 与 real-trained evaluator 的 CAM 一致性仍然偏低。
2. PathMNIST 和 BloodMNIST 中存在不少高置信正确、但 CAM 和 real reference 明显错位的样本。
3. 常见错位模式包括 diffuse、center-heavy、edge-heavy、corner-biased、low-hot-mass。
4. 这些现象说明 B 学到的可能是能分类的全局 shortcut、纹理 shortcut 或位置统计，而不是稳定的医学局部结构对齐。

因此，下一步最有针对性的改进是：

> 在 B 的全局 min-max NCFM 框架上，加入原图切块后的 patch feature local NCFD，用局部块级结构特征对齐补足全局 NCFD 的空间结构缺口。

---

## 2. 候选方案对比

这里把几个容易混淆的 local 方案明确区分。

### 2.1 方案 A: 原图 patch 像素直接做 local NCFD

流程：

```text
28x28 image
-> patchify
-> patch pixels
-> NCFD(pixel patch real, pixel patch syn)
```

优点：

- 形式上最接近“原图分块”。
- 不需要额外 patch encoder。
- 直观上好理解。

问题：

- 太接近像素分布匹配，不够结构感知。
- 容易把亮度、噪声、边界、设备差异也强行匹配进去。
- 医学判别结构通常不是逐像素等价，而是局部形态、局部纹理和局部组织统计等高层结构。

结论：

> A 不适合作为主方法。可以作为 appendix 里的 weak baseline，但不推荐第一版实现。

---

### 2.2 方案 B: 原图切块 -> patch feature -> local NCFD

流程：

```text
28x28 image
-> patchify into K patches
-> shared patch encoder h
-> patch feature z^(k)
-> NCFD(z_real^(k), z_syn^(k))
-> average over patches
```

这是当前最符合理论目标的方案。

原因：

1. 仍然在 feature space 里做 NCFD，符合 NCFM 的原始精神。
2. 局部单位来自原图 patch，和 MedMNIST 小图像中的局部医学结构直接对应。
3. 比纯 feature-map local 更容易讲清楚“28x28 原图局部块”与“局部诊断区域”的关系。
4. 比 pixel patch 匹配更结构感知，不会退化成低层像素对齐。

结论：

> B 是论文/方法定义上最应该采用的主方案。

---

### 2.3 方案 C: 把所有 patch 合成一个 batch 后并行过 patch encoder

流程：

```text
[B, C, 28, 28]
-> patchify 4x4
-> [B, 16, C, 7, 7]
-> reshape [B*16, C, 7, 7]
-> patch encoder h
-> [B*16, Dp]
-> reshape [B, 16, Dp]
```

C 不是新的理论方案，而是 B 的高效工程实现。

结论：

> 方法定义写 B，代码实现用 C。

---

### 2.4 Feature-map Local NCFD

流程：

```text
28x28 image
-> full image encoder
-> intermediate feature map [B, C, H, W]
-> local NCFD on feature-map locations or blocks
```

优点：

- 改动较干净。
- 可以直接复用 ConvNet 中间层 feature。
- 工程上比 patch encoder 更容易起步。

为什么这次不作为主方案：

- 当前理论强调的是原图 28x28 分块、4x4 grid、每块 7x7，与医学局部诊断区域直接对应。
- Feature-map local 更像网络内部表示局部，不如原图 patch 那么贴合“MedMNIST 小图局部医学结构”的叙事。

结论：

> Feature-map local 是干净工程起点，但若严格服务当前 multi-scale NCFD 理论，原图 patch-feature local NCFD 更合适。

---

## 3. 最终方法选择

最终选择：

> Global NCFD + Local Patch-feature NCFD。

也就是：

```text
L_total = L_global + lambda_local * L_local
```

其中：

```text
L_global = NCFD(f(x_real), f(x_syn))
```

```text
L_local = (1 / K) * sum_k NCFD(h(x_real^(k)), h(x_syn^(k)))
```

符号含义：

- `x_real`: 真实图像。
- `x_syn`: 合成图像。
- `f`: 原始 NCFM 的整图 feature extractor。
- `h`: 共享 patch encoder。
- `x^(k)`: 第 k 个原图 patch。
- `K`: patch 数量。
- `lambda_local`: 局部 NCFD 权重。

第一版只做：

```text
L_total = L_global + lambda_local * L_local
```

第一版不做：

- SSIM。
- local sampling net。
- raw pixel patch NCFD。
- CAM-guided loss。
- foreground mask loss。
- class-aware prototype loss。

---

## 4. 第一版推荐配置

### 4.1 主尺度

第一版主尺度：

```text
grid = 4x4
patch number K = 16
patch size = 7x7
```

原因：

1. `28 / 4 = 7`，整除，patch 边界清楚。
2. 7x7 对 28x28 小图来说足够局部。
3. 7x7 又不至于太小，仍能包含一块局部组织、细胞形态或纹理结构。
4. 比 2x2 更局部，比 7x7 grid 更稳定。

### 4.2 Patch feature 维度

推荐：

```text
patch feature dim Dp = 128
```

原因：

- 当前 ConvNet width 为 128。
- 与现有特征通道宽度一致。
- 不需要额外压得过低。
- 每个 patch 的 local NCFD 在 `[B, 128]` 上计算，维度清晰。

### 4.3 lambda_local

第一版只扫两个值：

```text
lambda_local = 0.3
lambda_local = 0.6
```

原因：

- 局部损失应该补充全局 NCFD，而不是覆盖全局语义。
- `0.3` 是保守局部约束。
- `0.6` 是较强局部结构约束。
- 第一版不把网格数和 lambda 同时扫太大，避免实验量失控。

### 4.4 先不加 SSIM

第一版不加 SSIM。

原因：

- SSIM 会引入新的结构正则和额外超参数。
- 一旦同时加入 local NCFD 和 SSIM，很难判断收益来自哪里。
- 当前最需要隔离验证的是：patch feature local NCFD 本身是否有效。

### 4.5 先不加 local sampling net

第一版不加 local sampling net。

严格来说，如果要叫“local NCFD”，后续可以加：

```text
local sampling net = SampleNet(feature_dim=128, t_batchsize=local_num_freqs)
```

但第一轮最好固定 local 频率采样，不引入新的 min-max 分支。

原因：

- B 基线已经包含全局 sampling net。
- 如果第一版同时加入 local sampling net，改动会变成：
  - local patch encoder
  - local NCFD
  - local sampling net
  - local min-max update
- 这样不利于判断“局部 patch feature 对齐”本身是否有效。

第一版目标是最小因果验证：

```text
B baseline
vs
B + fixed-frequency local patch-feature NCFD
```

---

## 5. 张量形状设计

以下默认 2D MedMNIST 输入：

```text
Input image: [B, C, 28, 28]
```

其中：

- PathMNIST / BloodMNIST: `C = 3`
- PneumoniaMNIST: `C = 1`

---

### 5.1 Global NCFD 分支

当前 B 基线的 global NCFD 使用整图 ConvNet flatten feature。

PathMNIST / BloodMNIST:

```text
[B, 3, 28, 28]
-> ConvNet
-> final feature [B, 1152]
```

PneumoniaMNIST:

```text
[B, 1, 28, 28]
-> ConvNet
-> final feature [B, 2048]
```

因此：

```text
L_global = NCFD([B, D_global], [B, D_global])
```

其中：

```text
D_global = 1152 for RGB 28x28
D_global = 2048 for grayscale 28x28
```

---

### 5.2 Local patch-feature NCFD: 1x1 grid

`1x1` 是无局部切分的对照。

```text
grid = 1x1
K = 1
patch size = 28x28
```

形状：

```text
[B, C, 28, 28]
-> [B, 1, C, 28, 28]
-> [B*1, C, 28, 28]
-> patch encoder h
-> [B*1, 128]
-> [B, 1, 128]
```

local loss:

```text
L_local_1x1 = NCFD(Z_real[:, 0, :], Z_syn[:, 0, :])
```

意义：

- 近似“patch-feature 版 global”。
- 主要作为尺度消融对照，不是主方法。

---

### 5.3 Local patch-feature NCFD: 2x2 grid

```text
grid = 2x2
K = 4
patch size = 14x14
```

形状：

```text
[B, C, 28, 28]
-> [B, 4, C, 14, 14]
-> [B*4, C, 14, 14]
-> patch encoder h
-> [B*4, 128]
-> [B, 4, 128]
```

local loss:

```text
L_local_2x2 = (1 / 4) * sum_{k=1}^4 NCFD(Z_real[:, k, :], Z_syn[:, k, :])
```

意义：

- 粗粒度局部结构。
- 可能更适合 PneumoniaMNIST 这种较依赖大区域肺野结构的任务。
- 作为 4x4 的粗尺度对照。

---

### 5.4 Local patch-feature NCFD: 4x4 grid

```text
grid = 4x4
K = 16
patch size = 7x7
```

形状：

```text
[B, C, 28, 28]
-> [B, 16, C, 7, 7]
-> [B*16, C, 7, 7]
-> patch encoder h
-> [B*16, 128]
-> [B, 16, 128]
```

local loss:

```text
L_local_4x4 = (1 / 16) * sum_{k=1}^{16} NCFD(Z_real[:, k, :], Z_syn[:, k, :])
```

意义：

- 第一版主尺度。
- 最符合“28x28 小图中局部医学结构被全局平均淹没”的问题设定。
- 每块 7x7 既局部又不至于太碎。

---

### 5.5 Local patch-feature NCFD: 7x7 grid

```text
grid = 7x7
K = 49
patch size = 4x4
```

严格地说，`28 / 7 = 4`，因此每块是 `4x4`。

形状：

```text
[B, C, 28, 28]
-> [B, 49, C, 4, 4]
-> [B*49, C, 4, 4]
-> patch encoder h
-> [B*49, 128]
-> [B, 49, 128]
```

local loss:

```text
L_local_7x7 = (1 / 49) * sum_{k=1}^{49} NCFD(Z_real[:, k, :], Z_syn[:, k, :])
```

意义：

- 极细局部尺度。
- 可能捕捉微小纹理或边缘片段。
- 风险是过细、噪声敏感、计算更重。
- 建议作为后续消融，不作为第一版主方法。

---

## 6. Patch encoder 建议

### 6.1 不能直接复用完整 3 层 ConvNet

当前完整 ConvNet 对 28x28 的空间变化是：

```text
28 -> 14 -> 7 -> 3
```

如果直接把 7x7 patch 输入完整 3 层 ConvNet：

```text
7 -> 3 -> 1 -> 0/报错
```

因此，4x4 grid 的 7x7 patch 不能直接过完整 3 层 ConvNet。

---

### 6.2 推荐 patch encoder

第一版推荐轻量 patch encoder：

```text
PatchEncoder:
  Conv3x3(C -> 128), padding=1
  Norm
  ReLU
  Conv3x3(128 -> 128), padding=1
  Norm
  ReLU
  AdaptiveAvgPool2d(1)
  flatten
```

输出：

```text
[B*K, C, patch_h, patch_w]
-> [B*K, 128]
```

优点：

- 兼容 28x28、14x14、7x7、4x4 patch。
- 不依赖固定 patch 尺寸。
- 输出固定为 128 维。
- 与现有 ConvNet width 对齐。

---

### 6.3 patch encoder 是否冻结

第一版建议：

```text
patch encoder 使用当前 pretrain / blended model 的浅层权重初始化，并冻结。
```

如果工程上先做最小原型，也可以：

```text
从当前 distillation step 选中的 model 里复用前 1-2 个 conv block，并对 patch 做 GAP。
```

但要注意：

- 如果 patch encoder 随 synthetic image 一起被优化，局部度量本身会漂。
- 如果 patch encoder 是额外随机网络，局部特征未必有医学判别意义。
- 更稳妥的是让 patch encoder 来自已经正常 pretrain 的 evaluator feature space。

---

## 7. 和当前 B 基线如何拼接

当前 B 基线可以理解为：

```text
sampling_net: True
num_freqs: 1024
iter_calib: 0
global NCFD with learnable psi
```

下一版主方法不是替换 B，而是在 B 上加 local 分支：

```text
B-local = B_global + local_patch_feature_NCFD
```

具体：

```text
L_total = L_global_B + lambda_local * L_local_patch
```

其中：

- `L_global_B` 保持 B 的论文式 min-max NCFM。
- `L_local_patch` 第一版使用固定 local frequency，不加 local sampling net。
- `patch encoder` 输出 128 维。
- 主尺度用 4x4 grid。

第一版不改变：

- 原始 global sampling net。
- 原始 global NCFD。
- 原始 condense 主循环逻辑。
- 预训练模型生成方式。
- evaluator 训练协议。

---

## 8. 现有 CAM / 预实验对该方法的支持程度

现有证据支持这条路线，但支持强度要分层写。

---

### 8.1 强支持

强支持的是：

> B 基线存在 real-vs-synthetic CAM 不一致问题，且这种不一致高度符合“局部空间结构约束不足”的解释。

证据包括：

1. B 与 real-trained CAM 的整体相似度偏低。
2. PathMNIST / BloodMNIST 中出现高置信正确但 low-similarity 的样本。
3. 错位样本常见 diffuse、edge-heavy、center-heavy、corner-biased、low-hot-mass。
4. 这些模式说明模型可能判对了，但关注结构不一定和 real-trained 模型一致。

解释：

```text
global NCFD 约束整图 feature 分布，
但不强制每个局部区域承载正确的医学判别结构。
```

因此，local patch-feature NCFD 正好补这个缺口。

---

### 8.2 中等支持

中等支持的是：

> PathMNIST / BloodMNIST 比 PneumoniaMNIST 更适合作为 local NCFD 的主验证数据集。

原因：

- PathMNIST 更像局部组织纹理和病理结构驱动。
- BloodMNIST 更像细胞形态和局部结构主体驱动。
- PneumoniaMNIST 更偏全局肺野纹理和整体结构，局部方法未必同等受益。

因此，第一版主实验应重点看：

```text
PathMNIST
BloodMNIST
```

PneumoniaMNIST 用作：

```text
全局结构型医学任务对照
```

---

### 8.3 弱支持 / 尚未证明

还没有被直接证明的是：

> local patch-feature NCFD 上线后一定提升性能和解释性指标。

目前还缺少直接因果实验：

- `B` vs `B + local patch-feature NCFD` 的正式对比。
- CAM similarity 是否上升。
- edge / corner bias 是否下降。
- occlusion behavior 是否更接近 real-trained evaluator。
- ACC / AUC / Macro-F1 / Balanced ACC 是否提升或至少不下降。

因此当前表述应该是：

> CAM 和预实验足够支持把 local patch-feature NCFD 作为下一步主线立项，但还不能宣称它已经被证明优于 B。

---

## 9. 第一轮最小验证闭环

第一轮不要铺太大，只做最小因果验证。

### 9.1 对比系统

| Run | 方法 | 目的 |
|---|---|---|
| B | global min-max NCFM | 当前主基线 / SOTA 代表 |
| B+Local-0.3 | B + 4x4 local patch-feature NCFD, lambda=0.3 | 验证保守局部约束 |
| B+Local-0.6 | B + 4x4 local patch-feature NCFD, lambda=0.6 | 验证较强局部约束 |

### 9.2 数据集

第一轮：

```text
PathMNIST
BloodMNIST
```

可选对照：

```text
PneumoniaMNIST
```

### 9.3 IPC

第一轮主设定：

```text
IPC = 10
```

如果算力允许，再补：

```text
IPC = 1
```

IPC=1 更容易暴露局部结构丢失，但方差可能更大。

### 9.4 指标

分类指标：

```text
ACC
AUC macro OvR
Macro-F1
Balanced ACC
```

二分类额外：

```text
Sensitivity
Specificity
AUPRC
```

解释性指标：

```text
CAM similarity to real
top-k IoU / Dice
Pearson / Spearman / Cosine
edge_mass
center_mass
corner_mass
cam_entropy
topk_activation_ratio
```

遮挡指标：

```text
hot_drop
random_drop
cold_drop
```

### 9.5 成功标准

强成功：

1. ACC / AUC / F1 / Balanced ACC 中至少主要指标不下降，最好提升。
2. CAM similarity to real 上升。
3. edge / corner bias 下降，或 CAM entropy / hot mass 更合理。
4. occlusion 中 hot_drop 相对 random_drop / cold_drop 更符合 real-trained 行为。

中等成功：

1. 分类指标小幅提升或持平。
2. CAM similarity 明显更接近 real。
3. 低相似高置信样本比例下降。

弱成功：

1. 分类指标有提升，但 CAM 没改善。
2. 说明 local patch-feature NCFD 可能提高了判别性能，但未必让模型看得更医学合理。

失败：

1. 分类指标下降。
2. CAM similarity 不升反降。
3. edge / corner bias 更强。
4. synthetic 图像质量变差。

---

## 10. 后续消融计划

如果第一轮 `4x4` 有正向信号，再做尺度消融。

### 10.1 尺度消融

| Variant | Grid | Patch number | Patch size | 目的 |
|---|---:|---:|---:|---|
| Global only | 1x1 | 1 | 28x28 | 对照 |
| Coarse local | 2x2 | 4 | 14x14 | 大区域局部结构 |
| Main local | 4x4 | 16 | 7x7 | 主方法 |
| Fine local | 7x7 | 49 | 4x4 | 极细纹理消融 |

预期：

- PathMNIST / BloodMNIST 可能更受益于 4x4。
- PneumoniaMNIST 可能 2x2 或 global already enough。
- 7x7 可能过细，容易噪声敏感。

### 10.2 lambda 消融

第一轮：

```text
lambda_local = 0.3, 0.6
```

后续如有必要：

```text
lambda_local = 0.1, 0.9, 1.0
```

### 10.3 local sampling net 消融

第一轮不加。

如果 local patch-feature NCFD 有效，再比较：

| Variant | local sampling net | 目的 |
|---|---|---|
| fixed local frequency | no | 隔离 local 特征对齐 |
| local psi | yes | 验证局部 min-max 是否进一步有用 |

local psi 形状：

```text
SampleNet(feature_dim=128, t_batchsize=local_num_freqs)
```

### 10.4 SSIM / 结构正则

第一轮不加。

如果局部 NCFD 有效，再加入：

```text
L_total = L_global + lambda_local * L_local + lambda_ssim * L_ssim
```

建议只作为第三阶段，因为它会引入额外解释变量。

---

## 11. 论文叙事建议

建议不要写成：

> 原始 NCFM 在 MedMNIST 上失败。

更稳的写法是：

> 原始 NCFM 的全局 NCFD 在某些医学任务上已经有效，尤其是全局结构较强的任务；但在局部组织纹理和细胞形态更关键的 MedMNIST 任务上，synthetic-trained evaluator 与 real-trained evaluator 的注意力存在明显空间错位，说明全局分布对齐没有充分约束局部医学判别结构。基于这一诊断，我们提出 local patch-feature NCFD，在保留全局 NCFM 的同时，对原图局部 patch 的结构特征分布进行对齐。

这条叙事链是：

```text
CAM 诊断问题
-> 发现局部结构错位
-> 设计 patch-feature local NCFD
-> 用指标和 CAM 共同验证
```

---

## 12. 当前结论

最终结论：

> 如果严格按当前 MedMNIST 多尺度 NCFD 理论目标来选，最符合的是“原图切块 -> patch feature NCFD”；方法定义写作 B，工程实现采用 C。

原因：

1. 它保留了 NCFM 的 feature-space NCFD 精神。
2. 它把局部单位直接绑定到原图 28x28 的医学结构区域。
3. 它比 pixel patch 匹配更结构感知。
4. 它比 feature-map local 更贴合当前论文叙事。
5. 现有 CAM 和预实验已经足够支持它作为下一步主线。

第一版建议：

```text
Base: B_minmax_ncfm_psi
Local grid: 4x4
Patch size: 7x7
Patch feature dim: 128
lambda_local: 0.3 / 0.6
No SSIM
No local sampling net
```

第一轮验证：

```text
B
vs
B + 4x4 local patch-feature NCFD, lambda=0.3
vs
B + 4x4 local patch-feature NCFD, lambda=0.6
```

主要看：

```text
ACC / AUC / Macro-F1 / Balanced ACC
CAM similarity to real
spatial bias
occlusion behavior
```

只有当解释性指标也向 real-trained evaluator 靠近时，这条 local patch-feature NCFD 路线才算真正被坐实。

---

## 13. 当前拍板意见

经过现有 CAM 诊断、B 基线结果、医学局部结构动机和工程可控性综合判断，当前正式选择是：

> 将“原图切块 -> patch feature NCFD”作为下一阶段主方法。

这个选择不是单纯为了多加一个正则项，而是直接服务当前核心研究问题：

> MedMNIST 这种医学小图里，NCFM 的全局 NCFD 是否没有充分保住关键局部结构？

当前选择成立的核心理由有五点。

---

### 13.1 它最贴合当前真正想研究的问题

当前研究问题不是：

> 再加一个工程 trick 能不能涨点？

而是：

> 医学小图中，局部病灶、局部细胞形态、局部组织纹理是否在 NCFM 的全局特征对齐中被淹没？

因此，局部单位最好直接来自原图 patch。

原图 patch 与医学局部区域之间有明确对应关系：

- PathMNIST: 局部组织结构、纹理、细胞排列。
- BloodMNIST: 局部细胞形态、边界、染色区域。
- PneumoniaMNIST: 肺野局部纹理和较大范围结构，更多作为全局结构型对照。

所以“原图切块 -> patch feature NCFD”是直接对当前问题下手，而不是旁敲侧击。

---

### 13.2 它比 patch 像素匹配更合理

直接对 patch 像素做 local NCFD 容易退化成低层统计匹配，例如：

- 亮度匹配。
- 纹理噪声匹配。
- 边界统计匹配。
- 设备风格匹配。

这些不一定是医学判别结构。

当前真正想保留的是：

> 局部结构语义，而不是局部像素逐点像。

因此，local loss 应该落在 patch feature space，而不是 raw pixel space。

---

### 13.3 它比 feature-map local 更符合当前论文叙事

Feature-map local NCFD 当然也能做，而且工程上可能更容易。

但是 feature-map local 的局部单位是：

```text
网络内部 feature map 的空间位置
```

而当前叙事已经明确强调：

```text
原图 28x28 小图
局部块 7x7
局部医学诊断区域
```

因此，从论文表达看，原图切块版更顺：

```text
28x28 medical image
-> 4x4 grid
-> 16 local patches
-> patch structural features
-> local NCFD
```

这个流程更容易让读者理解：

> 我们不是在网络内部随便切特征，而是在医学小图的原始空间里显式保留局部结构。

---

### 13.4 它和现有 CAM 证据对得上

当前 CAM 证据显示的问题不是：

> 模型完全学不会类别。

而是：

> 模型能判对，但关注区域和 real-trained evaluator 不一致。

典型现象包括：

- diffuse: 注意力过于分散。
- edge-heavy: 过度关注边缘。
- center-heavy: 过度依赖中心区域。
- corner-biased: 角落/位置偏置。
- low-hot-mass: 热点不集中，显著区域质量低。

这些现象最自然的解释是：

```text
global NCFD 已经能学到部分全局判别统计，
但没有强制每个局部区域保留正确医学结构。
```

因此，patch-feature local NCFD 正好对症：

```text
每个局部 patch 也要在 feature distribution 上靠近真实数据。
```

---

### 13.5 它最适合作为第一版主方法

第一版研究最怕一次加太多东西，最后不知道到底是谁起作用。

当前选择足够干净：

```text
保留 B 的 global min-max NCFM
只增加 local patch-feature NCFD
先不加 local sampling net
先不加 SSIM
先只看 4x4 主尺度
```

这样一旦有效，因果链非常清楚：

```text
收益主要来自局部 patch feature distribution alignment。
```

因此，当前拍板版本是：

```text
主方法: B + 4x4 local patch-feature NCFD
patch size: 7x7
patch feature dim: 128
lambda_local: 0.3 / 0.6
patch encoder: frozen
local psi: no
SSIM: no
```

---

## 14. 审稿人视角下必须守住的两点

这个选择成立，但必须守住两个边界，否则容易被审稿人质疑。

---

### 14.1 Patch encoder 不要变成另一个会漂的学习系统

如果 patch encoder 跟 synthetic image 一起训练或频繁更新，审稿人可能会问：

> 你到底是在约束局部医学结构，还是又引入了一个可变的局部判别器？

因此第一版建议：

```text
patch encoder 来自稳定 pretrain feature space
patch encoder frozen
```

这样 local NCFD 的意义更清楚：

> 在一个固定、已具备医学判别能力的局部 feature space 中，对齐真实与合成 patch 的局部分布。

第一版不建议：

- patch encoder 随 synthetic data 一起训练。
- patch encoder 从随机初始化开始学。
- 同时引入 local sampling net 和 trainable patch encoder。

后续如果要做可学习 patch encoder，应作为单独消融，而不是第一版主方法。

---

### 14.2 不要只报分类指标

这个方法的价值不只是 ACC 能不能涨。

它真正想证明的是：

> synthetic-trained evaluator 的局部关注是否更接近 real-trained evaluator。

因此，即使 ACC 只小幅提升，只要下面指标明显改善，也有研究价值：

- CAM similarity to real 提高。
- top-k IoU / Dice 提高。
- edge/corner bias 降低。
- CAM entropy 更合理。
- occlusion hot region 的行为更接近 real-trained evaluator。
- low-similarity high-confidence case 比例下降。

反过来，如果 ACC 提升但 CAM 更歪，需要谨慎解释：

> 这可能只是学到了新的 shortcut，而不是保住了更合理的医学结构。

因此第一版实验必须同时看：

```text
分类指标 + 可解释性指标 + 遮挡行为
```

---

## 15. 下一步 technical spec 要明确的问题

下一步不应直接盲改代码，而应先写一份实现规格书，明确最小改动边界。

technical spec 应至少回答下面问题。

### 15.1 patchify 接口

需要明确：

- 输入张量形状。
- grid 如何指定。
- patch 尺寸如何计算。
- 是否要求 `28 % grid == 0`。
- 输出 patch tensor 形状。
- 是否使用 `unfold`。

建议接口：

```python
patches = patchify_images(images, grid=4)
```

输入：

```text
images: [B, C, 28, 28]
```

输出：

```text
patches: [B, K, C, patch_h, patch_w]
```

其中：

```text
K = grid * grid
patch_h = 28 / grid
patch_w = 28 / grid
```

---

### 15.2 patch encoder 放在哪里

需要决定新模块位置。

建议：

```text
NCFM/PatchEncoder.py
```

或：

```text
condenser/local_patch_ncfd.py
```

第一版更推荐：

```text
condenser/local_patch_ncfd.py
```

原因：

- local patch NCFD 先作为 loss 层增强。
- 不污染 `NCFM/NCFM.py` 的核心全局 NCFD 定义。
- 方便后续回退和消融。

---

### 15.3 local loss 函数签名

建议第一版函数签名：

```python
def local_patch_feature_ncfd_loss(
    img_real,
    img_syn,
    patch_encoder,
    cf_loss_func,
    args,
    grid=4,
    lambda_local=0.3,
):
    ...
```

输出：

```text
loss_local
```

如需 logging，可额外返回：

```text
loss_local_raw
patch_shape
```

---

### 15.4 如何接入 compute_loss.py

第一版接入方式：

```text
loss_global = 原始 B 的 match_loss
loss_local = local_patch_feature_ncfd_loss(...)
loss_total = loss_global + lambda_local * loss_local
```

建议新增配置字段：

```yaml
use_local_patch_feature_ncfd: true
local_patch_grid: 4
lambda_local_patch_ncfd: 0.3
local_patch_feature_dim: 128
local_patch_encoder_frozen: true
use_local_patch_sampling_net: false
```

第一版不使用旧字段：

- `use_local_token_ncfm`
- `use_local_patch_ncfm`
- `local_patch_block_counts`
- `lambda_local`

避免和历史 raw patch 分支混淆。

---

### 15.5 哪些文件可以改，哪些不要动

第一版允许改：

```text
condenser/compute_loss.py
condenser/local_patch_ncfd.py
condense/condense_script.py
utils/init_script.py
utils/experiment_tracker.py
config/ipc10/*.yaml 或新增 config
```

第一版尽量不要改：

```text
NCFM/NCFM.py
NCFM/SampleNet.py
condenser/Condenser.py
models/convnet.py
```

如果必须改，也要把改动控制在接口读取或 logging，不改变原始 global NCFD 行为。

---

## 16. 下一步执行建议

建议按这个顺序推进：

1. **先写 technical spec**
   - 明确 patchify、patch encoder、local loss、配置字段、接入点。
   - 仍然不改训练代码。

2. **从当前基线开新分支**
   - 分支名建议：
     ```text
     local-patch-feature-ncfd-v1
     ```
   - 从当前提交之后继续，不污染 baseline tag。

3. **实现最小版本**
   - 只实现 `B + 4x4 local patch-feature NCFD`。
   - `lambda_local = 0.3 / 0.6`。
   - frozen patch encoder。
   - no SSIM。
   - no local psi。

4. **先跑 smoke test**
   - PathMNIST IPC=1 或小 niter。
   - 只验证 shape、loss、梯度、显存和日志。

5. **再跑第一轮正式对比**
   - PathMNIST IPC=10。
   - BloodMNIST IPC=10。
   - 对比 B、B+Local-0.3、B+Local-0.6。

6. **同步跑解释性评估**
   - CAM similarity to real。
   - spatial bias。
   - occlusion sensitivity。
   - low-similarity high-confidence case 比例。

7. **根据结果决定下一步**
   - 如果分类和解释性都改善：进入尺度消融。
   - 如果分类改善但 CAM 不改善：检查是否学到新 shortcut。
   - 如果 CAM 改善但分类不升：考虑调 lambda 或局部尺度。
   - 如果都不改善：回头检查 patch encoder 和频率采样设计。

