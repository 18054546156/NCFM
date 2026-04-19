# NCFM MedMNIST 适配指南

## 修改概述

将NCFM数据集蒸馏方法适配到MedMNIST医学图像数据集，支持均衡和不均衡数据集。

## 新增文件

### 1. `data/medmnist_dataset.py`
```python
# MedMNIST数据集加载器
- MedMNISTDataset: 将MedMNIST数据转换为NCFM兼容的张量格式
- load_medmnist_data: 加载训练和测试集
- get_medmnist_mean_std: 获取数据集统计信息
```

## 修改文件

### 1. `data/dataset_statistics.py`
添加了MedMNIST数据集的mean/std值：
```python
# 新增统计信息
MEANS['pathmnist'] = [0.7380, 0.5455, 0.6583]
STDS['pathmnist'] = [0.1678, 0.1880, 0.1775]
# ... 其他11个数据集
```

### 2. `utils/utils.py`
在`load_resized_data()`函数中添加MedMNIST支持：
```python
elif dataset in ["pathmnist", "chestmnist", "dermamnist", ...]:
    from data.medmnist_dataset import load_medmnist_data
    train_dataset, val_dataset = load_medmnist_data(...)
```

### 3. 配置文件
- `config/ipc10/pathmnist.yaml` - 均衡数据集配置
- `config/ipc10/dermamnist.yaml` - 不均衡数据集配置

## 使用方法

### 安装依赖
```bash
pip install medmnist
pip install efficientnet-pytorch  # NCFM依赖
```

### 运行数据蒸馏

#### 1. 均衡数据集 (PathMNIST)
```bash
cd /path/to/NCFM

# 单GPU
python condense/condense_script.py \
    --config_path config/ipc10/pathmnist.yaml \
    --gpu "0" \
    --ipc 10 \
    --run_mode Condense

# 多GPU
CUDA_VISIBLE_DEVICES=0,1 python condense/condense_script.py \
    --config_path config/ipc10/pathmnist.yaml \
    --gpu "0,1" \
    --ipc 10 \
    --run_mode Condense
```

#### 2. 不均衡数据集 (DermaMNIST)
```bash
python condense/condense_script.py \
    --config_path config/ipc10/dermamnist.yaml \
    --gpu "0" \
    --ipc 10 \
    --run_mode Condense
```

### 评估蒸馏结果
```bash
# 评估阶段
python evaluation/evaluation_script.py \
    --config_path config/ipc10/pathmnist.yaml \
    --gpu "0" \
    --ipc 10 \
    --run_mode Evaluation
```

## 数据集信息

### 均衡数据集: PathMNIST
- **类别数**: 9
- **不平衡比**: 1.63 (相对均衡)
- **样本数**: 训练集~90K, 测试集~7K
- **图像**: 3x28x28 RGB病理图像
- **任务**: 结直肠癌组织病理学多分类

### 不均衡数据集: DermaMNIST
- **类别数**: 7
- **不平衡比**: 58.66 (严重不均衡)
- **样本分布**:
  - 类别0: 867样本 (actinic keratoses)
  - 类别1: 1119样本 (basal cell carcinoma)
  - 类别2: 1030样本 (benign keratosis)
  - 类别3: 115样本 (dermatofibroma)
  - 类别4: 1113样本 (melanoma)
  - 类别5: 4693样本 (melanocytic nevi) - 最多
  - 类别6: 80样本 (vascular lesions) - 最少
- **图像**: 3x28x28 RGB皮肤镜图像
- **任务**: 皮肤病变多分类

## 配置参数说明

### 关键参数
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `ipc` | 每类蒸馏样本数 | 10-50 |
| `niter` | 蒸馏迭代次数 | 20000 |
| `lr_img` | 图像学习率 | 0.01 |
| `dis_metrics` | 分布匹配度量 | "NCFM" |
| `num_premodel` | 预训练模型数量 | 20 |

### 针对不均衡数据的调整
对于DermaMNIST等不均衡数据集：
1. 减小`batch_real` (避免过拟合)
2. 可能需要调整`ipc`以平衡各类
3. 考虑使用类别权重

## 输出文件

蒸馏完成后，结果保存在：
```
results/condense/
├── pathmnist_ipc10/
│   └── cond_data.pt  # 蒸馏数据
└── dermamnist_ipc10/
    └── cond_data.pt
```

## 扩展到其他MedMNIST数据集

1. 在`data/dataset_statistics.py`添加对应统计信息
2. 复制并修改配置文件，调整：
   - `dataset`: 数据集名称
   - `nclass`: 类别数
   - `nch`: 通道数 (1或3)
   - `size`: 图像尺寸 (通常28)

3. 运行蒸馏

## 实验建议

### 对比实验
1. 均衡 vs 不均衡数据集的蒸馏效果
2. 不同ipc值的影响 (1, 10, 50)
3. 不同蒸馏方法 (NCFM vs DC, DSA等)

### 评估指标
- 整体准确率
- 每类准确率 (特别关注少数类)
- F1-score (对于不均衡数据)
