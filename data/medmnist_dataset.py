"""
MedMNIST dataset wrapper for NCFM dataset condensation.
Supports both balanced and imbalanced medical datasets.
"""

import os

import medmnist
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data.dataset_statistics import MEANS, STDS


class MedMNISTDataset(Dataset):
    """
    MedMNIST dataset wrapper that converts MedMNIST data to tensor format
    compatible with NCFM's condensation pipeline.
    """

    def __init__(self, dataset_name, split='train', download=True, data_dir=None):
        """
        Args:
            dataset_name: Name of MedMNIST dataset (e.g., 'pathmnist', 'dermamnist')
            split: 'train', 'val', or 'test'
            download: Whether to download if not exists
            data_dir: Directory to store data (default: ~/.medmnist)
        """
        self.dataset_name = dataset_name
        self.info = INFO[dataset_name]

        # Set data directory
        if data_dir is None:
            import os
            data_dir = os.path.expanduser('~/.medmnist')
        else:
            import os
            data_dir = os.path.expanduser(data_dir)

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # 关键改动：
        # MedMNIST 原始返回通常是 PIL/uint8 图像，像素范围是 [0,255]。
        # 这里统一走 torchvision 的 ToTensor()，把它转成：
        #   类型: torch.float32
        #   形状: (C, H, W)
        #   数值范围: [0,1]
        # 这样就和原作者在 CIFAR 分支里的输入约定一致。
        to_tensor = transforms.ToTensor()

        # Get dataset class
        DatasetClass = getattr(medmnist, self.info['python_class'])

        # Download and load dataset
        self.dataset = DatasetClass(
            split=split,
            transform=to_tensor,
            download=download,
            root=data_dir,
        )

        # 提前把整个 split 读进内存，后面 NCFM 的 class sampler 直接拿 tensor 用。
        # self.images: Tensor, 形状 [N, C, H, W], 数值范围 [0,1]
        # self.labels: Tensor, 形状 [N]
        self.images = []
        self.labels = []

        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            # 正常情况下 DatasetClass 已经应用了 ToTensor()；
            # 这里保留兜底分支，避免未来改动 transform 后类型不一致。
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)

            self.images.append(img)
            # MedMNIST 的 label 可能是 numpy 标量/数组，这里统一压成 Python int。
            self.labels.append(int(torch.as_tensor(label).view(-1)[0].item()))

        # Stack into tensors
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

        # Dataset attributes
        self.nclass = len(self.info['label'])
        self.n_channel = self.info['n_channels']
        self.size = self.images.shape[-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def get_class_distribution(self):
        """Return class distribution as dict."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}


class MedMNISTTensorDataset(Dataset):
    """
    Tensor-backed dataset for the post-loaded MedMNIST pipeline.

    这里专门把三条链拆开：
    1. real train: transform=None，直接输出 float32 [0,1]
    2. real val/test: transform=Normalize(...)，直接输出标准化后的 tensor
    3. syn train: 不走这里，而是走 condenser/condense_transfom.py 里的 from_tensor=True 分支
    """

    def __init__(self, images, labels, transform=None):
        # images: [N, C, H, W], torch.float32, 默认数值范围 [0,1]
        # labels: [N], torch.long/int
        self.images = images.float()
        self.targets = labels.long()
        self.labels = self.targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform is not None:
            # 对 eval 链来说，这一步做的是:
            # normalized = (sample - mean) / std
            # 输入 sample:  torch.float32, [C,H,W], 范围 [0,1]
            # 输出 sample:  torch.float32, [C,H,W], 零均值/单位方差附近
            sample = self.transform(sample)
        return sample, self.targets[index]


def build_medmnist_eval_transform(dataset_name):
    """
    Build the upstream-style eval transform for MedMNIST.

    原作者 CIFAR 风格里:
    - train loader: 只 ToTensor -> [0,1]
    - val/test loader: ToTensor + Normalize

    MedMNIST 已经在上游包装里完成了 ToTensor()，所以这里 eval 只需要继续做:
        x_norm = (x - mean) / std
    """
    return transforms.Normalize(mean=MEANS[dataset_name], std=STDS[dataset_name])


def load_medmnist_data(dataset_name, data_dir=None, download=True):
    """
    Load MedMNIST train and test datasets for NCFM.

    Args:
        dataset_name: Name of MedMNIST dataset
        data_dir: Directory to store data (default: ~/.medmnist)
        download: Whether to download if not exists

    Returns:
        train_dataset, test_dataset

        这里按严格 upstream 风格返回两条不同语义的数据流:
        - train_dataset: train + val 合并，样本类型是 torch.float32，范围 [0,1]
        - test_dataset:  test split，但 __getitem__ 时已经执行 Normalize(mean, std)
    """
    # Set default data directory
    if data_dir is None:
        import os
        data_dir = os.path.expanduser('~/.medmnist')
    else:
        import os
        data_dir = os.path.expanduser(data_dir)

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading MedMNIST dataset: {dataset_name}")
    print(f"  Data directory: {data_dir}")
    info = INFO[dataset_name]
    print(f"  Task: {info['task']}")
    print(f"  Classes: {len(info['label'])}")
    print(f"  Channels: {info['n_channels']}")

    train_dataset = MedMNISTDataset(dataset_name, split='train', download=download, data_dir=data_dir)
    val_dataset = MedMNISTDataset(dataset_name, split='val', download=download, data_dir=data_dir)
    test_dataset = MedMNISTDataset(dataset_name, split='test', download=download, data_dir=data_dir)

    # 当前项目训练口径是把 train + val 合并后作为“训练集”。
    # 所以后面如果要重算 mean/std，也应该按同样口径来算。
    import torch
    combined_images = torch.cat([train_dataset.images, val_dataset.images], dim=0)
    combined_labels = torch.cat([train_dataset.labels, val_dataset.labels], dim=0)

    # train real 链:
    # train_loader / condense real loader 从这里拿到的是未标准化 tensor。
    # 类型: torch.float32
    # 范围: [0,1]
    train_combined = MedMNISTTensorDataset(combined_images, combined_labels)
    train_combined.nclass = train_dataset.nclass
    train_combined.n_channel = train_dataset.n_channel
    train_combined.size = train_dataset.size

    # val/test real 链:
    # validate() / evaluation 不再额外做 Normalize，所以这里直接把 Normalize
    # 绑定到 dataset.__getitem__ 上，确保送入模型前已经是标准化后的值。
    eval_transform = build_medmnist_eval_transform(dataset_name)
    test_final = MedMNISTTensorDataset(
        test_dataset.images,
        test_dataset.labels,
        transform=eval_transform,
    )
    test_final.nclass = test_dataset.nclass
    test_final.n_channel = test_dataset.n_channel
    test_final.size = test_dataset.size

    # Print class distribution
    print(f"\nTrain class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for c, count in sorted(train_dist.items()):
        print(f"  Class {c}: {count} samples")

    print(f"\nTest class distribution:")
    test_dist = test_dataset.get_class_distribution()
    for c, count in sorted(test_dist.items()):
        print(f"  Class {c}: {count} samples")

    # Calculate imbalance ratio
    counts = list(train_dist.values())
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    return train_combined, test_final


def compute_medmnist_mean_std(
    dataset_name,
    data_dir=None,
    download=True,
    splits=('train', 'val'),
    batch_size=1024,
    num_workers=0,
):
    """Compute exact per-channel mean/std after ToTensor() scaling to [0, 1]."""
    if data_dir is None:
        data_dir = os.path.expanduser('~/.medmnist')
    else:
        data_dir = os.path.expanduser(data_dir)

    os.makedirs(data_dir, exist_ok=True)

    DatasetClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    to_tensor = transforms.ToTensor()

    # total_sum[c]    = 第 c 个通道所有像素之和
    # total_sq[c]     = 第 c 个通道所有像素平方和
    # total_pixels    = 每个通道累计的像素数 = 样本数 * H * W
    total_sum = None
    total_sq = None
    total_pixels = 0

    for split in splits:
        dataset = DatasetClass(
            split=split,
            transform=to_tensor,
            download=download,
            root=data_dir,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        for images, _ in loader:
            # images: [B, C, H, W], float64, 数值范围 [0,1]
            images = images.double()
            _, channels, height, width = images.shape

            if total_sum is None:
                total_sum = torch.zeros(channels, dtype=torch.float64)
                total_sq = torch.zeros(channels, dtype=torch.float64)

            total_sum += images.sum(dim=(0, 2, 3))
            total_sq += (images * images).sum(dim=(0, 2, 3))
            total_pixels += images.shape[0] * height * width

    # 按通道计算总体均值和总体标准差：
    # mean = E[x]
    # std  = sqrt(E[x^2] - E[x]^2)
    mean = total_sum / total_pixels
    std = (total_sq / total_pixels - mean * mean).sqrt()
    return mean.tolist(), std.tolist()


def get_medmnist_mean_std(dataset_name):
    # 这里仍然是“硬编码表”，但现在这些值的口径已经统一成：
    # 1. 先 ToTensor()，即 [0,255] -> [0,1]
    # 2. 默认按 train + val 统计
    # 3. 每个通道单独计算 mean/std
    """Get configured per-channel mean/std values in [0, 1] space."""
    medmnist_stats = {
        'pathmnist': {'mean': [0.740638, 0.533137, 0.705940], 'std': [0.123628, 0.176805, 0.124425]},
        'chestmnist': {'mean': [0.5096], 'std': [0.2751]},
        'dermamnist': {'mean': [0.763445, 0.538597, 0.562011], 'std': [0.136320, 0.153896, 0.168832]},
        'octmnist': {'mean': [0.2462], 'std': [0.1966]},
        'pneumoniamnist': {'mean': [0.4784], 'std': [0.2157]},
        'retinamnist': {'mean': [0.5333, 0.4667, 0.4588], 'std': [0.1608, 0.1725, 0.1804]},
        'breastmnist': {'mean': [0.5176], 'std': [0.2510]},
        'bloodmnist': {'mean': [0.794354, 0.659687, 0.696210], 'std': [0.215488, 0.241463, 0.117844]},
        'tissuemnist': {'mean': [0.5804], 'std': [0.2431]},
        'organamnist': {'mean': [0.5686], 'std': [0.2627]},
        'organcmnist': {'mean': [0.5569], 'std': [0.2706]},
        'organsmnist': {'mean': [0.5608], 'std': [0.2667]},
    }

    if dataset_name in medmnist_stats:
        return medmnist_stats[dataset_name]['mean'], medmnist_stats[dataset_name]['std']
    else:
        # Default to ImageNet stats as approximation
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
