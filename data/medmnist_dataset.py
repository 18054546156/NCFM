"""
MedMNIST dataset wrapper for NCFM dataset condensation.
Supports both balanced and imbalanced medical datasets.
"""

import torch
import numpy as np
import medmnist
from medmnist import INFO
import os
from torch.utils.data import Dataset


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

        # Get dataset class
        DatasetClass = getattr(medmnist, self.info['python_class'])

        # Download and load dataset
        self.dataset = DatasetClass(split=split, download=download, root=data_dir)

        # Extract data and labels
        self.images = []
        self.labels = []

        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            # Convert to tensor if not already
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.array(img)).float()
            # Ensure channel-first format (C, H, W)
            if img.dim() == 2:  # Grayscale (H, W) -> (1, H, W)
                img = img.unsqueeze(0)
            elif img.dim() == 3 and img.shape[-1] in [1, 3]:  # (H, W, C) -> (C, H, W)
                img = img.permute(2, 0, 1)

            self.images.append(img)
            self.labels.append(int(label))

        # Stack into tensors
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

        # Dataset attributes
        self.nclass = len(self.info['label'])
        self.n_channel = self.info['n_channels']
        self.size = self.images.shape[1]  # Height (should equal width)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def get_class_distribution(self):
        """Return class distribution as dict."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}


def load_medmnist_data(dataset_name, data_dir=None, download=True):
    """
    Load MedMNIST train and test datasets for NCFM.

    Args:
        dataset_name: Name of MedMNIST dataset
        data_dir: Directory to store data (default: ~/.medmnist)
        download: Whether to download if not exists

    Returns:
        train_dataset, test_dataset
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

    # Combine train and val for training (common practice)
    import torch
    combined_images = torch.cat([train_dataset.images, val_dataset.images], dim=0)
    combined_labels = torch.cat([train_dataset.labels, val_dataset.labels], dim=0)

    # Create combined dataset
    try:
        from data.dataset import Dataset
    except ImportError:
        # Fallback to torch Dataset
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.targets = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, index):
                return self.images[index], self.targets[index]

    train_combined = Dataset(combined_images, combined_labels)
    train_combined.nclass = train_dataset.nclass

    # Test dataset
    test_final = Dataset(test_dataset.images, test_dataset.labels)
    test_final.nclass = test_dataset.nclass

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


def get_medmnist_mean_std(dataset_name):
    """Get approximate mean and std for MedMNIST datasets."""
    # These are approximate values - for exact values, compute from dataset
    medmnist_stats = {
        'pathmnist': {'mean': [0.7380, 0.5455, 0.6583], 'std': [0.1678, 0.1880, 0.1775]},
        'chestmnist': {'mean': [0.5096], 'std': [0.2751]},
        'dermamnist': {'mean': [0.6274, 0.5294, 0.5451], 'std': [0.1922, 0.2078, 0.2157]},
        'octmnist': {'mean': [0.2462], 'std': [0.1966]},
        'pneumoniamnist': {'mean': [0.4784], 'std': [0.2157]},
        'retinamnist': {'mean': [0.5333, 0.4667, 0.4588], 'std': [0.1608, 0.1725, 0.1804]},
        'breastmnist': {'mean': [0.5176], 'std': [0.2510]},
        'bloodmnist': {'mean': [0.6863, 0.6549, 0.6667], 'std': [0.1294, 0.1412, 0.1373]},
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
