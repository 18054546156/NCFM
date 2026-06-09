import json
import os

import numpy as np

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    medmnist = None
    INFO = {}

from data.dataset_statistics import MEANS, STDS


SUPPORTED_MEDMNIST_SIZES = {28, 64, 128, 224}
SUPPORTED_MEDMNIST_TASKS = {"binary-class", "multi-class"}


def is_supported_medmnist(dataset_name):
    return (
        dataset_name in INFO
        and INFO[dataset_name].get("task") in SUPPORTED_MEDMNIST_TASKS
    )


def get_medmnist_root(data_dir):
    root = os.path.join(data_dir, "medmnist")
    os.makedirs(root, exist_ok=True)
    return root


def _normalize_size(size):
    size = 28 if size is None else int(size)
    if size not in SUPPORTED_MEDMNIST_SIZES:
        raise ValueError(
            f"MedMNIST size must be one of {sorted(SUPPORTED_MEDMNIST_SIZES)}, got {size}."
        )
    return size


def get_medmnist_info(dataset_name):
    if medmnist is None:
        raise ModuleNotFoundError(
            "medmnist is required for MedMNIST support. Install requirements.txt first."
        )
    if not is_supported_medmnist(dataset_name):
        raise ValueError(
            f"Unsupported MedMNIST dataset: {dataset_name}. "
            "Only binary-class and multi-class MedMNIST tasks are supported."
        )
    return INFO[dataset_name]


def get_medmnist_class(dataset_name):
    class_name = get_medmnist_info(dataset_name)["python_class"]
    return getattr(medmnist, class_name)


def get_medmnist_nclass(dataset_name):
    return len(get_medmnist_info(dataset_name)["label"])


def get_medmnist_nch(dataset_name):
    return int(get_medmnist_info(dataset_name)["n_channels"])


def medmnist_target_transform(target):
    return int(np.asarray(target).squeeze())


def _size_arg(size):
    size = _normalize_size(size)
    return None if size == 28 else size


def _size_suffix(size):
    size = _normalize_size(size)
    return "" if size == 28 else f"_{size}"


def _stats_cache_path(root, dataset_name, size):
    return os.path.join(root, f"{dataset_name}{_size_suffix(size)}_stats.json")


def _register_stats(dataset_name, mean, std):
    MEANS[dataset_name] = [float(v) for v in mean]
    STDS[dataset_name] = [float(v) for v in std]


def _compute_channel_stats(imgs, chunk_size=2048):
    if imgs.ndim == 3:
        imgs = imgs[..., None]

    channel_sum = np.zeros(imgs.shape[-1], dtype=np.float64)
    channel_sq_sum = np.zeros(imgs.shape[-1], dtype=np.float64)
    pixel_count = 0

    for start in range(0, imgs.shape[0], chunk_size):
        chunk = imgs[start : start + chunk_size].astype(np.float64) / 255.0
        channel_sum += chunk.sum(axis=(0, 1, 2))
        channel_sq_sum += np.square(chunk).sum(axis=(0, 1, 2))
        pixel_count += chunk.shape[0] * chunk.shape[1] * chunk.shape[2]

    mean = channel_sum / pixel_count
    var = np.maximum(channel_sq_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


def register_medmnist_stats(dataset_name, root, size=None):
    size = _normalize_size(size)
    cache_path = _stats_cache_path(root, dataset_name, size)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        _register_stats(dataset_name, payload["mean"], payload["std"])
        return MEANS[dataset_name], STDS[dataset_name]

    dataset_cls = get_medmnist_class(dataset_name)
    train_dataset = dataset_cls(
        split="train",
        transform=None,
        target_transform=medmnist_target_transform,
        download=True,
        as_rgb=(get_medmnist_nch(dataset_name) == 3),
        root=root,
        size=_size_arg(size),
    )

    mean, std = _compute_channel_stats(train_dataset.imgs)
    _register_stats(dataset_name, mean, std)

    payload = {
        "dataset": dataset_name,
        "size": size,
        "mean": MEANS[dataset_name],
        "std": STDS[dataset_name],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return MEANS[dataset_name], STDS[dataset_name]


def build_medmnist_dataset(
    dataset_name,
    root,
    split,
    transform,
    size=None,
    download=True,
):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"MedMNIST split must be train, val, or test, got {split}.")

    size = _normalize_size(size)
    dataset_cls = get_medmnist_class(dataset_name)
    dataset = dataset_cls(
        split=split,
        transform=transform,
        target_transform=medmnist_target_transform,
        download=download,
        as_rgb=(get_medmnist_nch(dataset_name) == 3),
        root=root,
        size=_size_arg(size),
    )
    dataset.nclass = get_medmnist_nclass(dataset_name)
    dataset.targets = dataset.labels.squeeze().astype(int).tolist()
    return dataset
