from data.transform import (
    MEDMNIST_DATASETS,
    transform_imagenet,
    transform_cifar,
    transform_svhn,
    transform_mnist,
    transform_fashion,
    transform_tiny,
    transform_medmnist,
)


def get_train_transform(
    dataset,
    augment=True,
    from_tensor=True,
    size=0,
    rrc=False,
    rrc_size=None,
    device="cpu",
):
    if dataset in [
        "imagenette",
        "imagewoof",
        "imagemeow",
        "imagesquawk",
        "imagefruit",
        "imageyellow",
        "imagenet",
    ]:
        train_transform, _ = transform_imagenet(
            augment=augment,
            from_tensor=from_tensor,
            size=size,
            rrc=rrc,
            rrc_size=rrc_size,
            device=device,
        )
    elif dataset[:5] == "cifar":
        train_transform, _ = transform_cifar(augment=augment, from_tensor=from_tensor)
    elif dataset == "svhn":
        train_transform, _ = transform_svhn(augment=augment, from_tensor=from_tensor)
    elif dataset == "mnist":
        train_transform, _ = transform_mnist(augment=augment, from_tensor=from_tensor)
    elif dataset == "fashion":
        train_transform, _ = transform_fashion(augment=augment, from_tensor=from_tensor)
    elif dataset == "tinyimagenet":
        train_transform, _ = transform_tiny(augment=augment, from_tensor=from_tensor)
    elif dataset in MEDMNIST_DATASETS:
        # syn 链（严格 upstream 风格）:
        # Condenser 里的 synthetic images 本身已经是 torch.float32 tensor，范围 [0,1]。
        # 所以这里必须走 from_tensor=True:
        # 1. 不再重复 ToTensor()
        # 2. 直接在 tensor 上做增强/Normalize(mean, std)
        train_transform, _ = transform_medmnist(
            dataset,
            augment=augment,
            from_tensor=from_tensor,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_transform, _
