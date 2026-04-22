import argparse

from data.medmnist_dataset import compute_medmnist_mean_std


def main():
    # 这个脚本只是把 compute_medmnist_mean_std(...) 包成命令行入口，
    # 方便以后对任意 MedMNIST 数据集重算统计量，而不是手改硬编码。
    parser = argparse.ArgumentParser(
        description="Compute exact MedMNIST per-channel mean/std after ToTensor() scaling."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="MedMNIST dataset names, e.g. pathmnist dermamnist bloodmnist",
    )
    parser.add_argument(
        "--data-dir",
        default="~/.medmnist",
        help="Dataset root directory. Default: ~/.medmnist",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to include. Default: train val",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for statistics computation. Default: 1024",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt to download missing datasets.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        # 输出值的口径与训练保持一致：
        # ToTensor() 后的 [0,1] 图像，默认统计 train + val。
        mean, std = compute_medmnist_mean_std(
            dataset,
            data_dir=args.data_dir,
            download=not args.no_download,
            splits=tuple(args.splits),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(dataset)
        print("  mean =", [round(v, 6) for v in mean])
        print("  std  =", [round(v, 6) for v in std])


if __name__ == "__main__":
    main()
