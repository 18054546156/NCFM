import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.medmnist import get_medmnist_root, register_medmnist_stats
from evaluation.cam_utils import load_plain_state_dict
from utils.utils import define_model, load_resized_data


DATASETS = {
    "pneumoniamnist": {"nclass": 2, "nch": 1},
    "bloodmnist": {"nclass": 8, "nch": 3},
    "pathmnist": {"nclass": 9, "nch": 3},
}


def resolve_checkpoint(exp_root, dataset, group):
    if group == "real_train":
        return exp_root / "checkpoints" / "pretrain" / dataset / "premodel0_trained.pth.tar"
    return exp_root / "checkpoints" / "synthetic_train" / dataset / f"{group}_best.pth.tar"


def load_model(exp_root, dataset, group, device):
    info = DATASETS[dataset]
    model = define_model(
        dataset,
        "instance",
        "convnet",
        info["nch"],
        3,
        1.0,
        info["nclass"],
        logger=None,
        size=28,
    ).to(device)
    checkpoint = resolve_checkpoint(exp_root, dataset, group)
    model.load_state_dict(load_plain_state_dict(str(checkpoint)))
    model.eval()
    return model, checkpoint


def cam_mask(cam_path, mode, fraction, seed):
    cam = np.asarray(Image.open(cam_path).convert("RGB")).astype(np.float32) / 255.0
    values = cam[..., 0].reshape(-1)
    k = max(1, int(round(values.size * fraction)))
    if mode == "hot":
        idx = np.argsort(values)[-k:]
    elif mode == "cold":
        idx = np.argsort(values)[:k]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(values.size, size=k, replace=False)
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")
    mask = np.zeros(values.size, dtype=bool)
    mask[idx] = True
    return torch.from_numpy(mask.reshape(cam.shape[0], cam.shape[1]))


def apply_mask(image, mask):
    masked = image.clone()
    mask = mask.to(dtype=torch.bool, device=masked.device)
    if mask.shape[-2:] != masked.shape[-2:]:
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=masked.shape[-2:],
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()
    masked[:, mask] = 0.0
    return masked


def evaluate_drop(model, image, y_pred, masks, device):
    image = image.to(device)
    with torch.no_grad():
        probs = F.softmax(model(image.unsqueeze(0)), dim=1)
        original_prob = float(probs[0, y_pred].item())
        drops = {"original_prob": original_prob}
        for mode, mask in masks.items():
            masked = apply_mask(image, mask)
            masked_probs = F.softmax(model(masked.unsqueeze(0)), dim=1)
            masked_prob = float(masked_probs[0, y_pred].item())
            drops[f"{mode}_prob"] = masked_prob
            drops[f"{mode}_drop"] = original_prob - masked_prob
    return drops


def main():
    parser = argparse.ArgumentParser(description="Run CAM hot/random/cold occlusion sensitivity.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--datasets", default="pathmnist,bloodmnist,pneumoniamnist")
    parser.add_argument("--groups", default="real_train,ipc10_D_local_patch_feature_lam03,ipc10_E_local_patch_feature_lam06")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]

    rows = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        register_medmnist_stats(dataset, get_medmnist_root(args.exp_root / "data"), size=28)
        _, test_dataset = load_resized_data(
            dataset,
            str(args.exp_root / "data"),
            size=28,
            nclass=DATASETS[dataset]["nclass"],
            load_memory=False,
        )
        for group in groups:
            summary_path = args.exp_root / "results" / "cam" / dataset / group / "summary.csv"
            if not summary_path.exists():
                continue
            model, checkpoint = load_model(args.exp_root, dataset, group, device)
            summary = pd.read_csv(summary_path).head(args.max_samples)
            for _, item in summary.iterrows():
                index = int(item["index"])
                image, y_true = test_dataset[index]
                y_pred = int(item["y_pred"])
                cam_path = Path(str(item["cam_path"]))
                if not cam_path.exists():
                    cam_path = args.exp_root / cam_path
                masks = {
                    mode: cam_mask(cam_path, mode, args.fraction, seed=index)
                    for mode in ["hot", "random", "cold"]
                }
                drops = evaluate_drop(model, image, y_pred, masks, device)
                rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "index": index,
                        "y_true": int(y_true),
                        "y_pred": y_pred,
                        "correct": int(item["correct"]),
                        "checkpoint": str(checkpoint),
                        **drops,
                    }
                )

    report_dir = args.exp_root / "reports" / "cam"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_path = report_dir / "occlusion_sensitivity_all_rows.csv"
    grouped_path = report_dir / "occlusion_sensitivity_grouped.csv"
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(all_path, index=False)
        grouped = df.groupby(["dataset", "group"], as_index=False).agg(
            n=("index", "count"),
            original_prob=("original_prob", "mean"),
            hot_drop=("hot_drop", "mean"),
            random_drop=("random_drop", "mean"),
            cold_drop=("cold_drop", "mean"),
        )
        grouped.to_csv(grouped_path, index=False)
        print(grouped)
    else:
        with all_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "group", "index"])
    print(f"Saved: {all_path}")
    print(f"Saved: {grouped_path}")


if __name__ == "__main__":
    main()
