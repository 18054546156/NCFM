import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)

from data.dataset_statistics import MEANS, STDS
from data.medmnist import (
    build_medmnist_dataset,
    get_medmnist_nch,
    get_medmnist_nclass,
    get_medmnist_root,
    is_supported_medmnist,
    register_medmnist_stats,
)
from data.transform import transform_medmnist
import models.convnet as CN


def parse_layers(value):
    if value is None or value == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_plain_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = OrderedDict()
    for key, value in state.items():
        clean[key.replace("module.", "")] = value
    return clean


def tensor_to_uint8_image(tensor, mean, std):
    x = tensor.detach().cpu().float().clone()
    mean = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    x = (x * std + mean).clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def normalize_map(values, eps=1e-8):
    values = values.astype(np.float32)
    values = values - values.min()
    denom = values.max()
    if denom < eps:
        return np.zeros_like(values, dtype=np.float32)
    return values / denom


def make_heatmap(attention):
    attention = np.clip(attention, 0, 1).astype(np.float32)
    heatmap = np.zeros((*attention.shape, 3), dtype=np.float32)
    heatmap[..., 0] = attention
    heatmap[..., 1] = 0.35 * attention
    heatmap[..., 2] = 0.05 * attention
    return (heatmap * 255.0).round().astype(np.uint8)


def overlay_heatmap(image_uint8, heatmap_uint8, alpha=0.45):
    out = (1 - alpha) * image_uint8.astype(np.float32) + alpha * heatmap_uint8.astype(np.float32)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def resize_panel(image_uint8, panel_size, resample):
    if panel_size <= 0:
        return image_uint8
    return np.asarray(
        Image.fromarray(image_uint8).resize((panel_size, panel_size), resample=resample)
    )


def save_pair(
    path,
    left_uint8,
    right_uint8,
    panel_size=224,
    left_label="original",
    right_label="DAM energy",
):
    left_uint8 = resize_panel(left_uint8, panel_size, Image.Resampling.NEAREST)
    right_uint8 = resize_panel(right_uint8, panel_size, Image.Resampling.BILINEAR)
    pad = 8
    label_h = 18
    h, w = left_uint8.shape[:2]
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x0 = pad
    x1 = w + pad * 2
    y = pad + label_h
    canvas.paste(Image.fromarray(left_uint8), (x0, y))
    canvas.paste(Image.fromarray(right_uint8), (x1, y))
    draw.text((x0, pad), left_label, fill=(0, 0, 0))
    draw.text((x1, pad), right_label, fill=(0, 0, 0))
    canvas.save(path)


def entropy(attention):
    values = attention.astype(np.float64).reshape(-1)
    values = values / max(values.sum(), 1e-12)
    return float(-(values * np.log(values + 1e-12)).sum() / np.log(values.size))


def topk_mass(attention, top_fraction=0.1):
    values = np.sort(attention.reshape(-1))[::-1]
    k = max(1, int(round(values.size * top_fraction)))
    return float(values[:k].sum() / max(values.sum(), 1e-12))


def extract_spatial_features(model, x):
    if hasattr(model, "get_feature_from_layer"):
        _, features = model.get_feature_from_layer(x, return_features=True)
    elif hasattr(model, "get_feature"):
        features = model.get_feature(x, 0, getattr(model, "depth", 1) - 1)
    else:
        raise TypeError("Model must expose get_feature_from_layer(...) or get_feature(...).")
    return [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]


@torch.no_grad()
def dam_energy_map(model, image_batch, layers=None, p=2.0, norm="l2", eps=1e-6):
    features = extract_spatial_features(model, image_batch)
    if not features:
        raise ValueError("No spatial feature maps found.")

    if layers is None:
        layers = list(range(len(features)))
    selected = [idx for idx in layers if 0 <= idx < len(features)]
    if not selected:
        raise ValueError(f"No valid layers selected from {len(features)} spatial maps.")

    maps = []
    for idx in selected:
        attention = torch.sum(torch.abs(features[idx]).pow(float(p)), dim=1, keepdim=True)
        if norm == "l2":
            flat = F.normalize(attention.flatten(start_dim=1), p=2, dim=1, eps=eps)
            attention = flat.view_as(attention)
        elif norm == "l1":
            flat = attention.flatten(start_dim=1)
            flat = flat / flat.abs().sum(dim=1, keepdim=True).clamp_min(eps)
            attention = flat.view_as(attention)
        elif norm in {"none", None}:
            pass
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        attention = F.interpolate(
            attention,
            size=image_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        maps.append(attention)

    attention = torch.stack(maps, dim=0).mean(dim=0)
    attention = attention[0, 0].detach().cpu().numpy()
    return normalize_map(attention)


def load_dataset(args):
    if is_supported_medmnist(args.dataset):
        root = get_medmnist_root(args.data_dir)
        register_medmnist_stats(args.dataset, root, size=args.size)
        _, transform = transform_medmnist(
            args.dataset,
            size=args.size,
            augment=False,
            from_tensor=False,
            normalize=True,
        )
        return build_medmnist_dataset(
            args.dataset,
            root,
            split=args.split,
            transform=transform,
            size=args.size,
            download=True,
        )

    raise ValueError("This DAM energy diagnostic currently supports MedMNIST datasets only.")


def define_dam_model(args):
    if args.net_type != "convnet":
        raise ValueError("This DAM energy diagnostic currently supports --net_type convnet only.")
    return CN.ConvNet(
        args.nclass,
        net_norm=args.norm_type,
        net_depth=args.depth,
        net_width=int(128 * args.width),
        channel=args.nch,
        im_size=(args.size, args.size),
    )


def build_args():
    parser = argparse.ArgumentParser(
        description="Visualize M05/DataDAM ConvNet spatial energy maps."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--attention_p", type=float, default=2.0)
    parser.add_argument("--attention_norm", default="l2", choices=["l2", "l1", "none"])
    parser.add_argument("--net_type", default="convnet")
    parser.add_argument("--norm_type", default="instance")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--nch", type=int, default=None)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--panel_size", type=int, default=224)
    return parser.parse_args()


def main():
    args = build_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not is_supported_medmnist(args.dataset):
        raise ValueError("This DAM energy diagnostic currently supports MedMNIST datasets only.")
    if args.nclass is None:
        args.nclass = get_medmnist_nclass(args.dataset)
    if args.nch is None:
        args.nch = get_medmnist_nch(args.dataset)

    dataset = load_dataset(args)
    model = define_dam_model(args).to(device)
    checkpoint_note = "random_init"
    if args.checkpoint:
        model.load_state_dict(load_plain_state_dict(args.checkpoint))
        checkpoint_note = args.checkpoint
    model.eval()

    layers = parse_layers(args.layers)
    image_dir = os.path.join(args.save_dir, "images")
    heatmap_dir = os.path.join(args.save_dir, "heatmaps")
    overlay_dir = os.path.join(args.save_dir, "overlays")
    pair_dir = os.path.join(args.save_dir, "pairs")
    for path in [image_dir, heatmap_dir, overlay_dir, pair_dir]:
        os.makedirs(path, exist_ok=True)

    rows = []
    count = min(args.num_samples, len(dataset))
    mean, std = MEANS[args.dataset], STDS[args.dataset]
    for index in range(count):
        image, target = dataset[index]
        image_batch = image.unsqueeze(0).to(device)
        attention = dam_energy_map(
            model,
            image_batch,
            layers=layers,
            p=args.attention_p,
            norm=args.attention_norm,
        )
        image_uint8 = tensor_to_uint8_image(image, mean, std)
        heatmap = make_heatmap(attention)
        overlay = overlay_heatmap(image_uint8, heatmap)

        original_path = os.path.join(image_dir, f"original_{index:05d}.png")
        heatmap_path = os.path.join(heatmap_dir, f"dam_energy_{index:05d}.png")
        overlay_path = os.path.join(overlay_dir, f"overlay_{index:05d}.png")
        pair_path = os.path.join(pair_dir, f"pair_{index:05d}.png")
        Image.fromarray(image_uint8).save(original_path)
        Image.fromarray(heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)
        save_pair(
            pair_path,
            image_uint8,
            overlay if args.overlay else heatmap,
            panel_size=args.panel_size,
            right_label="DAM energy overlay" if args.overlay else "DAM energy",
        )

        rows.append(
            {
                "index": index,
                "split": args.split,
                "target": int(target),
                "layers": args.layers,
                "attention_p": args.attention_p,
                "attention_norm": args.attention_norm,
                "attention_entropy": entropy(attention),
                "top10_activation_mass": topk_mass(attention),
                "checkpoint": checkpoint_note,
                "original_path": original_path,
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
                "pair_path": pair_path,
            }
        )

    summary_path = os.path.join(args.save_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} DAM energy pairs to: {pair_dir}")
    print(f"Summary: {summary_path}")
    if not args.checkpoint:
        print("WARNING: no checkpoint was provided; maps come from a random-init model.")


if __name__ == "__main__":
    main()
