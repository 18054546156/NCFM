#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _steps_and_values(records: List[Dict], key: str):
    steps = [r["step"] for r in records if key in r and r[key] is not None]
    values = [r[key] for r in records if key in r and r[key] is not None]
    return steps, values


def plot_loss_curves(data: Dict, out_dir: str):
    records = data.get("loss", [])
    if not records:
        return

    steps, total_loss = _steps_and_values(records, "total_loss")
    _, match_loss = _steps_and_values(records, "match_loss_total")
    _, calib_loss = _steps_and_values(records, "calib_loss_total")

    plt.figure(figsize=(8, 5))
    if total_loss:
        plt.plot(steps, total_loss, label="total_loss", linewidth=2)
    if match_loss:
        plt.plot(steps, match_loss, label="match_loss_total", linewidth=1.8)
    if calib_loss and any(v != 0 for v in calib_loss):
        plt.plot(steps, calib_loss, label="calib_loss_total", linewidth=1.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss Dynamics")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_a_loss_curve.png"), dpi=220)
    plt.close()


def plot_per_class_loss(data: Dict, out_dir: str):
    records = data.get("per_class_loss", [])
    nclass = int(data.get("nclass", 0))
    if not records or nclass <= 0:
        return

    plt.figure(figsize=(9, 5.5))
    for c in range(nclass):
        xs = []
        ys = []
        for r in records:
            values = r.get("values", [])
            if c < len(values) and values[c] is not None:
                xs.append(r["step"])
                ys.append(values[c])
        if ys:
            plt.plot(xs, ys, label=f"class {c}", linewidth=1.5)

    plt.xlabel("step")
    plt.ylabel("match loss")
    plt.title("Per-Class Match Loss Dynamics")
    if nclass <= 10:
        plt.legend(ncol=min(4, nclass))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_b_per_class_loss.png"), dpi=220)
    plt.close()


def plot_gradient_frequency(data: Dict, out_dir: str):
    records = data.get("gradient_frequency", [])
    if not records:
        return

    plt.figure(figsize=(8.5, 5.2))
    for key in ["top20", "top50", "top100", "top200", "top400"]:
        xs, ys = _steps_and_values(records, key)
        if ys:
            plt.plot(xs, ys, label=key, linewidth=2 if key in {"top20", "top200"} else 1.5)

    plt.xlabel("step")
    plt.ylabel("cumulative gradient energy")
    plt.title("Gradient Frequency Concentration")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_c_gradient_frequency.png"), dpi=220)
    plt.close()


def plot_eval_metrics(data: Dict, out_dir: str):
    records = data.get("intermediate_eval", [])
    if not records:
        return

    plt.figure(figsize=(8.5, 5.2))
    for key, label in [
        ("acc", "ACC"),
        ("balanced_acc", "Balanced ACC"),
        ("macro_f1", "Macro F1"),
        ("macro_auc", "Macro AUC"),
    ]:
        xs, ys = _steps_and_values(records, key)
        if ys:
            plt.plot(xs, ys, label=label, linewidth=2 if key == "balanced_acc" else 1.7)

    plt.xlabel("step")
    plt.ylabel("score")
    plt.title("Intermediate Evaluation Metrics")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_d_eval_metrics.png"), dpi=220)
    plt.close()


def plot_loss_vs_balanced_acc(data: Dict, out_dir: str):
    loss_records = {r["step"]: r for r in data.get("loss", [])}
    eval_records = data.get("intermediate_eval", [])
    xs = []
    ys = []
    for r in eval_records:
        step = r["step"]
        if step in loss_records and r.get("balanced_acc") is not None:
            xs.append(loss_records[step]["total_loss"])
            ys.append(r["balanced_acc"])

    if not xs:
        return

    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(xs, ys, s=50, alpha=0.8)
    for loss, bacc, rec in zip(xs, ys, eval_records):
        plt.annotate(str(rec["step"]), (loss, bacc), fontsize=8, alpha=0.8)
    plt.xlabel("total loss")
    plt.ylabel("balanced accuracy")
    plt.title("Loss vs Balanced Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_e_loss_vs_bacc.png"), dpi=220)
    plt.close()


def plot_feature_metrics(data: Dict, out_dir: str):
    records = data.get("feature_metrics", [])
    if not records:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    xs, fisher = _steps_and_values(records, "fisher_retention")
    if fisher:
        axes[0].plot(xs, fisher, linewidth=2.0)
    axes[0].set_title("Fisher Retention")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("ratio")
    axes[0].grid(alpha=0.25)

    xs1, drift = _steps_and_values(records, "centroid_drift_mean")
    xs2, mmd = _steps_and_values(records, "mmd")
    if drift:
        axes[1].plot(xs1, drift, label="centroid_drift_mean", linewidth=2.0)
    if mmd:
        axes[1].plot(xs2, mmd, label="mmd", linewidth=2.0)
    axes[1].set_title("Feature-Space Gap")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_f_feature_metrics.png"), dpi=220)
    plt.close()


def write_summary(data: Dict, out_dir: str):
    lines = []
    lines.append(f"dataset: {data.get('dataset')}")
    lines.append(f"ipc: {data.get('ipc')}")
    lines.append(f"nclass: {data.get('nclass')}")

    loss_records = data.get("loss", [])
    if loss_records:
        first = loss_records[0]
        last = loss_records[-1]
        lines.append(
            f"loss: step {first['step']} total={first.get('total_loss')} -> "
            f"step {last['step']} total={last.get('total_loss')}"
        )

    grad_records = data.get("gradient_frequency", [])
    if grad_records:
        first = grad_records[0]
        last = grad_records[-1]
        lines.append(
            f"grad_top20: step {first['step']}={first.get('top20'):.4f} -> "
            f"step {last['step']}={last.get('top20'):.4f}"
        )
        lines.append(
            f"grad_top200: step {first['step']}={first.get('top200'):.4f} -> "
            f"step {last['step']}={last.get('top200'):.4f}"
        )

    eval_records = data.get("intermediate_eval", [])
    if eval_records:
        best_bacc = max(
            (r for r in eval_records if r.get("balanced_acc") is not None),
            key=lambda r: r["balanced_acc"],
            default=None,
        )
        if best_bacc is not None:
            lines.append(
                f"best balanced_acc: step {best_bacc['step']} value={best_bacc['balanced_acc']:.4f}"
            )
        best_f1 = max(
            (r for r in eval_records if r.get("macro_f1") is not None),
            key=lambda r: r["macro_f1"],
            default=None,
        )
        if best_f1 is not None:
            lines.append(f"best macro_f1: step {best_f1['step']} value={best_f1['macro_f1']:.4f}")

    with open(os.path.join(out_dir, "training_dynamics_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot NCFM training dynamics JSON")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    data = _load_json(args.json_path)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.json_path), "training_dynamics_plots"
    )
    _ensure_dir(output_dir)

    plot_loss_curves(data, output_dir)
    plot_per_class_loss(data, output_dir)
    plot_gradient_frequency(data, output_dir)
    plot_eval_metrics(data, output_dir)
    plot_loss_vs_balanced_acc(data, output_dir)
    plot_feature_metrics(data, output_dir)
    write_summary(data, output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
