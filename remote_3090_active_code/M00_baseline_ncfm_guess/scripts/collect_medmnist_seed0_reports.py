import argparse
import csv
import json
import re
from pathlib import Path


def group_family(group):
    if group.startswith("B_"):
        return "baseline"
    if group.startswith("LP_"):
        return "local_patch"
    if group.startswith("DAM_"):
        return "dam_attention"
    if group.startswith("SSIM_"):
        return "ssim"
    return "unknown"


def parse_group(group):
    out = {"group": group, "family": group_family(group)}
    if group.startswith("B_T"):
        out["num_freqs"] = group.replace("B_T", "")
    elif group.startswith("LP_"):
        m = re.search(r"lam(\d+)_g(\d+)_lf(\d+)", group)
        if m:
            lam = int(m.group(1)) / 100.0
            grid = int(m.group(2))
            out.update(
                {
                    "lambda_local_patch_ncfd": lam,
                    "local_patch_grid": grid,
                    "patch_count": grid * grid,
                    "local_patch_num_freqs": int(m.group(3)),
                }
            )
    elif group.startswith("DAM_"):
        m = re.search(r"w(\d+)_(L\d+)", group)
        if m:
            out.update({"dam_attention_weight": int(m.group(1)), "dam_attention_layers": m.group(2)})
    elif group.startswith("SSIM_"):
        m = re.search(r"w(\d+)_G(\d+)", group)
        if m:
            weights = {"005": 0.05, "01": 0.1, "05": 0.5, "10": 1.0}
            out.update({"ssim_weight": weights.get(m.group(1), m.group(1)), "ssim_grids": ",".join(m.group(2))})
    return out


def pick(metrics, *keys):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return ""


def load_metrics(exp_root, dataset, ipc):
    rows = []
    run_root = exp_root / "runs" / dataset / f"ipc{ipc}"
    if not run_root.exists():
        return rows
    for path in sorted(run_root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        group = metrics.get("group", path.parent.name)
        row = parse_group(group)
        row.update(
            {
                "dataset": metrics.get("dataset", dataset),
                "ipc": metrics.get("ipc", ipc),
                "acc_percent": pick(metrics, "acc_percent", "accuracy"),
                "auc_macro_ovr": pick(metrics, "auc_macro_ovr", "auc"),
                "macro_f1": pick(metrics, "macro_f1"),
                "balanced_acc": pick(metrics, "balanced_acc", "balanced_accuracy"),
                "epoch": pick(metrics, "epoch"),
                "metrics_path": str(path),
                "final_synthetic_pt": metrics.get("condensed_path", ""),
                "eval_checkpoint": metrics.get("checkpoint_path", ""),
            }
        )
        rows.append(row)
    return rows


def metric_float(row, key="acc_percent"):
    try:
        return float(row.get(key, -1))
    except (TypeError, ValueError):
        return -1.0


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Collect MedMNIST seed0 sweep reports.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ipc", type=int, default=10)
    args = parser.parse_args()

    rows = load_metrics(args.exp_root, args.dataset, args.ipc)
    rows_sorted = sorted(rows, key=lambda row: metric_float(row), reverse=True)
    report_dir = args.exp_root / "merged_reports"
    fields = [
        "dataset",
        "ipc",
        "family",
        "group",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "num_freqs",
        "lambda_local_patch_ncfd",
        "local_patch_grid",
        "patch_count",
        "dam_attention_weight",
        "dam_attention_layers",
        "ssim_weight",
        "ssim_grids",
        "epoch",
        "metrics_path",
        "final_synthetic_pt",
        "eval_checkpoint",
    ]
    stem = f"{args.dataset}_seed0_all_methods"
    write_csv(report_dir / f"{stem}.csv", rows_sorted, fields)
    write_md(report_dir / f"{stem}.md", rows_sorted, fields[:17])

    best_by_family = []
    for family in ["baseline", "local_patch", "dam_attention", "ssim"]:
        fam_rows = [row for row in rows_sorted if row.get("family") == family]
        if fam_rows:
            best_by_family.append(fam_rows[0])
            write_csv(report_dir / f"{args.dataset}_seed0_{family}.csv", fam_rows, fields)
            write_md(report_dir / f"{args.dataset}_seed0_{family}.md", fam_rows, fields[:17])
    write_csv(report_dir / f"{args.dataset}_seed0_best_configs.csv", best_by_family, fields)
    write_md(report_dir / f"{args.dataset}_seed0_best_configs.md", best_by_family, fields[:17])
    print(f"Collected {len(rows)} rows into {report_dir}")


if __name__ == "__main__":
    main()
