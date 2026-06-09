import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from run_medmnist_seed0_sweep import IPC, build_groups


METRIC_FIELDS = [
    "acc_percent",
    "auc_macro_ovr",
    "macro_f1",
    "balanced_acc",
    "condense_seconds",
    "eval_seconds",
    "cam_seconds",
    "total_seconds",
]


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def std(values):
    values = [float(x) for x in values if x is not None]
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else None
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def collect_rows(base_exp_root, datasets, seeds):
    groups, family = build_groups()
    rows = []
    missing = []
    for dataset in datasets:
        for seed in seeds:
            exp_root = base_exp_root / "experiments" / f"{dataset}_seed{seed}"
            for group in groups:
                metrics_path = exp_root / "runs" / dataset / f"ipc{IPC}" / group / "metrics.json"
                if not metrics_path.exists():
                    missing.append({"dataset": dataset, "seed": seed, "group": group, "metrics": str(metrics_path)})
                    continue
                item = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "family": family[group],
                    "group": group,
                    "metrics_path": str(metrics_path),
                }
                for field in METRIC_FIELDS:
                    row[field] = item.get(field)
                rows.append(row)
    return rows, missing


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, fields, title):
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["dataset"], row["family"], row["group"])].append(row)
    summary = []
    for (dataset, family, group), items in sorted(by_group.items()):
        out = {"dataset": dataset, "family": family, "group": group, "n": len(items)}
        for field in METRIC_FIELDS:
            vals = [item.get(field) for item in items]
            out[f"{field}_mean"] = mean(vals)
            out[f"{field}_std"] = std(vals)
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect lab 3090 multiseed MedMNIST reports.")
    parser.add_argument("--base_exp_root", type=Path, required=True)
    parser.add_argument("--datasets", default="bloodmnist,pneumoniamnist")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    seeds = parse_csv(args.seeds, cast=int)
    out_dir = args.out_dir or (args.base_exp_root / "merged_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = collect_rows(args.base_exp_root, datasets, seeds)
    run_fields = ["dataset", "seed", "family", "group"] + METRIC_FIELDS + ["metrics_path"]
    write_csv(out_dir / "lab3090_multiseed_runs.csv", rows, run_fields)
    write_markdown(out_dir / "lab3090_multiseed_runs.md", rows, run_fields[:-1], "Lab 3090 Multiseed Runs")

    summary = summarize(rows)
    summary_fields = ["dataset", "family", "group", "n"]
    for field in METRIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(out_dir / "lab3090_multiseed_summary.csv", summary, summary_fields)
    write_markdown(
        out_dir / "lab3090_multiseed_summary.md",
        summary,
        summary_fields,
        "Lab 3090 Multiseed Summary",
    )

    write_csv(out_dir / "lab3090_multiseed_missing.csv", missing, ["dataset", "seed", "group", "metrics"])
    print(f"runs={len(rows)} missing={len(missing)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
