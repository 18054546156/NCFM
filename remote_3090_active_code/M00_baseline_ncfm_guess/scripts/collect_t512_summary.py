#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DATASETS = ["pneumoniamnist", "bloodmnist", "pathmnist"]
GROUPS = ["B_NCFM_T512", "D_LOCAL_LAM03_T512", "F_DAM_L012_W10_T512"]
FIELDS = [
    "dataset",
    "ipc",
    "group",
    "acc_percent",
    "auc_macro_ovr",
    "macro_f1",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "auprc",
    "checkpoint_path",
    "cam_summary",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--prefix", default="main_t512_summary")
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        for group in GROUPS:
            metrics_path = args.exp_root / "runs" / dataset / f"ipc{args.ipc}" / group / "metrics.json"
            if not metrics_path.exists():
                rows.append({"dataset": dataset, "ipc": args.ipc, "group": group})
                continue
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    report_dir = args.exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{args.prefix}.json"
    csv_path = report_dir / f"{args.prefix}.csv"
    md_path = report_dir / f"{args.prefix}.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Dataset | IPC | Group | ACC | AUC | Macro-F1 | Balanced ACC | Sens | Spec | AUPRC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset')} | {row.get('ipc')} | {row.get('group')} | "
            f"{row.get('acc_percent', '')} | {row.get('auc_macro_ovr', '')} | "
            f"{row.get('macro_f1', '')} | {row.get('balanced_acc', '')} | "
            f"{row.get('sensitivity', '')} | {row.get('specificity', '')} | "
            f"{row.get('auprc', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
