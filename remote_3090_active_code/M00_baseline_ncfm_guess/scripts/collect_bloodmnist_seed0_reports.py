import argparse
import csv
import json
import re
from pathlib import Path


FAMILY_ORDER = ["baseline", "local_patch", "dam_attention", "ssim"]
REPORT_NAMES = {
    "baseline": "bloodmnist_baseline_t_sweep_seed0",
    "local_patch": "bloodmnist_local_patch_grid_lambda_seed0",
    "dam_attention": "bloodmnist_dam_attention_sweep_seed0",
    "ssim": "bloodmnist_ssim_sweep_seed0",
}


def group_family(group):
    if group.startswith("B_T"):
        return "baseline"
    if group.startswith("LP_"):
        return "local_patch"
    if group.startswith("DAM_"):
        return "dam_attention"
    if group.startswith("SSIM_"):
        return "ssim"
    return "unknown"


def parse_group(group):
    parsed = {"T": "", "grid": "", "patch_count": "", "lambda_local": "", "localT": "", "layers": "", "lambda_dam": "", "ssim_grids": "", "lambda_ssim": "", "kernel": ""}
    if group.startswith("B_T"):
        parsed["T"] = group.replace("B_T", "")
    elif group.startswith("LP_"):
        m = re.search(r"lam(\d+)_g(\d+)_lf(\d+)", group)
        if m:
            lam_raw, grid, local_t = m.groups()
            lam = int(lam_raw) / 100.0
            grid_i = int(grid)
            parsed.update({"lambda_local": lam, "grid": grid_i, "patch_count": grid_i * grid_i, "localT": local_t})
    elif group.startswith("DAM_"):
        m = re.search(r"w(\d+)_L(\d+)", group)
        if m:
            weight, layers = m.groups()
            parsed.update({"lambda_dam": weight, "layers": "[" + ",".join(layers) + "]"})
    elif group.startswith("SSIM_"):
        m = re.search(r"w(\d+)_G(\d+)", group)
        if m:
            weight_raw, grids = m.groups()
            if len(weight_raw) == 3 and weight_raw.startswith("0"):
                weight = int(weight_raw) / 100.0
            else:
                weight = int(weight_raw) / 10.0
            parsed.update({"lambda_ssim": weight, "ssim_grids": "[" + ",".join(grids) + "]", "kernel": 7})
    return parsed


def load_metrics(exp_root):
    rows = []
    run_root = exp_root / "runs" / "bloodmnist" / "ipc10"
    for path in sorted(run_root.glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        group = record.get("group", path.parent.name)
        row = {
            "family": group_family(group),
            "group": group,
            "acc_percent": record.get("acc_percent", record.get("accuracy", "")),
            "auc_macro_ovr": record.get("auc_macro_ovr", ""),
            "macro_f1": record.get("macro_f1", ""),
            "balanced_acc": record.get("balanced_acc", ""),
            "sensitivity": record.get("sensitivity", ""),
            "specificity": record.get("specificity", ""),
            "auprc": record.get("auprc", ""),
            "condensed_path": record.get("condensed_path", ""),
            "checkpoint_path": record.get("checkpoint_path", ""),
            "metrics_path": str(path),
        }
        row.update(parse_group(group))
        rows.append(row)
    return rows


def metric_float(row, key="acc_percent"):
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return float("-inf")
        return float(value)
    except Exception:
        return float("-inf")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, rows, fields, headers=None):
    headers = headers or fields
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_reports(report_dir, rows):
    specs = {
        "baseline": ["group", "T", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path"],
        "local_patch": ["group", "grid", "patch_count", "lambda_local", "localT", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path"],
        "dam_attention": ["group", "layers", "lambda_dam", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path"],
        "ssim": ["group", "ssim_grids", "lambda_ssim", "kernel", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path"],
    }
    for family, fields in specs.items():
        family_rows = [row for row in rows if row["family"] == family]
        family_rows.sort(key=lambda row: row["group"])
        stem = REPORT_NAMES[family]
        write_csv(report_dir / f"{stem}.csv", family_rows, fields)
        write_md(report_dir / f"{stem}.md", family_rows, fields)


def write_summary_reports(report_dir, rows):
    all_fields = ["family", "group", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path", "metrics_path"]
    rows_sorted = sorted(rows, key=lambda row: (FAMILY_ORDER.index(row["family"]) if row["family"] in FAMILY_ORDER else 99, row["group"]))
    write_csv(report_dir / "bloodmnist_all_methods_seed0.csv", rows_sorted, all_fields)
    write_md(report_dir / "bloodmnist_all_methods_seed0.md", rows_sorted, all_fields)

    best_rows = []
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        best = max(family_rows, key=metric_float)
        best_rows.append(best)
    write_csv(report_dir / "bloodmnist_best_configs_seed0.csv", best_rows, all_fields)
    write_md(report_dir / "bloodmnist_best_configs_seed0.md", best_rows, all_fields)


def main():
    parser = argparse.ArgumentParser(description="Collect BloodMNIST seed0 sweep metrics into paper-ready tables.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args()

    rows = load_metrics(args.exp_root)
    report_dir = args.report_dir or args.exp_root / "merged_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_family_reports(report_dir, rows)
    write_summary_reports(report_dir, rows)
    print(f"Collected {len(rows)} completed runs into {report_dir}")


if __name__ == "__main__":
    main()
