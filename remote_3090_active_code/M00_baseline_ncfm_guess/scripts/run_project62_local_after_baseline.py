import argparse
import json
import math
import os
import socket
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from run_lab3090_multiseed_queue import ensure_shared_assets, parse_csv, write_json
from run_medmnist_seed0_sweep import IPC, load_pipeline


BASELINE_GROUPS = ["B_T128", "B_T256", "B_T512", "B_T1024"]
DATASETS = {"bloodmnist", "pneumoniamnist", "pathmnist"}


def baseline_metrics_path(baseline_root, dataset, seed, group):
    return baseline_root / "experiments" / f"{dataset}_seed{seed}" / "runs" / dataset / f"ipc{IPC}" / group / "metrics.json"


def local_metrics_path(exp_root, dataset, group):
    return exp_root / "runs" / dataset / f"ipc{IPC}" / group / "metrics.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_acc(item):
    for key in ("acc_percent", "accuracy", "acc"):
        value = item.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"No accuracy field in {item}")


def baseline_status(baseline_root, datasets, seeds):
    rows = []
    missing = []
    for dataset in datasets:
        for seed in seeds:
            for group in BASELINE_GROUPS:
                path = baseline_metrics_path(baseline_root, dataset, seed, group)
                if not path.exists():
                    missing.append({"dataset": dataset, "seed": seed, "group": group, "metrics": str(path)})
                    continue
                item = read_json(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "group": group,
                        "T": int(group.split("_T", 1)[1]),
                        "acc_percent": metric_acc(item),
                        "auc_macro_ovr": item.get("auc_macro_ovr"),
                        "macro_f1": item.get("macro_f1"),
                        "balanced_acc": item.get("balanced_acc"),
                        "metrics_path": str(path),
                    }
                )
    return rows, missing


def choose_best_t(rows, datasets, seeds):
    by_dataset_t = defaultdict(list)
    for row in rows:
        by_dataset_t[(row["dataset"], row["T"])].append(row)

    chosen = {}
    summary = []
    for dataset in datasets:
        candidates = []
        for group in BASELINE_GROUPS:
            t_value = int(group.split("_T", 1)[1])
            items = by_dataset_t[(dataset, t_value)]
            if len(items) != len(seeds):
                continue
            accs = [item["acc_percent"] for item in items]
            mean_acc = sum(accs) / len(accs)
            std_acc = 0.0
            if len(accs) > 1:
                std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / (len(accs) - 1))
            baccs = [item["balanced_acc"] for item in items if item.get("balanced_acc") is not None]
            mean_bacc = sum(baccs) / len(baccs) if baccs else None
            candidates.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "T": t_value,
                    "n": len(items),
                    "acc_percent_mean": mean_acc,
                    "acc_percent_std": std_acc,
                    "balanced_acc_mean": mean_bacc,
                }
            )
        if not candidates:
            raise RuntimeError(f"No complete baseline candidates for {dataset}")
        candidates.sort(
            key=lambda item: (
                item["acc_percent_mean"],
                item["balanced_acc_mean"] if item["balanced_acc_mean"] is not None else -1,
                item["T"],
            ),
            reverse=True,
        )
        chosen[dataset] = candidates[0]
        summary.extend(candidates)
    return chosen, summary


def make_local_groups(chosen, local_t):
    groups = {}
    family = {}
    tags = {}
    for dataset, best in chosen.items():
        prefix = "Blood" if dataset == "bloodmnist" else "Pneu" if dataset == "pneumoniamnist" else dataset
        global_t = int(best["T"])
        for grid in (2, 4, 7):
            for lam_tag, lam in (("030", 0.3), ("050", 0.5), ("080", 0.8)):
                name = f"LP_{prefix}_T{global_t}_lam{lam_tag}_g{grid}_lf{local_t}"
                groups[name] = {
                    "iter_calib": 0,
                    "calib_weight": 1,
                    "sampling_net": True,
                    "num_freqs": global_t,
                    "use_local_patch_feature_ncfd": True,
                    "local_patch_grid": grid,
                    "lambda_local_patch_ncfd": lam,
                    "local_patch_feature_dim": 128,
                    "local_patch_encoder_blocks": 2,
                    "local_patch_num_freqs": local_t,
                    "local_patch_encoder_source": "premodel0_trained",
                    "local_patch_encoder_frozen": True,
                    "use_local_patch_sampling_net": False,
                    "dam_enabled": False,
                    "use_ssim_regularization": False,
                }
                family[name] = "local_patch"
                tags.setdefault(dataset, []).append(name)
    return groups, family, tags


def claim_task(lock_dir, task_id, worker_name):
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{task_id}.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task_id,
                "worker": worker_name,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )
    return lock_path


def write_csv(path, rows, fields):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, fields, title):
    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_group_plan(exp_root, dataset, seed, group_names, groups, family):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"group": group, "family": family[group], "config": json.dumps(groups[group], sort_keys=True)} for group in group_names]
    write_csv(report_dir / f"{dataset}_seed{seed}_local_after_bestT_plan.csv", rows, ["group", "family", "config"])
    md_rows = [{"Group": row["group"], "Family": row["family"], "Key Config": row["config"]} for row in rows]
    write_markdown(report_dir / f"{dataset}_seed{seed}_local_after_bestT_plan.md", md_rows, ["Group", "Family", "Key Config"], f"{dataset} seed{seed} Local After Best T Plan")


def wait_for_baseline(args, datasets, seeds):
    reports = args.base_exp_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    chosen_path = reports / "chosen_bestT.json"
    if chosen_path.exists():
        chosen = read_json(chosen_path)
        return chosen

    while True:
        rows, missing = baseline_status(args.baseline_exp_root, datasets, seeds)
        write_csv(reports / "baseline_status_rows.csv", rows, ["dataset", "seed", "group", "T", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "metrics_path"])
        write_csv(reports / "baseline_status_missing.csv", missing, ["dataset", "seed", "group", "metrics"])
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] baseline rows={len(rows)} "
            f"missing={len(missing)} required={len(datasets) * len(seeds) * len(BASELINE_GROUPS)}",
            flush=True,
        )
        if not missing:
            chosen, summary = choose_best_t(rows, datasets, seeds)
            out = {
                "chosen": chosen,
                "baseline_root": str(args.baseline_exp_root),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "selection_metric": "mean acc_percent across seeds; tie-break balanced_acc then larger T",
            }
            chosen_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            fields = ["dataset", "group", "T", "n", "acc_percent_mean", "acc_percent_std", "balanced_acc_mean"]
            write_csv(reports / "baseline_bestT_summary.csv", summary, fields)
            write_markdown(reports / "baseline_bestT_summary.md", summary, fields, "Baseline Best T Summary")
            return out
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Wait for project_6_2 baseline 3-seed runs, then launch local-patch sweep at each dataset's best baseline T.")
    parser.add_argument("--baseline_exp_root", type=Path, required=True)
    parser.add_argument("--base_exp_root", type=Path, required=True)
    parser.add_argument("--datasets", default="bloodmnist,pneumoniamnist")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--ipc", type=int, default=IPC)
    parser.add_argument("--cam_samples", type=int, default=100)
    parser.add_argument("--local_patch_num_freqs", type=int, default=512)
    parser.add_argument("--poll_seconds", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.ipc != IPC:
        raise ValueError("This launcher is fixed to IPC=10.")

    datasets = parse_csv(args.datasets, valid=DATASETS)
    seeds = parse_csv(args.seeds, cast=int)

    chosen_record = wait_for_baseline(args, datasets, seeds)
    chosen = chosen_record["chosen"]
    groups, family, groups_by_dataset = make_local_groups(chosen, args.local_patch_num_freqs)

    pipeline = load_pipeline()
    first_group = next(iter(groups.values()))
    pipeline.GROUPS = {"B_NCFM_T512": first_group, **groups}
    repo_dir = Path(__file__).resolve().parents[1]

    lock_dir = args.base_exp_root / "queue_locks"
    completed = []
    failed = []
    worker_name = f"worker{args.worker_id}_gpu{args.gpu}"

    for dataset in datasets:
        for seed in seeds:
            exp_root = args.base_exp_root / "experiments" / f"{dataset}_seed{seed}"
            exp_root.mkdir(parents=True, exist_ok=True)
            ensure_shared_assets(args.base_exp_root, exp_root, dataset, args.model_num)
            pipeline.ensure_headers(exp_root)
            save_group_plan(exp_root, dataset, seed, groups_by_dataset[dataset], groups, family)

            for group in groups_by_dataset[dataset]:
                if local_metrics_path(exp_root, dataset, group).exists() and not args.force:
                    continue
                task_id = f"{dataset}_seed{seed}_{group}"
                if claim_task(lock_dir, task_id, worker_name) is None:
                    continue
                if local_metrics_path(exp_root, dataset, group).exists() and not args.force:
                    continue

                args.exp_root = exp_root
                args.seed = seed
                status = {
                    "worker": worker_name,
                    "gpu": args.gpu,
                    "dataset": dataset,
                    "seed": seed,
                    "group": group,
                    "family": family[group],
                    "best_baseline_T": chosen[dataset]["T"],
                    "started": datetime.now().isoformat(timespec="seconds"),
                }
                write_json(args.base_exp_root / "queue_status" / f"{worker_name}.json", status)
                print(
                    f"CLAIM_LOCAL dataset={dataset} seed={seed} group={group} "
                    f"bestT={chosen[dataset]['T']} gpu={args.gpu}",
                    flush=True,
                )
                started = time.monotonic()
                try:
                    metrics = pipeline.run_condense_eval_cam(args, repo_dir, dataset, IPC, group)
                    metrics["family"] = family[group]
                    metrics["seed"] = seed
                    metrics["best_baseline_T"] = chosen[dataset]["T"]
                    metrics["queue_worker"] = worker_name
                    metrics["queue_wall_seconds"] = round(time.monotonic() - started, 3)
                    completed.append(metrics)
                    prefix = f"{dataset}_seed{seed}_local_after_bestT_{worker_name}"
                    pipeline.save_summary(exp_root, completed, prefix)
                    print(
                        f"DONE_LOCAL dataset={dataset} seed={seed} group={group} "
                        f"acc={metrics.get('acc_percent')} total_s={metrics.get('total_seconds')}",
                        flush=True,
                    )
                except Exception as exc:
                    error_record = {
                        "worker": worker_name,
                        "gpu": args.gpu,
                        "dataset": dataset,
                        "seed": seed,
                        "group": group,
                        "family": family[group],
                        "best_baseline_T": chosen[dataset]["T"],
                        "error": repr(exc),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                    failed.append(error_record)
                    write_json(args.base_exp_root / "queue_failed" / f"{task_id}_{worker_name}.json", error_record)
                    print(f"FAILED_LOCAL dataset={dataset} seed={seed} group={group} error={exc!r}", flush=True)

    write_json(
        args.base_exp_root / "queue_status" / f"{worker_name}.done.json",
        {
            "worker": worker_name,
            "gpu": args.gpu,
            "completed": [
                {"dataset": item.get("dataset"), "seed": item.get("seed"), "group": item.get("group")}
                for item in completed
            ],
            "failed": failed,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )


if __name__ == "__main__":
    main()
