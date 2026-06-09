import argparse
import csv
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

from run_medmnist_seed0_sweep import IPC, build_groups, load_pipeline


DATASETS = {"bloodmnist", "pneumoniamnist", "pathmnist"}


def parse_csv(value, valid=None, cast=str):
    items = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if valid is not None:
        unknown = sorted(set(items) - set(valid))
        if unknown:
            raise ValueError(f"Unknown values: {unknown}")
    return items


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def link_or_keep(target, link_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() != target.resolve():
            link_path.unlink()
        else:
            return
    try:
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        # Another queue worker may have created the same shared-asset link.
        return


def ensure_shared_assets(base_exp_root, exp_root, dataset, model_num):
    shared = base_exp_root / "shared_assets"
    data_target = shared / "data" / "medmnist" / f"{dataset}.npz"
    pretrain_target = shared / "checkpoints" / "pretrain" / dataset
    missing = []
    if not data_target.exists():
        missing.append(str(data_target))
    for idx in range(model_num):
        for suffix in ("init", "trained"):
            path = pretrain_target / f"premodel{idx}_{suffix}.pth.tar"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing shared assets:\n" + "\n".join(missing))

    link_or_keep(shared / "data", exp_root / "data")
    ckpt_dir = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    link_or_keep(shared / "checkpoints" / "pretrain", ckpt_dir / "pretrain")


def task_done(exp_root, dataset, group):
    return (exp_root / "runs" / dataset / f"ipc{IPC}" / group / "metrics.json").exists()


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


def save_group_plan(exp_root, dataset, seed, groups, family):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{dataset}_seed{seed}_group_plan.csv"
    md_path = report_dir / f"{dataset}_seed{seed}_group_plan.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "family", "config"])
        writer.writeheader()
        for group, config in groups.items():
            writer.writerow({"group": group, "family": family[group], "config": json.dumps(config, sort_keys=True)})
    rows = ["| Group | Family | Key Config |", "|---|---|---|"]
    for group, config in groups.items():
        rows.append(f"| {group} | {family[group]} | `{json.dumps(config, sort_keys=True)}` |")
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Lock-based 3-seed queue for the 28-group MedMNIST sweep on lab 3090.")
    parser.add_argument("--base_exp_root", type=Path, required=True)
    parser.add_argument("--datasets", default="bloodmnist,pneumoniamnist")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--ipc", type=int, default=IPC)
    parser.add_argument("--cam_samples", type=int, default=100)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.ipc != IPC:
        raise ValueError("This queue is fixed to IPC=10.")

    datasets = parse_csv(args.datasets, valid=DATASETS)
    seeds = parse_csv(args.seeds, cast=int)
    groups, family = build_groups()
    group_names = list(groups) if args.groups == "all" else parse_csv(args.groups, valid=set(groups))

    pipeline = load_pipeline()
    pipeline.GROUPS = {"B_NCFM_T512": groups["B_T512"], **groups}
    repo_dir = Path(__file__).resolve().parents[1]

    worker_name = f"worker{args.worker_id}_gpu{args.gpu}"
    lock_dir = args.base_exp_root / "queue_locks"
    completed = []
    failed = []

    for dataset in datasets:
        for seed in seeds:
            exp_root = args.base_exp_root / "experiments" / f"{dataset}_seed{seed}"
            exp_root.mkdir(parents=True, exist_ok=True)
            ensure_shared_assets(args.base_exp_root, exp_root, dataset, args.model_num)
            pipeline.ensure_headers(exp_root)
            save_group_plan(exp_root, dataset, seed, groups, family)

            for group in group_names:
                if task_done(exp_root, dataset, group) and not args.force:
                    continue
                task_id = f"{dataset}_seed{seed}_{group}"
                if claim_task(lock_dir, task_id, worker_name) is None:
                    continue
                if task_done(exp_root, dataset, group) and not args.force:
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
                    "started": datetime.now().isoformat(timespec="seconds"),
                }
                write_json(args.base_exp_root / "queue_status" / f"{worker_name}.json", status)
                print(f"CLAIM dataset={dataset} seed={seed} group={group} family={family[group]} gpu={args.gpu}", flush=True)

                started = time.monotonic()
                try:
                    metrics = pipeline.run_condense_eval_cam(args, repo_dir, dataset, IPC, group)
                    metrics["family"] = family[group]
                    metrics["seed"] = seed
                    metrics["queue_worker"] = worker_name
                    metrics["queue_wall_seconds"] = round(time.monotonic() - started, 3)
                    completed.append(metrics)
                    prefix = f"{dataset}_seed{seed}_queue_{worker_name}"
                    pipeline.save_summary(exp_root, completed, prefix)
                    print(
                        f"DONE dataset={dataset} seed={seed} group={group} "
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
                        "error": repr(exc),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                    failed.append(error_record)
                    write_json(args.base_exp_root / "queue_failed" / f"{task_id}_{worker_name}.json", error_record)
                    print(f"FAILED dataset={dataset} seed={seed} group={group} error={exc!r}", flush=True)

                if args.max_tasks and len(completed) >= args.max_tasks:
                    break
            if args.max_tasks and len(completed) >= args.max_tasks:
                break
        if args.max_tasks and len(completed) >= args.max_tasks:
            break

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
