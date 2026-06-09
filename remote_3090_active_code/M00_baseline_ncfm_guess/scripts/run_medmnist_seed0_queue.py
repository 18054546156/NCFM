import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path

from run_medmnist_seed0_sweep import IPC, assert_assets, build_groups, load_pipeline, save_group_plan


def claim_group(lock_dir, group, worker_name):
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{group}.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(
            {
                "group": group,
                "worker": worker_name,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )
    return lock_path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def group_done(exp_root, dataset, group):
    return (exp_root / "runs" / dataset / f"ipc{IPC}" / group / "metrics.json").exists()


def main():
    parser = argparse.ArgumentParser(description="Lock-based MedMNIST seed0 28-group queue worker.")
    parser.add_argument("--dataset", required=True, choices=["pneumoniamnist", "bloodmnist", "pathmnist"])
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker_id", required=True)
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
    parser.add_argument("--summary_prefix", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.ipc != IPC:
        raise ValueError("This queue is fixed to IPC=10.")
    if args.summary_prefix is None:
        args.summary_prefix = f"{args.dataset}_seed0_queue"

    args.exp_root.mkdir(parents=True, exist_ok=True)
    assert_assets(args.exp_root, args.dataset, args.model_num)

    pipeline = load_pipeline()
    groups, family = build_groups()
    pipeline.GROUPS = {"B_NCFM_T512": groups["B_T512"], **groups}
    pipeline.ensure_headers(args.exp_root)
    save_group_plan(args.exp_root, args.dataset, groups, family)

    repo_dir = Path(__file__).resolve().parents[1]
    lock_dir = args.exp_root / "queue_locks"
    worker_name = f"worker{args.worker_id}_gpu{args.gpu}"
    completed = []
    failed = []

    for group in groups:
        if group_done(args.exp_root, args.dataset, group) and not args.force:
            continue
        lock_path = claim_group(lock_dir, group, worker_name)
        if lock_path is None:
            continue
        if group_done(args.exp_root, args.dataset, group) and not args.force:
            continue

        status = {
            "worker": worker_name,
            "gpu": args.gpu,
            "dataset": args.dataset,
            "current_group": group,
            "family": family[group],
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(args.exp_root / "queue_status" / f"{worker_name}.json", status)
        print(f"CLAIM dataset={args.dataset} group={group} family={family[group]} gpu={args.gpu}", flush=True)
        try:
            metrics = pipeline.run_condense_eval_cam(args, repo_dir, args.dataset, IPC, group)
            metrics["family"] = family[group]
            metrics["seed"] = 0
            completed.append(metrics)
            pipeline.save_summary(args.exp_root, completed, f"{args.summary_prefix}_{worker_name}")
            print(f"DONE dataset={args.dataset} group={group} acc={metrics.get('acc_percent')}", flush=True)
        except Exception as exc:
            error_record = {
                "dataset": args.dataset,
                "group": group,
                "family": family[group],
                "worker": worker_name,
                "gpu": args.gpu,
                "error": repr(exc),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            write_json(args.exp_root / "queue_failed" / f"{group}_{worker_name}.json", error_record)
            failed.append(error_record)
            print(f"FAILED dataset={args.dataset} group={group} error={exc!r}", flush=True)

    write_json(
        args.exp_root / "queue_status" / f"{worker_name}.done.json",
        {
            "worker": worker_name,
            "gpu": args.gpu,
            "dataset": args.dataset,
            "completed": [item.get("group") for item in completed],
            "failed": failed,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )


if __name__ == "__main__":
    main()
