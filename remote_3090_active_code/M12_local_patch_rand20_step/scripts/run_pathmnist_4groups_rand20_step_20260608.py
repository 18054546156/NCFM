
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Thread

import yaml

DATASET = "pathmnist"
IPC = 10
NCLASS = 9
NCH = 3


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def compact_float_token(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        digits = text[2:]
        return "0" + digits if len(digits) == 1 else digits
    return text.replace(".", "")


def copy_assets(exp_root, data_source, pretrain_source):
    data_dst = exp_root / "data" / "medmnist"
    pretrain_dst = exp_root / "checkpoints" / "pretrain" / DATASET
    data_dst.mkdir(parents=True, exist_ok=True)
    pretrain_dst.mkdir(parents=True, exist_ok=True)

    data_target = data_dst / "pathmnist.npz"
    if not data_target.exists():
        shutil.copy2(data_source, data_target)

    copied = 0
    for src in Path(pretrain_source).glob("premodel*.pth.tar"):
        dst = pretrain_dst / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1
    if copied < 40:
        raise RuntimeError(f"Expected about 40 premodel init/trained files, found {copied} in {pretrain_source}")


def make_config(args, exp_root, group_name, save_root, job):
    rdzv_dir = exp_root / "rdzv"
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    store = rdzv_dir / f"{DATASET}_{group_name}_{int(time.time() * 1000000)}.store"
    init_method = "file:///" + str(store).replace("\\", "/") + "?rank=0&world_size=1"

    cfg = {
        "distibution_train": {
            "backend": "gloo",
            "init_method": init_method,
            "workers": args.workers,
        },
        "dataset": {
            "dataset": DATASET,
            "nclass": NCLASS,
            "size": 28,
            "data_dir": str(exp_root / "data"),
            "load_memory": True,
            "batch_real": args.batch_real,
            "nch": NCH,
        },
        "network": {
            "net_type": "convnet",
            "norm_type": "instance",
            "depth": 3,
            "width": 1.0,
        },
        "train": {
            "evaluation_epochs": args.eval_epochs,
            "epoch_print_freq": 10,
            "epoch_eval_interval": args.epoch_eval_interval,
            "pertrain_epochs": 60,
            "batch_size": args.batch_size,
            "lr": 0.01,
            "adamw_lr": 0.001,
            "eval_optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "seed": args.seed,
            "model_num": args.model_num,
        },
        "augmentation": {
            "mixup": "cut",
            "beta": 1.0,
            "mix_p": 0.5,
            "rrc": True,
            "dsa": True,
            "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
            "aug_type": "color_crop_cutout",
        },
        "optimization": {
            "optimizer": "adamw",
            "lr_scale_adam": 0.1,
            "lr_img": 0.01,
            "mom_img": 0.5,
            "lr_sampling_net": 1e-3,
        },
        "save_path": {
            "save_dir": str(save_root),
            "pretrain_dir": str(exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "ipc": IPC,
            "num_premodel": args.model_num,
            "niter": args.niter,
            "iter_calib": 0,
            "calib_weight": 1,
            "sampling_net": True,
            "num_freqs": int(job["global_T"]),
            "dis_metrics": "NCFM",
            "factor": 2,
            "alpha_for_loss": 0.5,
            "beta_for_loss": 0.5,
            "decode_type": "single",
            "teacher_model_epoch": 20,
        },
    }
    if job["method"] == "m12":
        cfg["condense"].update({
            "use_local_patch_feature_ncfd": True,
            "lambda_local_patch_ncfd": float(job["lam"]),
            "local_patch_grid": int(job["grid"]),
            "local_patch_num_freqs": int(job["localT"]),
            "local_patch_loss_scale": 1.0,
            "local_patch_feature_dim": 0,
            "local_patch_encoder_source": "random_trained_step",
            "local_patch_encoder_blocks": int(job["blocks"]),
            "local_patch_encoder_frozen": True,
            "use_local_patch_sampling_net": False,
            "local_patch_model_num": args.model_num,
            "local_patch_encoder_seed": args.seed,
        })
    return cfg


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_command(cmd, cwd, stdout_path, stderr_path, gpu):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LOCAL_RANK"] = "0"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_WORLD_SIZE"] = "1"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29500 + int(gpu)))
    env.setdefault("PYTHONUTF8", "1")
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True, env=env)
    return proc.returncode


def latest_distilled_data(root, start_time):
    candidates = []
    for path in root.glob(f"**/{DATASET}/ipc{IPC}/**/distilled_data/data_*.pt"):
        if path.name == "data_init.pt":
            continue
        if path.stat().st_mtime >= start_time:
            candidates.append(path)
    if not candidates:
        candidates = [p for p in root.glob(f"**/{DATASET}/ipc{IPC}/**/distilled_data/data_*.pt") if p.name != "data_init.pt"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_best_metrics(metrics_jsonl):
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    best = {}
    if metrics_jsonl.exists():
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("is_best"):
                best = record
    return best


def run_one(job, args, gpu):
    group = job["group"]
    method = job["method"]
    repo_dir = args.m00_repo if method == "baseline" else args.m12_repo
    exp_root = args.root / method
    save_root = exp_root / "results" / "condense" / group
    exp_root.mkdir(parents=True, exist_ok=True)
    copy_assets(exp_root, args.data_source, args.pretrain_source)

    config_path = exp_root / "configs" / DATASET / f"ipc{IPC}_{group}.yaml"
    run_dir = exp_root / "runs" / DATASET / f"ipc{IPC}" / group
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(run_dir / "method_plan.json", json.dumps(job | {"gpu": gpu}, indent=2))
    cfg = make_config(args, exp_root, group, save_root, job)
    write_config(config_path, cfg)

    start_time = time.time()
    condense_cmd = [
        sys.executable,
        "condense/condense_script.py",
        "--config_path", str(config_path),
        "--gpu", str(gpu),
        "-i", str(IPC),
        "--run_mode", "Condense",
        "--init", "mix",
    ]
    write_text(run_dir / "condense_command.txt", " ".join(map(str, condense_cmd)) + "\n")
    rc = run_command(condense_cmd, repo_dir, run_dir / "condense_stdout.log", run_dir / "condense_stderr.log", gpu)
    if rc != 0:
        raise RuntimeError(f"Condense failed for {group}; see {run_dir}")
    condensed_path = latest_distilled_data(save_root, start_time)
    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {group}")
    write_text(run_dir / "condensed_path.txt", str(condensed_path) + "\n")

    checkpoint_path = exp_root / "checkpoints" / "synthetic_train" / DATASET / f"ipc{IPC}_{group}_best.pth.tar"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_cmd = [
        sys.executable,
        "evaluation/evaluation_script.py",
        "--config_path", str(config_path),
        "--gpu", str(gpu),
        "-i", str(IPC),
        "--run_mode", "Evaluation",
        "--load_path", str(condensed_path),
        "--val_repeat", "1",
        "--eval_checkpoint_path", str(checkpoint_path),
        "--eval_metrics_path", str(eval_metrics_path),
    ]
    write_text(run_dir / "eval_command.txt", " ".join(map(str, eval_cmd)) + "\n")
    rc = run_command(eval_cmd, repo_dir, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log", gpu)
    if rc != 0:
        raise RuntimeError(f"Eval failed for {group}; see {run_dir}")

    best = read_best_metrics(eval_metrics_path)
    metrics = {
        "dataset": DATASET,
        "seed": args.seed,
        "ipc": IPC,
        "group": group,
        "method": method,
        "global_T": job["global_T"],
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
        **best,
    }
    if method == "m12":
        metrics.update({
            "local_patch_lambda": job["lam"],
            "local_patch_grid": job["grid"],
            "local_patch_num_freqs": job["localT"],
            "local_patch_encoder_source": "random_trained_step",
            "local_patch_model_num": args.model_num,
            "local_patch_encoder_blocks": job["blocks"],
        })
    write_text(run_dir / "metrics.json", json.dumps(metrics, indent=2))
    append_text(args.root / "RESULTS.jsonl", json.dumps(metrics) + "\n")
    save_summary(args.root)
    return metrics


def metric_acc(item):
    acc = item.get("acc_percent", item.get("acc"))
    if isinstance(acc, (int, float)) and acc <= 1:
        acc *= 100
    return acc


def save_summary(root):
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    path = root / "RESULTS.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                results.append(json.loads(line))
    fields = ["method", "group", "global_T", "local_patch_num_freqs", "local_patch_lambda", "local_patch_grid", "local_patch_encoder_source", "local_patch_encoder_blocks", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc"]
    with (report_dir / "pathmnist_4groups_seed0.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["acc_percent"] = metric_acc(r)
            writer.writerow(row)
    rows = ["| Method | Group | global T | localT | lambda | grid | encoder | blocks | ACC | AUC | Macro-F1 | BACC |", "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in results:
        rows.append(
            f"| {r.get('method')} | {r.get('group')} | {r.get('global_T')} | {r.get('local_patch_num_freqs','')} | {r.get('local_patch_lambda','')} | {r.get('local_patch_grid','')} | {r.get('local_patch_encoder_source','')} | {r.get('local_patch_encoder_blocks','')} | {metric_acc(r)} | {r.get('auc_macro_ovr')} | {r.get('macro_f1')} | {r.get('balanced_acc')} |"
        )
    write_text(report_dir / "pathmnist_4groups_seed0.md", "\n".join(rows) + "\n")


def gpu_worker(gpu, jobs, args):
    append_text(args.root / f"gpu{gpu}_queue.log", f"[{datetime.now().isoformat(timespec='seconds')}] start queue {len(jobs)} jobs\n")
    for job in jobs:
        group = job["group"]
        try:
            append_text(args.root / f"gpu{gpu}_queue.log", f"[{datetime.now().isoformat(timespec='seconds')}] START {group}\n")
            res = run_one(job, args, gpu)
            append_text(args.root / f"gpu{gpu}_queue.log", f"[{datetime.now().isoformat(timespec='seconds')}] DONE {group} acc={metric_acc(res)}\n")
        except Exception as exc:
            fail = {"group": group, "gpu": gpu, "error": str(exc), "traceback": traceback.format_exc()}
            append_text(args.root / "FAILURES.jsonl", json.dumps(fail) + "\n")
            append_text(args.root / f"gpu{gpu}_queue.log", f"[{datetime.now().isoformat(timespec='seconds')}] FAIL {group}: {exc}\n")
    append_text(args.root / f"gpu{gpu}_queue.log", f"[{datetime.now().isoformat(timespec='seconds')}] queue done\n")


def main():
    parser = argparse.ArgumentParser(description="Run PathMNIST baseline and M12 rand20-step local patch 4 groups on 2x3090.")
    parser.add_argument("--root", type=Path, default=Path("/data/zengqiang/experiments/NCFMproject_0603/experiments/pathmnist_m12_rand20_step_4groups_20260608"))
    parser.add_argument("--m00_repo", type=Path, default=Path("/data/zengqiang/experiments/NCFMproject_0603/active_code/M00_baseline_ncfm/code"))
    parser.add_argument("--m12_repo", type=Path, default=Path("/data/zengqiang/experiments/NCFMproject_0603/active_code/M12_local_patch_rand20_step/code"))
    parser.add_argument("--data_source", type=Path, default=Path("/data/zengqiang/experiments/project_20260419_143900_medmnist_stats/data/pathmnist.npz"))
    parser.add_argument("--pretrain_source", type=Path, default=Path("/data/zengqiang/experiments/project_20260419_143900_medmnist_stats/pretrained_models/pathmnist/pathmnist"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    jobs_gpu0 = [
        {"method": "baseline", "group": "Path_B_T1024", "global_T": 1024},
        {"method": "m12", "group": "Path_M12_lam03_g4_lT1024_rand20_b2", "global_T": 1024, "localT": 1024, "lam": 0.3, "grid": 4, "blocks": 2},
    ]
    jobs_gpu1 = [
        {"method": "m12", "group": "Path_M12_lam06_g4_lT256_rand20_b2", "global_T": 1024, "localT": 256, "lam": 0.6, "grid": 4, "blocks": 2},
        {"method": "m12", "group": "Path_M12_lam06_g4_lT1024_rand20_b2", "global_T": 1024, "localT": 1024, "lam": 0.6, "grid": 4, "blocks": 2},
    ]
    status = {"updated_at": datetime.now().isoformat(timespec="seconds"), "stage": "started", "gpu0": jobs_gpu0, "gpu1": jobs_gpu1}
    write_text(args.root / "RUN_STATUS_PATHMNIST_4GROUPS.json", json.dumps(status, indent=2))
    t0 = Thread(target=gpu_worker, args=(0, jobs_gpu0, args), daemon=False)
    t1 = Thread(target=gpu_worker, args=(1, jobs_gpu1, args), daemon=False)
    t0.start(); t1.start(); t0.join(); t1.join()
    save_summary(args.root)
    final = {"updated_at": datetime.now().isoformat(timespec="seconds"), "stage": "completed" if not (args.root / "FAILURES.jsonl").exists() else "partial"}
    write_text(args.root / "RUN_STATUS_PATHMNIST_4GROUPS.json", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
