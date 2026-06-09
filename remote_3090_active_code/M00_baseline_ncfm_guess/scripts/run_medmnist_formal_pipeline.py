import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


DATASETS = {
    "pneumoniamnist": {"nclass": 2, "nch": 1},
    "bloodmnist": {"nclass": 8, "nch": 3},
    "pathmnist": {"nclass": 9, "nch": 3},
}

GROUPS = {
    "B_NCFM_T512": {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": False,
        "dam_enabled": False,
    },
    "D_LOCAL_LAM03_T512": {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": True,
        "local_patch_grid": 4,
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_feature_dim": 128,
        "local_patch_encoder_blocks": 2,
        "local_patch_num_freqs": 256,
        "local_patch_encoder_source": "premodel0_trained",
        "local_patch_encoder_frozen": True,
        "use_local_patch_sampling_net": False,
        "dam_enabled": False,
    },
    "F_DAM_L012_W10_T512": {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": False,
        "dam_enabled": True,
        "dam_objective": "ncfm_attention",
        "dam_feature_weight": 1.0,
        "dam_attention_weight": 10.0,
        "dam_attention_layers": [0, 1, 2],
        "dam_attention_p": 2,
        "dam_attention_norm": "l2",
        "dam_detach_real": True,
        "dam_log_components": True,
    },
}


def run_command(command, cwd, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    env.setdefault("TORCH_USE_LIBUV", "0")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 1000)))
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("LOCAL_WORLD_SIZE", "1")
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, text=True, env=env)
    return proc.returncode


def append(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def make_config(args, dataset, ipc, group_name=None, niter=None, metrics_path=None):
    info = DATASETS[dataset]
    group = GROUPS.get(group_name, GROUPS["B_NCFM_T512"])
    rdzv_dir = args.exp_root / "rdzv"
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    safe_group = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_name or "pretrain")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    rdzv_file = rdzv_dir / f"{dataset}_{safe_group}_{stamp}.store"
    init_method = f"file:///{rdzv_file.as_posix()}?rank=0&world_size=1"
    cfg = {
        "distibution_train": {
            "backend": "gloo",
            "init_method": init_method,
            "workers": args.workers,
        },
        "dataset": {
            "dataset": dataset,
            "nclass": info["nclass"],
            "size": 28,
            "data_dir": str(args.exp_root / "data"),
            "load_memory": True,
            "batch_real": args.batch_real,
            "nch": info["nch"],
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
            "pertrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "lr": 0.01,
            "adamw_lr": 0.001,
            "eval_optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "seed": int(getattr(args, "seed", 0)),
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
            "save_dir": str(args.exp_root / "results" / "condense"),
            "pretrain_dir": str(args.exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "exp_tag": group_name or "pretrain",
            "ipc": ipc,
            "num_premodel": args.model_num,
            "niter": int(niter),
            "iter_calib": group["iter_calib"],
            "calib_weight": group["calib_weight"],
            "sampling_net": group["sampling_net"],
            "num_freqs": group["num_freqs"],
            "dis_metrics": "NCFM",
            "factor": 2,
            "alpha_for_loss": 0.5,
            "beta_for_loss": 0.5,
            "decode_type": "single",
            "teacher_model_epoch": 20,
        },
    }
    local_keys = [
        "use_local_patch_feature_ncfd",
        "local_patch_grid",
        "lambda_local_patch_ncfd",
        "local_patch_feature_dim",
        "local_patch_encoder_blocks",
        "local_patch_num_freqs",
        "local_patch_encoder_source",
        "local_patch_encoder_frozen",
        "use_local_patch_sampling_net",
        "dam_enabled",
        "dam_objective",
        "dam_feature_weight",
        "dam_attention_weight",
        "dam_attention_layers",
        "dam_attention_p",
        "dam_attention_norm",
        "dam_detach_real",
        "dam_log_components",
        "use_ssim_regularization",
        "ssim_weight",
        "ssim_grids",
        "ssim_kernel_size",
        "ssim_data_range",
        "ssim_pairing",
        "ssim_log_components",
    ]
    for key in local_keys:
        if key in group:
            cfg["condense"][key] = group[key]
    if metrics_path:
        cfg["records"] = {"metrics_path": str(metrics_path)}
    return cfg


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def torchrun_command(script, config_path, gpu, ipc=None, extra=None):
    if os.name == "nt":
        command = [sys.executable, script, "--config_path", str(config_path), "--gpu", str(gpu)]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            script,
            "--config_path",
            str(config_path),
            "--gpu",
            str(gpu),
        ]
    if ipc is not None:
        command.extend(["-i", str(ipc)])
    if extra:
        command.extend(extra)
    return command


def newest_distilled_data(exp_root, dataset, ipc, start_time):
    root = exp_root / "results" / "condense"
    candidates = []
    if root.exists():
        pattern = f"**/{dataset}/ipc{ipc}/**/distilled_data/data_*.pt"
        for path in root.glob(pattern):
            if path.name == "data_init.pt":
                continue
            if path.stat().st_mtime >= start_time:
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_distilled_data(exp_root, dataset, ipc):
    root = exp_root / "results" / "condense"
    candidates = []
    if root.exists():
        pattern = f"**/{dataset}/ipc{ipc}/**/distilled_data/data_*.pt"
        for path in root.glob(pattern):
            if path.name == "data_init.pt":
                continue
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_best_accuracy(stdout_path):
    text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    matches = re.findall(r"Best\s+accuracy \(top-1 and 5\):\s*([0-9.]+)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"Mean Accuracy:\s*([0-9.]+)", text)
    if matches:
        return float(matches[-1])
    return None


def read_best_metrics(metrics_jsonl):
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    if not metrics_jsonl.exists():
        return {}
    best = {}
    for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("is_best"):
            best = record
    return best


def ensure_headers(exp_root):
    append(
        exp_root / "EXPERIMENT_LOG.md",
        "# Experiment Log\n\n"
        "| Time | Dataset | IPC | Group | Stage | ACC | AUC | Macro-F1 | Balanced ACC | Notes |\n"
        "|---|---|---:|---|---|---:|---:|---:|---:|---|\n",
    )
    append(
        exp_root / "RUNS.md",
        "# Runs\n\n| Time | Dataset | IPC | Group | Command File | Status |\n|---|---|---:|---|---|---|\n",
    )
    append(
        exp_root / "CAM_RESULTS.md",
        "# CAM Results\n\n| Dataset | Source | Checkpoint | Summary | Overlays | Notes |\n|---|---|---|---|---|---|\n",
    )
    append(
        exp_root / "PATCH_NOTES.md",
        "# Patch Notes\n\n"
        "- Added `utils/metrics.py` for ACC, macro OvR AUC, Macro-F1, Balanced ACC, and binary Sensitivity/Specificity/AUPRC.\n"
        "- Added metric JSONL output to pretrain and synthetic evaluator loops.\n"
        "- Fixed pretrain LR scheduler so very short runs do not start with milestone 0 decay.\n",
    )


def ensure_pretrain(args, repo_dir, dataset):
    metrics_path = args.exp_root / "reports" / "pretrain" / dataset / "metrics.jsonl"
    config_path = args.exp_root / "configs" / dataset / "pretrain_formal.yaml"
    config = make_config(args, dataset, ipc=1, niter=args.niter, metrics_path=metrics_path)
    write_config(config_path, config)

    expected = (
        args.exp_root
        / "checkpoints"
        / "pretrain"
        / dataset
        / f"premodel{args.model_num - 1}_trained.pth.tar"
    )
    run_dir = args.exp_root / "runs" / dataset / "pretrain_formal"
    if expected.exists() and not args.force:
        return expected

    command = torchrun_command("pretrain/pretrain_script.py", config_path, args.gpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    append(
        args.exp_root / "RUNS.md",
        f"| {datetime.now().isoformat(timespec='seconds')} | {dataset} | - | pretrain | `{run_dir / 'command.txt'}` | started |\n",
    )
    rc = run_command(command, repo_dir, run_dir / "stdout.log", run_dir / "stderr.log")
    if rc != 0:
        raise RuntimeError(f"Pretrain failed for {dataset}; see {run_dir}")
    append(
        args.exp_root / "RUNS.md",
        f"| {datetime.now().isoformat(timespec='seconds')} | {dataset} | - | pretrain | `{run_dir / 'command.txt'}` | done |\n",
    )
    return expected


def run_real_cam(args, repo_dir, dataset):
    checkpoint_path = args.exp_root / "checkpoints" / "pretrain" / dataset / "premodel0_trained.pth.tar"
    cam_dir = args.exp_root / "results" / "cam" / dataset / "real_train"
    run_dir = args.exp_root / "runs" / dataset / "real_train_cam"
    run_dir.mkdir(parents=True, exist_ok=True)
    cam_cmd = [
        sys.executable,
        "evaluation/run_cam.py",
        "--dataset",
        dataset,
        "--data_dir",
        str(args.exp_root / "data"),
        "--checkpoint",
        str(checkpoint_path),
        "--save_dir",
        str(cam_dir),
        "--num_samples",
        str(args.cam_samples),
        "--nclass",
        str(DATASETS[dataset]["nclass"]),
        "--nch",
        str(DATASETS[dataset]["nch"]),
        "--size",
        "28",
        "--gpu",
        str(args.gpu),
    ]
    (run_dir / "cam_command.txt").write_text(" ".join(cam_cmd) + "\n", encoding="utf-8")
    rc = run_command(cam_cmd, repo_dir, run_dir / "cam_stdout.log", run_dir / "cam_stderr.log")
    if rc != 0:
        raise RuntimeError(f"Real CAM failed for {dataset}; see {run_dir}")
    append(
        args.exp_root / "CAM_RESULTS.md",
        f"| {dataset} | real_train | `{checkpoint_path}` | `{cam_dir / 'summary.csv'}` | `{cam_dir / 'overlays'}` | formal pretrain |\n",
    )


def run_condense_eval_cam(args, repo_dir, dataset, ipc, group_name):
    config_dir = args.exp_root / "configs" / dataset
    config_path = config_dir / f"ipc{ipc}_{group_name}.yaml"
    config = make_config(args, dataset, ipc=ipc, group_name=group_name, niter=args.niter)
    write_config(config_path, config)

    run_dir = args.exp_root / "runs" / dataset / f"ipc{ipc}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)

    condensed_record = run_dir / "condensed_path.txt"
    condensed_path = None
    if condensed_record.exists() and not args.force:
        candidate = Path(condensed_record.read_text(encoding="utf-8").strip())
        if candidate.exists():
            condensed_path = candidate

    if condensed_path is None:
        start_time = datetime.now().timestamp()
        condense_started = time.monotonic()
        condense_cmd = torchrun_command(
            "condense/condense_script.py",
            config_path,
            args.gpu,
            ipc=ipc,
            extra=["--run_mode", "Condense", "--init", "mix"],
        )
        (run_dir / "condense_command.txt").write_text(" ".join(condense_cmd) + "\n", encoding="utf-8")
        rc = run_command(condense_cmd, repo_dir, run_dir / "condense_stdout.log", run_dir / "condense_stderr.log")
        condense_seconds = round(time.monotonic() - condense_started, 3)
        if rc != 0:
            raise RuntimeError(f"Condense failed for {dataset} {ipc} {group_name}; see {run_dir}")

        condensed_path = newest_distilled_data(args.exp_root, dataset, ipc, start_time)
        if condensed_path is None:
            condensed_path = latest_distilled_data(args.exp_root, dataset, ipc)
    else:
        condense_seconds = 0.0

    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {dataset} {ipc} {group_name}")
    condensed_record.write_text(str(condensed_path) + "\n", encoding="utf-8")

    checkpoint_path = (
        args.exp_root
        / "checkpoints"
        / "synthetic_train"
        / dataset
        / f"ipc{ipc}_{group_name}_best.pth.tar"
    )
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_cmd = torchrun_command(
        "evaluation/evaluation_script.py",
        config_path,
        args.gpu,
        ipc=ipc,
        extra=[
            "--run_mode",
            "Evaluation",
            "--load_path",
            str(condensed_path),
            "--val_repeat",
            "1",
            "--eval_checkpoint_path",
            str(checkpoint_path),
            "--eval_metrics_path",
            str(eval_metrics_path),
        ],
    )
    (run_dir / "eval_command.txt").write_text(" ".join(eval_cmd) + "\n", encoding="utf-8")
    eval_started = time.monotonic()
    rc = run_command(eval_cmd, repo_dir, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log")
    eval_seconds = round(time.monotonic() - eval_started, 3)
    if rc != 0:
        raise RuntimeError(f"Eval failed for {dataset} {ipc} {group_name}; see {run_dir}")

    best_metrics = read_best_metrics(eval_metrics_path)
    accuracy = best_metrics.get("acc_percent") or parse_best_accuracy(run_dir / "eval_stdout.log")

    cam_dir = args.exp_root / "results" / "cam" / dataset / f"ipc{ipc}_{group_name}"
    cam_cmd = [
        sys.executable,
        "evaluation/run_cam.py",
        "--dataset",
        dataset,
        "--data_dir",
        str(args.exp_root / "data"),
        "--checkpoint",
        str(checkpoint_path),
        "--save_dir",
        str(cam_dir),
        "--num_samples",
        str(args.cam_samples),
        "--nclass",
        str(DATASETS[dataset]["nclass"]),
        "--nch",
        str(DATASETS[dataset]["nch"]),
        "--size",
        "28",
        "--gpu",
        str(args.gpu),
    ]
    (run_dir / "cam_command.txt").write_text(" ".join(cam_cmd) + "\n", encoding="utf-8")
    cam_started = time.monotonic()
    rc = run_command(cam_cmd, repo_dir, run_dir / "cam_stdout.log", run_dir / "cam_stderr.log")
    cam_seconds = round(time.monotonic() - cam_started, 3)
    if rc != 0:
        raise RuntimeError(f"CAM failed for {dataset} {ipc} {group_name}; see {run_dir}")

    metrics = {
        "dataset": dataset,
        "ipc": ipc,
        "group": group_name,
        "seed": int(getattr(args, "seed", 0)),
        "niter": args.niter,
        "accuracy": accuracy,
        "condense_seconds": condense_seconds,
        "eval_seconds": eval_seconds,
        "cam_seconds": cam_seconds,
        "total_seconds": round(condense_seconds + eval_seconds + cam_seconds, 3),
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
        "cam_summary": str(cam_dir / "summary.csv"),
        **best_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifact_manifest = {
        "dataset": dataset,
        "ipc": ipc,
        "group": group_name,
        "config": str(config_path),
        "final_synthetic_pt": str(condensed_path),
        "eval_checkpoint": str(checkpoint_path),
        "metrics": str(run_dir / "metrics.json"),
        "eval_metrics_jsonl": str(eval_metrics_path),
        "eval_metrics_best": str(eval_metrics_path.with_name(eval_metrics_path.stem + "_best.json")),
        "condense_command": str(run_dir / "condense_command.txt"),
        "eval_command": str(run_dir / "eval_command.txt"),
        "cam_summary": str(cam_dir / "summary.csv"),
        "condense_seconds": condense_seconds,
        "eval_seconds": eval_seconds,
        "cam_seconds": cam_seconds,
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8"
    )
    append(
        args.exp_root / "EXPERIMENT_LOG.md",
        f"| {datetime.now().isoformat(timespec='seconds')} | {dataset} | {ipc} | {group_name} | done | "
        f"{metrics.get('acc_percent')} | {metrics.get('auc_macro_ovr')} | {metrics.get('macro_f1')} | "
        f"{metrics.get('balanced_acc')} | `{run_dir / 'metrics.json'}` |\n",
    )
    append(
        args.exp_root / "CAM_RESULTS.md",
        f"| {dataset} | ipc{ipc}_{group_name} | `{checkpoint_path}` | `{cam_dir / 'summary.csv'}` | `{cam_dir / 'overlays'}` | formal protocol |\n",
    )
    return metrics


def save_summary(exp_root, metrics, prefix="formal_ablation_summary"):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{prefix}.json"
    csv_path = report_dir / f"{prefix}.csv"
    md_path = report_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fields = [
        "dataset",
        "ipc",
        "group",
        "seed",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "sensitivity",
        "specificity",
        "auprc",
        "condense_seconds",
        "eval_seconds",
        "cam_seconds",
        "total_seconds",
        "checkpoint_path",
        "cam_summary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    rows = [
        "| Dataset | IPC | Group | Seed | ACC | AUC | Macro-F1 | Balanced ACC | Sens | Spec | AUPRC | Condense s | Eval s | Total s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        rows.append(
            f"| {item.get('dataset')} | {item.get('ipc')} | {item.get('group')} | "
            f"{item.get('seed')} | {item.get('acc_percent')} | {item.get('auc_macro_ovr')} | {item.get('macro_f1')} | "
            f"{item.get('balanced_acc')} | {item.get('sensitivity')} | {item.get('specificity')} | "
            f"{item.get('auprc')} | {item.get('condense_seconds')} | {item.get('eval_seconds')} | "
            f"{item.get('total_seconds')} |"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved summary: {md_path}")


def parse_datasets(value):
    if value == "all":
        return list(DATASETS)
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return datasets


def parse_groups(value):
    if value in {"main", "abc", "all"}:
        return list(GROUPS)
    if value == "baseline":
        return ["B_NCFM_T512"]
    if value == "local":
        return ["D_LOCAL_LAM03_T512"]
    if value == "dam":
        return ["F_DAM_L012_W10_T512"]
    groups = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}")
    return groups


def main():
    parser = argparse.ArgumentParser(description="Run formal MedMNIST A/B/C + metrics + CAM pipeline.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--stage", choices=["pretrain", "cam_real", "abc", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--cam_samples", type=int, default=100)
    parser.add_argument("--groups", default="abc")
    parser.add_argument("--summary_prefix", default="formal_ablation_summary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.exp_root.mkdir(parents=True, exist_ok=True)
    ensure_headers(args.exp_root)
    datasets = parse_datasets(args.datasets)
    groups = parse_groups(args.groups)

    if args.stage in {"pretrain", "cam_real", "all"}:
        for dataset in datasets:
            ensure_pretrain(args, repo_dir, dataset)

    if args.stage in {"cam_real", "all"}:
        for dataset in datasets:
            run_real_cam(args, repo_dir, dataset)

    all_metrics = []
    if args.stage in {"abc", "all"}:
        for dataset in datasets:
            for group_name in groups:
                all_metrics.append(run_condense_eval_cam(args, repo_dir, dataset, args.ipc, group_name))
        save_summary(args.exp_root, all_metrics, args.summary_prefix)


if __name__ == "__main__":
    main()
