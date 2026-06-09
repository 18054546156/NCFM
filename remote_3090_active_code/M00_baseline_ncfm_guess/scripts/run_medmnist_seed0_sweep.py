import argparse
import csv
import importlib.util
import json
from pathlib import Path


IPC = 10


def load_pipeline():
    here = Path(__file__).resolve().parent
    path = here / "run_medmnist_formal_pipeline.py"
    spec = importlib.util.spec_from_file_location("formal_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lam_tag(value):
    return f"{int(round(float(value) * 100)):03d}"


def layer_tag(layers):
    return "L" + "".join(str(x) for x in layers)


def build_groups():
    groups = {}
    family = {}

    for num_freqs in [128, 256, 512, 1024]:
        name = f"B_T{num_freqs}"
        groups[name] = {
            "sampling_net": True,
            "num_freqs": num_freqs,
            "iter_calib": 0,
            "calib_weight": 1,
            "use_local_patch_feature_ncfd": False,
            "dam_enabled": False,
            "use_ssim_regularization": False,
        }
        family[name] = "baseline"

    for grid in [2, 4, 7]:
        for lam in [0.3, 0.5, 0.8]:
            name = f"LP_lam{lam_tag(lam)}_g{grid}_lf512"
            groups[name] = {
                "sampling_net": True,
                "num_freqs": 512,
                "iter_calib": 0,
                "calib_weight": 1,
                "use_local_patch_feature_ncfd": True,
                "local_patch_grid": grid,
                "lambda_local_patch_ncfd": lam,
                "local_patch_feature_dim": 128,
                "local_patch_encoder_blocks": 2,
                "local_patch_num_freqs": 512,
                "local_patch_encoder_source": "premodel0_trained",
                "local_patch_encoder_frozen": True,
                "use_local_patch_sampling_net": False,
                "dam_enabled": False,
                "use_ssim_regularization": False,
            }
            family[name] = "local_patch"

    for layers in [[0, 1], [1, 2], [0, 1, 2]]:
        for weight in [10.0, 50.0, 100.0]:
            name = f"DAM_w{int(weight)}_{layer_tag(layers)}"
            groups[name] = {
                "sampling_net": True,
                "num_freqs": 512,
                "iter_calib": 0,
                "calib_weight": 1,
                "use_local_patch_feature_ncfd": False,
                "dam_enabled": True,
                "dam_objective": "ncfm_attention",
                "dam_feature_weight": 1.0,
                "dam_attention_weight": weight,
                "dam_attention_layers": layers,
                "dam_attention_p": 2,
                "dam_attention_norm": "l2",
                "dam_detach_real": True,
                "dam_log_components": True,
                "use_ssim_regularization": False,
            }
            family[name] = "dam_attention"

    for name, weight, grids in [
        ("SSIM_w005_G124", 0.05, [1, 2, 4]),
        ("SSIM_w01_G1", 0.1, [1]),
        ("SSIM_w01_G12", 0.1, [1, 2]),
        ("SSIM_w01_G124", 0.1, [1, 2, 4]),
        ("SSIM_w05_G124", 0.5, [1, 2, 4]),
        ("SSIM_w10_G124", 1.0, [1, 2, 4]),
    ]:
        groups[name] = {
            "sampling_net": True,
            "num_freqs": 512,
            "iter_calib": 0,
            "calib_weight": 1,
            "use_local_patch_feature_ncfd": False,
            "dam_enabled": False,
            "use_ssim_regularization": True,
            "ssim_weight": weight,
            "ssim_grids": grids,
            "ssim_kernel_size": 7,
            "ssim_data_range": 1.0,
            "ssim_pairing": "prototype",
            "ssim_log_components": True,
        }
        family[name] = "ssim"

    return groups, family


def select_groups(group_names, requested, chunk_id=None, num_chunks=None):
    if requested == "all":
        selected = list(group_names)
    else:
        wanted = [item.strip() for item in requested.split(",") if item.strip()]
        unknown = sorted(set(wanted) - set(group_names))
        if unknown:
            raise ValueError(f"Unknown groups: {unknown}")
        selected = wanted
    if chunk_id is not None and num_chunks is not None:
        selected = [name for idx, name in enumerate(selected) if idx % num_chunks == chunk_id]
    return selected


def assert_assets(exp_root, dataset, model_num):
    missing = []
    data_path = exp_root / "data" / "medmnist" / f"{dataset}.npz"
    pretrain_dir = exp_root / "checkpoints" / "pretrain" / dataset
    if not data_path.exists():
        missing.append(str(data_path))
    for idx in range(model_num):
        for suffix in ("init", "trained"):
            path = pretrain_dir / f"premodel{idx}_{suffix}.pth.tar"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Required {dataset} assets are missing:\n" + "\n".join(missing))


def save_group_plan(exp_root, dataset, groups, family):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{dataset}_seed0_group_plan.csv"
    md_path = report_dir / f"{dataset}_seed0_group_plan.md"
    fields = ["group", "family", "config"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, config in groups.items():
            writer.writerow({"group": group, "family": family[group], "config": json.dumps(config, sort_keys=True)})
    rows = ["| Group | Family | Key Config |", "|---|---|---|"]
    for group, config in groups.items():
        rows.append(f"| {group} | {family[group]} | `{json.dumps(config, sort_keys=True)}` |")
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_progress(exp_root, dataset, selected, completed, failed):
    status_path = exp_root / "RUN_STATUS.md"
    lines = [
        f"# {dataset} Seed0 Sweep Status",
        "",
        f"- Planned in this worker: {len(selected)}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(failed)}",
        "",
        "## Completed",
        "",
    ]
    for group in completed:
        lines.append(f"- {group}")
    lines.extend(["", "## Failed", ""])
    for group, error in failed:
        lines.append(f"- {group}: `{error}`")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run a MedMNIST seed0 28-group NCFM method sweep.")
    parser.add_argument("--dataset", required=True, choices=["pneumoniamnist", "bloodmnist", "pathmnist"])
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--chunk_id", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
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
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--list_only", action="store_true")
    args = parser.parse_args()

    if args.ipc != IPC:
        raise ValueError("This runner is fixed to IPC=10 for the 20260530 sweep.")
    if (args.chunk_id is None) != (args.num_chunks is None):
        raise ValueError("--chunk_id and --num_chunks must be provided together.")
    if args.summary_prefix is None:
        args.summary_prefix = f"{args.dataset}_seed0_worker_summary"

    pipeline = load_pipeline()
    groups, family = build_groups()
    group_names = list(groups)
    selected = select_groups(group_names, args.groups, args.chunk_id, args.num_chunks)
    pipeline.GROUPS = {"B_NCFM_T512": groups["B_T512"], **groups}

    if args.list_only:
        for group in selected:
            print(group, family[group], groups[group])
        return

    args.exp_root.mkdir(parents=True, exist_ok=True)
    assert_assets(args.exp_root, args.dataset, args.model_num)
    save_group_plan(args.exp_root, args.dataset, groups, family)
    pipeline.ensure_headers(args.exp_root)

    repo_dir = Path(__file__).resolve().parents[1]
    all_metrics = []
    completed = []
    failed = []
    for group_name in selected:
        print(f"RUN dataset={args.dataset} group={group_name} family={family[group_name]} gpu={args.gpu}", flush=True)
        try:
            metrics = pipeline.run_condense_eval_cam(args, repo_dir, args.dataset, IPC, group_name)
            metrics["family"] = family[group_name]
            metrics["seed"] = 0
            all_metrics.append(metrics)
            completed.append(group_name)
            pipeline.save_summary(args.exp_root, all_metrics, f"{args.summary_prefix}_gpu{args.gpu}")
        except Exception as exc:
            failed.append((group_name, str(exc)))
            save_progress(args.exp_root, args.dataset, selected, completed, failed)
            raise
        save_progress(args.exp_root, args.dataset, selected, completed, failed)
        if args.max_runs and len(completed) >= args.max_runs:
            break

    pipeline.save_summary(args.exp_root, all_metrics, f"{args.summary_prefix}_gpu{args.gpu}")


if __name__ == "__main__":
    main()
