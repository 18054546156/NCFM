#!/bin/bash
# Run MedMNIST IPC=50 condensation with training dynamics, reusing existing pretrained models.

set -euo pipefail

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
LOG_DIR="${NCFM_DIR}/ncfm_logs/ipc50_only"

GPU="${1:-1}"
DATASETS=("pathmnist" "dermamnist" "bloodmnist")

mkdir -p "${LOG_DIR}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate py311

cd "${NCFM_DIR}"

find_latest_run_dir() {
    local dataset="$1"
    local base="${PROJECT_DIR}/results/condense/condense/${dataset}/ipc50"
    ls -td "${base}"/* 2>/dev/null | head -n 1
}

run_condense() {
    local dataset="$1"
    local port="$2"
    local log_file="${LOG_DIR}/${dataset}_ipc50.log"

    echo "=== Condense ${dataset} IPC=50 on GPU ${GPU} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" \
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}" \
    python condense/condense_script.py \
      --config_path "config/ipc50/${dataset}.yaml" \
      --gpu "${GPU}" \
      --ipc 50 \
      --run_mode Condense | tee "${log_file}"
}

plot_dynamics_if_exists() {
    local dataset="$1"
    local run_dir
    run_dir="$(find_latest_run_dir "${dataset}")"
    if [ -n "${run_dir}" ] && [ -f "${run_dir}/training_dynamics.json" ]; then
        echo "=== Plot dynamics ${dataset} IPC=50 ==="
        MPLCONFIGDIR=/tmp/matplotlib-cache \
        python plot_training_dynamics.py \
          --json_path "${run_dir}/training_dynamics.json"
    fi
}

echo "=== MedMNIST IPC50 only ==="
echo "GPU: ${GPU}"
echo "Datasets: ${DATASETS[*]}"

port_seed=29700
for dataset in "${DATASETS[@]}"; do
    run_condense "${dataset}" "${port_seed}"
    plot_dynamics_if_exists "${dataset}"
    port_seed=$((port_seed + 10))
done

echo "=== IPC50 runs completed ==="
