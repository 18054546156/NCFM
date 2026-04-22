#!/bin/bash
# Pretrain once per dataset, then run IPC {1,10,50} with training dynamics enabled.

set -euo pipefail

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
LOG_DIR="${NCFM_DIR}/ncfm_logs/dynamics_matrix"

GPU="${1:-0}"
DATASETS=("pathmnist" "dermamnist" "bloodmnist")
IPCS=(1 10 50)

mkdir -p "${LOG_DIR}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate py311

cd "${NCFM_DIR}"

find_latest_run_dir() {
    local dataset="$1"
    local ipc="$2"
    local base="${PROJECT_DIR}/results/condense/condense/${dataset}/ipc${ipc}"
    ls -td "${base}"/* 2>/dev/null | head -n 1
}

find_latest_syn_pt() {
    local run_dir="$1"
    find "${run_dir}/distilled_data" -maxdepth 1 -type f -name 'data_20000.pt' | head -n 1
}

run_pretrain() {
    local dataset="$1"
    local log_file="${LOG_DIR}/${dataset}_pretrain.log"
    echo "=== Pretrain ${dataset} on GPU ${GPU} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" \
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
    python pretrain/pretrain_script.py \
      --config_path "config/ipc10/${dataset}.yaml" \
      --gpu "${GPU}" \
      --ipc 10 \
      --run_mode Pretrain | tee "${log_file}"
}

run_condense_plot_eval() {
    local dataset="$1"
    local ipc="$2"
    local port_base="$3"
    local log_file="${LOG_DIR}/${dataset}_ipc${ipc}.log"

    echo "=== Condense ${dataset} IPC=${ipc} on GPU ${GPU} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" \
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${port_base}" \
    python condense/condense_script.py \
      --config_path "config/ipc${ipc}/${dataset}.yaml" \
      --gpu "${GPU}" \
      --ipc "${ipc}" \
      --run_mode Condense | tee "${log_file}"

    local run_dir
    run_dir="$(find_latest_run_dir "${dataset}" "${ipc}")"
    if [ -z "${run_dir}" ]; then
        echo "No run dir found for ${dataset} IPC=${ipc}" >&2
        return 1
    fi

    if [ -f "${run_dir}/training_dynamics.json" ]; then
        echo "=== Plot dynamics ${dataset} IPC=${ipc} ==="
        MPLCONFIGDIR=/tmp/matplotlib-cache \
        python plot_training_dynamics.py \
          --json_path "${run_dir}/training_dynamics.json"
    fi

    local syn_pt
    syn_pt="$(find_latest_syn_pt "${run_dir}")"
    if [ -n "${syn_pt}" ]; then
        echo "=== Evaluate ${dataset} IPC=${ipc} ==="
        CUDA_VISIBLE_DEVICES="${GPU}" \
        RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
        MASTER_ADDR=127.0.0.1 MASTER_PORT="$((port_base + 1))" \
        python evaluation/evaluation_script.py \
          --config_path "config/ipc${ipc}/${dataset}.yaml" \
          --gpu "${GPU}" \
          --ipc "${ipc}" \
          --run_mode Evaluation \
          --load_path "${syn_pt}" | tee -a "${log_file}"
    else
        echo "No data_20000.pt found for ${dataset} IPC=${ipc}, skip evaluation" >&2
    fi
}

echo "=== MedMNIST dynamics matrix ==="
echo "GPU: ${GPU}"
echo "Datasets: ${DATASETS[*]}"
echo "IPCs: ${IPCS[*]}"

for dataset in "${DATASETS[@]}"; do
    run_pretrain "${dataset}"
    port_seed=29610
    for ipc in "${IPCS[@]}"; do
        run_condense_plot_eval "${dataset}" "${ipc}" "${port_seed}"
        port_seed=$((port_seed + 10))
    done
done

echo "=== All runs completed ==="
