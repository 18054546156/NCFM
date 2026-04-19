#!/bin/bash
# NCFM MedMNIST Complete Experiment Launcher - Fixed for background execution

set -e

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
LOG_DIR="${NCFM_DIR}/ncfm_logs"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"

mkdir -p ${LOG_DIR}
mkdir -p ${RESULTS_DIR}

# Full path to conda
CONDA_BASE="/root/miniconda3"

cd ${NCFM_DIR}

echo "========================================"
echo "NCFM MedMNIST Strict Reproduction"
echo "========================================"
echo "Start time: $(date)"
echo "========================================"

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local ipc=$2
    local gpu=$3

    local exp_name="${dataset}_ipc${ipc}"
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo ""
    echo "========================================"
    echo "Starting: ${exp_name}"
    echo "GPU: ${gpu}"
    echo "Time: $(date)"
    echo "========================================"

    # Pretrain phase
    echo "[${exp_name}] Phase 1: Pretraining 20 models..."
    CUDA_VISIBLE_DEVICES=${gpu} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
        ${CONDA_BASE}/envs/py311/bin/python pretrain/pretrain_script.py \
        --config_path config/ipc${ipc}/${dataset}.yaml \
        --gpu "0" \
        --ipc ${ipc} \
        --run_mode Pretrain >> ${log_file} 2>&1

    if [ $? -ne 0 ]; then
        echo "[${exp_name}] ✗ Pretraining failed!"
        cat ${log_file} | tail -50
        return 1
    fi

    echo "[${exp_name}] ✓ Pretraining completed"

    # Condense phase
    echo "[${exp_name}] Phase 2: Condensing dataset..."
    CUDA_VISIBLE_DEVICES=${gpu} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
        ${CONDA_BASE}/envs/py311/bin/python condense/condense_script.py \
        --config_path config/ipc${ipc}/${dataset}.yaml \
        --gpu "0" \
        --ipc ${ipc} \
        --run_mode Condense >> ${log_file} 2>&1

    if [ $? -ne 0 ]; then
        echo "[${exp_name}] ✗ Condensation failed!"
        cat ${log_file} | tail -50
        return 1
    fi

    echo "[${exp_name}] ✓ Condensation completed"

    # Evaluate phase
    echo "[${exp_name}] Phase 3: Evaluating..."
    CUDA_VISIBLE_DEVICES=${gpu} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
        ${CONDA_BASE}/envs/py311/bin/python evaluation/evaluation_script.py \
        --config_path config/ipc${ipc}/${dataset}.yaml \
        --gpu "0" \
        --ipc ${ipc} \
        --run_mode Evaluation >> ${log_file} 2>&1

    if [ $? -ne 0 ]; then
        echo "[${exp_name}] ✗ Evaluation failed!"
        cat ${log_file} | tail -50
        return 1
    fi

    echo "[${exp_name}] ✓ Evaluation completed"

    # Copy results
    mkdir -p ${RESULTS_DIR}/${dataset}
    cp ${NCFM_DIR}/results/*${dataset}*ipc${ipc}* ${RESULTS_DIR}/${dataset}/ 2>/dev/null || true

    echo "========================================"
    echo "✓ ${exp_name} COMPLETED"
    echo "Time: $(date)"
    echo "========================================"
}

# Run all experiments sequentially
echo ""
echo "=== Starting Experiment Queue ==="

# PathMNIST (Balanced)
echo ""
echo "Group 1: PathMNIST (Balanced, IR=1.63)"
run_experiment "pathmnist" 10 1
run_experiment "pathmnist" 1 1
run_experiment "pathmnist" 50 1

# DermaMNIST (Severely Imbalanced)
echo ""
echo "Group 2: DermaMNIST (Severely Imbalanced, IR=58.66)"
run_experiment "dermamnist" 10 1
run_experiment "dermamnist" 1 1
run_experiment "dermamnist" 50 1

# BloodMNIST (Moderately Imbalanced)
echo ""
echo "Group 3: BloodMNIST (Moderately Imbalanced, IR=2.74)"
run_experiment "bloodmnist" 10 1
run_experiment "bloodmnist" 1 1
run_experiment "bloodmnist" 50 1

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "End time: $(date)"
echo "========================================"
