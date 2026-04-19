#!/bin/bash
# DermaMNIST experiments on GPU 1

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
LOG_DIR="${NCFM_DIR}/ncfm_logs"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"
CONDA_BASE="/root/miniconda3"

cd ${NCFM_DIR}
mkdir -p ${LOG_DIR} ${RESULTS_DIR}

echo "========================================"
echo "DermaMNIST on GPU 1 (IR=58.66, Severe Imbalance)"
echo "Start time: \$(date)"
echo "========================================"

run_dermamnist() {
    local ipc=$1
    local exp_name="dermamnist_ipc\${ipc}"
    local log_file="\${LOG_DIR}/\${exp_name}.log"
    
    echo ""
    echo "Starting: \${exp_name}"
    
    # Pretrain
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        \${CONDA_BASE}/envs/py311/bin/python pretrain/pretrain_script.py \
        --config_path config/ipc\${ipc}/dermamnist.yaml \
        --gpu "0" --ipc \${ipc} --run_mode Pretrain >> \${log_file} 2>&1
    
    echo "✓ Pretraining done"
    
    # Condense
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        \${CONDA_BASE}/envs/py311/bin/python condense/condense_script.py \
        --config_path config/ipc\${ipc}/dermamnist.yaml \
        --gpu "0" --ipc \${ipc} --run_mode Condense >> \${log_file} 2>&1
    
    echo "✓ Condensation done"
    
    # Evaluate
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        \${CONDA_BASE}/envs/py311/bin/python evaluation/evaluation_script.py \
        --config_path config/ipc\${ipc}/dermamnist.yaml \
        --gpu "0" --ipc \${ipc} --run_mode Evaluation >> \${log_file} 2>&1
    
    echo "✓ \${exp_name} COMPLETED"
    mkdir -p \${RESULTS_DIR}/dermamnist
    cp \${NCFM_DIR}/results/*dermamnist*ipc\${ipc}* \${RESULTS_DIR}/dermamnist/ 2>/dev/null || true
}

# Run all IPC values for DermaMNIST
run_dermamnist 10
run_dermamnist 1
run_dermamnist 50

echo ""
echo "========================================"
echo "DermaMNIST ALL COMPLETED!"
echo "End time: \$(date)"
echo "========================================"
