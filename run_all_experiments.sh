#!/bin/bash
# NCFM Data Condensation Experiments on MedMNIST
# Datasets: PathMNIST (balanced), DermaMNIST (imbalanced), BloodMNIST (imbalanced)
# IPC: 1, 10, 50

set -e

# Paths
PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"

mkdir -p ${RESULTS_DIR}

# Activate conda
source /root/miniconda3/condabin/conda shell.bash hook
conda activate py311

cd ${NCFM_DIR}

echo "========================================"
echo "NCFM MedMNIST Experiments"
echo "========================================"
echo "Datasets: PathMNIST, DermaMNIST, BloodMNIST"
echo "IPC: 1, 10, 50"
echo "========================================"

# Datasets and IPCs
DATASETS=("pathmnist" "dermamnist" "bloodmnist")
IPCS=(1 10 50)

# Function to run condensation
run_condense() {
    local DATASET=$1
    local IPC=$2
    local GPU=$3

    echo ""
    echo "========================================"
    echo "Running: ${DATASET} | IPC=${IPC}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python condense/condense_script.py \
        --config_path config/ipc${IPC}/${DATASET}.yaml \
        --gpu "0" \
        --ipc ${IPC} \
        --run_mode Condense
}

# Check available GPUs
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{if ($2/$3 < 0.1) print $1}' > /tmp/available_gpus.txt

AVAILABLE_GPUS=$(cat /tmp/available_gpus.txt | wc -l)
echo "Available GPUs: ${AVAILABLE_GPUS}"

if [ ${AVAILABLE_GPUS} -eq 0 ]; then
    echo "ERROR: No available GPUs!"
    exit 1
fi

# Get first available GPU
GPU_ID=$(head -1 /tmp/available_gpus.txt)
echo "Using GPU: ${GPU_ID}"

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do
        echo ""
        echo "========================================"
        echo "Starting: ${DATASET} with IPC=${IPC}"
        echo "Time: $(date)"
        echo "========================================"

        run_condense ${DATASET} ${IPC} ${GPU_ID}

        # Move results to organized directory
        mkdir -p ${RESULTS_DIR}/${DATASET}
        find ${NCFM_DIR}/results -name "*${DATASET}*ipc${IPC}*" -type f -exec cp {} ${RESULTS_DIR}/${DATASET}/ \; 2>/dev/null || true
        find ${NCFM_DIR}/results -name "*ipc${IPC}*${DATASET}*" -type f -exec cp {} ${RESULTS_DIR}/${DATASET}/ \; 2>/dev/null || true

        echo "✓ Completed: ${DATASET} with IPC=${IPC}"
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "========================================"

# Summary
echo ""
echo "========================================"
echo "Experiment Summary"
echo "========================================"
echo "Datasets: ${DATASETS[*]}"
echo "IPC values: ${IPCS[*]}"
echo "Total experiments: $((${#DATASETS[@]} * ${#IPCS[@]}))"
echo ""
echo "Results directory structure:"
ls -la ${RESULTS_DIR}/*/
echo "========================================"
