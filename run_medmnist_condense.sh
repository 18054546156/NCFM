#!/bin/bash
# Script to run NCFM condensation on MedMNIST datasets

# Set paths
PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
RESULTS_DIR="${PROJECT_DIR}/results"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Activate conda environment
source /root/miniconda3/condabin/conda shell.bash hook
conda activate py311

cd ${NCFM_DIR}

echo "======================================"
echo "NCFM MedMNIST Data Condensation"
echo "======================================"

# Function to run condensation
run_condense() {
    local DATASET=$1
    local IPC=$2
    local GPU=$3

    echo ""
    echo "--------------------------------------"
    echo "Running: ${DATASET} with IPC=${IPC}"
    echo "--------------------------------------"

    python condense/condense_script.py \
        --config_path config/ipc${IPC}/${DATASET}.yaml \
        --gpu "${GPU}" \
        --ipc ${IPC} \
        --run_mode Condense
}

# Check which dataset to run
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name> [ipc] [gpu]"
    echo ""
    echo "Examples:"
    echo "  $0 pathmnist 10 0"
    echo "  $0 dermamnist 10 0,1"
    echo ""
    echo "Available datasets:"
    echo "  - pathmnist (balanced, 9 classes)"
    echo "  - dermamnist (imbalanced, 7 classes, IR=58.66)"
    exit 1
fi

DATASET=$1
IPC=${2:-10}
GPU=${3:-0}

run_condense ${DATASET} ${IPC} ${GPU}

echo ""
echo "======================================"
echo "Done! Results saved to: ${RESULTS_DIR}"
echo "======================================"
