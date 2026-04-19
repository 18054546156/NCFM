#!/bin/bash
# Run a single NCFM experiment
# Usage: ./run_single.sh <dataset> <ipc> <gpu>

DATASET=${1:-"pathmnist"}
IPC=${2:-10}
GPU=${3:-0}

echo "Running NCFM condensation:"
echo "  Dataset: ${DATASET}"
echo "  IPC: ${IPC}"
echo "  GPU: ${GPU}"

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"

source /root/miniconda3/condabin/conda shell.bash hook
conda activate py311
cd ${NCFM_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python condense/condense_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Condense

echo "Done!"
