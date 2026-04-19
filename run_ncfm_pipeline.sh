#!/bin/bash
# Complete NCFM Pipeline: Pretrain + Condense for MedMNIST

set -e

DATASET=${1:-"pathmnist"}
IPC=${2:-10}
GPU=${3:-1}

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
PRETRAIN_DIR="${NCFM_DIR}/pretrained_models/${DATASET}"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"

mkdir -p ${PRETRAIN_DIR}
mkdir -p ${RESULTS_DIR}

source /root/miniconda3/condabin/conda shell.bash hook
conda activate py311

echo "========================================"
echo "NCFM Pipeline for ${DATASET} (IPC=${IPC})"
echo "========================================"

cd ${NCFM_DIR}

# Step 1: Pretrain models
echo ""
echo "Step 1: Pretraining 20 models on ${DATASET}..."
echo "This may take a while..."

CUDA_VISIBLE_DEVICES=${GPU} python pretrain/pretrain_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Pretrain

echo "✓ Pretraining completed"

# Step 2: Condense data
echo ""
echo "Step 2: Condensing ${DATASET} dataset..."

CUDA_VISIBLE_DEVICES=${GPU} python condense/condense_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Condense

echo "✓ Condensation completed"

# Step 3: Evaluate
echo ""
echo "Step 3: Evaluating condensed data..."

CUDA_VISIBLE_DEVICES=${GPU} python evaluation/evaluation_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Evaluation

echo "✓ Evaluation completed"

# Copy results
mkdir -p ${RESULTS_DIR}/${DATASET}
find ${NCFM_DIR}/results -name "*${DATASET}*ipc${IPC}*" -type f -exec cp {} ${RESULTS_DIR}/${DATASET}/ \; 2>/dev/null || true

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "Results: ${RESULTS_DIR}/${DATASET}/"
echo "========================================"
