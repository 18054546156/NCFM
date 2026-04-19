#!/bin/bash
# NCFM Complete Pipeline: Pretrain (20 models) + Condense + Evaluate
# Following the original paper configuration

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

cd ${NCFM_DIR}

echo "========================================"
echo "NCFM Full Pipeline for ${DATASET} (IPC=${IPC})"
echo "Following paper: 20 pre-trained models"
echo "========================================"
echo "Start time: $(date)"

# Step 1: Pretrain 20 models (~60 epochs each)
echo ""
echo "========================================"
echo "Step 1: Pretraining 20 models"
echo "========================================"
echo "This will take several hours..."

CUDA_VISIBLE_DEVICES=${GPU} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
    python pretrain/pretrain_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Pretrain

echo "✓ Pretraining completed"

# Step 2: Condense data using pre-trained models
echo ""
echo "========================================"
echo "Step 2: Condensing dataset"
echo "========================================"
echo "Using 20 pre-trained models for matching..."

CUDA_VISIBLE_DEVICES=${GPU} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
    python condense/condense_script.py \
    --config_path config/ipc${IPC}/${DATASET}.yaml \
    --gpu "0" \
    --ipc ${IPC} \
    --run_mode Condense

echo "✓ Condensation completed"

# Step 3: Evaluate condensed data
echo ""
echo "========================================"
echo "Step 3: Evaluating condensed data"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
    python evaluation/evaluation_script.py \
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
echo "End time: $(date)"
echo "========================================"

# Show results summary
echo ""
echo "=== Results Summary ==="
ls -lh ${RESULTS_DIR}/${DATASET}/ 2>/dev/null || echo "No results found"
