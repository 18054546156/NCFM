#!/bin/bash
# NCFM MedMNIST Complete Experiments
# 3 datasets × 3 IPC values = 9 experiments
# Each experiment: Pretrain 20 models + Condense + Evaluate

set -e

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"
LOG_DIR="${PROJECT_DIR}/ncfm_logs"

mkdir -p ${RESULTS_DIR}
mkdir -p ${LOG_DIR}

source /root/miniconda3/condabin/conda shell.bash hook
conda activate py311

cd ${NCFM_DIR}

echo "========================================"
echo "NCFM MedMNIST Full Experiment Suite"
echo "========================================"
echo "Datasets: PathMNIST (Balanced), DermaMNIST (Imbalanced), BloodMNIST (Imbalanced)"
echo "IPC values: 1, 10, 50"
echo "Total experiments: 9"
echo "Start time: $(date)"
echo "========================================"

# Experiment matrix
DATASETS=("pathmnist" "dermamnist" "bloodmnist")
IPCS=(1 10 50)

# Check GPU availability
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

GPU_ID=$(head -1 /tmp/available_gpus.txt)
echo "Using GPU: ${GPU_ID}"

# Run experiments sequentially
for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do
        echo ""
        echo "========================================"
        echo "Experiment: ${DATASET} | IPC=${IPC}"
        echo "Start time: $(date)"
        echo "========================================"

        LOG_FILE="${LOG_DIR}/${DATASET}_ipc${IPC}.log"

        # Run full pipeline with logging
        bash run_full_pipeline.sh ${DATASET} ${IPC} ${GPU_ID} 2>&1 | tee ${LOG_FILE}

        echo "✓ Completed: ${DATASET} with IPC=${IPC}"
        echo "End time: $(date)"
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "=== Results Summary ==="
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "${DATASET}:"
    ls -lh ${RESULTS_DIR}/${DATASET}/ 2>/dev/null || echo "  No results"
done

echo ""
echo "=== Log files ==="
ls -lh ${LOG_DIR}/
