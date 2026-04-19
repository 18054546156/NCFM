#!/bin/bash
# Monitor NCFM experiment progress

PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
LOG_DIR="${PROJECT_DIR}/NCFM/ncfm_logs"

echo "========================================"
echo "NCFM Experiment Monitor"
echo "========================================"
echo "Time: $(date)"
echo ""

# Check all log files
for log_file in ${LOG_DIR}/*.log; do
    if [ -f "$log_file" ]; then
        exp_name=$(basename "$log_file" .log)
        echo "[$exp_name]"

        # Get last line with epoch info
        last_epoch=$(tail -100 "$log_file" 2>/dev/null | grep "Epoch" | tail -1)
        if [ -n "$last_epoch" ]; then
            echo "  $last_epoch"
        fi

        # Check if completed
        if grep -q "completed" "$log_file" 2>/dev/null; then
            echo "  ✓ Completed"
        elif grep -q "Error\|Traceback" "$log_file" 2>/dev/null; then
            echo "  ✗ Error detected"
        else
            echo "  Running..."
        fi
        echo ""
    fi
done

# Check pre-trained models count
echo "========================================"
echo "Pre-trained Models Status"
echo "========================================"
for dataset in pathmnist dermamnist bloodmnist; do
    pretrain_dir="${PROJECT_DIR}/NCFM/pretrained_models/${dataset}"
    if [ -d "$pretrain_dir" ]; then
        init_count=$(ls "$pretrain_dir"/*_init.pth.tar 2>/dev/null | wc -l)
        trained_count=$(ls "$pretrain_dir"/*_trained.pth.tar 2>/dev/null | wc -l)
        echo "${dataset}: ${trained_count}/20 models trained"
    fi
done

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
