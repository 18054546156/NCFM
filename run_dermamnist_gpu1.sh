#!/bin/bash
PROJECT_DIR="/data/zengqiang/experiments/project_20260419_143900_medmnist_stats"
NCFM_DIR="${PROJECT_DIR}/NCFM"
LOG_DIR="${NCFM_DIR}/ncfm_logs"
RESULTS_DIR="${PROJECT_DIR}/ncfm_results"

cd ${NCFM_DIR}

echo "DermaMNIST on GPU 1 starting..."

for ipc in 10 1 50; do
    echo "Starting dermamnist IPC=${ipc}"
    
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        /root/miniconda3/envs/py311/bin/python ${NCFM_DIR}/pretrain/pretrain_script.py \
        --config_path ${NCFM_DIR}/config/ipc${ipc}/dermamnist.yaml \
        --gpu "0" --ipc ${ipc} --run_mode Pretrain 2>&1 | tee ${LOG_DIR}/dermamnist_ipc${ipc}.log
    
    echo "Pretraining done for IPC=${ipc}"
    
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        /root/miniconda3/envs/py311/bin/python ${NCFM_DIR}/condense/condense_script.py \
        --config_path ${NCFM_DIR}/config/ipc${ipc}/dermamnist.yaml \
        --gpu "0" --ipc ${ipc} --run_mode Condense 2>&1 | tee -a ${LOG_DIR}/dermamnist_ipc${ipc}.log
    
    echo "Condensation done for IPC=${ipc}"
    
    CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29501 \
        /root/miniconda3/envs/py311/bin/python ${NCFM_DIR}/evaluation/evaluation_script.py \
        --config_path ${NCFM_DIR}/config/ipc${ipc}/dermamnist.yaml \
        --gpu "0" --ipc ${ipc} --run_mode Evaluation 2>&1 | tee -a ${LOG_DIR}/dermamnist_ipc${ipc}.log
    
    echo "dermamnist IPC=${ipc} COMPLETED"
done

echo "DermaMNIST ALL COMPLETED!"
