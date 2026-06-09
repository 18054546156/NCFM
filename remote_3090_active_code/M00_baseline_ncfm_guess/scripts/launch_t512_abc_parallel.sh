#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-tmp/ncfm_t512_main_20260528}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
IPC="${IPC:-10}"
NITER="${NITER:-20000}"
MODEL_NUM="${MODEL_NUM:-20}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-60}"
EVAL_EPOCHS="${EVAL_EPOCHS:-2000}"
EPOCH_EVAL_INTERVAL="${EPOCH_EVAL_INTERVAL:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_REAL="${BATCH_REAL:-1024}"
RUN_WORKERS="${RUN_WORKERS:-2}"
CAM_SAMPLES="${CAM_SAMPLES:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS:-pneumoniamnist,bloodmnist,pathmnist}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS:-B_NCFM_T512,D_LOCAL_LAM03_T512,F_DAM_L012_W10_T512}"

LOG_ROOT="$EXP_ROOT/launcher_logs"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

echo "[$(date -Is)] START abc_parallel max_parallel=$MAX_PARALLEL workers=$RUN_WORKERS" | tee -a "$EXP_ROOT/RUN_STATUS.md"

running=0
status=0
for dataset in "${DATASET_LIST[@]}"; do
  for group in "${METHOD_LIST[@]}"; do
    dataset="$(echo "$dataset" | xargs)"
    group="$(echo "$group" | xargs)"
    log_path="$LOG_ROOT/parallel_${dataset}_${group}.log"
    echo "[$(date -Is)] START ${dataset} ${group}" | tee -a "$EXP_ROOT/RUN_STATUS.md"
    (
      set -euo pipefail
      cd "$PROJECT_ROOT"
      "$PYTHON" scripts/run_medmnist_formal_pipeline.py \
        --exp_root "$EXP_ROOT" \
        --gpu "$GPU" \
        --datasets "$dataset" \
        --groups "$group" \
        --stage abc \
        --workers "$RUN_WORKERS" \
        --batch_size "$BATCH_SIZE" \
        --batch_real "$BATCH_REAL" \
        --model_num "$MODEL_NUM" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --eval_epochs "$EVAL_EPOCHS" \
        --epoch_eval_interval "$EPOCH_EVAL_INTERVAL" \
        --niter "$NITER" \
        --ipc "$IPC" \
        --cam_samples "$CAM_SAMPLES" \
        --summary_prefix "single_${dataset}_${group}"
    ) > "$log_path" 2>&1 &
    echo $! > "$LOG_ROOT/parallel_${dataset}_${group}.pid"
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      if ! wait -n; then
        status=1
      fi
      running=$((running - 1))
    fi
  done
done

while (( running > 0 )); do
  if ! wait -n; then
    status=1
  fi
  running=$((running - 1))
done

"$PYTHON" scripts/collect_t512_summary.py --exp_root "$EXP_ROOT" --ipc "$IPC" --prefix main_t512_summary
echo "[$(date -Is)] DONE abc_parallel status=$status" | tee -a "$EXP_ROOT/RUN_STATUS.md"
exit "$status"
