#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-tmp/ncfm_t512_main_20260528}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-pneumoniamnist,bloodmnist,pathmnist}"

PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-8}"
RUN_WORKERS="${RUN_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_REAL="${BATCH_REAL:-1024}"
MODEL_NUM="${MODEL_NUM:-20}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-60}"
EVAL_EPOCHS="${EVAL_EPOCHS:-2000}"
EPOCH_EVAL_INTERVAL="${EPOCH_EVAL_INTERVAL:-100}"
NITER="${NITER:-20000}"
IPC="${IPC:-10}"
CAM_SAMPLES="${CAM_SAMPLES:-100}"
FORCE="${FORCE:-0}"

LOG_ROOT="$EXP_ROOT/launcher_logs"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name" | tee -a "$EXP_ROOT/RUN_STATUS.md"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  echo "[$(date -Is)] DONE  $name" | tee -a "$EXP_ROOT/RUN_STATUS.md"
}

common_args=(
  --exp_root "$EXP_ROOT"
  --gpu "$GPU"
  --datasets "$DATASETS"
  --batch_size "$BATCH_SIZE"
  --batch_real "$BATCH_REAL"
  --model_num "$MODEL_NUM"
  --pretrain_epochs "$PRETRAIN_EPOCHS"
  --eval_epochs "$EVAL_EPOCHS"
  --epoch_eval_interval "$EPOCH_EVAL_INTERVAL"
  --niter "$NITER"
  --ipc "$IPC"
  --cam_samples "$CAM_SAMPLES"
)

force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args=(--force)
fi

{
  echo "# T512 Main Run Status"
  echo
  echo "started: $(date -Is)"
  echo "project_root: $PROJECT_ROOT"
  echo "exp_root: $EXP_ROOT"
  echo "datasets: $DATASETS"
  echo "methods: B_NCFM_T512, D_LOCAL_LAM03_T512, F_DAM_L012_W10_T512"
  echo "num_freqs: 512"
  echo "ipc: $IPC"
  echo "niter: $NITER"
  echo "model_num: $MODEL_NUM"
  echo "pretrain_epochs: $PRETRAIN_EPOCHS"
  echo
} > "$EXP_ROOT/RUN_STATUS.md"

run_step pretrain_all \
  "$PYTHON" scripts/run_medmnist_formal_pipeline.py \
    "${common_args[@]}" \
    --workers "$PRETRAIN_WORKERS" \
    --stage pretrain \
    --groups baseline \
    "${force_args[@]}"

run_step cam_real_all \
  "$PYTHON" scripts/run_medmnist_formal_pipeline.py \
    "${common_args[@]}" \
    --workers "$RUN_WORKERS" \
    --stage cam_real \
    --groups baseline

launch_group() {
  local group_key="$1"
  local group_name="$2"
  local summary_prefix="$3"
  local log_path="$LOG_ROOT/${group_name}.log"
  echo "[$(date -Is)] START $group_name" | tee -a "$EXP_ROOT/RUN_STATUS.md"
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    "$PYTHON" scripts/run_medmnist_formal_pipeline.py \
      "${common_args[@]}" \
      --workers "$RUN_WORKERS" \
      --stage abc \
      --groups "$group_key" \
      --summary_prefix "$summary_prefix"
  ) > "$log_path" 2>&1 &
  echo $! > "$LOG_ROOT/${group_name}.pid"
}

launch_group baseline B_NCFM_T512 main_B_NCFM_T512
launch_group dam F_DAM_L012_W10_T512 main_F_DAM_L012_W10_T512
launch_group local D_LOCAL_LAM03_T512 main_D_LOCAL_LAM03_T512

status=0
for pid_file in "$LOG_ROOT"/*.pid; do
  group_name="$(basename "$pid_file" .pid)"
  pid="$(cat "$pid_file")"
  if wait "$pid"; then
    echo "[$(date -Is)] DONE  $group_name" | tee -a "$EXP_ROOT/RUN_STATUS.md"
  else
    echo "[$(date -Is)] FAIL  $group_name" | tee -a "$EXP_ROOT/RUN_STATUS.md"
    status=1
  fi
done

echo "[$(date -Is)] ALL_GROUPS_FINISHED status=$status" | tee -a "$EXP_ROOT/RUN_STATUS.md"
exit "$status"
