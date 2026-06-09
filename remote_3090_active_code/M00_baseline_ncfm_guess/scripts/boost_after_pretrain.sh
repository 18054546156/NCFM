#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/NCFM_methods_T512_20260528}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-tmp/ncfm_t512_main_20260528}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
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

LOG_ROOT="$EXP_ROOT/launcher_logs"
mkdir -p "$LOG_ROOT"

echo "[$(date -Is)] BOOST waiting for all pretrain checkpoints" | tee -a "$EXP_ROOT/RUN_STATUS.md"
while true; do
  ready=1
  for dataset in pneumoniamnist bloodmnist pathmnist; do
    expected="$EXP_ROOT/checkpoints/pretrain/$dataset/premodel$((MODEL_NUM - 1))_trained.pth.tar"
    if [[ ! -f "$expected" ]]; then
      ready=0
      break
    fi
  done
  [[ "$ready" == "1" ]] && break
  sleep 20
done

sleep 10
echo "[$(date -Is)] BOOST pretrain complete; replacing old launcher with abc_parallel" | tee -a "$EXP_ROOT/RUN_STATUS.md"

old_pid="$(cat "$EXP_ROOT/launcher.pid" 2>/dev/null || true)"
if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
  pgrep -P "$old_pid" | xargs -r kill || true
  sleep 2
  kill "$old_pid" || true
fi

for pid in $(pgrep -f "run_medmnist_formal_pipeline.py --exp_root $EXP_ROOT" || true); do
  cmd="$(ps -p "$pid" -o args= || true)"
  if [[ "$cmd" != *"--stage pretrain"* ]]; then
    kill "$pid" || true
  fi
done

cd "$PROJECT_ROOT"
nohup env \
  PYTHON="$PYTHON" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  EXP_ROOT="$EXP_ROOT" \
  GPU="$GPU" \
  IPC="$IPC" \
  NITER="$NITER" \
  MODEL_NUM="$MODEL_NUM" \
  PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS" \
  EVAL_EPOCHS="$EVAL_EPOCHS" \
  EPOCH_EVAL_INTERVAL="$EPOCH_EVAL_INTERVAL" \
  BATCH_SIZE="$BATCH_SIZE" \
  BATCH_REAL="$BATCH_REAL" \
  RUN_WORKERS="$RUN_WORKERS" \
  CAM_SAMPLES="$CAM_SAMPLES" \
  MAX_PARALLEL="$MAX_PARALLEL" \
  bash scripts/launch_t512_abc_parallel.sh \
  > "$LOG_ROOT/nohup_abc_parallel.log" 2>&1 &

echo $! > "$EXP_ROOT/abc_parallel.pid"
echo "[$(date -Is)] BOOST abc_parallel pid=$(cat "$EXP_ROOT/abc_parallel.pid")" | tee -a "$EXP_ROOT/RUN_STATUS.md"
