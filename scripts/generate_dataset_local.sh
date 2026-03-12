#!/usr/bin/env bash

# Local (non-Slurm) prior dataset generation.
# Run this script after SSH-ing into the target machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_HOME="${PROJECT_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional conda activation (skip if unavailable).
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV:-tabicl}" || true
fi

# Avoid nested BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PYTHONPATH="${PROJECT_HOME}/src:${PROJECT_HOME}:${PYTHONPATH:-}"
# Safety: force CPU-only generation path to avoid any accidental model/GPU training workflow.
export CUDA_VISIBLE_DEVICES=""

# ---------------------------- User knobs -----------------------------
PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/prior_data/stage1_int8}"
TOTAL_BATCHES="${TOTAL_BATCHES:-160000}"
N_WORKERS="${N_WORKERS:-8}"                      # e.g. 8 workers on 8 GPUs / 8 NUMA domains
SAVE_DTYPE="${SAVE_DTYPE:-int8}"                 # float32/float16/bfloat16/int16/int8
PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"              # mlp_scm/tree_scm/mix_scm
BATCH_SIZE="${BATCH_SIZE:-256}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-4}"
MIN_FEATURES="${MIN_FEATURES:-2}"
MAX_FEATURES="${MAX_FEATURES:-100}"
MAX_CLASSES="${MAX_CLASSES:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-0.1}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-0.9}"
THREADS_PER_GENERATE="${THREADS_PER_GENERATE:-1}"
NP_SEED="${NP_SEED:-43}"
TORCH_SEED="${TORCH_SEED:-42}"
LOG_DIR="${LOG_DIR:-${PRIOR_SAVE_DIR}/logs}"

CPU_TOTAL="$(nproc)"
if [[ -z "${N_JOBS_PER_WORKER:-}" ]]; then
  N_JOBS_PER_WORKER="$(( CPU_TOTAL / N_WORKERS ))"
  if [[ "${N_JOBS_PER_WORKER}" -lt 1 ]]; then
    N_JOBS_PER_WORKER=1
  fi
fi
EXPECTED_CPU="$(( N_WORKERS * N_JOBS_PER_WORKER * THREADS_PER_GENERATE ))"

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] PROJECT_HOME=${PROJECT_HOME}"
echo "[$(date)] PRIOR_SAVE_DIR=${PRIOR_SAVE_DIR}"
echo "[$(date)] TOTAL_BATCHES=${TOTAL_BATCHES}, N_WORKERS=${N_WORKERS}"
echo "[$(date)] SAVE_DTYPE=${SAVE_DTYPE}, PRIOR_TYPE=${PRIOR_TYPE}"
echo "[$(date)] CPU_TOTAL=${CPU_TOTAL}, N_JOBS_PER_WORKER=${N_JOBS_PER_WORKER}, THREADS_PER_GENERATE=${THREADS_PER_GENERATE}"
echo "[$(date)] EXPECTED_CPU_LOAD=${EXPECTED_CPU}"
echo "[$(date)] Safety check: this script only runs src/tabicl/prior/genload.py with --device cpu."
if [[ "${EXPECTED_CPU}" -gt "${CPU_TOTAL}" ]]; then
  echo "[$(date)] Warning: expected CPU load > available cores, consider lowering N_WORKERS or N_JOBS_PER_WORKER."
fi

mkdir -p "${PRIOR_SAVE_DIR}" "${LOG_DIR}"
START_COUNT="$(find "${PRIOR_SAVE_DIR}" -maxdepth 1 -name 'batch_*.pt' | wc -l | tr -d ' ')"
START_TS="$(date +%s)"

# Initialize metadata once.
"${PYTHON_BIN}" "${PROJECT_HOME}/src/tabicl/prior/genload.py" \
  --save_dir "${PRIOR_SAVE_DIR}" \
  --np_seed "${NP_SEED}" \
  --torch_seed "${TORCH_SEED}" \
  --num_batches 0 \
  --resume_from 0 \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_per_gp "${BATCH_SIZE_PER_GP}" \
  --min_features "${MIN_FEATURES}" \
  --max_features "${MAX_FEATURES}" \
  --max_classes "${MAX_CLASSES}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --min_train_size "${MIN_TRAIN_SIZE}" \
  --max_train_size "${MAX_TRAIN_SIZE}" \
  --prior_type "${PRIOR_TYPE}" \
  --save_dtype "${SAVE_DTYPE}" \
  --n_jobs 1 \
  --num_threads_per_generate 1 \
  --device cpu >/dev/null 2>&1 || true

pids=()
for ((rank = 0; rank < N_WORKERS; rank++)); do
  base=$(( TOTAL_BATCHES / N_WORKERS ))
  rem=$(( TOTAL_BATCHES % N_WORKERS ))
  if [[ "${rank}" -lt "${rem}" ]]; then
    num_local=$(( base + 1 ))
    start=$(( rank * (base + 1) ))
  else
    num_local=${base}
    start=$(( rem * (base + 1) + (rank - rem) * base ))
  fi

  if [[ "${num_local}" -le 0 ]]; then
    continue
  fi

  (
    set -euo pipefail
    export JOBLIB_TEMP_FOLDER="/tmp/${USER}/joblib_prior_${$}_${rank}"
    mkdir -p "${JOBLIB_TEMP_FOLDER}"

    echo "[$(date)] [worker ${rank}] start=${start}, num_local=${num_local}"
    "${PYTHON_BIN}" "${PROJECT_HOME}/src/tabicl/prior/genload.py" \
      --save_dir "${PRIOR_SAVE_DIR}" \
      --np_seed "$(( NP_SEED + rank ))" \
      --torch_seed "$(( TORCH_SEED + rank ))" \
      --num_batches "${num_local}" \
      --resume_from "${start}" \
      --batch_size "${BATCH_SIZE}" \
      --batch_size_per_gp "${BATCH_SIZE_PER_GP}" \
      --min_features "${MIN_FEATURES}" \
      --max_features "${MAX_FEATURES}" \
      --max_classes "${MAX_CLASSES}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --min_train_size "${MIN_TRAIN_SIZE}" \
      --max_train_size "${MAX_TRAIN_SIZE}" \
      --prior_type "${PRIOR_TYPE}" \
      --save_dtype "${SAVE_DTYPE}" \
      --n_jobs "${N_JOBS_PER_WORKER}" \
      --num_threads_per_generate "${THREADS_PER_GENERATE}" \
      --device cpu
    echo "[$(date)] [worker ${rank}] done"
  ) >"${LOG_DIR}/worker_${rank}.log" 2>&1 &

  pids+=("$!")
done

format_hms() {
  local total="$1"
  if [[ "${total}" -lt 0 ]]; then
    total=0
  fi
  local h=$(( total / 3600 ))
  local m=$(( (total % 3600) / 60 ))
  local s=$(( total % 60 ))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${s}"
}

print_progress() {
  local done_now="$1"
  local total="$2"
  local elapsed="$3"
  local width=40
  local pct=0
  if [[ "${total}" -gt 0 ]]; then
    pct=$(( done_now * 100 / total ))
  fi
  if [[ "${pct}" -gt 100 ]]; then
    pct=100
  fi
  local fill=$(( pct * width / 100 ))
  local empty=$(( width - fill ))
  local bar
  bar="$(printf "%${fill}s" "" | tr ' ' '#')$(printf "%${empty}s" "" | tr ' ' '-')"

  local eta_str="--:--:--"
  if [[ "${done_now}" -gt 0 && "${total}" -gt "${done_now}" ]]; then
    local eta=$(( elapsed * (total - done_now) / done_now ))
    eta_str="$(format_hms "${eta}")"
  elif [[ "${total}" -le "${done_now}" ]]; then
    eta_str="00:00:00"
  fi

  printf "\r[progress] [%s] %3d%%  %d/%d  elapsed=%s  eta=%s" \
    "${bar}" "${pct}" "${done_now}" "${total}" "$(format_hms "${elapsed}")" "${eta_str}"
}

# Progress monitor (global): based on number of batch_*.pt files produced.
while true; do
  now_count="$(find "${PRIOR_SAVE_DIR}" -maxdepth 1 -name 'batch_*.pt' | wc -l | tr -d ' ')"
  produced=$(( now_count - START_COUNT ))
  if [[ "${produced}" -lt 0 ]]; then
    produced=0
  fi
  if [[ "${produced}" -gt "${TOTAL_BATCHES}" ]]; then
    produced="${TOTAL_BATCHES}"
  fi
  now_ts="$(date +%s)"
  elapsed=$(( now_ts - START_TS ))
  print_progress "${produced}" "${TOTAL_BATCHES}" "${elapsed}"

  all_done=1
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      all_done=0
      break
    fi
  done
  if [[ "${all_done}" -eq 1 ]]; then
    break
  fi
  sleep 5
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

# Final progress line
final_count="$(find "${PRIOR_SAVE_DIR}" -maxdepth 1 -name 'batch_*.pt' | wc -l | tr -d ' ')"
final_done=$(( final_count - START_COUNT ))
if [[ "${final_done}" -lt 0 ]]; then
  final_done=0
fi
if [[ "${final_done}" -gt "${TOTAL_BATCHES}" ]]; then
  final_done="${TOTAL_BATCHES}"
fi
final_elapsed=$(( $(date +%s) - START_TS ))
print_progress "${final_done}" "${TOTAL_BATCHES}" "${final_elapsed}"
echo

if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date)] One or more workers failed. Check logs in ${LOG_DIR}."
  exit 1
fi

FIRST_FILE="$(ls "${PRIOR_SAVE_DIR}"/batch_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -n "${FIRST_FILE}" ]]; then
  bytes="$(stat -c%s "${FIRST_FILE}")"
  "${PYTHON_BIN}" - "$bytes" "$TOTAL_BATCHES" <<'PY'
import sys
b = float(sys.argv[1]); n = int(sys.argv[2])
est = b * n
print(f"[estimate] first_batch={b/1024/1024:.2f}MB, total={est/1024**4:.3f}TB")
PY
fi

echo "[$(date)] Prior generation finished successfully."
