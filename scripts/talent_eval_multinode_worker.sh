#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabicl

if [[ -z "${PROJECT_HOME:-}" || -z "${OUTDIR_ROOT:-}" || -z "${ALL_CKPTS_FILE:-}" || -z "${DATA_ROOT:-}" ]]; then
  echo "Missing required env vars. Need PROJECT_HOME/OUTDIR_ROOT/ALL_CKPTS_FILE/DATA_ROOT"
  exit 2
fi

NODE_RANK="${SLURM_NODEID:-0}"
NUM_NODES="${SLURM_NNODES:-1}"
NODE_ROOT="${OUTDIR_ROOT}/node_${NODE_RANK}"
NODE_LINK_DIR="${NODE_ROOT}/ckpts"
NODE_OUTDIR="${NODE_ROOT}/eval"
NODE_ASSIGNED_FILE="${NODE_ROOT}/assigned_ckpts.txt"

mkdir -p "${NODE_LINK_DIR}" "${NODE_OUTDIR}"
rm -f "${NODE_LINK_DIR}"/*

python - "${ALL_CKPTS_FILE}" "${NODE_RANK}" "${NUM_NODES}" "${NODE_LINK_DIR}" "${NODE_ASSIGNED_FILE}" <<'PY'
import os
import sys
from pathlib import Path

all_file = Path(sys.argv[1])
rank = int(sys.argv[2])
world = int(sys.argv[3])
link_dir = Path(sys.argv[4])
assigned_file = Path(sys.argv[5])

ckpts = [line.strip() for line in all_file.read_text(encoding="utf-8").splitlines() if line.strip()]
assigned = []
for idx, ck in enumerate(ckpts):
    if idx % world != rank:
        continue
    src = Path(ck)
    # Prefix index to avoid potential filename collision across folders.
    dst = link_dir / f"{idx:06d}__{src.name}"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)
    assigned.append(str(src))

assigned_file.parent.mkdir(parents=True, exist_ok=True)
assigned_file.write_text("\n".join(assigned) + ("\n" if assigned else ""), encoding="utf-8")
print(len(assigned))
PY

ASSIGNED_COUNT="$(wc -l < "${NODE_ASSIGNED_FILE}" | tr -d '[:space:]')"
echo "[$(date)] [node ${NODE_RANK}] assigned checkpoints: ${ASSIGNED_COUNT}"

if [[ "${ASSIGNED_COUNT}" == "0" ]]; then
  echo "[$(date)] [node ${NODE_RANK}] no work, exit."
  exit 0
fi

# Force single-GPU mode per node for talent_eval_online.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH:-}"

cd "${PROJECT_HOME}"
python scripts/talent_eval_online.py \
  --models_dir "${NODE_LINK_DIR}" \
  --data_root "${DATA_ROOT}" \
  --outdir "${NODE_OUTDIR}" \
  --poll_sec "${POLL_SEC:-5}" \
  --stable_sec "${STABLE_SEC:-1}" \
  --idle_exit_sec "${IDLE_EXIT_SEC:-120}" \
  --step_mod "${STEP_MOD:-1}" \
  --clf_n_estimators "${CLF_N_ESTIMATORS:-32}" \
  --clf_batch_size "${CLF_BATCH_SIZE:-8}" \
  --clf_n_jobs "${CLF_N_JOBS:-1}" \
  --cpu_threads "${CPU_THREADS:-1}"

echo "[$(date)] [node ${NODE_RANK}] done."
