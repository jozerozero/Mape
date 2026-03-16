#!/bin/bash
#SBATCH --job-name=talent_eval_multi
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "[$(date)] Start multi-node talent evaluation..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabicl

export PROJECT_HOME="${PROJECT_HOME:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon}"
export MODELS_DIR="${MODELS_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon/stabe1/checkpoint/dir2}"
export DATA_ROOT="${DATA_ROOT:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/datasets}"
export OUTDIR_ROOT="${OUTDIR_ROOT:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon/evaluation_results_multinode/${SLURM_JOB_ID}}"

# Forward talent_eval_online.py knobs if provided.
export CLF_N_ESTIMATORS="${CLF_N_ESTIMATORS:-32}"
export CLF_BATCH_SIZE="${CLF_BATCH_SIZE:-8}"
export CLF_N_JOBS="${CLF_N_JOBS:-1}"
export CPU_THREADS="${CPU_THREADS:-1}"
export POLL_SEC="${POLL_SEC:-5}"
export STABLE_SEC="${STABLE_SEC:-1}"
export IDLE_EXIT_SEC="${IDLE_EXIT_SEC:-120}"
export STEP_MOD="${STEP_MOD:-1}"

mkdir -p "${OUTDIR_ROOT}"
export ALL_CKPTS_FILE="${OUTDIR_ROOT}/all_ckpts_sorted.txt"

echo "[$(date)] PROJECT_HOME=${PROJECT_HOME}"
echo "[$(date)] MODELS_DIR=${MODELS_DIR}"
echo "[$(date)] DATA_ROOT=${DATA_ROOT}"
echo "[$(date)] OUTDIR_ROOT=${OUTDIR_ROOT}"

python - "${MODELS_DIR}" "${ALL_CKPTS_FILE}" <<'PY'
import re
import sys
from pathlib import Path

models_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])

if not models_dir.exists():
    raise FileNotFoundError(f"models_dir not found: {models_dir}")

allowed = {".ckpt", ".pt", ".pth"}
files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]

def last_int(stem: str):
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None

def sort_key(p: Path):
    step = last_int(p.stem)
    if step is not None:
        return (0, step, p.stem)
    return (1, int(p.stat().st_mtime), p.stem)

ordered = sorted(files, key=sort_key)
out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open("w", encoding="utf-8") as f:
    for p in ordered:
        f.write(str(p.resolve()) + "\n")
print(f"discovered={len(ordered)}")
PY

TOTAL_CKPTS="$(wc -l < "${ALL_CKPTS_FILE}" | tr -d '[:space:]')"
if [[ "${TOTAL_CKPTS}" == "0" ]]; then
  echo "[$(date)] No checkpoints found in ${MODELS_DIR}"
  exit 1
fi
echo "[$(date)] Total checkpoints to evaluate: ${TOTAL_CKPTS}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  bash "${PROJECT_HOME}/scripts/talent_eval_multinode_worker.sh"

python "${PROJECT_HOME}/scripts/merge_talent_summaries.py" \
  --input_glob "${OUTDIR_ROOT}/node_*/eval/all_models_summary.tsv" \
  --output "${OUTDIR_ROOT}/all_models_summary_sorted.tsv"

echo "[$(date)] Finished. Sorted summary: ${OUTDIR_ROOT}/all_models_summary_sorted.tsv"
