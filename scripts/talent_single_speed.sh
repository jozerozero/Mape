#!/bin/bash
#SBATCH --job-name=talent_single_speed
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=00:40:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabicl

PROJECT_HOME=${PROJECT_HOME:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon}
MODELS_DIR=${MODELS_DIR:-${PROJECT_HOME}/stabe1/checkpoint/dir2}
OUTDIR_ROOT=${OUTDIR_ROOT:-${PROJECT_HOME}/evaluation_results_multinode}

cd "${PROJECT_HOME}"

if [[ -n "${MODEL_PATH:-}" ]]; then
  CKPT="${MODEL_PATH}"
else
  CKPT=$(ls "${MODELS_DIR}"/*.ckpt 2>/dev/null | sort | head -n 1)
fi

if [[ -z "${CKPT:-}" ]]; then
  echo "No checkpoint found under ${MODELS_DIR}"
  exit 2
fi

OUTDIR="${OUTDIR_ROOT}/single_speed_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "CKPT=${CKPT}"
echo "OUTDIR=${OUTDIR}"

/usr/bin/time -p python scripts/talent_eval_online.py \
  --model_path "${CKPT}" \
  --outdir "${OUTDIR}"
