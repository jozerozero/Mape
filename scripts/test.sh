#!/bin/bash
#SBATCH --job-name=gen_prior_only
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "[$(date)] Prior generation only (no model forward/backward)"

source ~/.bashrc
source ~/slurm_tools/mi.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabicl

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

export PROJECT_HOME="${PROJECT_HOME:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon}"
export PYTHONPATH="$PROJECT_HOME:$PYTHONPATH"

# -------- Generation controls --------
export PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/prior_data/stage1_int8}"
export NUM_BATCHES="${NUM_BATCHES:-1}"                     # 按你的要求，默认只生成1个batch
export ESTIMATE_TOTAL_BATCHES="${ESTIMATE_TOTAL_BATCHES:-160000}"
export SAVE_DTYPE="${SAVE_DTYPE:-int8}"                    # 目标把总量压到10TB以内
export PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"

echo "[$(date)] PROJECT_HOME=$PROJECT_HOME"
echo "[$(date)] PRIOR_SAVE_DIR=$PRIOR_SAVE_DIR"
echo "[$(date)] NUM_BATCHES=$NUM_BATCHES ESTIMATE_TOTAL_BATCHES=$ESTIMATE_TOTAL_BATCHES"
echo "[$(date)] SAVE_DTYPE=$SAVE_DTYPE PRIOR_TYPE=$PRIOR_TYPE"

python "$PROJECT_HOME/src/tabicl/prior/genload.py" \
  --save_dir "$PRIOR_SAVE_DIR" \
  --np_seed 43 \
  --torch_seed 42 \
  --num_batches "$NUM_BATCHES" \
  --estimate_total_batches "$ESTIMATE_TOTAL_BATCHES" \
  --batch_size 256 \
  --batch_size_per_gp 4 \
  --min_features 2 \
  --max_features 100 \
  --max_classes 10 \
  --max_seq_len 2048 \
  --min_train_size 0.1 \
  --max_train_size 0.9 \
  --prior_type "$PRIOR_TYPE" \
  --save_dtype "$SAVE_DTYPE" \
  --n_jobs 32 \
  --num_threads_per_generate 1 \
  --device cpu

echo "[$(date)] Generation finished."
