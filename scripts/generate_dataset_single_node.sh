#!/bin/bash
#SBATCH --job-name=prior_gen_1n8w
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=1500G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] JobID: ${SLURM_JOB_ID:-N/A}"
echo "[$(date)] NodeList: ${SLURM_JOB_NODELIST:-N/A}"
echo "[$(date)] Prior generation only (no model forward/backward)."

source ~/.bashrc
source ~/slurm_tools/mi.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabicl

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

# Avoid nested BLAS/OpenMP oversubscription inside joblib workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PROJECT_HOME="${PROJECT_HOME:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl_muon}"
export PYTHONPATH="$PROJECT_HOME:$PYTHONPATH"

# ---------------------------- User knobs -----------------------------
export PRIOR_SAVE_DIR="${PRIOR_SAVE_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/prior_data/stage1_int8}"
export TOTAL_BATCHES="${TOTAL_BATCHES:-160000}"
export SAVE_DTYPE="${SAVE_DTYPE:-int8}"          # float32/float16/bfloat16/int16/int8
export PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"       # mlp_scm/tree_scm/mix_scm
export BATCH_SIZE="${BATCH_SIZE:-256}"
export BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-4}"
export MIN_FEATURES="${MIN_FEATURES:-2}"
export MAX_FEATURES="${MAX_FEATURES:-100}"
export MAX_CLASSES="${MAX_CLASSES:-10}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
export MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-0.1}"
export MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-0.9}"
export N_JOBS_PER_TASK="${N_JOBS_PER_TASK:-16}"  # recommended == cpus-per-task
export THREADS_PER_GENERATE="${THREADS_PER_GENERATE:-1}"
export NP_SEED="${NP_SEED:-43}"
export TORCH_SEED="${TORCH_SEED:-42}"

echo "[$(date)] PROJECT_HOME=$PROJECT_HOME"
echo "[$(date)] PRIOR_SAVE_DIR=$PRIOR_SAVE_DIR"
echo "[$(date)] TOTAL_BATCHES=$TOTAL_BATCHES SAVE_DTYPE=$SAVE_DTYPE PRIOR_TYPE=$PRIOR_TYPE"
echo "[$(date)] BATCH_SIZE=$BATCH_SIZE N_JOBS_PER_TASK=$N_JOBS_PER_TASK THREADS_PER_GENERATE=$THREADS_PER_GENERATE"

mkdir -p "$PRIOR_SAVE_DIR"

# Use all 8 tasks on the single node; each task writes a disjoint batch range.
srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=16 bash -lc '
  set -euo pipefail
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tabicl

  RANK=${SLURM_PROCID}
  WORLD=${SLURM_NTASKS}
  BASE=$(( TOTAL_BATCHES / WORLD ))
  REM=$(( TOTAL_BATCHES % WORLD ))

  if [ "$RANK" -lt "$REM" ]; then
    NUM_LOCAL=$(( BASE + 1 ))
    START=$(( RANK * (BASE + 1) ))
  else
    NUM_LOCAL=$BASE
    START=$(( REM * (BASE + 1) + (RANK - REM) * BASE ))
  fi

  if [ "$NUM_LOCAL" -le 0 ]; then
    echo "[rank ${RANK}] no batches assigned, skip."
    exit 0
  fi

  export JOBLIB_TEMP_FOLDER="/tmp/$USER/joblib_${SLURM_JOB_ID}_${RANK}"
  mkdir -p "$JOBLIB_TEMP_FOLDER"

  echo "[rank ${RANK}] START=${START}, NUM_LOCAL=${NUM_LOCAL}"

  python "$PROJECT_HOME/src/tabicl/prior/genload.py" \
    --save_dir "$PRIOR_SAVE_DIR" \
    --np_seed $((NP_SEED + RANK)) \
    --torch_seed $((TORCH_SEED + RANK)) \
    --num_batches "$NUM_LOCAL" \
    --resume_from "$START" \
    --batch_size "$BATCH_SIZE" \
    --batch_size_per_gp "$BATCH_SIZE_PER_GP" \
    --min_features "$MIN_FEATURES" \
    --max_features "$MAX_FEATURES" \
    --max_classes "$MAX_CLASSES" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --min_train_size "$MIN_TRAIN_SIZE" \
    --max_train_size "$MAX_TRAIN_SIZE" \
    --prior_type "$PRIOR_TYPE" \
    --save_dtype "$SAVE_DTYPE" \
    --n_jobs "$N_JOBS_PER_TASK" \
    --num_threads_per_generate "$THREADS_PER_GENERATE" \
    --device cpu
'

# Optional: estimate final size from one batch file if report not present.
if [ ! -f "$PRIOR_SAVE_DIR/size_estimate.json" ]; then
  FIRST_FILE=$(ls "$PRIOR_SAVE_DIR"/batch_*.pt 2>/dev/null | head -n 1 || true)
  if [ -n "$FIRST_FILE" ]; then
    BYTES=$(stat -c%s "$FIRST_FILE")
    python - "$BYTES" "$TOTAL_BATCHES" <<'PY'
import sys
b = float(sys.argv[1]); n = int(sys.argv[2])
est = b * n
print(f"[estimate] first_batch={b/1024/1024:.2f}MB, total={est/1024**4:.3f}TB")
PY
  fi
fi

echo "[$(date)] Prior generation finished."
