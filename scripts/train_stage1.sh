#!/bin/bash
#SBATCH --job-name=mape_stage1_v2
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}"

source ~/.bashrc
source ~/slurm_tools/mi.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabicl

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

PROJECT_HOME=${PROJECT_HOME:-/vast/users/guangyi.chen/causal_group/zijian.li/zh/Mape}
RUN_FILE="${PROJECT_HOME}/src/tabicl/train/run.py"
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_HOME}/outputs/stage1_v2}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints}
WANDB_DIR=${WANDB_DIR:-${OUTPUT_ROOT}/wandb}

mkdir -p "${CKPT_DIR}" "${WANDB_DIR}"

export PYTHONPATH="${PROJECT_HOME}/src:${PYTHONPATH:-}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$(( NUM_NODES * GPUS_PER_NODE ))

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

echo "[$(date)] PROJECT_HOME=${PROJECT_HOME}"
echo "[$(date)] CKPT_DIR=${CKPT_DIR}"
echo "[$(date)] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[$(date)] NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"

srun --ntasks="${NUM_NODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail

  source ~/.bashrc
  source ~/slurm_tools/mi.sh
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tabicl

  NODE_RANK=${SLURM_NODEID}
  GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}

  echo "[${HOSTNAME}] NODE_RANK=${NODE_RANK}"
  echo "[${HOSTNAME}] GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "[${HOSTNAME}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes='"${NUM_NODES}"' \
    --node_rank="${NODE_RANK}" \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    '"${RUN_FILE}"' \
    --wandb_log True \
    --wandb_project Mape \
    --wandb_name stage1-v2 \
    --wandb_dir '"${WANDB_DIR}"' \
    --wandb_mode offline \
    --device cuda \
    --dtype bfloat16 \
    --amp True \
    --np_seed 43 \
    --torch_seed 42 \
    --max_steps 160000 \
    --batch_size 256 \
    --micro_batch_size 4 \
    --lr 2e-4 \
    --optimizer muon \
    --scheduler cosine_warmup \
    --cosine_lr_end 2e-5 \
    --warmup_proportion 0.05 \
    --gradient_clipping 1.0 \
    --weight_decay 0.05 \
    --prior_type mix_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --max_seq_len 2048 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --col_affine True \
    --col_feature_group same \
    --col_feature_group_size 3 \
    --col_target_aware True \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 5000 \
    --row_last_cls_only True \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --dropout 0.0 \
    --norm_first True \
    --bias_free_ln False \
    --arch_mode v2 \
    --checkpoint_dir '"${CKPT_DIR}"' \
    --save_temp_every 50 \
    --save_perm_every 5000 \
    --freeze_col False \
    --freeze_row False \
    --freeze_icl False
'

echo "[$(date)] Job finished."
