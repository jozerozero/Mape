#!/bin/bash
#SBATCH --job-name=mape_stage1
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
#SBATCH --exclude=auh7-1b-gpu-[214-221,260-267,268-275,282-289],auh7-1b-gpu-259
#SBATCH --nodelist=auh7-1b-gpu-[222-229,234-241,306-313]

echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "[$(date)] Starting distributed training job..."

########################################
# 环境设置
########################################
source ~/.bashrc
source ~/slurm_tools/mi.sh
conda activate tabicl
mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

########################################
# 分布式环境变量
########################################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NUM_NODES=${SLURM_NNODES}
WORLD_SIZE=$(( NUM_NODES * GPUS_PER_NODE ))

export MASTER_ADDR MASTER_PORT WORLD_SIZE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
export OPTIMIZER=${OPTIMIZER:-muon}

export PROJECT_HOME="/vast/users/guangyi.chen/causal_group/zijian.li/zh/Mape"
export RUN_FILE="${PROJECT_HOME}/src/tabicl/train/run.py"
export OUTPUT_ROOT="${PROJECT_HOME}/outputs/stage1"
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints"
export PYTHONPATH="${PROJECT_HOME}/src:${PYTHONPATH}"

mkdir -p "${WANDB_DIR}" "${CHECKPOINT_DIR}"

echo "[$(date)] PROJECT_HOME=$PROJECT_HOME"
echo "[$(date)] RUN_FILE=$RUN_FILE"
echo "[$(date)] CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "[$(date)] OPTIMIZER=$OPTIMIZER"

########################################
# 启动训练
########################################
srun --ntasks=${NUM_NODES} --ntasks-per-node=1 bash -lc '
  NODE_RANK=${SLURM_NODEID}
  echo "[${HOSTNAME}] NODE_RANK=${NODE_RANK}"

  GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
  echo "[${HOSTNAME}] GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "[${HOSTNAME}] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tabicl

  torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes='"${NUM_NODES}"' --node_rank=${NODE_RANK} --master_addr='"${MASTER_ADDR}"' --master_port='"${MASTER_PORT}"' "${RUN_FILE}" \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir "${WANDB_DIR}" \
            --wandb_mode offline \
            --device cuda \
            --dtype float32 \
            --np_seed 43 \
            --torch_seed 42 \
            --max_steps 160000 \
            --batch_size 256 \
            --micro_batch_size 4 \
            --lr 2e-4 \
            --optimizer '"${OPTIMIZER}"' \
            --scheduler cosine_warmup \
            --cosine_lr_end 2e-5 \
            --warmup_proportion 0.05 \
            --gradient_clipping 1.0 \
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
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 5000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --save_temp_every 50 \
            --save_perm_every 5000 \
            --only_load_model True \
            --freeze_col False \
            --freeze_row False
'

echo "[$(date)] Job finished."
