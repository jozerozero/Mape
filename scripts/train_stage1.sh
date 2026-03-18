#!/bin/bash
#SBATCH --job-name=mape_stage1      # 作业名称，在队列中显示的名称
#SBATCH --partition=faculty          # 分区名称，决定了作业在哪些机器上运行
#SBATCH --account=test-acc           # 账户名称，用于计费或权限控制
#SBATCH --qos=bgqos                  # 服务质量(Quality of Service)，影响优先级
#SBATCH --nodes=1                    # 申请的节点数量
#SBATCH --ntasks-per-node=1          # 每个节点运行的任务数
#SBATCH --gpus-per-node=8            # 每个节点申请的 GPU 数量
#SBATCH --cpus-per-task=128          # 每个任务申请的 CPU 核心数
#SBATCH --mem=240G                   # 申请的内存大小
#SBATCH --time=3-00:00:00            # 运行最长时间 (天-小时:分钟:秒)
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.out # 标准输出日志文件路径
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/slurm_tools/logs/%x-%j.err  # 错误输出日志文件路径
#SBATCH --export=ALL                 # 导出所有环境变量到作业中
#SBATCH --exclude=auh7-1b-gpu-[214-221,260-267,268-275,282-289],auh7-1b-gpu-259 # 排除有问题的节点

# 遇到任何错误立即退出脚本，防止错误级联
set -euo pipefail

# 获取脚本的绝对路径和所在目录
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-unset}"
echo "[$(date)] Starting single-node 8-GPU training job..."
echo "[$(date)] SCRIPT_PATH=${SCRIPT_PATH}"
echo "[$(date)] SCRIPT_DIR=${SCRIPT_DIR}"

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "[$(date)] Running outside sbatch allocation; falling back to local GPU detection."
fi

parse_slurm_gpu_count() {
  local spec="${1:-}"
  spec="${spec// /}"
  [[ -n "${spec}" ]] || return 1
  spec="${spec%%(*}"
  spec="${spec##*:}"
  [[ "${spec}" =~ ^[0-9]+$ ]] || return 1
  echo "${spec}"
}

count_visible_devices() {
  local visible="${1:-}"
  local count=0
  local dev
  local -a devices

  visible="${visible// /}"
  [[ -n "${visible}" ]] || return 1
  [[ "${visible}" != "NoDevFiles" ]] || return 1
  [[ "${visible}" != "none" ]] || return 1

  IFS=',' read -r -a devices <<< "${visible}"
  for dev in "${devices[@]}"; do
    [[ -n "${dev}" ]] && ((count += 1))
  done

  (( count > 0 )) || return 1
  echo "${count}"
}

detect_gpu_count() {
  local count

  if count=$(parse_slurm_gpu_count "${SLURM_GPUS_PER_NODE:-}"); then
    GPU_COUNT_SOURCE="SLURM_GPUS_PER_NODE"
    GPUS_PER_NODE="${count}"
    return 0
  fi

  if count=$(parse_slurm_gpu_count "${SLURM_GPUS_ON_NODE:-}"); then
    GPU_COUNT_SOURCE="SLURM_GPUS_ON_NODE"
    GPUS_PER_NODE="${count}"
    return 0
  fi

  if count=$(count_visible_devices "${CUDA_VISIBLE_DEVICES:-}"); then
    GPU_COUNT_SOURCE="CUDA_VISIBLE_DEVICES"
    GPUS_PER_NODE="${count}"
    return 0
  fi

  if count=$(count_visible_devices "${ROCR_VISIBLE_DEVICES:-}"); then
    GPU_COUNT_SOURCE="ROCR_VISIBLE_DEVICES"
    GPUS_PER_NODE="${count}"
    return 0
  fi

  if count=$(count_visible_devices "${HIP_VISIBLE_DEVICES:-}"); then
    GPU_COUNT_SOURCE="HIP_VISIBLE_DEVICES"
    GPUS_PER_NODE="${count}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      GPU_COUNT_SOURCE="nvidia-smi"
      GPUS_PER_NODE="${count}"
      return 0
    fi
  fi

  if command -v rocminfo >/dev/null 2>&1; then
    count=$(rocminfo 2>/dev/null | grep -Ec '^[[:space:]]*Name:[[:space:]]*gfx')
    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      GPU_COUNT_SOURCE="rocminfo"
      GPUS_PER_NODE="${count}"
      return 0
    fi
  fi

  GPU_COUNT_SOURCE="default"
  GPUS_PER_NODE="1"
}

########################################
# 环境设置 (加载依赖、Python环境等)
########################################
if ! command -v conda >/dev/null 2>&1; then
  had_nounset=0
  if [[ -o nounset ]]; then
    had_nounset=1
    set +u
  fi

  if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc"
  fi

  if (( had_nounset )); then
    set -u
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found after sourcing ${HOME}/.bashrc"
  exit 1
fi

source ~/slurm_tools/mi.sh
source "$(conda info --base)/etc/profile.d/conda.sh" # 加载 conda 命令
conda activate tabicl                                # 激活名为 'tabicl' 的虚拟环境

# 设置临时目录，防止写满系统默认的 /tmp
mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

########################################
# 训练路径与运行环境配置
########################################
# 设置优化器，默认为 muon
export OPTIMIZER=${OPTIMIZER:-muon}
# 项目根目录路径 (重要：如果你的路径不同，需要修改这里)
export PROJECT_HOME="/vast/users/guangyi.chen/causal_group/zijian.li/zh/Mape"
# 训练脚本的入口文件
export RUN_FILE="${PROJECT_HOME}/src/tabicl/train/run.py"
# 输出文件的根目录
export OUTPUT_ROOT="${PROJECT_HOME}/outputs/stage1"
# WandB 日志目录
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
# 模型检查点保存目录
export CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints"
# 将 src 目录添加到 PYTHONPATH，确保能导入 tabicl 包
export PYTHONPATH="${PROJECT_HOME}/src:${PYTHONPATH:-}"

# 创建必要的目录
mkdir -p "${WANDB_DIR}" "${CHECKPOINT_DIR}"

# 获取当前节点的 GPU 数量，优先使用 Slurm 分配信息
GPU_COUNT_SOURCE=""
GPUS_PER_NODE=""
detect_gpu_count

# 分布式通信 (NCCL) 相关设置，通常不需要修改
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}

echo "[$(date)] PROJECT_HOME=${PROJECT_HOME}"
echo "[$(date)] RUN_FILE=${RUN_FILE}"
echo "[$(date)] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[$(date)] WANDB_DIR=${WANDB_DIR}"
echo "[$(date)] GPUS_PER_NODE=${GPUS_PER_NODE} (source: ${GPU_COUNT_SOURCE})"
echo "[$(date)] OPTIMIZER=${OPTIMIZER}"

# 检查脚本中是否还残留着模板路径
if grep -n "/path/to/tabicl" "${SCRIPT_PATH}" >/dev/null 2>&1; then
  echo "[ERROR] ${SCRIPT_PATH} still contains template path /path/to/tabicl"
  exit 1
fi

if [[ "${SCRIPT_DIR}" != "${PROJECT_HOME}/scripts" ]]; then
  echo "[ERROR] Script is not being executed from the expected repository."
  echo "[ERROR] Expected script dir: ${PROJECT_HOME}/scripts"
  echo "[ERROR] Actual script dir:   ${SCRIPT_DIR}"
  exit 1
fi

if [[ ! -f "${RUN_FILE}" ]]; then
  echo "[ERROR] Training entrypoint not found: ${RUN_FILE}"
# torchrun: PyTorch 提供的分布式启动工具
# --nproc_per_node: 启动的进程数，通常等于 GPU 数量
  exit 1
fi

python - <<'PY'
import os
print(f"[PYTHONPATH_CHECK] {os.environ.get('PYTHONPATH', '')}")
PY

########################################
# 启动训练
########################################
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" "${RUN_FILE}" \
          --wandb_log True \
          --wandb_project TabICL \
          --wandb_name Stage1 \
          --wandb_dir "${WANDB_DIR}" \
          --wandb_mode offline \
          --device cuda \
          --dtype float16 \
          --np_seed 43 \
          --torch_seed 42 \
          --max_steps 160000 \
          --batch_size 256 \
          --micro_batch_size 4 \
          --lr 2e-4 \
          --optimizer "${OPTIMIZER}" \
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

echo "[$(date)] Job finished."
