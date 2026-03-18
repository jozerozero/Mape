# This script is used to train TabICL for the first stage of the curriculum learning

PROJECT_HOME="/vast/users/guangyi.chen/causal_group/zijian.li/zh/Mape"
RUN_FILE="${PROJECT_HOME}/src/tabicl/train/run.py"
PRIOR_GEN_FILE="${PROJECT_HOME}/src/tabicl/prior/genload.py"
OUTPUT_ROOT="${PROJECT_HOME}/outputs/stage1_v2"
WANDB_DIR="${OUTPUT_ROOT}/wandb"
CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints"
PRIOR_DIR="${OUTPUT_ROOT}/prior"

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=8 "${RUN_FILE}" \
            --wandb_log True \
            --wandb_project Mape \
            --wandb_name stage1-v2 \
            --wandb_dir "${WANDB_DIR}" \
            --wandb_mode offline \
            --device cuda \
            --dtype bfloat16 \
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
            --norm_first True \
            --bias_free_ln False \
            --arch_mode v2 \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --save_temp_every 50 \
            --save_perm_every 5000


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
python "${PRIOR_GEN_FILE}" \
    --save_dir "${PRIOR_DIR}" \
    --np_seed 43 \
    --torch_seed 42 \
    --num_batches 160000 \
    --resume_from 0 \
    --batch_size 256 \
    --batch_size_per_gp 4 \
    --prior_type mix_scm \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --max_seq_len 2048 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --n_jobs -1 \
    --num_threads_per_generate 1 \
    --device cpu

# Loading from disk and training
torchrun --standalone --nproc_per_node=8 "${RUN_FILE}" \
            --wandb_log True \
            --wandb_project Mape \
            --wandb_name stage1-v2 \
            --wandb_dir "${WANDB_DIR}" \
            --wandb_mode offline \
            --device cuda \
            --dtype bfloat16 \
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
            --prior_dir "${PRIOR_DIR}" \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cpu \
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
            --norm_first True \
            --bias_free_ln False \
            --arch_mode v2 \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --save_temp_every 50 \
            --save_perm_every 5000 \
            --only_load_model True
