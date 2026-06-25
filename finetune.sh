## launch command for this script:
# cd ~/H_RDT && conda activate hrdt && \
# mkdir -p ./checkpoints/T_R2_tableware_stack_bowls_two && \
# bash finetune.sh 2>&1 | tee ./checkpoints/FT_dense90backbone_seed42/train_log.txt

# Remove/disable cluster-specific NCCL settings:
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=bond0

#local NCCL setting
unset NCCL_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
#GPUs
export CUDA_VISIBLE_DEVICES=0,1
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
#export CUTLASS_PATH="/data/lingxuan/cutlass"

export RUN="FT_dense90backbone_seed42"
export PRETRAINED_CHECKPOINT="./checkpoints/adjusting_bottle/pretrains/S2_denseLSS/pytorch_model.bin"
export OUT_RUN="./checkpoints/adjusting_bottle/finetunes/$RUN"
export OUTPUT_DIR="$OUT_RUN"
export WANDB_PROJECT="$RUN"


export VISION_ENCODER_NAME="dino-siglip"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# 
log_manifest () {
  local OUT="$1" SCRIPT="$2" M="$1/RUN_MANIFEST.txt"
  mkdir -p "$OUT"; cp "$SCRIPT" "$OUT/$(basename "$SCRIPT").asrun"
  { echo "=== $(date -u) ==="; echo "host: $(hostname)  cwd: $(pwd)"
    echo "git: $(git rev-parse HEAD) [$(git rev-parse --abbrev-ref HEAD)]"
    echo "--- uncommitted ---"; git status --porcelain
    echo "EGODEX_DATA_ROOT=$EGODEX_DATA_ROOT"; echo "EGODEX_TILE_DENSE=$EGODEX_TILE_DENSE"
    echo "PRETRAINED_CHECKPOINT=$PRETRAINED_CHECKPOINT"; echo "OUTPUT_DIR=$OUTPUT_DIR"
    find "$EGODEX_DATA_ROOT" -name '*_dense.pt' 2>/dev/null | wc -l | sed 's/^/dense_pt_files: /'
    python -c "import torch,deepspeed,transformers,accelerate as a;print('torch',torch.__version__,'ds',deepspeed.__version__,'tf',transformers.__version__,'acc',a.__version__)"
  } > "$M"; pip freeze > "$OUT/pip_freeze.txt"; echo "manifest -> $M"
}

log_manifest "$OUT_RUN" "$0"


#note: train_batch_size is per device

# Original setup: accelerate launch --main_process_port 29500 main.py \
# --deepspeed="./configs/zero1.json" \
# --max_train_steps=1000000 \
#WANDB_RUN_ID="5ogbe6z1" WANDB_RESUME="allow" 
accelerate launch main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero2.json" \
    --config_path="configs/hrdt_finetune.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=1 \
    --max_train_steps=22000 \
    --checkpointing_period=3000 \
    --sample_period=500 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=2 \
    --dataset_type="robotwin_agilex" \
    --report_to=wandb \
    --upsample_rate=3 \
    --image_aug \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --max_robot_episodes=50 \
    --seed=42 \
    --pretrained_backbone_path="$PRETRAINED_CHECKPOINT" \
    --task_name="adjust_bottle" \
    #--resume_from_checkpoint="checkpoint-6000" \

    #--gradient_checkpointing \
    
    #pretrained_backbone_path="./checkpoints/_base/egodex_foundation/checkpoint-500000/pytorch_model.bin"

    # For finetune mode with specific robot embodiment, use these parameters instead:
    # --mode="finetune" \
    # --pretrained_backbone_path="./checkpoints/_base/egodex_foundation/pytorch_model.bin" \
    # --config_path="configs/hrdt_finetune.yaml" \  # Config with different action_dim for target robot
    # --dataset_type="finetune" \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training