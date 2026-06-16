## launch command for this script:
# cd ~/H_RDT && conda activate hrdt && \
# mkdir -p ./checkpoints/T_R2_tableware_stack_bowls_two && \
# bash finetune.sh 2>&1 | tee ./checkpoints/T_R2_tableware_stack_bowls_two/train_log.txt

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

export WANDB_PROJECT="T_R3_pooledLSS_tableware_stack_bowls_two"
export PRETRAINED_CHECKPOINT="./checkpoints/tidying_tableware/pretrains/S2_pooledLSS/checkpoint-10000/pytorch_model.bin"
export OUTPUT_DIR="./checkpoints/tidying_tableware/finetunes/R3_pooledLSS_stack_bowls_two"


export VISION_ENCODER_NAME="dino-siglip"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

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
    --pretrained_backbone_path="$PRETRAINED_CHECKPOINT" \
    --task_name="stack_bowls_two" \
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