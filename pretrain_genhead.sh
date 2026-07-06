# launch command:
# cd ~/H_RDT && conda activate hrdt && \
# mkdir -p ./checkpoints/adjusting_bottle/pretrains/S2_noLSS_seed42_batch64 && \
# bash pretrain.sh 2>&1 | tee ./checkpoints/adjusting_bottle/pretrains/S2_noLSS_seed42_batch64/train_log.txt


# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=0
# export CUDA_VISIBLE_DEVICES=0,1

# --- [YOUR LOCAL HARDWARE SETTINGS] ---
unset NCCL_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1  
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export CUDA_VISIBLE_DEVICES=0,1


export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/data/lingxuan/cutlass"

export WANDB_PROJECT="S2_noLSS_seed42_batch64"
export OUTPUT_DIR="./checkpoints/adjusting_bottle/pretrains/S2_genhead_bottle_seed42"
export PRETRAINED_FOLDER="./checkpoints/_base/egodex_foundation/checkpoint-500000"

export VISION_ENCODER_NAME="dino-siglip"
source datasets/pretrain/setup_pretrain.sh

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

export EGODEX_TILE_DENSE=0
export EGODEX_DATA_ROOT="$HOME/human-policy/data/recordings/processed_baseline_adjust_bottle"
export EGODEX_TILE_DENSE=0   # dense-tiling: deterministic windows (dense pretrain only)
echo "DeepSpeed Launching with Data Root: $EGODEX_DATA_ROOT"

#WANDB_RUN_ID="89f02271" WANDB_RESUME="allow" 
accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero2_bs64.json" \
    --config_path="configs/hrdt_pretrain.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=10000 \
    --checkpointing_period=2000 \
    --sample_period=99999 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=5e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --dataset_type="egodex" \
    --report_to=wandb \
    --upsample_rate=3 \
    --image_aug \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="pretrain" \
    --pretrained_model_name_or_path=$PRETRAINED_FOLDER \
    --use_lora \
    --lora_rank=8 \
    --lora_alpha=16 \
    --seed=42 \
    --use_reasoning_head \
    --reasoning_lambda=1.0 \
    
    #--use_lsa \
    #--lsa_lambda=0.1 \
    #--use_dense_lsa \
    #--resume_from_checkpoint="checkpoint-2000" \
    #--gradient_checkpointing
    
    # For finetune mode with specific robot embodiment, use these parameters instead:
    # --mode="finetune" \
    # --pretrained_backbone_path="./checkpoints/_base/egodex_foundation/pytorch_model.bin" \
    # --config_path="configs/hrdt_finetune.yaml" \  # Config with different action_dim for target robot
    # --dataset_type="finetune" \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training