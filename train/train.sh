#!/bin/bash
set -e

# 路径配置
MODEL_PATH="deepseek7b"
DATASET_PATH="merged_data.jsonl"
OUTPUT_DIR="Mcoder-deepseekmath"
SCRIPT_PATH="train.py"

# 环境配置
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.6"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate trainenv

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 2 \
    --master_port 29504 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29504 \
    "$SCRIPT_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --source_max_len 660 \
    --target_max_len 448 \
    --eval_dataset_size 500 \
    --max_train_samples 4900 \
    --max_eval_samples 500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_steps 700 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 150 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 150 \
    --do_eval True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 True \
    --gradient_checkpointing True \
    --gradient_clip_mode norm \
    --max_grad_norm 0.5 \
    --lora_r 24 \
    --lora_alpha 48 \
    --lora_dropout 0.1 \
    --bits 4 \
    --double_quant True \
    --quant_type nf4 \
    --group_by_length True \
    --seed 42 \
    --report_to none \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --use_math_assistant_prompt True \
    --use_chinese_prompt False \
    --optim paged_adamw_32bit 2>&1 | tee "$OUTPUT_DIR/training.log"
