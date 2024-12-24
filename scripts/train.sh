#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="./src:$PYTHONPATH"
export WANDB_MODE="offline"
# export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="keler_flare_detection"

RUN_NAME="fcn4flare-kepler_flare"
OUTPUT_DIR="./results/$RUN_NAME"
SEED=2024

python src/train.py \
    --dataset_name "Maxwell-Jia/kepler_flare" \
    --config_name_or_path "./configs/fcn4flare_config.json" \
    --max_seq_length 5000 \
    --pad_to_max_length true \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --eval_strategy "epoch" \
    --per_device_train_batch_size 192 \
    --per_device_eval_batch_size 192 \
    --learning_rate 3e-5 \
    --num_train_epochs 50 \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 20 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --seed $SEED \
    --dataloader_num_workers 8 \
    --run_name $RUN_NAME \
    --load_best_model_at_end true \
    --metric_for_best_model "dice" \
    --greater_is_better true \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.005 \