#!/bin/bash

export PYTHONPATH="./src:$PYTHONPATH"
export WANDB_MODE="offline"
export WANDB_PROJECT="flare_detection"
export WANDB_NAME="experiment_1"

python src/train.py \
    --config_name_or_path "./configs/fcn4flare_config.json" \
    --dataset_name "Maxwell-Jia/kepler_flare" \
    --output_dir "./results" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_predict_samples 4 \
    --max_train_samples 4 \
    --max_eval_samples 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --logging_dir "./results" \
    --logging_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end "True" \
    --metric_for_best_model "dice" \
    --greater_is_better "True" \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --report_to "wandb" \
    --max_seq_length 5000 \
    --pad_to_max_length "True" \
    --overwrite_cache "True"