#!/usr/bin/env bash
set -e
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main_finetune.py \
  --teacher_mode true \
  --fsdp_cpu_offload false \
  --ema_device cpu \
  --model convnextv2_base \
  --dual_view true \
  --input_size 320 \
  --drop_path 0.1 \
  --batch_size 6 \
  --epochs 200 \
  --lr 5e-4 --min_lr 1e-6 --warmup_epochs 5 \
  --num_classes 15 \
  --multi_label true \
  --eval_threshold 0.5 \
  --fuse_mode add \
  --fuse_levels C3 C4 C5 \
  --head_type c5 \
  --train_list annotations/DvXray_train.txt \
  --val_list   annotations/DvXray_val.txt \
  --classes_file annotations/classes.txt \
  --finetune ./output/teacher_v2b320_ema_add_bs4/checkpoint_best_ema.pth \
  --model_prefix '' \
  --output_dir ./output/teacher_v2b320_ema_add_bs4_cont200
