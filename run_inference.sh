#!/bin/bash

# Configurations
MODEL_PATH="/scratch/izar/cizinsky/pretrained/human3r.pth"
SIZE=512
SEQ_PATH="/home/cizinsky/zurihack/converted_mp4s/initial_demo.mp4"
SUBSAMPLE=1
VIS_THRESHOLD=2
DOWNSAMPLE_FACTOR=1
RESET_INTERVAL=100
OUTPUT_DIR="/scratch/izar/cizinsky/zurihack/human3r_output"

# Run inference
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path $MODEL_PATH \
    --size $SIZE \
    --seq_path $SEQ_PATH \
    --subsample $SUBSAMPLE \
    --use_ttt3r \
    --vis_threshold $VIS_THRESHOLD \
    --downsample_factor $DOWNSAMPLE_FACTOR \
    --reset_interval $RESET_INTERVAL \
    --output_dir $OUTPUT_DIR
