#!/bin/bash

module load gcc ffmpeg

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate human3r

# Configurations
SCENE_NAME="taichi"
SEQ_PATH="/scratch/izar/cizinsky/thesis/preprocessing/$SCENE_NAME/image"
MODEL_PATH="/scratch/izar/cizinsky/pretrained/human3r.pth"
SIZE=960
SUBSAMPLE=1
VIS_THRESHOLD=2
DOWNSAMPLE_FACTOR=1
RESET_INTERVAL=100
OUTPUT_DIR="/scratch/izar/cizinsky/thesis/preprocessing/$SCENE_NAME"
mkdir -p $OUTPUT_DIR

echo "Running inference"
echo "---------------------------------------------------------------------------------------------"
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
    --output_dir $OUTPUT_DIR \
    --save_video \
    --save_smpl