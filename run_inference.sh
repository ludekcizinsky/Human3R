#!/bin/bash

module load gcc ffmpeg

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate human3r

# Configurations
SEQ_NAME="pushups"
INPUT_MOV="/home/cizinsky/zurihack/iphone_vids/$SEQ_NAME.mov"
TARGET_FPS="15"
START_FRAME="0"
END_FRAME="2000"

MODEL_PATH="/scratch/izar/cizinsky/pretrained/human3r.pth"
SIZE=512
SEQ_PATH="/home/cizinsky/zurihack/converted_mp4s/$SEQ_NAME.mp4"
SUBSAMPLE=1
VIS_THRESHOLD=2
DOWNSAMPLE_FACTOR=1
RESET_INTERVAL=100
OUTPUT_DIR="/scratch/izar/cizinsky/zurihack/human3r/$SEQ_NAME"
mkdir -p $OUTPUT_DIR

select_filter="select='between(n\\,$START_FRAME\\,$END_FRAME)'"

echo "Converting $INPUT_MOV to $SEQ_PATH with fps=$TARGET_FPS from frame $START_FRAME to $END_FRAME"
echo "---------------------------------------------------------------------------------------------"
ffmpeg -y -i "$INPUT_MOV" \
    -vf "$select_filter,fps=$TARGET_FPS" \
    -vsync 0 \
    "$SEQ_PATH"

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