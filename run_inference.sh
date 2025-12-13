#!/bin/bash
set -euo pipefail

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate human3r
module load gcc ffmpeg

# Configurations
SCENE_NAME=$1
BASE_DIR=/scratch/izar/cizinsky/thesis/preprocessing/$SCENE_NAME
SEQ_PATH=$BASE_DIR/frames
MODEL_PATH=/scratch/izar/cizinsky/pretrained/human3r_896L.pth
OUTPUT_DIR=$BASE_DIR/motion_human3r
mkdir -p "$OUTPUT_DIR"

SIZE=512
SUBSAMPLE=1
RESET_INTERVAL=100


cd submodules/human3r
python inference.py \
    --model_path "$MODEL_PATH" \
    --size "$SIZE" \
    --seq_path "$SEQ_PATH" \
    --subsample "$SUBSAMPLE" \
    --use_ttt3r \
    --reset_interval "$RESET_INTERVAL" \
    --output_dir "$OUTPUT_DIR" \
