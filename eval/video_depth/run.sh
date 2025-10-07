#!/bin/bash

set -e

workdir='.'
model_name='human3r'
ckpt_name='human3r'
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('bonn' 'bonn_50' 'bonn_100' 'bonn_150' 'bonn_200' 'bonn_250' 'bonn_300' 'bonn_350' 'bonn_400' 'bonn_450' 'bonn_500')


for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1  eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --reset_interval 1000000000 \
        --use_ttt3r
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "metric"
done
