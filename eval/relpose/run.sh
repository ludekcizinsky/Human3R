#!/bin/bash

set -e

workdir='.'
model_name='human3r'
ckpt_name='human3r'
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('tum' 'tum_50' 'tum_100' 'tum_150' 'tum_200' 'tum_300' 'tum_400' 'tum_500' 'tum_600' 'tum_700' 'tum_800' 'tum_900' 'tum_1000')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29565 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --reset_interval 100 \
        --use_ttt3r
done


