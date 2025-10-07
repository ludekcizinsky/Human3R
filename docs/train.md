# Datasets
We freeze all weights of pretrained CUT3R and Multi-HMR encoder, and fine-tune the human-related modules on BEDLAM. 
This dataset provides 3D scene depth and SMPL-X meshes, with 1–10 people per scene, captured from diverse known camera viewpoints. 
Following CUT3R, we exclude BEDLAM sequences where the environment is represented by a panoramic HDRI image (sequences w/o correct scene depth), resulting in 5k sequences for training and 1k for validation, with each sequence averaging 30 frames.

Please download the image sequences (PNG), depth (EXR, 32-bit), segmentation masks (PNG), and ground truth for sequences (CSV) from [BEDLAM](https://bedlam.is.tue.mpg.de/) official sources into the `/path/to/bedlam/` directory, 
download SMPL ground truth parameters into the `/path/to/bedlam/processed_labels/` directory, 
```
# Download SMPL annotations. 
# The command will prompt you to register and log in to access data.

bash scripts/fetch_bedlam.sh
```


We provide the processing code [preprocess_bedlam.py](../datasets_preprocess/preprocess_bedlam.py), please refer to the following command to process data into the `/path/to/processed_bedlam/` directory.
```
cd datasets_preprocess/
python preprocess_bedlam.py --root /path/to/bedlam/ --outdir /path/to/processed_bedlam/ --annot_dir /path/to/bedlam/processed_labels/
```


# Training

For each iteration, we randomly sample 4 views from each sequence and train Human3R on one NVIDIA 48G GPU.

```
# Remember to replace the dataset path '/path/to/processed_bedlam/' to your own BEDLAM path
cd src/

CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu \
    --main_process_port=29506 train.py --config-name trian_human3r
```