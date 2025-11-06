#!/bin/bash
#SBATCH --job-name=00_train_finetune
#SBATCH --output=Output_logs/00_train_finetune_%j.log
#SBATCH --error=Output_logs/00_train_finetune_%j.log
#SBATCH --partition=gpu

module load miniforge3/24.1  cudnn/8.4.0.27_cuda11.x compilers/cuda/11.8 
source activate SwiftWind 
export PYTHONUNBUFFERED=1
mkdir -p Output_logs
mkdir -p Plot_all

# Train
python train_finetune.py --gpu 4 --seed 123 --max_epochs 4000 \
 --batch_frames 128 --enc_preproc 16 --dec_num_latent_channels 16 \
 --enc_num_latent_channels 16 --num_latents 256 --dec_preproc_ch 16 --test False --batch_pixels 8192 \
 --num_workers 6 --load_model_num xxx --lr 0.0001

# Test
python inference_1h_ascat.py --gpu 1 --seed 123 \
 --enc_preproc 16 --dec_num_latent_channels 16 --enc_num_latent_channels 16 --num_latents 256 \
 --dec_preproc_ch 16 --load_model_num xxx
