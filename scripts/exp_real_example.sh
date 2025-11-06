#!/bin/bash
#SBATCH --job-name=00_real
#SBATCH --output=Output_logs/00_real_%j.log
#SBATCH --error=Output_logs/00_real_%j.log
#SBATCH --partition=gpu

module load miniforge3/24.1  cudnn/8.4.0.27_cuda11.x compilers/cuda/11.8 
source activate SwiftWind 
export PYTHONUNBUFFERED=1
mkdir -p Output_logs
mkdir -p Plot_all

# Train
python train_real.py \
  --gpu 4 --seed 123 --max_epochs 4000 \
  --batch_frames 32 \
  --batch_pixels 8192 \
  --num_workers 6 \
  --test False \
  --lr 0.00002 \
  \
  --enc_preproc 64 \
  --dec_preproc_ch 64 \
  --enc_num_latent_channels 64 \
  --dec_num_latent_channels 64 \
  --num_latents 256 \
  --num_layers 2 \
  --num_self_attention_layers_per_block 4 \
  --num_cross_attention_heads 4 \
  --enc_num_self_attention_heads 4 \
  --dec_num_cross_attention_heads 4 \
  --load_model_num xxx

# Test
python test_real.py \
 --gpu 1 --seed 123 \
 --batch_frames 32 \
 --batch_pixels 8192 \
 \
 --enc_preproc 64 \
 --dec_preproc_ch 64 \
 --enc_num_latent_channels 64 \
 --dec_num_latent_channels 64 \
 --num_latents 256 \
 --num_layers 2 \
 --num_self_attention_layers_per_block 4 \
 --num_cross_attention_heads 4 \
 --enc_num_self_attention_heads 4 \
 --dec_num_cross_attention_heads 4 \
 --load_model_num xxx




