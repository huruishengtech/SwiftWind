#!/bin/bash
#SBATCH --job-name=00_swiftwind
#SBATCH --output=Output_logs/00_swiftwind_%j.log
#SBATCH --error=Output_logs/00_swiftwind%j.log
#SBATCH --partition=gpu

module load miniforge3/24.1  cudnn/8.4.0.27_cuda11.x compilers/cuda/11.8 
source activate SwiftWind 
export PYTHONUNBUFFERED=1
mkdir -p Output_logs
mkdir -p Plot_all

# Train
python train_osse.py --gpu 4 --num_sensors 500 --seed 123 --max_epochs 2000 \
  --batch_frames 32 --enc_preproc 128 --dec_num_latent_channels 128 --num_layers 3 \
  --num_self_attention_layers_per_block 6 --num_cross_attention_heads 8 --enc_num_self_attention_heads 8 \
  --dec_num_cross_attention_heads 8 --enc_num_latent_channels 128 --num_latents 512 --dec_preproc_ch 128 \
  --test False --batch_pixels 8192 --num_workers 8 --lr 0.0001

# Test
python test_osse.py --gpu 1 --num_sensors 500 --seed 123 \
  --enc_preproc 128 --dec_num_latent_channels 128 --num_layers 3 \
  --num_self_attention_layers_per_block 6 --num_cross_attention_heads 8 --enc_num_self_attention_heads 8 \
  --dec_num_cross_attention_heads 8 --enc_num_latent_channels 128 --num_latents 512 --dec_preproc_ch 128 \
  --batch_frames 32 --batch_pixels 8192 --load_model_num xxx






