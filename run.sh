#!/bin/bash
#PBS -q qnvidia
#PBS -A OPEN-26-11
#PBS -N nanogpt-22-01
#PBS -l select=1,walltime=48:00:00
#PBS -o nanogpt-22-01.out
module purge
ml load Anaconda3
ml load cuDNN/8.4.1.50-CUDA-11.7.0
module load GCC/8.3.0

nvidia-smi

conda init
source ~/.bashrc
conda activate nanogpt

cd /home/hereldav/project-startup/nanoGPT

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_2022_12.py
