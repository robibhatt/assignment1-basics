#!/bin/bash
#SBATCH --job-name=ibor_train_bpe
#SBATCH --output=logs/train_bpe.out
#SBATCH --error=logs/train_bpe.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=30:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run cs336_basics/train_bpe.py data/owt/owt_train.txt