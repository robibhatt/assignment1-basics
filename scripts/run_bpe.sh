#!/bin/bash
#SBATCH --job-name=jalani_train_bpe
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/train_bpe.out
#SBATCH --error=logs/train_bpe.err
#SBATCH --partition=standard
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --account=jalaniw0

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run scripts/run_train_bpe.py
