#!/bin/bash
#SBATCH --job-name=make_sweep_bori
#SBATCH --output=logs/start_sweep.out
#SBATCH --error=logs/start_sweep.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=30:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run scripts/create_sweep.py