#!/bin/bash
#SBATCH --job-name=ibor_tests
#SBATCH --output=logs/robi_tests.out
#SBATCH --error=logs/robi_tests.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=30:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

set -e
srun ./rb_tests.sh