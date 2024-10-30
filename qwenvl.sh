#!/bin/bash
#SBATCH --cpus-per-task=4              # No CPU allocation
#SBATCH --job-name=evaluation            # Job name
#SBATCH --output=qwenvl_log_2           # Standard output log
#SBATCH --error=qwenvl_err_2           # Error log
#SBATCH --time=120:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:2
#SBATCH --nodes=1
#SBATCH --mem=120g
#SBATCH --ntasks=1
srun python3 qwenvl.py
