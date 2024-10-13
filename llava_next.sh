#!/bin/bash
#SBATCH --cpus-per-task=4              # No CPU allocation
#SBATCH --job-name=evaluation            # Job name
#SBATCH --output=llava_next_log           # Standard output log
#SBATCH --error=llava_next_err           # Error log
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:2
#SBATCH --nodes=1
#SBATCH --mem=120g

srun python3 llava_next.py
