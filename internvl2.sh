#!/bin/bash
#SBATCH --cpus-per-task=1              # No CPU allocation
#SBATCH --job-name=evaluation            # Job name
#SBATCH --output=output.log           # Standard output log
#SBATCH --error=error.log             # Error log
#SBATCH --time=60:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1

conda init
conda activate llava_next
srun python3 internvl2.py
