#!/bin/bash
#SBATCH --cpus-per-task=4              # No CPU allocation
#SBATCH --job-name=evaluation            # Job name
#SBATCH --output=output2.log           # Standard output log
#SBATCH --error=error2.log             # Error log
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:2
#SBATCH --nodes=4

conda init
conda activate llava_next
srun python3 eval_test.py
