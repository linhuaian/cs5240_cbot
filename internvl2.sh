#!/bin/bash
#SBATCH --cpus-per-task=1              # No CPU allocation
#SBATCH --job-name=evaluation            # Job name
#SBATCH --output=internvl2_out           # Standard output log
#SBATCH --error=internvl2_err             # Error log
#SBATCH --time=60:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=150g
#SBATCH --nodes=1
#SBATCH --ntasks=1

conda activate eval
srun python3 internvl2.py
