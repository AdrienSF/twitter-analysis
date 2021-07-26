#!/bin/bash

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --time=0-4:00:00
#SBATCH --job-name=tlda
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
eval "$(conda shell.bash hook)"
conda init bash
conda activate rapids-0.19

python3 gpu_tlda.py


