#!/bin/bash

#SBATCH --partition=gpu-long.q
#SBATCH --mem=64G
#SBATCH --nodes=2
#SBATCH --time=2-0:00:00
#SBATCH --job-name=tlda
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
eval "$(conda shell.bash hook)"
conda init bash
conda activate rapids-0.19

python3 gpu_tlda.py
python3 mem_compromize.py

