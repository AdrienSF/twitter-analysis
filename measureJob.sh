#!/bin/bash

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --time=0-4:00:00
#SBATCH --job-name=complexity_measure
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
eval "$(conda shell.bash hook)"
conda init bash
conda activate rapids-0.19


for ((i=2;i<=9;i++)); do
   python3 cpu_complexity_measure.py "$((2**$i))"
done

python complexity_measure.py 600

