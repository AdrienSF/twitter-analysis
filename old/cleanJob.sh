#!/bin/bash

#SBATCH --mem=32GB
#SBATCH --partition=cpu-long.q
#SBATCH --nodes=4
#SBATCH --time=7-00:00:00
#SBATCH --job-name=TLDAlloc
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
eval "$(conda shell.bash hook)"
conda init bash
conda activate twittenv

python3 clean_tweets.py
