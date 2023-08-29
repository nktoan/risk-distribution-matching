#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

module load Anaconda3
source activate domainbed

cd ~/source_code_conf/risk-distribution-matching/DomainBed
which python 

pwd

DATA=~/causal_optimisation_dg/quantile_rm/DomainBed/domainbed/data/
OUTPUT=~/source_code_conf/risk-distribution-matching/DomainBed/domainbed
MyLauncher="submitit"

echo $DATA
echo $OUTPUT

python3 -m domainbed.scripts.train --data_dir=$DATA  --output_dir=$OUTPUT --algorithm RDM --dataset PACS --test_env 0