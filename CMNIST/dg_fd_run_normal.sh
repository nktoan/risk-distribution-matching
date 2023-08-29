#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

module load Anaconda3
source activate domainbed

cd ~/source_code_conf/risk-distribution-matching/CMNIST
which python 

pwd

DATA=~/causal_optimisation_dg/quantile_rm/DomainBed/domainbed/data/
OUTPUT=~/source_code_conf/risk-distribution-matching/CMNIST
MyLauncher="submitit"

echo $DATA

python train.py --algorithm rdm --variance_weight 0.0 --batch_size 25000 --penalty_weight 10000 --erm_pretrain_iters 0 --data_dir=$DATA --output_dir=$OUTPUT --save_ckpts --lr_cos_sched
