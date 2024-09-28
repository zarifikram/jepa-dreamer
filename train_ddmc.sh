#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 4-22:00:00
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1 
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM

module load python/3.9

source $HOME/scratch/dmr/bin/activate

mkdir logdir

game_name=$1
seed=$2
exp_name=ddmc_${game_name}_dv3_${seed}
proj_name=ddmc
type=dv3
nohup python dreamer.py --task=ddmc_${game_name} --wandb_proj ${proj_name} --wandb_exp ${exp_name} --configs dmc_vision debug updates --logdir ./logdir/${exp_name} &> ./logdir/${exp_name}.log 2> ./logdir/${exp_name}.err
