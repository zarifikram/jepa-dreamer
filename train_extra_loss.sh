#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 4-22:00:00
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1 
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM

# module load python/3.9

# source $HOME/scratch/dmr/bin/activate

# mkdir logdir

game_name=$1
extra_loss=$2
seed=$3
exp_name=${game_name}_extra_loss${extra_loss}_atc_${seed}
proj_name=ablation-extra-loss
type=atc
nohup python dreamer.py --task=atari_${game_name} --wandb_proj ${proj_name} --wandb_exp ${exp_name} --use_extra_loss ${extra_loss} --configs atari100k updates atp --logdir ./logdir/${exp_name} &> ./logdir/${exp_name}.log 2> ./logdir/${exp_name}.err