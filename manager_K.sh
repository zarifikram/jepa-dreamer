#!/bin/bash

module load python/3.9

source $HOME/scratch/dmr/bin/activate


declare -a All_env_names=(breakout asterix alien ms_pacman krull)
declare -a All_seeds=(0 1 2)
declare -a All_K=(4 8 12)

for env in "${All_env_names[@]}"; do
    for k in "${All_K[@]}"; do
        for seed in "${All_seeds[@]}"; do
            sbatch train_ablationK.sh $env $k $seed
        done
    done
done

