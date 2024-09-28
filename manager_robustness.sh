#!/bin/bash

module load python/3.9

source $HOME/scratch/dmr/bin/activate


declare -a All_env_names=(breakout asterix alien ms_pacman krull)
declare -a All_seeds=(0 1 2)
declare -a All_probs=(0.01 0.05)

for env in "${All_env_names[@]}"; do
    for prob in "${All_probs[@]}"; do
        for seed in "${All_seeds[@]}"; do
            sbatch train_robust.sh $env $prob $seed
        done
    done
done

