#!/bin/bash

module load python/3.9

source $HOME/scratch/dmr/bin/activate


declare -a All_env_names=(breakout asterix alien ms_pacman krull)
declare -a All_seeds=(0 1 2)
declare -a All_extra_loss=(True False)

for env in "${All_env_names[@]}"; do
    for extra_loss in "${All_extra_loss[@]}"; do
        for seed in "${All_seeds[@]}"; do
            sbatch train_ablationK.sh $env $extra_loss $seed
        done
    done
done

