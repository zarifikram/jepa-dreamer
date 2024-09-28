#!/bin/bash

module load python/3.9

source $HOME/scratch/dmr/bin/activate




declare -a All_env_names=(amidar asterix battle_zone boxing gopher hero james_bond kangaroo krull kung_fu_master ms_pacman private_eye qbert road_runner )
declare -a All_seeds=(0 1 2)

for env in "${All_env_names[@]}"; do
    for seed in "${All_seeds[@]}"; do
        sbatch train_cluster.sh $env $seed
    done
done


# coinrun 0 procgen 