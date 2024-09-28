#!/bin/bash

module load python/3.9

source $HOME/scratch/dmr/bin/activate


declare -a All_env_names=(cup_catch cartpole_swingup cheetah_run finger_spin reacher_easy walker_walk)
declare -a All_diff_names=(easy medium hard)
declare -a All_seeds=(0 1 2)

# the env names and difficulty names are joined by '_'. e.g., cup_catch_easy
# so we should run sbatch train_cluster.sh cup_catch easy 0
for env in "${All_env_names[@]}"; do
    for difficulty in "${All_diff_names[@]}"; do
        for seed in "${All_seeds[@]}"; do
            sbatch train_cluster.sh ${env}_${difficulty} $seed
        done
    done
done

