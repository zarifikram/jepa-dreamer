game_name=$1
seed=$2
cuda=$3
wandb_enabled=False
type=seed_${seed}_kl_tcl_gb
wandb_exp=${game_name}_${type}
# nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# run it for three seeds concurrently
mkdir logdir


# nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atc --logdir /data0/zikram/dreamer/${game_name}_${type}_seed${seed} &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err &
# before running the command, make sure to empty the log§§dir directory
rm -rf /data0/zikram/dreamer/${game_name}_${type}_seed${seed}
python dreamer.py --task=atari_${game_name} --seed ${seed} --wandb_enabled=${wandb_enabled} --wandb_exp=${wandb_exp} --device=cuda:${cuda} --configs atari100k updates dreamer_k_step --logdir /data0/zikram/dreamer/${game_name}_${type}_seed${seed}


# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm

# nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err