game_name=breakout
type=bottleneck_atc
wandb_enabled=False
# nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# run it for three seeds concurrently 
mkdir logdir

for seed in 0
do
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atc --logdir /data/zikram/dreamer/${game_name}_${type}_seed${seed} &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err &
    python dreamer.py --task=atari_${game_name} --seed ${seed} --wandb_enabled=${wandb_enabled} --configs atari100k updates atc delusion --logdir /data/zikram/dreamer/${game_name}_${type}_seed${seed} 
done

 
# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
 
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err