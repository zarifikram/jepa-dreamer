game_name=bank_heist
type=bottleneck_atc
# nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# run it for three seeds concurrently 
mkdir logdir

for seed in 0 1
do
    nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atc --logdir /data/zikram/dreamer/${game_name}_${type}_seed${seed} &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err &
done

 
# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
 
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err