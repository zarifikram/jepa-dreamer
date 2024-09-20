game_name=breakout
type=icm
nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates icm --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
