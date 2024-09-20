game_name=frostbite
type=acro
nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates acro --logdir ./logdir/atari-acro &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# python dreamer.py --task=atari_frostbite --configs atari100k updates debug --logdir ./logdir/atari-acro
