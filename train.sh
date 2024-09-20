game_name=breakout
type=novelty
nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log &

# python dreamer.py --task=atari_frostbite --configs atari100k updates debug --logdir ./logdir/atari-noreconstruction &> ./logdir/test.log