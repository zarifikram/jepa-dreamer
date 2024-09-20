game_name=breakout
type=mlr
nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates mlr --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# python dreamer.py --task=atari_breakout --configs atari100k updates mlr debug --logdir ./logdir/atari-mlr
