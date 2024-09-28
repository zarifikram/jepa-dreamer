game_name=asterix
type=bottleneck_atc
proj_name=ablation
exp_name=atari_${game_name}
use_extra_loss=True
# nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# run it for three seeds concurrently 
#mkdir logdir

for seed in 0 6 42
do
    nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --use_extra_loss ${use_extra_loss} --wandb_proj ${proj_name} --wandb_exp ${exp_name}_seed${seed}_use_extra_loss${use_extra_loss} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type}_seed${seed} &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err &
done
 
# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
 
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err