game_name=breakout
type=robust_breakout
# nohup python dreamer.py --task=atari_${game_name} --configs atari100k updates atp debug --logdir ./logdir/${game_name}_${type} &> ./logdir/${game_name}_${type}.log 2> ./logdir/${game_name}_${type}.err

# run it for three seeds concurrently 
mkdir logdir

gpu=cuda:2
pixel_prob=0.05
exp_name=atp_pixel_shift_${pixel_prob}_${game_name}
proj_name=robustness
for seed in 0 1 
do
    nohup python dreamer.py --task=atari_${game_name} --device ${gpu} --seed ${seed} --wandb_proj ${proj_name} --wandb_exp ${exp_name}_seed${seed} --pixel_shift_prob ${pixel_prob} --configs atari100k updates pixel_shift atp --logdir /data/zikram/dreamer/${exp_name}_seed${seed} &> ./logdir/${exp_name}_seed${seed}.log 2> ./logdir/${exp_name}_seed${seed}.err &
done

 
# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
 
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err