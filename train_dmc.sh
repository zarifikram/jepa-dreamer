game_name=reacher_easy
type=reacher_easy
proj_name=dmc
exp_name=${game_name}_atp
nohup python dreamer.py --task=dmc_${game_name} --configs dmc_vision updates atp debug --logdir /data/zikram/dreamer/${exp_name} --wandb_proj ${proj_name} --wandb_exp ${exp_name}  &> ./logdir/${exp_name}.log 2> ./logdir/${exp_name}.err &

exp_name=${game_name}_dv3
nohup python dreamer.py --task=dmc_${game_name} --configs dmc_vision updates debug --logdir ./logdir/${exp_name} --wandb_proj ${proj_name} --wandb_exp ${exp_name}  &> ./logdir/${exp_name}.log 2> ./logdir/${exp_name}.err &
# run it for three seeds concurrently 
# mkdir logdir

# gpu=cuda:2
# pixel_prob=0.05
# for seed in 0 1 
# do
#     nohup python dreamer.py --task=atari_${game_name} --device ${gpu} --seed ${seed} --wandb_proj ${proj_name} --wandb_exp ${exp_name}_seed${seed} --pixel_shift_prob ${pixel_prob} --configs atari100k updates pixel_shift atp --logdir /data/zikram/dreamer/${exp_name}_seed${seed} &> ./logdir/${exp_name}_seed${seed}.log 2> ./logdir/${exp_name}_seed${seed}.err &
# done

 
# python dreamer.py --task=atari_frostbite --configs atari100k updates icm --logdir ./logdir/atari-icm
 
    # nohup python dreamer.py --task=atari_${game_name} --seed ${seed} --configs atari100k updates atp --logdir ./logdir/${game_name}_${type}_seed${seed}  &> ./logdir/${game_name}_${type}_seed${seed}.log 2> ./logdir/${game_name}_${type}_seed${seed}.err