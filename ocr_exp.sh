game_name=$1
seed=$2
exp_name=${game_name}_atc_${seed}_ocr
proj_name=ocr_exp
type=atc
python dreamer.py --task=atari_${game_name} --wandb_proj ${proj_name} --wandb_exp ${exp_name}  --configs atari100k updates atp steve --logdir ./logdir/${exp_name} 
# &> ./logdir/${exp_name}.log 2> ./logdir/${exp_name}.err