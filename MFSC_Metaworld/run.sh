task=bc
exp_id=3003
seed=3

mkdir -p log

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=0 nohup python3 train.py \
    --algo ppokeypoint \
    --frame_stack 1 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --total_timesteps 4000000 \
    --seed ${seed} \
    -u > log/${task}_${exp_id}_${seed}.log &