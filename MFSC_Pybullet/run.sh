task=antnc
exp_id=9000
seed=0

mkdir -p log_antnc

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=4 nohup python3 train.py \
    --algo ppokeypoint \
    --frame_stack 2 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --total_timesteps 5000000 \
    --seed ${seed} \
    -u > log_antnc/${task}_${exp_id}_${seed}.log &

task=antnc
exp_id=9001
seed=1

mkdir -p log_antnc

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
    --algo ppokeypoint \
    --frame_stack 2 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --total_timesteps 5000000 \
    --seed ${seed} \
    -u > log_antnc/${task}_${exp_id}_${seed}.log &

task=antnc
exp_id=9002
seed=2

mkdir -p log_antnc

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=6 nohup python3 train.py \
    --algo ppokeypoint \
    --frame_stack 2 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --total_timesteps 5000000 \
    --seed ${seed} \
    -u > log_antnc/${task}_${exp_id}_${seed}.log &

task=antnc
exp_id=9003
seed=3

mkdir -p log_antnc

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=7 nohup python3 train.py \
    --algo ppokeypoint \
    --frame_stack 2 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --total_timesteps 5000000 \
    --seed ${seed} \
    -u > log_antnc/${task}_${exp_id}_${seed}.log &