task=bc
exp_id=3002
seed=2

mkdir -p log

Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99

MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=0 nohup python3 visualize.py \
    --algo ppokeypoint \
    --frame_stack 1 \
    -t ${task} \
    -v 1 \
    -e ${exp_id} \
    --seed ${seed} \
    -u > log/${task}_MFSC.log &
