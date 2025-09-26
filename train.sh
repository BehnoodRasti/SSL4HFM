#!/bin/bash

DEVICES=5
NUM_WORKERS=8

MODE=easy

MODEL=unaim
mr=0.75

TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=128

LOSS=mse
Lambda=0.01


LEARNING_RATE=1e-4
EPOCHS=300

Data_Path=SpectralEarth
WEIGHT_FILE=None  # Path to the weight file (set to None if not using pre-trained weights)
FIXED_WEIGHTS=false                   # Set to true to keep weights fixed during training
DIC_CHANNELS=100

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
\nohup \
  python -u train.py \
    --dic-channels ${DIC_CHANNELS} \
    --devices ${DEVICES} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --learning-rate ${LEARNING_RATE} \
    --mode ${MODE} \
    --model ${MODEL} \
    --loss ${LOSS} \
    --epochs ${EPOCHS} \
    --masking-ratio ${mr}\
    --l1_lambda ${Lambda}\
    --dataset ${Data_Path}\
    --dataset ${Data_Path} \
    --patch-size "${PATCH_SIZE}" \
    --vit-model "${VIT_MODEL}" \
    --weight-file ${WEIGHT_FILE} \
    $( [ "${FIXED_WEIGHTS}" = true ] && echo "--fixed-weights" )> nohup_10.out 2>&1 &
