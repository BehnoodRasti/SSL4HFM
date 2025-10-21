#!/bin/bash
#SBATCH --job-name=HyperMAE_Training  # Job name
#SBATCH --output=logs/mimaim_50K_ENC%j.out # Standard output and error log (%x = job name, %j = job ID)
#SBATCH --error=logs/mimaim_50K_ENC%j.out  # Error log
#SBATCH --mem=256G                         # replace with amount suitable for your job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --time=5-20:00:00                   # replace with amount suitable for your job
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --partition=LocalQ            # Partition name

DEVICES=0
NUM_WORKERS=8

MODE=easy

MODEL=mae
mr=0.75

TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=128
LOSS=mse
Lambda=0.0005

Lambda2=0.0
Lambda_mv=100
Temp=0.001

LEARNING_RATE=1e-4
EPOCHS=300
VIT_MODEL="vit_base_patch16_224"
PATCH_SIZE=16

Data_Path=SpectralEarth
WEIGHT_FILE=None 
FIXED_WEIGHTS=false
DIC_CHANNELS=500

ada_mask=true # flase for no adaptive masking, true for adaptive masking
group_num=0
spec_reduct=true
CONSTRAINT=abs
ATTENSION=0.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

srun python -u train.py \
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
    --mv_lambda ${Lambda_mv} \
    --l2_lambda ${Lambda2} \
    --temperature ${Temp} \
    --dataset ${Data_Path} \
    --weight-file ${WEIGHT_FILE} \
    --patch-size ${PATCH_SIZE} \
    --vit-model "${VIT_MODEL}" \
    --group ${group_num} \
    --constraint-type ${CONSTRAINT} \
    --attention-strength ${ATTENSION} \
    $( [ "${FIXED_WEIGHTS}" = true ] && echo "--fixed-weights" )\
    $( [ "${spec_reduct}" = true ] && echo "--use-spectral-reduction" ) \
    $( [ "${ada_mask}" = true ] && echo "--adaptive_masking" ) 
