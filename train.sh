#!/usr/bin/env bash
MVS_TRAINING="/data/jh/data/dtu/dtu_train/"
BLEND_TRAINING="/data/jh/data/dataset_low_res"
LOG_DIR="./checkpoints"
LOG_DIR_CKPT="./checkpoints/DTU.ckpt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
filename="ACP-MVS"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

##DTU
python -u train.py --mode='train' --epochs=20 --numdepth=384 --trainviews=5 --testviews=5 --logdir $LOG_DIR --dataset=dtu_yao_1to8_inverse --batch_size=12 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --ndepths=48 --CostNum=4 --lr=0.001 | tee -i $dirAndName

##finetune on BlendedMVS
#python -u train.py --mode='finetune' --epochs=20 --numdepth=768 --trainviews=5 --testviews=5 --logdir $LOG_DIR --loadckpt $LOG_DIR_CKPT --dataset=blend --batch_size=8 --trainpath=$BLEND_TRAINING \
#                --ndepths=96 --CostNum=4 --lr=0.0004 | tee -i $dirAndName