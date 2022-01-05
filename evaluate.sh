#!/bin/bash

##! 주요 디렉토리 변수 설정
# 데이터셋 상위 디렉토리 (서브 디렉토리 : cache, en, ko)
# checkpoint 상위 디렉토리 (서브 디렉토리 : 하이퍼파라미터 > 실행 일시)
# logging 상위 디렉토리 (서브 디렉토리 : 하이퍼파라미터)
# 하이퍼파라미터 서브디렉토리 설정 -> for loop에서 직접 전달

export ROOT_DIR=~/zzoliman/+MRL/Pytorch # [CHANGE] : this should be changed to your own Root Directory; where you cloned the repo.
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints
export LOG_DIR=$ROOT_DIR/logs

##! 주 사용 옵션
# 기타(misc)
seed=42
gpu_id=3
# 데이터(data)
max_len=128
train_langs="en" # currently not used; manually set in code
dev_langs="en" # currently not used; manually set in code
test_langs="ko" # currently not used; manually set in code
# 모델(model)
pretrain_model="bert-base-multilingual-cased"
freeze_layer="-1"
dropout=0.2
use_crf="False"
# 학습(train)
# report & eval
report_eval_every_epoch="True"
report_frequency=100 # 110000 / 16 = 7000 step
eval_frequency=700
eval_batch_size=128

for bs in 16 32; do
# for bs in 16; do
    for lr in 2e-5 3e-5 5e-5; do
    # for lr in 2e-5; do
        for ep in 3 4; do
        # for ep in 3; do
            python src/run.py \
                --data_dir "$DATA_DIR" \
                --ckpt_dir "$CKPT_DIR" \
                --log_dir "$LOG_DIR" \
                --exp_name bs$bs-lr$lr-ep$ep \
                --seed "$seed" \
                --gpu_id "$gpu_id" \
                --max_len $max_len \
                --train_langs "$train_langs" \
                --dev_langs "$dev_langs" \
                --test_langs "$test_langs" \
                --pretrain_model "$pretrain_model" \
                --freeze_layer "$freeze_layer" \
                --dropout $dropout \
                --use_crf "$use_crf" \
                --batch_size $bs \
                --learning_rate $lr \
                --max_epochs $ep \
                --report_eval_every_epoch "$report_eval_every_epoch" \
                --report_frequency $report_frequency \
                --eval_frequency $eval_frequency \
                --eval_batch_size $eval_batch_size
        done
    done
done