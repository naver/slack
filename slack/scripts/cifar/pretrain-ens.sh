#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/
save=$ROOT/data/outputs/c$1-$2
tag=c$1-$2-Uni-.5-s$3
if [[ $1 == '10' && $2 == '40x2' ]]
then
    decay=0.0002
else
    decay=0.0005
fi
if [[ $2 == '40x2' ]]
then
    model_type=wresnet40_2
else
    model_type=wresnet28_10
fi

for i in {1..3}
do
    python train.py -c $ROOT/configs/train/cifar.yaml --dataroot $root --dataset cifar$1 --optimizer_decay $decay --model_type $model_type --save $save-s$3/Uni$i.pth --tag $tag-$i  --cv-ratio 0.5 --cv $3 --seed $i
done
