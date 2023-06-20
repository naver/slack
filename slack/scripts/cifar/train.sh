#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/
if [ "$3" = "uniform" ]; then  # Uniform policy
    tag=c$1-$2-Uni
    resume="None"
elif [ -z "$4" ]; then  # End-to-end, extracts learned policy
    tag=c$1-$2-slack-s$3
    resume=$ROOT/data/outputs/c$1-$2-s$3/search
elif [ "$4" = "git-policies" ]; then  # Evaluates paper's policy
    tag=c$1-$2-slack-git-s$3
    resume=$ROOT/checkpoints/cifar/models/c$1-$2-s$3.ckpt
else
    tag=c$1-$2-slack-$4-s$3  # Ablations
    resume=$ROOT/data/outputs/c$1-$2-s$3/search-$4
fi
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

for i in {0..3}
do
    python train.py -c $ROOT/configs/train/cifar.yaml --dataroot $root --dataset cifar$1 --optimizer_decay $decay --optimizer_nesterov --model_type $model_type --save ckpt/$tag-$i.pth --tag $tag-$i --resume $resume --seed $i
done
