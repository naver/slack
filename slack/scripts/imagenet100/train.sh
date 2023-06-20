#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/imagenet-100/
if [ "$1" = "uniform" ]; then  # Uniform policy
    tag=imagenet100-Uni-$2
    resume="None"
elif [ -z "$3" ]; then
    tag=imagenet100-slack-s$1-$2  # End-to-end, extracts learned policy
    resume=$ROOT/data/outputs/imagenet100-s$1/search
else
    assert "$3" = "git-policies" "undefined argument \$3"  # Evaluates paper's policy
    tag=imagenet100-slack-git-s$1-$2
    resume=$ROOT/checkpoints/imagenet100/models/s$1.ckpt
fi

python train.py -c $ROOT/configs/train/imagenet100.yaml --dataroot $root --dataset imagenet100 --optimizer_decay 0.001 --optimizer_nesterov --model_type resnet18 --save ckpt/$tag.pth --tag $tag --resume $resume --seed $2
