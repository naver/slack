#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/domain_net/
if [ "$2" = "uniform" ]; then  # Uniform policy
    tag=domainnet-$1-Uni-$3
    resume="None"
elif [ -z "$4" ]; then  # End-to-end, extracts learned policy
    tag=domainnet-$1-slack-s$2-$3
    resume=$ROOT/data/outputs/domainnet-$1-s$2/search
else
    assert "$4" = "git-policies" "undefined argument \$3"  # Evaluates paper's policy
    tag=domainnet-$1-slack-git-s$2-$3
    resume=$ROOT/checkpoints/domainnet/models/$1-s$2.ckpt
fi

python train.py -c $ROOT/configs/train/domainnet.yaml --dataroot $root --dataset domainnet_$1 --optimizer_decay 0.001 --optimizer_nesterov --model_type resnet18 --save ckpt/$tag.pth --tag $tag --seed $3 --resume $resume
