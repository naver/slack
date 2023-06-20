#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/domain_net/
save=$ROOT/data/outputs/domainnet-$1-s$2
tag=domainnet-$1-Uni-.5-s$2

python train.py -c $ROOT/configs/train/domainnet.yaml --dataroot $root --dataset domainnet_$1 --optimizer_decay 0.001 --model_type resnet18 --save $save/Uni0.pth --tag $tag-0 --cv-ratio 0.5 --cv $2 --seed 0
