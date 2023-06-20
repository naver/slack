#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/domain_net/
tag=domainnet-$1-$2-$3

python train.py -c $ROOT/configs/train/domainnet_$2.yaml --dataroot $root --dataset domainnet_$1 --optimizer_decay 0.001 --optimizer_nesterov --model_type resnet18 --save ckpt/$tag.pth --tag $tag --seed $3
