#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment
root=$ROOT/data/datasets/imagenet-100/
tag=imagenet100-$1-$2

python train.py -c $ROOT/configs/train/imagenet100_$1.yaml --dataroot $root --dataset imagenet100 --optimizer_decay 0.001 --model_type resnet18 --save ckpt/$tag.pth --tag $tag  --seed $2
