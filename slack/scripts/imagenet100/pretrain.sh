#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT/TrivialAugment

root=$ROOT/data/datasets/imagenet-100/
save=$ROOT/data/outputs/imagenet100-s$1
tag=imagenet100-Uni-.5-s$1-0
python train.py -c $ROOT/configs/train/imagenet100.yaml --dataroot $root --dataset imagenet100 --optimizer_decay 0.001 --model_type resnet18 --save $save/Uni0.pth  --tag $tag --cv-ratio 0.5 --cv $1 --seed 0
