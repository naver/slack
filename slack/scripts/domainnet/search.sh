#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT

inner_lr=.1
b_size=128
n_iter=1000
n_outer_iter=400
if [ $1 = 'painting' -o $1 = 'sketch' ]; then
    n_iter=1200;
    outer_lr=.8;
    entropy_reg=.0125
elif [ $1 = 'clipart' ]; then
    n_iter=800;
    outer_lr=1;
    entropy_reg=.01
elif [ $1 = 'infograph' ]; then
    n_iter=800;
    outer_lr=.625;
    entropy_reg=.016
elif [ $1 = 'quickdraw' ]; then
    n_iter=1000;
    outer_lr=1.25;
    entropy_reg=.008
elif [ $1 = 'real' ]; then
    n_iter=1000;
    outer_lr=.8;
    entropy_reg=.0125
else
    n_iter=1000;
    outer_lr=1;
    entropy_reg=.01
fi

log_name=domainnet-$1-s$2/search
outer_n_iter=$((10*n_outer_iter))
dtype=32
loss=domainnet
disp_freq=10
data=domainnet
device=0

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python search.py --multirun solver=slack \
                loss=$loss \
                data=$data \
                logs.log_name=$log_name \
                ++epochs=$epochs \
                system.dtype=$dtype \
                ++solver.outer.n_iter=$outer_n_iter \
                metrics.disp_freq=$disp_freq \
                metrics.max_outer_iter=10 \
                metrics.max_inner_iter=10 \
                data.b_size=$b_size \
                system.device=$device \
                ++data.split=0.5 \
                ++data.cv=$2 \
                ++system.num_workers=6 \
                ++solver.outer.lr=$outer_lr \
                ++loss.outer.model.m_opt=True \
                ++solver.inner_forward.optimizer.lr=$inner_lr \
                ++solver.inner_forward.n_iter=$n_iter \
                ++solver.inner_forward.n_outer_iter=$n_outer_iter \
                ++solver.inner_forward.cold_start=True \
                ++solver.inner_backward.entropy_reg=$entropy_reg \
                ++solver.inner_backward.kl=True \
                ++solver.inner_forward.clip=5 \
                ++inner_init=domainnet-$1-s$2/Uni0.pth \
                ++solver.inner_backward.p_opt=True \
                ++clip_m=0.1 \
                ++solver.inner_backward.divide_mu_grad=40 \
                ++data.root=data/datasets/domain_net/$1 \
