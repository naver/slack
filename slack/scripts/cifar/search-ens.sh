#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT

if [[ $1 == '10' ]]; then
    inner_lr=.4
else
    inner_lr=.1
fi
if [[ $1 == '10' && $2 == '40x2' ]]; then
    reg=0.0002
else
    reg=0.0005
fi

outer_lr=4
p_opt=True
m_opt=True
divide_mu_grad=60
cold_start=True
entropy_reg=.005

log_name=c$1-$2-s$3/search-ens
outer_n_iter=4000
dtype=32
loss=cifar-$2
disp_freq=10
data=cifar$1
device=0

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python search-mp.py --multirun solver=slack \
                loss=$loss \
                data=$data \
                logs.log_name=$log_name \
                ++epochs=$epochs \
                system.dtype=$dtype \
                ++solver.outer.n_iter=$outer_n_iter \
                metrics.disp_freq=$disp_freq \
                metrics.max_outer_iter=10 \
                metrics.max_inner_iter=10 \
                data.b_size=128 \
                system.device=$device \
                ++data.split=0.5 \
                ++data.cv=$3 \
                ++system.num_workers=6 \
                ++loss.inner.model.num_classes=$1 \
                ++solver.outer.lr=$outer_lr \
                ++loss.outer.model.m_opt=$m_opt \
                ++solver.inner_forward.optimizer.lr=$inner_lr \
                ++solver.inner_forward.n_iter=1000 \
                ++solver.inner_forward.n_outer_iter=400 \
                ++solver.inner_forward.cold_start=$cold_start \
                ++solver.inner_backward.entropy_reg=$entropy_reg \
                ++solver.inner_backward.kl=True \
                ++solver.inner_forward.clip=5 \
                ++inner_init=c$1-$2-s$3/Uni.pth \
                ++solver.inner_backward.p_opt=$p_opt \
                ++clip_m=0.1 \
                ++loss.inner.reg=$reg \
                ++solver.inner_backward.divide_mu_grad=$divide_mu_grad \
                ++solver.inner_forward.reduce_gradients=False \
                ++system.dataparallel=True \
                ++rank_init=True \
                #++solver.inner_backward.average_mu_grad=True \
                #++solver.inner_backward.mu_entropy=True \
                #++data.val_b_size=256 \
