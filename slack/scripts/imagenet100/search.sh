#!/bin/bash

# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

cd $ROOT

s=$1
inner_lr=.1
outer_lr=.5
entropy_reg=.01
log_name=imagenet100-s$s/search
outer_n_iter=8000
dtype=32
loss=imagenet100
disp_freq=10
data=imagenet100
device=0

HYDRA_FULL_ERROR=1   OC_CAUSE=1 python search.py --multirun solver=slack \
                loss=$loss \
                data=$data \
                logs.log_name=$log_name \
                ++epochs=$epochs \
                system.dtype=$dtype \
                ++solver.outer.n_iter=$outer_n_iter \
                metrics.disp_freq=$disp_freq \
                metrics.max_outer_iter=2 \
                metrics.max_inner_iter=2 \
                data.b_size=256 \
                system.device=$device \
                ++data.split=0.5 \
                ++data.cv=$s \
                ++system.num_workers=6 \
                ++loss.inner.model.num_classes=100 \
                ++solver.outer.lr=$outer_lr \
                ++loss.outer.model.m_opt=True \
                ++solver.inner_forward.optimizer.lr=$inner_lr \
                ++solver.inner_forward.n_iter=2000 \
                ++solver.inner_forward.n_outer_iter=800 \
                ++solver.inner_forward.cold_start=True \
                ++solver.inner_backward.entropy_reg=$entropy_reg \
                ++solver.inner_backward.kl=True \
                ++solver.inner_backward.p_opt=True \
                ++inner_init=imagenet100-s$s/Uni0.pth \
                ++clip_m=0.1 \
                ++solver.inner_backward.divide_mu_grad=40 \
                #++solver.inner_forward.clip=5 \
