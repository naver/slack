# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch
import torch.nn as nn
from torch import autograd
from utils.helpers import get_optimizer, config_to_instance
import core.utils as utils
from warmup_scheduler import GradualWarmupScheduler
import torch.distributed as dist
import copy
import numpy as np
import torch.nn.functional as F


def reduce_tensor(rt: torch.Tensor):
    """Reduces tensor across all nodes."""
    world_size = dist.get_world_size()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def reduce_gradients(params):
    """Reduces tensor across all nodes."""
    size = dist.get_world_size()
    for param in params:
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


class ForwardSolver(object):
    """Base class."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.counter = 0
        self.counter_grad = 0

    def run(self, func, generator, params):
        return NotImplementedError("Not implemented")


class ForwardREINFORCE(ForwardSolver):
    """Solves the inner problem."""
    def __init__(self, ctx):
        super(ForwardREINFORCE, self).__init__(ctx)
        self.optimizer = self.ctx.optimizer
        self.schedule = self.ctx.get("schedule", None)
        self.warmup = None
        if self.schedule is not None:
            self.warmup = self.schedule.pop("warmup", None)
        self.t = 0

    def reinit(self, inner_params):
        """Reinitializes optimizer and network parameters if at beginning of a round."""
        noi = self.ctx.get("n_outer_iter", None)
        if noi and self.t % noi == 0:
            n_iter = self.ctx.get("n_iter", None)
            if self.ctx.get("cold_start", True):
                for p, p0 in zip(inner_params, self.ctx["y0"]):
                    p.data = copy.deepcopy(p0.data)
            self._optimizer = get_optimizer(self.optimizer, inner_params)
            self._scheduler = None
            if self.schedule is not None:
                self._scheduler = config_to_instance(
                    optimizer=self._optimizer,
                    T_max=n_iter if not noi else n_iter + noi - 1,
                    eta_min=0.0,
                    **self.schedule,
                )
                if self.warmup is not None:
                    print(f"Using warmup {self.warmup}")
                    self._scheduler = GradualWarmupScheduler(
                        optimizer=self._optimizer,
                        after_scheduler=self._scheduler,
                        **self.warmup,
                    )
        else:
            n_iter = 1
        return n_iter

    def run(self, func, generator, inner_params):
        """
        Makes either a full re-training or a single gradient step. For each step, generates
        a batch and augments it with 8 composite transformations independently sampled.

        Args:
            func: subclass of core.losses.Loss
            generator: loader from utils.loaders, encapsulated the outer model
            inner_params: network parameters

        Returns:
            for_backward: tuple to pass to the backward solver, with grads containing
                the 8 inner gradients on the transformed training batches
            mean_val: training loss averaged over the 8 batches
            mean_acc: training accuracy averaged over the 8 batches

        """
        n_iter = self.reinit(inner_params)
        with torch.enable_grad():
            for i in range(n_iter):
                for p in inner_params:
                    if p.grad is not None:
                        p.grad.zero_()
                generator.outer_model.sample()
                d_aug = generator.data_augmentation
                assert d_aug is not None
                inputs, label = next(generator)
                inputs, label = [
                    utils.to_device(data, generator.device, generator.dtype)
                    for data in [inputs, label]
                ]
                inputs = [
                    d_aug(inputs, None, _pi, _mu)[0]
                    for _pi, _mu in zip(
                        generator.outer_model._pi, generator.outer_model._mu
                    )
                ]
                vals = []
                accs = []
                grads = []
                for j, inp in enumerate(inputs):
                    _label = label
                    val, acc = func([inp, _label], with_acc=True)
                    jac = autograd.grad(
                        outputs=val,
                        inputs=inner_params,
                        grad_outputs=None,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                        allow_unused=True,
                    )
                    grads.append(jac)
                    accs.append(acc)
                    vals.append(val)
                mean_acc = sum(accs) / len(accs)
                mean_val = sum(vals) / len(vals)
                self.counter_grad += 1
                if self.ctx.get("clip", None):
                    for g in grads:
                        utils.clip_norm_(g, self.ctx.clip)
                grad_sum = [sum(p) / len(p) for p in zip(*grads)]
                for p, g in zip(inner_params, grad_sum):
                    p.grad = g
                if self.ctx.distributed and self.ctx.get("reduce_gradients", True):
                    reduce_gradients(inner_params)
                self._optimizer.step()
                if self._scheduler is not None:
                    self._scheduler.step()

        self.t += 1
        for_backward = (grads, self.ctx.optimizer.lr, (n_iter > 1))
        return (
            for_backward,
            mean_val,
            mean_acc,
        )


class BackwardSolver(object):
    """Base class."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.counter = 0
        self.counter_grad = 0
        self.counter_jac = 0

    def run(
        self,
        generator,
        outer_params,
        inner_params,
        iterates,
        grad_output,
    ):
        return NotImplementedError("Not implemented")


class BackwardREINFORCE(BackwardSolver):
    """Solves the outer problem."""
    def __init__(self, ctx):
        super(BackwardREINFORCE, self).__init__(ctx)
        self.step = 0
        self.anchor = None

    def run(
        self,
        generator,
        outer_params,
        inner_grad,
        outer_grad,
    ):
        """
        Computes the outer gradient.

        Args:
            generator: loader from utils.loaders, encapsulated the outer model
            outer_params: outer parameters
            inner_grad: 8 gradients over the training data transformed with the sampled augmentations
            outer_grad: gradient over the validation data

        Returns:
            for_backward: tuple to pass to the backward solver, with grads containing
                the 8 inner gradients on the transformed training batches
            mean_val: training loss averaged over the 8 batches
            mean_acc: training accuracy averaged over the 8 batches

        """
        inner_grad, flr, update_anchor = inner_grad
        if self.ctx.get("kl", False) and (
            (isinstance(update_anchor, bool) and update_anchor) or self.anchor is None
        ):
            self.anchor = copy.deepcopy(generator.outer_model.pi.detach())
        for g in outer_grad:
            utils.clip_norm_(g, 5)
        with torch.enable_grad():
            val = [
                -1 / len(inner_grad) * sum(
                    [
                        torch.einsum("...i,...i->", ig, og)
                        for ig, og in zip(inner_gt, outer_grad)
                    ]
                )
                for inner_gt in inner_grad
            ]
            if self.ctx.distributed:
                val = torch.Tensor(val).cuda()
                val = reduce_tensor(val)
                val = list(val)
            logp = generator.outer_model.logp
            reg = None
            if "entropy_reg" in self.ctx:
                reg = self.ctx["entropy_reg"]
            beta = torch.autograd.grad(
                tuple(torch.sum(logp, -1)),
                inputs=outer_params,
                grad_outputs=val,
                retain_graph=bool(reg),
            )
            info = dict()
            wb = flr if self.ctx.get("lr_multiply", True) else 1
            if reg:
                KLDiv = nn.KLDivLoss(reduction="sum")
                _h = KLDiv(
                    F.log_softmax(self.anchor, -1),
                    F.softmax(generator.outer_model.pi, -1),
                )
                inputs = [generator.outer_model.pi]
                dh = torch.autograd.grad(
                    sum(_h) if isinstance(_h, list) else _h,
                    inputs=inputs,
                    allow_unused=True,
                )
                wh = 1
                dh += (None,) * (len(beta) - len(dh))
                nbe = torch.linalg.norm(beta[0].flatten())
                ratio = torch.linalg.norm(wh * reg * dh[0].flatten()) / (wb * nbe)
                info.update({"nh/nb": ratio.item(), "nb": wb * nbe.item()})

                beta = [
                    wb * b + wh * reg * (_dh if _dh is not None else 0)  # !! flr
                    for b, _dh in zip(beta, dh)
                ]
                self.step += 1
            else:
                beta = [wb * b for b in beta]
            if self.ctx.get("divide_mu_grad", False):
                beta[-1] /= self.ctx.divide_mu_grad
            if self.ctx.get("average_mu_grad", False):
                beta[-1] = beta[-1].mean() * torch.ones_like(beta[-1])
            if not self.ctx.get("p_opt", True):
                beta[0] = torch.zeros_like(beta[0])

        return tuple(beta), info
