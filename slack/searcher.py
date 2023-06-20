# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch
import numpy as np
import os
import time
import copy
import yaml

import utils.helpers as hp
import torch.distributed as dist

from core import utils
from utils.helpers import Config, config_to_dict, config_to_instance
from warmup_scheduler import GradualWarmupScheduler


class Trainer(object):
    def __init__(self, args, rank=0):
        args = config_to_dict(args)
        self.args = Config(args)
        if rank == 0:
            logdir = os.path.join(self.args.logs.log_dir, self.args.logs.log_name)
            os.mkdir(logdir)
            with open(os.path.join(logdir, "metadata.yaml"), "w") as f:
                yaml.dump(args, f)
        self.rank = rank
        self.train_outer = self.args.get("train_outer", True)
        self.rand_augment = self.args.get("rand_augment", True)
        name = self.args.logs.log_name
        self.distributed = self.args.system.get("dataparallel", False)
        self.device = (
            hp.assign_device(self.args.system.device)
            if not self.distributed
            else self.rank
        )
        self.dtype = hp.get_dtype(self.args.system.dtype)
        self.build_model()
        self.iteration = self._iteration
        self.eval_losses = self._eval_losses

    def build_model(self):

        # create data loaders for inner and outer problems
        self.count_max = self.args["solver"]["outer"].pop("n_iter")
        self.inner_loader, self.outer_loader, self.data_info = hp.get_data_loaders(
            self.args.data,
            self.args.system.num_workers,
            self.dtype,
            self.device,
            self.distributed,
        )
        # create either a pytorch Module or a list of parameters
        self.inner_model = hp.get_model(
            self.args["loss"]["inner"].pop("model"), self.device
        )
        self.inner_model = self.inner_model.to(self.device)
        self.inner_model = utils.to_type(self.inner_model, self.dtype)
        self.outer_model = utils.to_type(
            hp.get_model(
                self.args["loss"]["outer"].pop("model"),
                self.device if self.train_outer else "cpu",
            ),
            self.dtype,
        )

        # create a pytorch Modules whose output is a scalar
        self.inner_loss, self.outer_loss = hp.get_losses(
            self.args.loss,
            self.outer_model,
            self.inner_model,
            self.data_info,
            self.device,
        )
        # Construct the approximate solution to inner problem
        self.inner_params = self.inner_loss.inner_params
        self.outer_params = self.inner_loss.outer_params

        self.optimizer = None
        inner_optimizer = None
        self.scheduler = None
        if self.train_outer:
            self.optimizer = hp.get_optimizer(self.args.solver.outer, self.outer_params)
        if "inner" in self.args.solver:
            schedule = self.args["solver"]["inner"].pop("schedule", None)
            warmup_schedule = self.args["solver"]["inner"].pop("warmup_schedule", None)
            inner_optimizer = hp.get_optimizer(
                self.args.solver.inner, self.inner_params
            )
            self.args["solver"]["inner_forward"]["optimizer"] = inner_optimizer
            if schedule:
                print(f"Using scheduler {schedule}.")
                self.scheduler = config_to_instance(
                    optimizer=inner_optimizer,
                    T_max=self.count_max,
                    eta_min=0.0,
                    **schedule,
                )
            if warmup_schedule:
                print(f"Using warmup {warmup_schedule}")
                self.scheduler = GradualWarmupScheduler(
                    optimizer=inner_optimizer,
                    after_scheduler=self.scheduler,
                    **warmup_schedule,
                )
        self.forward_solver, self.backward_solver = hp.get_solvers(self.args.solver)

        if self.train_outer or self.rand_augment:
            self.inner_loader.build(self.outer_loss.outer_model)
        else:
            self.inner_loader.build()
        self.outer_loader.build()

        self.counter = 0
        self.outer_grad = None
        self.alg_time = 0.0

        if "inner_init" in self.args:
            if self.args.get("rank_init", False):
                inner_init = (
                    str(self.rank)
                    .join(os.path.splitext(self.args.inner_init))
                    .split("/")
                )
                path = os.path.join(self.args.logs.log_dir, *inner_init)
            else:
                path = os.path.join(
                    self.args.logs.log_dir, *self.args.inner_init.split("/")
                )
            print(f"Resuming inner_model from {path}.")
            state_dict = torch.load(path)
            if "inner_model" in state_dict:
                state_dict = state_dict.get("inner_model")
            else:
                state_dict = state_dict.get("model")
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.inner_loss.inner_model.load_state_dict(state_dict)
        self.forward_solver.ctx["y0"] = copy.deepcopy(self.inner_params)
        self.forward_solver.ctx["distributed"] = self.distributed
        self.backward_solver.ctx["distributed"] = self.distributed

        if "outer_init" in self.args:
            path = os.path.join(
                self.args.logs.log_dir, *self.args.outer_init.split("/"), "models.ckpt"
            )
            if os.path.exists(path):
                print(f"Resuming outer_model from {path}.")
                state_dict = torch.load(path)["outer_model"]
                self.outer_loss.outer_model.load_state_dict(state_dict)
            else:
                print(path, "does not exist, starting from random augmentation model.")

        if "resume" in self.args:
            lid = self.args.resume
            path = os.path.join(
                self.args.logs.log_dir, self.args.logs.log_name, str(lid), "models.ckpt"
            )
            state_dict = torch.load(path)
            print(f"Resuming from {path} at step {state_dict['step']}")
            if self.args.get("load_inner", True):
                print("Loading inner model.")
                self.inner_loss.inner_model.load_state_dict(state_dict["inner_model"])
                # self.forward_solver.ctx.optimizer.load_state_dict(
                #    state_dict["inner_optimizer"]
                # )
            if "outer_model" in state_dict:
                self.outer_loss.outer_model.load_state_dict(state_dict["outer_model"])
            if not self.args.data.get("finetune", False):
                self.counter = state_dict["step"]
                self.alg_time = state_dict["time"]
                if self.scheduler is not None:
                    for _ in range(self.counter):
                        self.scheduler.step()
                if "outer_optimizer" in state_dict:
                    self.optimizer.load_state_dict(state_dict["outer_optimizer"])

    def main(self):
        done = False
        while not done:
            self.train_epoch()
            done = True

    def save_model(self):
        log_path = os.path.join(self.args.logs.log_dir, self.args.logs.log_name)
        state_dict = dict(
            step=self.counter,
            time=self.alg_time,
            # inner_model=self.inner_loss.inner_model.state_dict(),
            # inner_optimizer=self.forward_solver.ctx.optimizer.state_dict(),
        )
        if self.scheduler:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if "inner_params" not in self.backward_solver.ctx:
            try:
                state_dict[
                    "am_optimizer"
                ] = self.backward_solver.am_optimizer.state_dict()
            except:
                pass
        if self.train_outer:
            self.outer_model.save_genotype(log_path, "%05d" % self.counter)
            state_dict["outer_model"] = self.outer_loss.outer_model.state_dict()
            state_dict["outer_optimizer"] = self.optimizer.state_dict()
        torch.save(state_dict, os.path.join(log_path, "models.ckpt"))

    def train_epoch(self):
        if self.train_outer:
            self.optimizer.zero_grad()

        accum_dict = dict()
        log_path = os.path.join(self.args.logs.log_dir, self.args.logs.log_name)

        try:
            self.outer_model.make_log_dirs(log_path)
        except:
            pass

        while self.counter <= self.count_max:
            for batch_idx, data in enumerate(self.outer_loader):
                display_every = self.forward_solver.ctx.get("n_outer_iter", 250)
                if display_every > 1000:
                    display_every = 250
                if self.counter % display_every == 0:
                    if self.rank == 0:
                        self.save_model()
                    print("Evaluating losses.")
                    out_dict = self.eval_losses(max_iter=50)
                    print("Done.")
                    if self.distributed:
                        keys = ["outer_loss", "outer_acc", "inner_loss", "inner_acc"]
                        out_tensor = torch.Tensor([out_dict[k] for k in keys]).cuda()
                        world_size = dist.get_world_size()
                        dist.all_reduce(out_tensor, op=dist.ReduceOp.SUM)
                        out_tensor /= world_size
                        out_tensor = out_tensor.cpu().numpy()
                        out_dict = {k: v for k, v in zip(keys, out_tensor)}

                    log_path = os.path.join(
                        self.args.logs.log_dir, self.args.logs.log_name
                    )
                    if self.rank == 0:
                        open(os.path.join(log_path, "val.txt"), "a").write(
                            " ".join(
                                list(
                                    map(
                                        lambda x: str(round(x, 4)),
                                        (
                                            self.counter,
                                            out_dict["outer_loss"],
                                            out_dict["outer_acc"],
                                            out_dict["inner_loss"],
                                            out_dict["inner_acc"],
                                        ),
                                    )
                                )
                            )
                            + "\n"
                        )

                if (
                    self.counter + 1
                ) % self.args.metrics.disp_freq == 0 and self.rank == 0:
                    accum_dict = {
                        key: sum(value) / len(value)
                        if (isinstance(value, list) or isinstance(value, tuple))
                        else value
                        for key, value in accum_dict.items()
                    }
                    metrics = {"iter": self.counter + 1, "time": self.alg_time}
                    metrics.update(accum_dict)
                    metrics = {k: round(v, 3) for k, v in metrics.items()}
                    # self.log_metrics(metrics)
                    print(metrics)
                    if self.train_outer:
                        self.outer_loss.outer_model.print_genotype()

                    accum_dict = {key: [] for key in accum_dict.keys()}
                self.counter += 1
                out_dict = self.iteration(data)
                accum_dict = {
                    key: accum_dict.get(key, []) + [value]
                    for key, value in out_dict.items()
                }
                if self.counter >= self.count_max:
                    if self.rank == 0:
                        self.save_model()
                    out_dict = self.eval_losses(max_iter=False)
                    if self.distributed:
                        keys = ["outer_loss", "outer_acc", "inner_loss", "inner_acc"]
                        out_tensor = torch.Tensor([out_dict[k] for k in keys]).cuda()
                        world_size = dist.get_world_size()
                        dist.all_reduce(out_tensor, op=dist.ReduceOp.SUM)
                        out_tensor /= world_size
                        out_tensor = out_tensor.cpu().numpy()
                        out_dict = {k: v for k, v in zip(keys, out_tensor)}
                    metrics = {"iter": self.counter, "time": self.alg_time}
                    metrics.update(out_dict)
                    log_path = os.path.join(
                        self.args.logs.log_dir, self.args.logs.log_name, "val.txt"
                    )
                    if self.rank == 0:
                        open(log_path, "a").write(
                            " ".join(
                                list(
                                    map(
                                        lambda x: str(round(x, 4)),
                                        (
                                            self.counter,
                                            out_dict["outer_loss"],
                                            out_dict["outer_acc"],
                                            out_dict["inner_loss"],
                                            out_dict["inner_acc"],
                                        ),
                                    )
                                )
                            )
                            + "\n"
                        )
                        metrics = {
                            "Final " + k: np.round(v).astype(np.int).tolist()
                            if (isinstance(v, list) or isinstance(v, tuple))
                            else round(v, 3)
                            for k, v in metrics.items()
                        }
                        # self.log_metrics(metrics)
                    return

    def _iteration(self, data):
        start_time_iter = time.time()
        self.outer_model.sample()
        if len(data) == 1 and isinstance(data, list):
            data = data[0]
        data = utils.to_device(data, self.device, self.dtype)

        inner_loader, inner_params = self.inner_loader, self.inner_params
        iterates, *eval_args = self.forward_solver.run(
            self.inner_loss, inner_loader, inner_params
        )
        _, _, update_anchor = iterates
        if (
            isinstance(update_anchor, bool)
            and isinstance(update_anchor, bool)
            and update_anchor
        ):
            print("Resetting optimizer.")
            self.optimizer = hp.get_optimizer(self.args.solver.outer, self.outer_params)
        utils.zero_grad(inner_params)
        info = dict()
        if self.train_outer and self.counter > self.args.get("burn", -1):

            utils.zero_grad(self.outer_params)

            loss = self.outer_loss(data)
            torch.autograd.backward(
                loss,
                retain_graph=False,
                create_graph=False,
                inputs=inner_params + self.outer_params,
            )
            if self.distributed:
                size = float(torch.distributed.get_world_size())
                for param in inner_params:
                    dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                    param.grad.data /= size
            loss = loss.detach().cpu()
            for p in inner_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            inner_grads = [p.grad for p in inner_params]

            out, info = self.backward_solver.run(
                inner_loader,
                self.outer_params,
                iterates,
                inner_grads,
            )

            for p, o in zip(self.outer_params, out):
                if p.grad is not None:
                    p.grad = p.grad + o
                else:
                    p.grad = 1.0 * o
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.args.get("clip_m", None):
                mu = tuple(self.outer_model.parameters())[-1]
                mu.data.clamp_(self.args.clip_m, 1)
            self.outer_grad = None
            end_time_iter = time.time()
            self.alg_time += end_time_iter - start_time_iter

        out_dict = self.eval_losses(*eval_args)
        out_dict.update(info)
        if self.scheduler is not None:
            self.scheduler.step()

        return out_dict

    def _eval_losses(self, inner_loss=None, inner_acc=None, max_iter=True):
        max_iter = max_iter if max_iter else -1
        out_dict = {}
        self.inner_loss.inner_model.eval()
        with torch.no_grad():
            out_dict.update(
                self.eval_loss(
                    self.data_info["eval_outer_loader"],
                    self.outer_loss,
                    max_iter=self.args.metrics.max_outer_iter
                    if isinstance(max_iter, bool)
                    else max_iter,
                    inner=False,
                )
            )
            if inner_loss is None:
                print("Done for outer, evaluating inner.")
                out_dict.update(
                    self.eval_loss(
                        self.data_info["eval_inner_loader"],
                        self.inner_loss,
                        max_iter=self.args.metrics.max_inner_iter
                        if isinstance(max_iter, bool)
                        else max_iter,
                        inner=True,
                    )
                )
            else:
                inner_loss = inner_loss.item()
                out_dict.update({"inner_loss": inner_loss, "inner_acc": 100*inner_acc})

        self.inner_loss.inner_model.train()

        return out_dict

    def eval_loss(self, loader, func, max_iter, inner):
        loss = 0.0
        acc = 0.0
        cnt = 0
        if max_iter <= 0:
            max_iter = len(loader)
        for index in range(max_iter):
            data = next(loader)
            data = utils.to_device(data, self.device, self.dtype)
            if inner:
                data = loader.data_augmentation(*data)
            _loss, _acc = func(data, with_acc=True, reg=False)
            b = len(data[0])
            loss += _loss.item() * b
            acc += _acc * b
            cnt += b
        acc = acc / cnt
        loss = loss / cnt
        out = {"loss": loss, "acc": 100 * acc}
        if inner:
            utils.add_prefix_to_keys_dict(out, "inner_")
        else:
            utils.add_prefix_to_keys_dict(out, "outer_")
        return out
