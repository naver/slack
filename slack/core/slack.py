# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.distributions import Categorical
from utils.helpers import config_to_instance


def identity(x):
    return x


def sample_d(x, Distrib, size=()):
    """Instantiates a distribution with logits values 'x' and samples 'size' elements from it."""
    dp = Distrib(logits=x)
    _p = dp.sample(size)
    logp = dp.log_prob(_p)
    return _p, logp


class SLACK(nn.Module):
    """Augmentation model."""

    def __init__(
        self,
        augmentations,
        n_op,
        Dmu,
        dmu,
        m_opt=True,
        mu_prior=1,
        init_mu=None,
        b_size=1,
        mu_d=None,
    ):
        """
        Constructor. Initializes he augmentation parameters.

        Args:
            augmentations: dict, instantiable core.augmentations_torch.Augmentation subclass.
            n_op: int, number of augmentations composed together
            Dmu: str, class name for magnitude parametrization, e.g. core.smoothed_uniform.SmoothedUniform
            dmu: dict of parameters passed as arguments to Dmu, with instantiable pre-processing functions (e.g. softmax,
                sigmoid). If no pre-processing function is provided, the magnitude parameter is optimized as is.
            m_opt: bool, whether to optimize the magnitude
            init_mu: str, instantiable method for initializing the magnitude
            b_size: int, number of composite augmentations to independently sample
            mu_d: int, dimension of magnitude distribution (only for multi-dimensional distributions)

        """
        super(SLACK, self).__init__()
        self.augmentations = config_to_instance(**augmentations)
        self.n_op = n_op
        self.Dmu = config_to_instance(name=Dmu)
        self.dmu = [
            (k, config_to_instance(name=v) if v else identity) for k, v in dmu.items()
        ]
        if init_mu:
            init_mu = config_to_instance(name=init_mu)
        self.init_mu = init_mu
        self.mu_prior = mu_prior
        self.mu_d = mu_d
        self.b_size = b_size
        self.m_opt = m_opt
        self.n_aug = len(self.augmentations.names)
        self._initialize_augment_parameters()
        self.batch_idx = -1
        self.b_idx = 0
        self.imq = []

    def _initialize_augment_parameters(self):
        """Initializes the model parameters."""
        self.n_aug = len(self.augmentations.names)
        pi = torch.zeros(self.n_op, self.n_aug).cuda()
        self.pi = nn.Parameter(pi, requires_grad=True)
        mu_shape = (self.n_aug, len(self.dmu))
        if self.mu_d is not None:
            mu_shape += (self.mu_d,)
        if self.init_mu is not None:
            mu = self.init_mu(mu_shape)
        else:
            log_prior = not (any([y == identity for (x, y) in self.dmu]))
            mu = (np.log(self.mu_prior) if log_prior else self.mu_prior) * torch.ones(
                *mu_shape
            ).cuda()
        if self.m_opt:
            self.mu = nn.Parameter(mu, requires_grad=self.m_opt)
            self._augment_parameters = [self.pi, self.mu]
        else:
            self.mu = mu
            self._augment_parameters = [self.pi]

    def sample(self, b_size=None):
        """Sampled b_size composite augmentations with their magnitudes."""
        if b_size is None:
            b_size = self.b_size
        self._pi, self._logpi = sample_d(self.pi, Categorical, (b_size,))
        _mu = self.mu[self._pi]
        _mu = _mu.transpose(0, 2).transpose(1, 2)
        dmu = self.Dmu(**{k: v(m) for (k, v), m in zip(self.dmu, _mu)})
        _mu = dmu.sample()
        self._logmu = dmu.log_prob(_mu)
        self._mu = _mu
        self.logp = self._logpi
        if self.m_opt:
            self.logp += self._logmu
        return self._pi, self._mu

    def genotype(self):
        """Returns the n_op lists of transformations with their probabilities and magnitude upper-bounds."""
        m = self.dmu[0][1](self.mu)
        p = nn.functional.softmax(self.pi, dim=-1)  # (15)
        ids = torch.argsort(p, dim=-1)
        gene = []
        for jl in zip(*[reversed(x) for x in ids]):
            gene.append([])
            for i, j in enumerate(jl):
                if self.mu_d or not self.m_opt:
                    _m = []
                else:
                    _m = (m[j] if len(m.shape) <= 2 else m[i, j]).data.detach().tolist()
                    _m = list(map(lambda x: str(round(x, 2)), _m))
                _p = round(p[i, j].data.detach().item(), 3)
                gene[-1] += [self.augmentations.names[j], str(_p), ";".join(_m)]
        return gene

    def plot_genotype(self, log_path, counter=None):
        """Plots a visualisation of the current policy."""
        log_path = os.path.join(log_path, "plt_genotype")
        if counter is not None:
            log_path = os.path.join(log_path, str(counter))
        p = nn.functional.softmax(self.pi, dim=-1).detach().cpu().numpy()  # (15)
        N = len(self.augmentations.names)
        npi = len(p) if len(p.shape) == 2 else 1
        mu = self.mu
        if len(self.mu.shape) <= 2 or self.mu_d is not None:
            mu = mu[None]
        nm = len(mu)
        cols = nm + 1 + 2 * npi
        plt.figure(figsize=(cols, self.n_aug), constrained_layout=True)
        for i, _m in enumerate(mu):
            for j, __m in enumerate(_m):
                if self.augmentations.names[j] in [
                    "Identity",
                    "AutoContrast",
                    "Invert",
                    "Equalize",
                ]:
                    continue
                ax = plt.subplot2grid((N, cols), (j, i), colspan=2, rowspan=1)
                x = torch.arange(0.01, 1, 0.01).to(self.mu.device)
                dmu = self.Dmu(**{k: v(m) for (k, v), m in zip(self.dmu, __m)})
                try:
                    pmu = torch.zeros_like(x)
                    pmu[dmu.support.check(x)] = (
                        dmu.log_prob(x[dmu.support.check(x)]).detach().exp()
                    )
                except:
                    pmu = dmu.log_prob(x).detach().exp()
                x = x.detach().cpu().numpy()
                ax.plot(x, pmu.cpu().numpy())
                ax.set_xticks([x[0], x[-1]])
                ax.set_ylim(0)
                name = self.augmentations.names[j]
                if name[-1].isdigit():
                    xticks = self.augmentations.augment_dict[name[:-1]][int(name[-1])][1:]
                else:
                    xticks = self.augmentations.augment_dict[self.augmentations.names[j]][1:]
                ax.set_xticklabels(xticks)
        if len(p.shape) == 1:
            p = p[None]
        for i, _p in enumerate(p):
            ax1 = plt.subplot2grid((N, cols), (0, nm + 1 + i * 2), rowspan=N, colspan=2)
            ax1.imshow(_p[:, None], cmap="BuPu")
            ax1.set_xticks([])
            if i:
                ax1.set_yticks([])
            else:
                ax1.set_yticks(np.arange(len(self.augmentations.names)))
                ax1.set_yticklabels(self.augmentations.names)
        plt.tight_layout()
        plt.savefig(log_path)
        plt.close()

    def make_log_dirs(self, log_path):
        """Creates log directories."""
        for name in ["genotype", "pi", "mu", "plt_genotype"]:
            os.mkdir(os.path.join(log_path, name))

    def save_genotype(self, log_path, counter=None):
        """Saves model parameters and plots."""
        self.plot_genotype(log_path, counter)
        genotype = self.genotype()
        dst_path = os.path.join(log_path, "genotype")
        np.savetxt(
            os.path.join(dst_path, f"{counter}.txt")
            if counter is not None
            else dst_path + ".txt",
            genotype,
            fmt="%s",
            header="name pi mu",
        )
        for name, p in zip(
            ["pi", "mu"] if self.m_opt else ["pi"], self._augment_parameters
        ):
            p = p.detach().cpu()
            dst_path = (log_path, name)
            if counter is not None:
                dst_path += (str(counter),)
            np.save(
                os.path.join(*dst_path),
                nn.functional.softmax(p, dim=-1).numpy() if name == "pi" else p.numpy(),
            )

    def print_genotype(self):
        """Prints the result from self.genotype()."""
        print(", ".join(["{} p:{} m:{}"] * self.n_op).format(*self.genotype()[0]))

    def forward(self, data):
        return data
