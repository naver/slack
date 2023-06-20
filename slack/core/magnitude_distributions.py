# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch
from torch import distributions as D


class SmoothedUniform(D.Distribution):
    def __init__(self, high, scale=0.1, validate_args=False):
        """
        Initializes a smoothed uniform distribution, where a Gaussian CDF
        replaces the indicator from a standard uniform distribution.

        Args:
            high: torch.nn.Parameter, upper-bound of the smoothed distribution
            scale: scale of the Gaussian, controls the slope of the smoothed indactor
            validate_args: bool, checks validity of arguments

        """
        self.high = high
        self.scale = scale
        super(SmoothedUniform, self).__init__(validate_args=validate_args)

    def log_prob(self, x):
        """Log of density function at x."""
        return torch.log(D.Normal(loc=x, scale=self.scale).cdf(self.high) / self.high)

    def sample(self, sample_shape=torch.Size()):
        """Samples a magnitude value."""
        return D.Uniform(0, self.high).sample(sample_shape) + self.scale * D.Normal(
            0, 1
        ).sample(sample_shape)
