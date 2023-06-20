# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    "Base class."
    def __init__(self, outer_model, inner_model, device=None):
        super(Loss, self).__init__()
        self.outer_model = outer_model
        self.inner_model = inner_model
        self.device = device
        self.recompute_features = True
        self.features = None
        if isinstance(inner_model, nn.Module):
            self.inner_params = tuple(inner_model.parameters())
        elif isinstance(inner_model, list) and isinstance(inner_model[0], nn.Module):
            self.inner_params = [tuple(m.parameters()) for m in inner_model]
        else:
            self.inner_params = inner_model
        if isinstance(outer_model, nn.Module):
            self.outer_params = tuple(p for p in outer_model.parameters())
        elif isinstance(outer_model, list) and isinstance(outer_model[0], nn.Module):
            self.outer_params = [tuple(m.parameters()) for m in outer_model]
        else:
            self.outer_params = outer_model

    def forward(self, inputs):
        raise NotImplementedError("Abstract class")


class AugCE(Loss):
    "Loss used for SLACK."
    def __init__(self, outer_model, inner_model, reg=None, device=None, **kwargs):
        """
        Constructor.

        Args:
            outer_model: instance of core.slack.SLACK
            inner_model: network
            reg: float, optinal weight decay

        """
        super(AugCE, self).__init__(outer_model, inner_model, device=device)
        self.reg = reg

    # @torch.autocast('cuda')
    def forward(self, data, with_acc=False, reduction="mean", reg=True):
        """
        Runs a forward pass on the network with the data and computes the loss.

        Args:
            data: torch.Tensor, batch of images
            with_acc: bool, whether to also return the accuracy
            reduction: str, how to aggregate the loss/accuracy over the batch
            reg: bool, whether to apply the weight decay

        """
        x, y = data
        y = y.long()
        if self.outer_model is not None:
            x = self.outer_model(x)
        y_pred = self.inner_model(x)
        out = F.cross_entropy(y_pred, y, reduction=reduction)
        if self.reg and reg:
            out = out + 0.5 * self.reg * sum(
                p.pow(2.0).sum() for p in self.inner_params
            )  # self.reg_term()
        if with_acc:
            pred = y_pred.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
            return out, acc
        else:
            return out
