# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch
import torch.nn as nn


def to_device(data, device, dtype):
    if isinstance(data, list):
        data = tuple(data)
    if type(data) is tuple:

        data = tuple(
            [
                to_type(d.to(device), dtype) if isinstance(d, torch.Tensor) else d
                for d in data
            ]
        )
        # for d in data:
        # 	if not d.is_sparse:
        # 		d.requires_grad = True
        if len(data) == 1:
            data = data[0]
    elif isinstance(data, torch.Tensor):
        data = to_type(data.to(device), dtype)
        # if not data.is_sparse:
        # 	data.requires_grad = True
    else:
        data = to_type(data, dtype)
        # if not data.is_sparse:
        # 	data.requires_grad = True
    return data


def to_type(data, dtype):
    if dtype == torch.double:
        return data.double()
    else:
        return data.float()


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def add_prefix_to_keys_dict(dico, prefix):
    keys = list(dico.keys())
    for key in keys:
        dico[prefix + key] = dico.pop(key)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return 100.0 * (predictions == targets).sum().float() / targets.size(0)


def clip_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p for p in parameters if p is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    if norm_type == "inf":
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm
