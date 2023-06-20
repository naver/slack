# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import os
import torch
import torchvision
import importlib
import omegaconf
import utils.loaders as loaders
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler, Sampler


class Config(dict):
    """Subclass of dict for accessing keys with '.' like class attributes."""
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = Config(value)
        return value


def config_to_dict(config):
    """Creates dict from omegaconf DictConfig."""
    out_dict = {}
    for key, value in config.items():
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            out_dict[key] = config_to_dict(value)
        else:
            out_dict[key] = value
    return out_dict


def config_to_instance(**config):
    """
        Instantites the classr or method config[name].
        For a class, arguments are specified as additional entries in the dict.

        Args:
            config: dict cainting the name of the class or method to create

    """
    module, attr = os.path.splitext(config.pop("name"))
    module = importlib.import_module(module)
    attr = getattr(module, attr[1:])
    if config:
        attr = attr(**config)
    return attr


class SubsetSampler(Sampler):
    """Cretes a subsampler."""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def get_optimizer(args, params):
    """Instantiates an optimizer."""
    return config_to_instance(params=params, **args)


def build_solver(args):
    """Instantiates a solver from core.solvers."""
    mod = __import__("core.solvers", fromlist=[args.name])
    klass = getattr(mod, args.name)
    return klass(args)


def get_solvers(args):
    """Instantiated the forward (inner) and backward (outer) solvers."""
    forward_solver = build_solver(args.inner_forward)
    backward_solver = build_solver(args.inner_backward)
    return forward_solver, backward_solver


def get_data_loaders(args, num_workers, dtype, device, distributed=False):
    info = None
    it_kwargs = dict(
        name=args.name,
        root=args.root,
        batch_size=args.b_size,
        device=device,
        dtype=dtype,
        distributed=distributed,
    )
    info = dict()
    name = args.name.upper()
    if name in ["IMAGENET", "DOMAINNET"]:
        targets = torchvision.datasets.ImageFolder(
            root=os.path.join(args.root, "train")
        ).targets
    else:
        name_to_cls = {
            "CIFAR10": torchvision.datasets.CIFAR10,
            "CIFAR100": torchvision.datasets.CIFAR100,
        }
        Dataset = name_to_cls[name.upper().split("_")[0]]
        targets = Dataset(
            root=args.root, train=True, transform=None, download=False
        ).targets
    random_state = args.get("random_state", 0)
    split_idx = args.get("cv", 0)
    sss = StratifiedShuffleSplit(
        n_splits=5, test_size=args.get("split", 0.5), random_state=random_state
    )
    sss = sss.split(list(range(len(targets))), targets)
    for _ in range(split_idx + 1):
        train_idx, valid_idx = next(sss)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    print(f"{len(train_idx)} samples for training, {len(valid_idx)} for validation.")

    inner_loader = loaders.SLACKIterator(
        sampler=train_sampler,
        train=True,
        project_mag=args.get("project_mag", False),
        **it_kwargs,
    )
    if args.get("val_b_size", False):
        it_kwargs["batch_size"] = args.val_b_size
    outer_loader = loaders.SLACKIterator(
        sampler=valid_sampler, train=False, **it_kwargs
    )
    test_loader = loaders.SLACKIterator(train=False, test=True, **it_kwargs)
    info["eval_test_loader"] = test_loader
    eval_inner_loader = inner_loader
    eval_outer_loader = outer_loader
    info.update(
        {
            "eval_outer_loader": eval_outer_loader,
            "eval_inner_loader": eval_inner_loader,
        }
    )

    return inner_loader, outer_loader, info


def get_model(args, device):
    """Instantiates the network."""
    model_path = args.pop("model_path", None)
    model = config_to_instance(**args)
    if model_path:
        state_dict_model = torch.load(model_path, map_location=device)
        model = model.load_state_dict(state_dict_model).to(device)
    return model


def get_loss(args, outer_model, inner_model, info, device):
    """Instantiates the loss from core.losses."""
    args.update(info)
    loss = config_to_instance(
        **args, outer_model=outer_model, inner_model=inner_model, device=device
    )
    return loss


def get_losses(args, outer_model, inner_model, info, device):
    """Instantiates the inner and outer losses."""
    if info is None:
        info = {}
    info.update({"is_outer": False, "params": None})
    inner_loss = get_loss(args.inner, outer_model, inner_model, info, device)
    info["is_outer"] = True
    outer_loss = get_loss(args.outer, outer_model, inner_model, info, device)
    return inner_loss, outer_loss


def assign_device(device):
    if device > -1:
        device = (
            "cuda:" + str(device)
            if torch.cuda.is_available() and device > -1
            else "cpu"
        )
    elif device == -1:
        device = "cuda"
    elif device == -2:
        device = "cpu"
    return device


def get_dtype(dtype):
    if dtype == 64:
        return torch.double
    elif dtype == 32:
        return torch.float
    else:
        raise NotImplementedError("Unkown type")
