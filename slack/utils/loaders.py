# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

from operator import itemgetter
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torchvision import transforms
from .augment_dataset import DataAugmentation
import numpy as np
import os
import glob
import cv2
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


CIFAR_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_SIZE = (224, 224)  # (width, height) may set to (244, 224)

MEANS = dict(
    CIFAR10=CIFAR_MEAN,
    CIFAR100=CIFAR_MEAN,
    IMAGENET=[0.485, 0.456, 0.406],
    DOMAINNET=[0.485, 0.456, 0.406],
)
STDS = dict(
    CIFAR10=CIFAR_STD,
    CIFAR100=CIFAR_STD,
    IMAGENET=[0.229, 0.224, 0.225],
    DOMAINNET=[0.229, 0.224, 0.225],
)
MEANS = {k: np.array(v, dtype=np.float32) for k, v in MEANS.items()}
STDS = {k: np.array(v, dtype=np.float32) for k, v in STDS.items()}


class SLACKIterator:
    def __init__(
        self,
        root,
        train,
        batch_size,
        device,
        dtype,
        sampler=None,
        test=False,
        name="CIFAR10",
        distributed=False,
        project_mag=False,
        **loader_kwargs,
    ):
        """
        Constructor.

        Args:
            root: str, path to the data directory
            train: bool, whether this iterates over the training data
            batch_size: int
            device: int
            dtype: dtype
            sampler: instance of torch.utils.data.Sampler, give the indices
                of the train or val split from the full training data
            test: bool, whether to fetch data from the 'test' directory
            name: str, name of the dataset
            distributed: bool, whether to distribute the dataset
            project_mag: bool, whether to re-normalize magnitudes

        """
        self.name = name.upper()
        self.root = root
        self.epoch = 0
        self.train = train
        self.test = test
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.sampler = sampler
        self.distributed = distributed
        self.project_mag = project_mag
        self.loader_kwargs = loader_kwargs

    def build(self, outer_model=None):
        """Initializes the data loader."""
        self.outer_model = outer_model
        self.train_outer = False
        try:
            self.train_outer = (
                outer_model is not None and next(outer_model.parameters()).is_cuda
            )
        except:
            pass
        if self.train:
            self.data_augmentation = DataAugmentation(
                augmentations=outer_model.augmentations,
                dataset=self.name,
                project_mag=self.project_mag,
            ).cuda()
            if self.name == "IMAGENET":
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            256, interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.RandomCrop(224),
                        transforms.ToTensor(),
                    ]
                )
            elif self.name == "DOMAINNET":
                domain = os.path.basename(self.root)
                assert domain in ['sketch', 'painting', 'real', 'infograph', 'quickdraw', 'clipart'], f'Invalid domain {domain}.'
                if domain in ['sketch', 'clipart', 'quickdraw']:
                    transform = transforms.Compose(
                        [
                            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC,),
                            transforms.ToTensor(),
                        ]
                    )
                else:
                    transform = transforms.Compose(
                        [
                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                        ]
                    )
            else:
                transform = transforms.Compose([transforms.ToTensor()])
        else:
            if "CIFAR" in self.name:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
                )
            elif self.name == "IMAGENET":
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            256, interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(MEANS["IMAGENET"], STDS["IMAGENET"]),
                    ]
                )
            elif self.name == "DOMAINNET":
                domain = os.path.basename(self.root)
                assert domain in ['sketch', 'painting', 'real', 'infograph', 'quickdraw', 'clipart'], f'Invalid domain {domain}.'
                if domain in ['sketch', 'clipart', 'quickdraw']:
                    transform = transforms.Compose(
                        [
                            transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(MEANS["DOMAINNET"], STDS["DOMAINNET"]),
                        ]
                    )
                else:
                    transform = transforms.Compose(
                        [
                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(MEANS["DOMAINNET"], STDS["DOMAINNET"]),
                        ]
                    )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(MEANS[self.name], STDS[self.name]),
                    ]
                )

        if self.name.upper() in ["IMAGENET", "DOMAINNET"]:
            split = "train"
            if self.test:
                split = "val" if self.name.upper() == "IMAGENET" else "test"
            self.dataset = ImageFolder(
                root=os.path.join(self.root, split), transform=transform
            )
            n_workers = 6  # 3*(self.train+1)
        else:
            name_to_cls = {
                "CIFAR10": CIFAR10,
                "CIFAR100": CIFAR100,
            }
            CDataset = name_to_cls[self.name.upper()]
            self.dataset = CDataset(
                root=self.root,
                transform=transform,
                train=(not self.test),
                download=False,
            )
            n_workers = 4 if self.distributed else 6 * (self.train + 1)
        print(transform.transforms)
        batch_size = self.batch_size
        rank = 0
        if self.distributed:
            rank = dist.get_rank()
            size = dist.get_world_size()
            self.sampler = DistributedSamplerWrapper(self.sampler, size, rank)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=n_workers,  # 0 if self.distributed else n_workers,
            drop_last=self.train,
            shuffle=self.train and self.sampler is None,
            pin_memory=True,
            **self.loader_kwargs,
        )
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        """Fetches a new batch from the loader."""
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
            self.dataset.batch_idx = 0
            self.epoch += 1
        return idx

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.iterator)


class DatasetFromSampler(Dataset):
    """Dataset created from a torch Sampler."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Distributed sampler that subsamples from a sampler."""
    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        """
        Constructor.

        Args:
            sampler: instance of torch.utils.data.Sampler
            num_replicates: int, number of independent subsamplers
            rank: int, rank of current device
            shuffle: bool, whether to shuffle data before sampling

        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
