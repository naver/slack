# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import numpy as np
import torch
from torchvision import transforms
from core.augmentations_torch import CutoutAbs

CIFAR_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
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


class DataAugmentation(torch.nn.Module):
    """Class for applying the augmentations sampled from the current policy."""
    def __init__(
        self,
        augmentations,
        dataset,
        pre_aug=None,
        post_aug=None,
        sampler=None,
        project_mag=False,
    ):
        """
        Constructor, intiializes transformations applied before/after SLACK.

        Args:
            augmentations: Augmentation subclass from core.augmentations (train) or core.augmentations_torch (search).
            dataset: str, name of the dataset
            pre_aug: instance of torchvision.transforms.Compose, transformations to apply before SLACK
            post_aug: instance of torchvision.transforms.Compose, transformations to apply after SLACK
            sampler: sampling method from core.slack.SLACK (used for training)
            project_mag: bool, whether to re-normalize sampled magnitudes for them to have an average of 0.5

        """
        super(DataAugmentation, self).__init__()
        self.augmentations = augmentations
        self.dataset = dataset
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        if self.post_aug is None:
            self.post_aug = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(MEANS[self.dataset], STDS[self.dataset]),
                ]
            )
        self.sampler = sampler
        self.project_mag = project_mag
        if self.project_mag and not isinstance(project_mag, float):
            self.project_mag = 0.5

    def __call__(self, images, labels=None, *args):
        """Applies the augmentations passed through *args to the image(s)."""
        if self.pre_aug is not None:
            images = self.pre_aug(images)
        cutout = False
        cutout_level = 0
        if args:
            pi, mu = args
            if not bool(pi.shape):
                pi, mu = pi[None], mu[None]
            if self.project_mag:
                mu = self.project_mag * len(mu) * mu / mu.sum()
            if any(["ResizeCrop" in self.augmentations.names[p] for p in pi]):  # if
                i = next(
                    i
                    for i, p in enumerate(pi)
                    if "ResizeCrop" in self.augmentations.names[p]
                )
                _, low, high = self.augmentations.get_augment(self.augmentations.names[pi[i]])
                assert high == 0.95
                m = mu[i].data.detach().item()
                m = min(m, 1) * high
                images = transforms.RandomResizedCrop(
                    224,
                    scale=(1 - float(m), 1 - float(m)),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )(images)
            if isinstance(images, torch.Tensor):
                images = (images * 255).byte()
            p_all = set()
            for i, (p, m) in enumerate(zip(pi, mu)):
                name = self.augmentations.names[p]
                if name in p_all or "ResizeCrop" in name:
                    continue
                if name == "Cutout":
                    cutout = i + 1
                    cutout_level = m
                else:
                    images = self.augmentations.apply_augment(
                        images, self.augmentations.names[p], m
                    )
            if isinstance(images, torch.Tensor):
                images = images.float() / 255
            else:
                images = transforms.ToTensor()(images)
        if self.post_aug is not None:
            images = self.post_aug(images)
        if cutout:
            _, low, high = self.augmentations.get_augment("Cutout")
            cutout_level = torch.clamp(cutout_level, 0, 1)
            cutout_level = cutout_level * (high - low) + low
            v = int(cutout_level * images.shape[-1])
            images = CutoutAbs(images, v)
        return images, labels

    def sample_and_transform(self, images):
        """Samples a composite transformation and applies it to the image(s)."""
        assert self.sampler is not None
        pi, mu = self.sampler()
        out = self(images, None, pi[0], mu[0])[0]
        return out
