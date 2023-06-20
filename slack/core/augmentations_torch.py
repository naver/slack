# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0
# code modified from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T


def affine_fun(img, angle=0, translate=(0, 0), scale=1, shear=0, center=None):
    return F.affine(
        img, angle=angle, translate=translate, scale=scale, shear=shear, center=center
    )


class Augmentations:
    """Base class for Torch augmentations (used for batch augmentation during search),
        with the default setting found in the litterature."""
    def __init__(
        self,
        random_mirror=True,
        augment_dict=None,
        augment_list=None,
    ):
        """
        Constructor. Defines list of augmentation and magnitude ranges, with optional filtering.

        Args:
            random_mirror: bool, whether to mirror transformations w.r.t. which
                the data is usually symmetric (e.g. Shear, Translate, Rotate).
            augment_dict: dict, to remove specific transformations (set value to 'None') or redefine their ranges (set a,b)
            augment_list: list, discards all transformations that are not contains in the list
        """
        self.random_mirror = random_mirror
        if augment_dict is None:
            augment_dict = dict()
        self.names = list(map(lambda x: x[0].__name__, self.default_augment_list()))
        if augment_list is not None:
            self.names = augment_list.split(",")
        for k, v in augment_dict.items():
            augment_dict[k] = tuple(map(float, v.split(","))) if v != "None" else None
        self.augment_dict = {
            fn.__name__: (fn,) + augment_dict.get(fn.__name__, (v1, v2))
            for fn, v1, v2 in self.default_augment_list()
            if (fn.__name__ in self.names and augment_dict.get(fn.__name__, True))
        }
        self.names = list(filter(lambda x: x in self.augment_dict, self.names))

    def ShearX(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return affine_fun(img, shear=[v * 55, 0], center=[0, 0])

    def ShearY(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return affine_fun(img, shear=[0, v * 55], center=[0, 0])

    def TranslateX(self, img, v):
        v = -v * min(img.shape[-2], img.shape[-1])
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return affine_fun(img, translate=[v, 0])

    def TranslateY(self, img, v):
        v = -v * min(img.shape[2], img.shape[-1])
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return affine_fun(img, translate=[0, v])

    def Rotate(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return affine_fun(img, angle=-v.item())

    def Cutout(self, img, v):
        if v <= 0.0:
            return img
        v = int(v * min(img.shape[-2], img.shape[-1]))
        return CutoutAbs(img, v)

    def default_augment_list(self):
        l = [
            (self.ShearX, -0.30, 0.30),  # 0
            (self.ShearY, -0.30, 0.30),  # 1
            (self.TranslateX, -0.45, 0.45),  # 2
            (self.TranslateY, -0.45, 0.45),  # 3
            (self.Rotate, -30, 30),  # 4
            (AutoContrast, 0, 1),  # 5
            (Invert, 0, 1),  # 6
            (Equalize, 0, 1),  # 7
            (Solarize, 0, 256),  # 8
            (Posterize, 0, 4),  # 9
            (Contrast, 0.1, 1.9),  # 10
            (Color, 0.1, 1.9),  # 11
            (Brightness, 0.1, 1.9),  # 12
            (Sharpness, 0.1, 1.9),  # 13
            (self.Cutout, 0, 0.7),  # 14
            (Identity, 0, 1), # 16
        ]
        return l

    def get_augment(self, name, i=None):
        return self.augment_dict[name] if i is None else self.augment_dict[name][i]

    def apply_augment(self, img, name, level):
        uint_input = True
        assert isinstance(img, torch.Tensor), "Image should be Tensor"
        if img.dtype != torch.uint8:
            uint_input = False
            img = (img * 255).byte()
        if name[-1].isdigit():
            _name, i = name[:-1], name[-1]
            augment_fn, low, high = self.get_augment(_name, int(i))
        else:
            augment_fn, low, high = self.get_augment(name)
        level = torch.clamp(level, 0, 1)
        img = augment_fn(img, level * (high - low) + low)
        if not uint_input:
            img = img.float() / 255
        return img


class AugmentationsOptUMirror(Augmentations):
    """Augmentation policy used for SLACK."""
    def default_augment_list(self):
        l = [
            (self.ShearX, 0, 1),  # 0
            (self.ShearY, 0, 1),  # 1
            (self.TranslateX, 0, 0.75),  # 2
            (self.TranslateY, 0, 0.75),  # 3
            (self.Rotate, 0, 90),  # 4
            (AutoContrast, 0, 1),  # 5
            (Invert, 0, 1),  # 6
            (Equalize, 0, 1),  # 7
            (Solarize, 0, 255),  # 8
            (Posterize, 0, 6),  # 9
            (ContrastMirror, 0, 0.99),  # 10
            (ColorMirror, 0, 0.99),  # 11
            (BrightnessMirror, 0, 0.99),  # 12
            (SharpnessMirror, 0, 0.99),  # 13
            (self.Cutout, 0, 1),  # 14
            (RandomCrop, 0, 0.5), # 15
            (Identity, 0, 1), # 16
        ]
        return l


class AugmentationsOptUMirrorImagenet(AugmentationsOptUMirror):
    """Augmentation policy used for SLACK on ImageNet."""
    def default_augment_list(self):
        l = super().default_augment_list()
        l.insert(-3, (HueMirror, 0, 0.5))
        l[-2] = (RandResizeCrop, 0, 0.95)
        return l


class AugmentationsOptUMirrorDomainnet(AugmentationsOptUMirror):
    """Augmentation policy used for SLACK on DomainNet."""
    def default_augment_list(self):
        l = super().default_augment_list()
        l.insert(-3, (HueMirror, 0, 0.5))
        l.insert(-2, (RandResizeCrop, 0, 0.95))
        l.insert(8, (Grayscale, 0, 1))
        return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def RandResizeCrop(img, v):
    scale = (1 - abs(float(v)), 1 - abs(float(v)))
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    size = (224, 224)  # (224, 224)
    return T.RandomResizedCrop(size, scale, ratio)(img)


def AutoContrast(img, _):
    return F.autocontrast(img)


def Invert(img, _):
    return F.invert(img)


def Equalize(img, _):
    return F.equalize(img)


def Solarize(img, v):
    v = 255 - torch.abs(v)
    return F.solarize(img, v)


def Posterize(img, v):
    v = 8 - abs(int(torch.round(v)))
    return F.posterize(img, v)


def Contrast(img, v):
    return F.adjust_contrast(img, v)


def Color(img, v):
    return F.adjust_saturation(img, v)


def Brightness(img, v):
    return F.adjust_brightness(img, v)


def Sharpness(img, v):
    return F.adjust_sharpness(img, v)


def Hue(img, v):
    return F.adjust_hue(img, v)


def ContrastMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return F.adjust_contrast(img, 1 - v)


def ColorMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return F.adjust_saturation(img, 1 - v)


def BrightnessMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return F.adjust_brightness(img, 1 - v)


def SharpnessMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return F.adjust_sharpness(img, 1 - v)


def HueMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return F.adjust_hue(img, v)


def Grayscale(img, _):
    return transforms.Grayscale(num_output_channels=3)(img)


def CutoutAbs(img, v):
    if v < 0:
        return img

    h, w = img.shape[-2], img.shape[-1]
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    i = int(max(0, x0 - v / 2.0))
    j = int(max(0, y0 - v / 2.0))
    vi = min(w - i, v)
    vj = min(h - j, v)

    return F.erase(
        img, i, j, vi, vj, 0
    )


def RandomCrop(img, v):
    h, w = img.shape[-2], img.shape[-1]
    v = abs(int(torch.round(v * min(h, w))))
    return T.RandomCrop((h, w), v)(img)


def Identity(img, v):
    return img
