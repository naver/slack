# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0
# code modified from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class Augmentations:
    """Base class for PIL augmentations (used for standard training), with the default setting found in the litterature."""
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
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

    def ShearY(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

    def TranslateX(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    def TranslateY(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        v = v * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

    def Rotate(self, img, v):
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return img.rotate(v)

    def Cutout(self, img, v):
        v = np.sqrt(v)
        if v <= 0.0:
            return img
        v = v * img.size[0]
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
            (Solarize, 0, 255),  # 8
            (Posterize, 0, 4),  # 9
            (Contrast, 0.1, 1.9),  # 10
            (Color, 0.1, 1.9),  # 11
            (Brightness, 0.1, 1.9),  # 12
            (Sharpness, 0.1, 1.9),  # 13
            (self.Cutout, 0, 0.2),  # 14
            (Identity, 0, 1), # 15
        ]
        return l

    def get_augment(self, name, i=None):
        return self.augment_dict[name] if i is None else self.augment_dict[name][i]

    def apply_augment(self, img, name, level):
        if name[-1].isdigit():
            _name, i = name[:-1], name[-1]
            augment_fn, low, high = self.get_augment(_name, int(i))
        else:
            augment_fn, low, high = self.get_augment(name)
        level = torch.clamp(level, 0, 1)
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        return augment_fn(img.copy(), level * (high - low) + low)


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
        l[-2] = (RandomResizeCrop, 0, 0.95)
        return l


class AugmentationsOptUMirrorDomainnet(AugmentationsOptUMirror):
    """Augmentation policy used for SLACK on DomainNet."""
    def default_augment_list(self):
        l = super().default_augment_list()
        l.insert(-3, (HueMirror, 0, 0.5))
        l.insert(-2, (RandomResizeCrop, 0, 0.95))
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


def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):
    v = 255 - torch.abs(v)
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):
    v = 8 - abs(int(torch.round(v)))
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def ContrastMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v)


def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)


def ColorMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Color(img).enhance(1 + v)


def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def BrightnessMirror(img, v):  # [0.1,1.9]
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v)


def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def SharpnessMirror(img, v):  # [0.1,1.9]
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v)


def Hue(img, v):
    return transforms.functional.adjust_hue(img, v)


def HueMirror(img, v):
    if random.random() > 0.5:
        v = -v
    return transforms.functional.adjust_hue(img, v)


def Grayscale(img, _):
    return transforms.Grayscale(num_output_channels=3)(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def RandomCrop(img, v):
    v = abs(int(torch.round(v * img.size[0])))
    padded_size = (img.size[0] + 2 * v, img.size[1] + 2 * v)
    new_img = PIL.Image.new("RGB", padded_size, color=(0, 0, 0))
    new_img.paste(img, (v, v))
    top = random.randint(0, v * 2)
    left = random.randint(0, v * 2)
    new_img = new_img.crop((left, top, left + img.size[0], top + img.size[1]))
    return new_img


def RandomResizeCrop(img, v):
    scale = (0.05 + abs(v), 1.0)
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    size = img.size

    def get_params(img, scale, ratio):
        width, height = img.size
        area = float(width * height)
        log_ratio = [np.log(r) for r in ratio]

        for _ in range(10):
            target_area = area * random.uniform(scale[0], scale[1]).cpu()
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = np.round(np.sqrt(target_area * float(aspect_ratio)))
            h = np.round(np.sqrt(target_area / float(aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return left, top, w, h

            # fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(ratio):
                w = width
                h = round(w / min(ratio))
            elif in_ratio > max(ratio):
                h = height
                w = round(h * max(ratio))
            else:
                w = width
                h = height
            top = (height - h) // 2
            left = (width - w) // 2
            return left, top, w, h

    left, top, w_box, h_box = get_params(img, scale, ratio)
    box = (left, top, left + w_box, top + h_box)
    img = img.resize(size=size, resample=PIL.Image.CUBIC, box=box)
    return img


def Identity(img, v):
    return img
