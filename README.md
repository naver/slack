# SLACK: Stable Learning of Augmentations with Cold-start and KL regularization (CVPR23)

This repository contains the Pytorch implementation for the paper "SLACK: Stable Learning of Augmentations with Cold-start and KL regularization", which was published in the Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in June 2023.

## Citation

If you find this work useful for your research, please cite the paper using the following BibTeX entry:

```
@InProceedings{Marrie_2023_CVPR,
  author = {Marrie, Juliette and Arbel, Michael and Larlus, Diane and Mairal, Julien},
  title = {SLACK: Stable Learning of Augmentations with Cold-start and KL regularization},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2023}
}
```

## Table of Contents
- [Setup](#setup)
   - [Dependencies](#dependencies)
   - [Search](#search)
   - [(Pre)training](#pretraining)
- [Evaluating our policies](#evaluation)
   - [CIFAR](#cifar)
   - [ImageNet-100](#imagenet-100)
   - [DomainNet](#domainnet)
- [End-to-end search and evaluation](#end-to-end)
   - [CIFAR](#cifar-1)
   - [ImageNet-100](#imagenet-100-1)
   - [DomainNet](#domainnet-1)

## Setup <a name="setup"></a>

SLACK is evaluated based on $n_{search}$ independent search runs. Each run uses a different train/val split of the full training data and consists of: 

1. A pretraining phase
2. A search phase, where a policy is found
3. The evaluation phase, where a network is trained with the found policy on the full training data, with $n_{train}$ independent runs (seeds).

The final reported accuracy is the average over the $n_{search} \times n_{train}$ runs.

### Dependencies <a name="dependencies"></a>

Create conda environment: `conda create -n slack python=3.9.7`

Install dependencies: `pip install -r requirements.txt`

Before running the script, `export ROOT=$HOME/SLACK/slack`.
The data should be contained in `$ROOT/data/datasets`.

### Search <a name="search"></a>

The search logs are saved in `$ROOT/data/outputs/[SEARCH_DIR]`, in a `SEARCH_DIR` directory specific to each search experiment (and automatically created). They contain the augmentation model in different formats: as checkpoint (`models.ckpt`), as numpy files (in `pi`, `mu`), as .txt (in `genotype`) and with simple visualizations (in `plt_genotype`). Also, `metadata.yaml` reports the hyperparameters used for the search and `val.txt` reports validation and training metrics.

### (Pre)training <a name="pretraining"></a>

The (pre)training is performed using TrivialAugment's framework, located in `$ROOT/TrivialAugment`.

The (pre)training logs are saved in `$ROOT/TrivialAugment/logs` and can be evaluated with the $ROOT/TrivialAugment/aggregate_results.py script.
```bash
python aggregate_results.py --logdirs [DIRS] --split [train|test] --metric [top1|top5|loss] --step [STEP]
```

The pretraining checkpoint is directly saved in `$ROOT/data/outputs/[SEARCH_DIR]`. The other checkpoints are saved in `$ROOT/TrivialAugment/ckpt`.


## Evaluating our policies <a name="evaluation"></a>

### CIFAR [WideResNet-40x2, WideResNet-28x10] <a name="cifar"></a>

Evaluate our policy on SPLIT with 8 seeds:
```bash
sh $ROOT/scripts/cifar/train.sh [10|100] [40x2|28x10] [SPLIT] git-policies
```

Evaluate our Uniform policy with 8 seeds:
```bash
sh $ROOT/scripts/cifar/train.sh [10|100] [40x2|28x10] uniform
```

**Test accuracies on CIFAR** (average over $4\times 4$ seeds for SLACK and $8$ seeds for our Uniform policy).
|                    | CIFAR10 WRN-40-2 | CIFAR10 WRN-28-10 | CIFAR100 WRN-40-2 | CIFAR100 WRN-28-10 |
|--------------------|------------------|-------------------|-------------------|--------------------|
| **TA (Wide)**      | 96.32 | 97.46 | 79.76 | 84.33 |
| **Uniform policy** | 96.12 | 97.26 | 78.79 | 82.82 |
| **SLACK**          | 96.29 | 97.46 | 79.87 | 84.08 |

**Best policies found for CIFAR10 (1,2) and CIFAR100 (3,4) with WRN-40-2 (1,3) and WRN-28-10 (2,4).**
<p align="center">
  <img alt="CIFAR10, WideResNet-40x2" src="slack/checkpoints/cifar/pies/c10-40x2-s1.png?raw=true" title="CIFAR10, WideResNet-40x2" width="24%">
  <img alt="CIFAR10, WideResNet-28x10" src="slack/checkpoints/cifar/pies/c10-28x10-s2.png?raw=true" title="CIFAR10, WideResNet-28x10" width="24%">
  <img alt="CIFAR100, WideResNet-40x2" src="slack/checkpoints/cifar/pies/c100-40x2-s2.png?raw=true" title="CIFAR100, WideResNet-40x2" width="24%">
  <img alt="CIFAR100, WideResNet-28x10" src="slack/checkpoints/cifar/pies/c100-28x10-s3.png?raw=true"  title="CIFAR100, WideResNet-28x10" width="24%">
</p>

### ImageNet-100 [ResNet-18] <a name="imagenet-100"></a>                                                                 

Class IDs for ImageNet-100 can be found [here](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt).

Evaluate our policy on SPLIT with SEED
```bash
sh $ROOT/scripts/imagenet100/train.sh [SPLIT] [SEED] git-policies
```

Evaluate our Uniform policy with SEED
```bash
sh $ROOT/scripts/imagenet100/train.sh uniform [SEED]
```

Evaluate TrivialAugment with SEED
```bash
sh $ROOT/scripts/imagenet100/train-TA.sh [ta|ta_wide] [SEED]
```

**Test accuracies on ImageNet-100** (average over $4\times 4$ seeds for SLACK and $8$ seeds for search-free methods).
| Method  | ImageNet-100, ResNet-18 |
| ------------- | ------------- |
| **TA (RA)**  | 85.87  |
| **TA (Wide)**  | 86.39  |
| **Uniform policy**  | 85.78  |
| **SLACK**  | 86.06  |

**Best policy found for ImageNet-100.**
<p align="center">
  <img alt="ImageNet-100" src="slack/checkpoints/imagenet100/pies/s0.png?raw=true" title="ImageNet-100, ResNet-18" width="33%">
</p>


### DomainNet [ResNet-18] <a name="domainnet"></a>

DomainNet is a dataset commonly used for domain generalization that contains 345 classes of images from six different domains: painting, clipart, sketch, infograph, quickdraw, real. 
It can be downloaded from the [DomainBed](https://github.com/facebookresearch/DomainBed) suite.

**Download**

We evaluate on the six domains, with a reduced version of 50,000 training images for the two largest (real, quickdraw) and use the remaining of the data for testing. For the other domains, we isolate 20% of the data for testing. The filenames belonging to the train/test splits are stored in `$ROOT/domein_net_splits/`.
Use the following script to separatesthe data into training and testing folders for a DOMAIN from {painting, clipart, sketch, infograph, quidraw, real}:
```bash
python domain_net_splits/split_dataset.py --data_dir data/datasets/domain_net/[DOMAINET] --train_id domain_net_splits/npz/[DOMAIN]_train.npz
```

**Evaluation**

Evaluate our policy on SPLIT with SEED
```bash
sh $ROOT/scripts/domainnet/train.sh [DOMAIN] [SPLIT] [SEED] git-policies
```

Evaluate our Uniform policy with SEED
```bash
sh $ROOT/scripts/domainnet/train.sh [DOMAIN] uniform [SEED]
```

Evaluate TrivialAugment with SEED
```bash
sh $ROOT/scripts/domainnet/train-TA.sh [DOMAIN] [ta_imagenet|ta_cifar|ta_imagenet_wide|ta_cifar_wide] [SEED]
```

**Test accuracies on DomainNet** (average over $4\times 4$ seeds for SLACK and $8$ seeds for search-free methods).
| Method                  | Real-50k | Quickdraw-50k | Inforgraph | Sketch | Painting | Clipart | Average |
|-------------------------|----------|---------------|------------|--------|----------|---------|---------|
| **DomainBed**               | 62.54    | 66.54         | 26.76      | 59.54  | 58.31    | 66.23   | 57.23   |
| **TA (RA) ImageNet**        | 70.85    | 67.85         | **35.24**  | 65.63  | 64.75    | 70.29   | 62.43   |
| **TA (Wide) ImageNet**      | **71.56**    | 68.60         | **35.44**  | **66.21**  | **65.15**    | 71.19   | **63.03**   |
| **TA (RA) CIFAR**           | 70.28    | 68.35         | 33.85      | 64.13  | 64.73    | 70.33   | 61.94   |
| **TA (Wide) CIFAR**         | 71.12    | **69.29**         | 34.21      | 65.52  | 64.81    | 71.01   | 62.66   |
| **Uniform policy**          | 70.37    | 68.27         | 34.11      | 65.22  | 63.97    | 72.26   | 62.37   |
| **SLACK**                   | 71.00    | 68.14         | 34.78      | 65.41  | 64.83    | **72.65**   | 62.80   |


**Best policies found for Sketch (left), Clipart (center) and Painting (right).**
<p align="center">
  <img alt="Sketch" src="slack/checkpoints/domainnet/pies/sketch-s3.png?raw=true" title="Sketch" width="32%">
  <img alt="Clipart" src="slack/checkpoints/domainnet/pies/clipart-s3.png?raw=true" title="Clipart" width="32%">
  <img alt="Painting" src="slack/checkpoints/domainnet/pies/painting-s2.png?raw=true" title="Painting" width="32%">
</p>

## End-to-end search and evaluation <a name="end-to-end"></a>

### CIFAR [WideResNet-40x2, WideResNet-28x10] <a name="cifar-1"></a>

1. Pretrain on SPLIT
    ```bash
    sh $ROOT/scripts/cifar/pretrain.sh [10|100] [40x2|28x10] [SPLIT]
    ```

2. Search on SPLIT
    ```bash
    sh $ROOT/scripts/cifar/search.sh [10|100] [40x2|28x10] [SPLIT]
    ```

3. Evaluate on 4 seeds for SPLIT
    ```bash
    sh $ROOT/scripts/cifar/train.sh [10|100] [40x2|28x10] [SPLIT]
    ```

You can also run our ablations (no-kl, warm-start, unrolled, pi-only, mu-only):
    ```bash
    sh $ROOT/scripts/cifar/search.sh [10|100] [40x2|28x10] [SPLIT] [ABLATION]
    ```

### ImageNet-100 [ResNet-18] <a name="imagenet-100-1"></a>                                                                       

1. Pretrain on SPLIT
    ```bash
    sh $ROOT/scripts/imagenet100/pretrain.sh [SPLIT]
    ```

2. Search on SPLIT
    ```bash
    sh $ROOT/scripts/imagenet100/search.sh [SPLIT]
    ```

3. Evaluate on SPLIT with SEED
    ```bash
    sh $ROOT/scripts/imagenet100/train.sh [SPLIT] [SEED]
    ```

### DomainNet [ResNet-18] <a name="domainnet-1"></a>

1. Pretrain on SPLIT
    ```bash
    sh $ROOT/scripts/domainnet/pretrain.sh [DOMAIN] [SPLIT]
    ```

2. Search on SPLIT
    ```bash
    sh $ROOT/scripts/domainnet/search.sh [DOMAIN] [SPLIT]
    ```

3. Evaluate on SPLIT with SEED
    ```bash
    sh $ROOT/scripts/domainnet/train.sh [DOMAIN] [SPLIT] [SEED]
    ```
