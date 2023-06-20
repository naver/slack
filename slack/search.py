# Slack
# Copyright 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

from __future__ import print_function
import argparse
import yaml
import torch
import os
import numpy as np
import random
from searcher import Trainer

import hydra

# check whether we want to load a pretrained model depending on the given parameters

work_dir = os.getcwd()

def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_name="config.yaml", config_path="./configs")
def run(cfg):
    reproducibility(0)
    os.chdir(work_dir)
    trainer = Trainer(cfg)
    trainer.main()
    print("Finished!")


if __name__ == "__main__":
    run()
