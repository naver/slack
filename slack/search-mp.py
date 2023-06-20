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
import torch.distributed as dist
import torch.multiprocessing as mp
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

def spawn_process(rank, worldsize, port_suffix, cfg):
    started_with_spawn = worldsize is not None and worldsize > 0
    if worldsize != 0:
        os.environ['MASTER_ADDR'] =  'localhost'
        os.environ['MASTER_PORT'] = f'12{port_suffix}'
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=worldsize)
        rank, worlsize = dist.get_rank(), dist.get_world_size()
    os.chdir(work_dir)
    reproducibility(0)
    trainer = Trainer(cfg, rank=rank)
    trainer.args['worldsize'] = worldsize
    trainer.main()

    if worldsize:
        dist.destroy_process_group()

@hydra.main(config_name="config.yaml", config_path="./configs")
def run(cfg):
    worldsize = torch.cuda.device_count()
    port_suffix = str(random.randint(10,99))
    if worldsize > 1 and cfg.system.dataparallel:
        outcome = mp.spawn(
            spawn_process,
            args=(worldsize,port_suffix,cfg),
            nprocs=worldsize,
            join=True
        )
    else:
        outcome = spawn_process(0, 0, 0, cfg)
    print("Finished!")


if __name__ == "__main__":
    run()
