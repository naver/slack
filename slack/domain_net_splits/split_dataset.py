import shutil
import os
import numpy as np
import argparse
import random

def main(data_dir, train_id):
    classes = os.listdir(data_dir)
    test_dir = os.path.join(data_dir, 'test')
    os.mkdir(test_dir)
    for c in classes:
        shutil.move(os.path.join(data_dir, c), test_dir)
    train_dir = os.path.join(data_dir, 'train')
    os.mkdir(train_dir)
    train_id = np.load(train_id)
    for c in classes:
        train_c_dir = os.path.join(train_dir, c)
        os.mkdir(train_c_dir)
        for train_f in train_id[c]:
            shutil.move(os.path.join(test_dir, c, train_f), train_c_dir)

def parse_args():
  parser = argparse.ArgumentParser(description="Dataset divider")
  parser.add_argument("--data_dir", required=True)
  parser.add_argument("--train_id", required=True)
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.data_dir, args.train_id)
