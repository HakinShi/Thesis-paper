import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pickle as pkl
import argparse
import gc
import os
import numpy as np
from tools import transform_input


class Dataloader(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        data_path = args.data
        self.working_num = args.num_worker
        self.bs = args.bs
        self.targets_dir = os.path.join(data_path, 'New_Normalized_X')
        self.input_dir = os.path.join(data_path, 'New_Normalized_Y')

        self.setup()

    def prepare_data(self):
        # data is processed
        print('call prepare data')

    def setup(self, stage=None):
        targets_files = os.listdir(self.targets_dir)
        input_files = os.listdir(self.input_dir)
        train_data = [[os.path.join(self.input_dir, i), os.path.join(self.targets_dir, t)] for i, t in zip(input_files[:int(0.8 * len(input_files))], targets_files[:int(0.8 * len(targets_files))])]
        val_data = [[os.path.join(self.input_dir, i), os.path.join(self.targets_dir, t)] for i, t in zip(input_files[:int(0.8 * len(input_files))], targets_files[:int(0.8 * len(targets_files))])]
        self.train_data = MyDataset(train_data)
        self.val_data = MyDataset(val_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.bs, num_workers=self.working_num)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.bs * 2, num_workers=self.working_num)

    # def teardown(self):
    #     print('called teardown!')


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx][0], 'rb') as f:
            input_x = np.load(f)['normalized_pap']
        # input_x = transform_input(input_x)

        with open(self.data[idx][1], 'rb') as f:
            target = np.load(f)['normalize_x_pahse']
        return torch.DoubleTensor(input_x), torch.FloatTensor(target)


if __name__ == '__main__':
    Dataloader('./data')
