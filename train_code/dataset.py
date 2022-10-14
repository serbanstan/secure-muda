import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import config as config
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class FrozenDataset():
    def __init__(self, file_name, train_ratio = 1):
        file_name = os.path.join(config.server_root_path, config.settings['dataset_dir'], file_name)

        data = pd.read_csv(file_name)
        data = np.asarray(data)

        x = data[:, :-1]
        y = data[:, -1]

        perm = np.random.RandomState(42).permutation(x.shape[0])
        limit_train = int(x.shape[0] * train_ratio)

        self.img = torch.tensor(x[perm[:limit_train]], dtype=torch.float32)
        self.label = torch.tensor(y[perm[:limit_train]], dtype=torch.long)

        self.val_img = torch.tensor(x[perm[limit_train:]], dtype=torch.float32)
        self.val_label = torch.tensor(y[perm[limit_train:]], dtype=torch.long)

        if train_ratio == 1:
            assert x.shape[0] == self.img.shape[0]

    def sample(self, batch_size):
        assert self.img.shape[0] >= batch_size

        idx = random.sample(range(self.img.shape[0]), batch_size)
        return self.img[idx], self.label[idx]

    def sample_val(self, batch_size):
        assert self.val_img.shape[0] >= batch_size

        idx = random.sample(range(self.val_img.shape[0]), batch_size)
        return self.val_img[idx], self.val_label[idx]


# https://medium.com/@shashikachamod4u/excel-csv-to-pytorch-dataset-def496b6bcc1
class FeatureDataset(Dataset):

    def __init__(self, file_name):

        file_name = os.path.join(config.server_root_path, config.settings['dataset_dir'], file_name)

        data = pd.read_csv(file_name)
        data = np.asarray(data)

        x = data[:, :-1]
        y = data[:, -1]

        self.img = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return idx, self.img[idx], self.label[idx]