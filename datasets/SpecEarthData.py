import csv
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
def Normalize(img):
    minimum_value = 0
    maximum_value = 10000
    # load patch
    # dataset = rasterio.open(patch_path)
    # # remove nodata channels
    # src = dataset.read(valid_channels_ids)
    # clip data to remove uncertainties
    clipped = np.clip(img, a_min=minimum_value, a_max=maximum_value)
    # min-max normalization
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    # save npy
    # out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
    # np.save(out_path, out_data)
    return out_data

class SpecEarthdata(Dataset):

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir

        self.csv_path = os.path.join(self.root_dir, "splits", f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.data_path = sum(csv_data, [])
        self.data_path = [os.path.join(self.root_dir, "spectral_earth50K", x) for x in self.data_path]

        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # get full numpy path
        data_path = self.data_path[index]
        # read numpy data
        with rasterio.open(data_path) as f:
            img = Normalize(f.read().astype(np.int16))
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        return img

