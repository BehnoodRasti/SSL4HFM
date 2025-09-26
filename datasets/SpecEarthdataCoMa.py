
import csv
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
def Normalize(img):
    minimum_value = 0
    maximum_value = 10000
    clipped = np.clip(img, a_min=minimum_value, a_max=maximum_value)
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    return out_data

class SpecEarthdataCoMa(Dataset):
    def __init__(self, root_dir, split="train", transform=None, spectral_mask_ratio=0.75, patch_mask_ratio=0.75, patch_size=16):
        self.root_dir = root_dir
        self.spectral_mask_ratio = spectral_mask_ratio
        self.patch_mask_ratio = patch_mask_ratio
        self.patch_size = patch_size

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
        data_path = self.data_path[index]
        with rasterio.open(data_path) as f:
            img = Normalize(f.read().astype(np.int16))  # (C, H, W)

        img = torch.from_numpy(img)  # (C, H, W)

        if self.transform:
            img = self.transform(img)

        img1 = self.apply_spectral_mask(img.clone())      # View 1: Spectral mask
        img2 = img.clone() #apply_spatial_patch_mask(img.clone()) # View 2: Patch mask

        return img1, img2

    def apply_spectral_mask(self, img):
        # img: (C, H, W)
        C = img.shape[0]
        num_mask = int(self.spectral_mask_ratio * C)
        mask_indices = torch.randperm(C)[:num_mask]
        img[mask_indices] = 0
        return img

    def apply_spatial_patch_mask(self, img):
        # img: (C, H, W)
        C, H, W = img.shape
        num_patches_x = W // self.patch_size
        num_patches_y = H // self.patch_size
        total_patches = num_patches_x * num_patches_y
        num_masked = int(self.patch_mask_ratio * total_patches)

        mask = torch.ones_like(img)
        patch_indices = torch.randperm(total_patches)[:num_masked]
        for idx in patch_indices:
            i = (idx // num_patches_x) * self.patch_size
            j = (idx % num_patches_x) * self.patch_size
            mask[:, i:i+self.patch_size, j:j+self.patch_size] = 0
        return img * mask
