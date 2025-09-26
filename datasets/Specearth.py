import csv
import torch
from torch.utils.data import Dataset
import os
import random
from typing import Callable, Optional, TypedDict
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
import imageio
import matplotlib.pyplot as plt
from torchgeo.datasets.geo import NonGeoDataset

class SpecEarthDataset(NonGeoDataset):
    rgb_indices = {
        "enmap": [43, 28, 10],
    }

    valid_sensors = ['enmap', 'enmap_vnir', 'enmap_swir']
    def __init__(self, root, data_dir, sensor, split="train", transforms=None, return_two_views=False):
        """
        Args:
            root: Root directory of the dataset.
            data_dir: Directory where the data is stored.
            sensor: Sensor type ('enmap', 'enmap_vnir', 'enmap_swir').
            num_bands: Number of spectral bands.
            patch_size: Size of the patches.
            split: Dataset split ('train', 'val', 'test').
            transforms: Optional transforms to apply to the samples.
            return_two_views: If True, returns two random views of the same patch.
        """
        assert (
            sensor in self.valid_sensors
        ), f"Only supports one of {self.valid_sensors}, but found '{sensor}'."
        self.sensor = sensor
        self.root = root
        self.data_dir = data_dir
        self.transforms = transforms
        self.return_two_views = return_two_views
        self.patch_paths = []

        # Load the file paths from the CSV file
        csv_path = os.path.join(self.root, "splits", f"{split}.csv")
        with open(csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.patch_paths = sum(csv_data, [])
            #print("Loaded paths:", self.patch_paths)
        self.patch_paths = [os.path.join(self.root, x) for x in self.patch_paths]
        #print("Loaded paths:", self.patch_paths)
        # for patch_dir in self.patch_paths:
        #     patch_path = os.path.join(self.root, self.data_dir, patch_dir)
        #     if os.path.isdir(patch_path):
        #         image_files = [
        #             os.path.join(patch_path, f)
        #             for f in os.listdir(patch_path)
        #             if f.endswith(".tif")
        #         ]
        #         if image_files:
        #             self.patch_paths[patch_dir] = image_files

    # def __len__(self):
    #     return len(self.patch_paths)
    # def __getitem__(self, index: int) -> dict:
    #     # Get full numpy path
    #     npy_path = self.patch_paths[index]
    #     print(f"Loading file: {npy_path}")
    #     # Read numpy data
    #     # Check file extension and load accordingly
    #     if npy_path.endswith('.npy'):
    #         img = np.load(npy_path, allow_pickle=True)
    #     elif npy_path.endswith('.tif'):
    #         img = rasterio.imread(npy_path)
    #     else:
    #         raise ValueError(f"Unsupported file format: {npy_path}")

    #     # img = np.load(npy_path, allow_pickle=True)
    #     # Convert numpy array to pytorch tensor
    #     img = torch.from_numpy(img)
    #     # Apply transformations
    #     if self.transforms:
    #         img = self.transforms(img)
    #     return img
    # def __getitem__(self, index: int) -> dict[str, Tensor]:
    #     """
    #     Return a data sample from the dataset.

    #     Args:
    #         index: Index of the sample to retrieve.

    #     Returns:
    #         A dictionary containing the image(s).
    #     """
    #     # patch_id = list(self.patch_paths.keys())[index]
    #     # image_paths = self.patch_paths[patch_id]
    #     npy_path = self.patch_paths[index]
    #     if self.return_two_views:
    #         if len(npy_path) >= 2:
    #             chosen_paths = random.sample(npy_path, 2)
    #         else:
    #             # If less than 2 images are available, duplicate the same image
    #             chosen_paths = [random.choice(npy_path)] * 2

    #         images = []
    #         for path in chosen_paths:
    #             with rasterio.open(path) as f:
    #                 image = torch.from_numpy(f.read().astype(np.int16))
    #             images.append(image)

    #         sample = {"image1": images[0], "image2": images[1]}
    #     else:
    #         chosen_path = random.choice(npy_path)
    #         with rasterio.open(chosen_path) as f:
    #             image = torch.from_numpy(f.read().astype(np.int16))
    #         sample = {"image": image}

    #     if self.transforms is not None:
    #         sample = self.transforms(sample)

    #     return sample

    # def __getitem__(self, index: int) -> dict:
    #     # Get full numpy path
    #     npy_path = self.patch_paths[index]
    #     #print(f"Loading file: {npy_path}")
    #     # Read numpy data
    #     img = np.load(npy_path, allow_pickle=True)
    #     # Convert numpy array to pytorch tensor
    #     img = torch.from_numpy(img)
    #     # Apply transformations
    #     if self.transforms:
    #         img = self.transforms(img)
    #     return img

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            show_titles: Whether to show titles above each panel.
            suptitle: Optional string for the figure's suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        if self.return_two_views:
            image = sample["image1"][self.rgb_indices[self.sensor]].numpy()
        else:
            image = sample["image"][self.rgb_indices[self.sensor]].numpy()

        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
# # Example usage
# root = "/data/behnood/spectral_earth"
# data_dir = "enmap"
# sensor = "enmap"
# train_dataset = SpecEarthDataset(root, data_dir, sensor, split="train")
# val_dataset = SpecEarthDataset(root, data_dir, sensor, split="val")
# test_dataset = SpecEarthDataset(root, data_dir, sensor, split="test")

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Check the number of samples
# print("Number of samples in train dataset:", len(train_dataset))
# print("Number of samples in val dataset:", len(val_dataset))
# print("Number of samples in test dataset:", len(test_dataset))