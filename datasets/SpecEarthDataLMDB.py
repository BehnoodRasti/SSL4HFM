import csv
import os
import numpy as np
import torch
import lmdb
from torch.utils.data import Dataset
import pickle
from safetensors.numpy import load
from pathlib import Path
def Normalize(img):
    minimum_value = 0
    maximum_value = 10000
    clipped = np.clip(img, a_min=minimum_value, a_max=maximum_value)
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    return out_data
class SpecEarthdataLMDB(Dataset):

    def __init__(self, root_dir, split="train", csv_file=None, transform=None):
        self.root_dir = root_dir
        self.lmdb_file = os.path.join(root_dir, split)  # Point to the directory, not a specific file
        self.env = None
        self.transform = transform

        # Read keys from the CSV file
        self.keys = []
        if csv_file:
            with open(csv_file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.keys.append(row[0])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.env = lmdb.open(
                self.lmdb_file,
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=100* 1024**3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )
        #self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if index >= len(self.keys):
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.keys)}")

        key = self.keys[index].encode()  # Ensure the key format matches the LMDB keys

        # read numpy data
        with self.env.begin(write=False) as txn:
            data = txn.get(key)

        if data is None:
            raise ValueError(f"No data found for key '{key.decode()}'")

        safetensor_dict = load(data)
        img = Normalize(np.stack([safetensor_dict[b] for b in safetensor_dict.keys()]))
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        return img
# class SpecEarthdataLMDB(Dataset):

#     def __init__(self, root_dir, split="train", transform=None):
#         self.root_dir = root_dir
#         self.lmdb_file = os.path.join(root_dir, split)  # Point to the directory, not a specific file
#         self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
#         with self.env.begin(write=False) as txn:
#                 self.length = txn.stat()['entries']     
#         self.transform = transform

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         # get full numpy path
#      # open lmdb file if not opened yet
#         # self.open_env()
#         # read numpy data
#         with self.env.begin(write=False) as txn:
#             data = txn.get(str(index).encode())

#         scenes, s_shape = pickle.loads(data)
#         img = np.frombuffer(scenes, dtype=np.int16).reshape(s_shape)
#         # convert numpy array to pytorch tensor
#         img = torch.from_numpy(img)
#         # apply transformations
#         if self.transform:
#             img = self.transform(img)
#         return img
