import csv
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class RFCDataset(Dataset):
    """RibFrac Challenge dataset."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (string): Path to the directory of preprepared data files.
        """
        self.data_dir = data_dir

        paths = [path for path in Path(data_dir).glob("*.pt")]

        self.paths = paths

        # Maps slice index to the index of the file it came from.
        slice_to_path = {}

        total_slices = 0
        for i, path in enumerate(paths):
            (img, label) = torch.load(path)
            num_slices = img.shape[0]
            for j in range(num_slices):
                slice_to_path[total_slices + j] = (i, j)

            total_slices += num_slices

        self.slice_to_path = slice_to_path
        self.total_slices = total_slices

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_i, img_slice_j = self.slice_to_path[idx]

        img, label = self.load_data_file(img_i)

        img_slice = img[img_slice_j]
        label_slice = label[img_slice_j]

        example = {"image": img_slice, "label": label_slice}

        return example

    @lru_cache(maxsize=50)
    def load_data_file(self, i):
        (img, label) = torch.load(self.paths[i])
        return (img, label)
