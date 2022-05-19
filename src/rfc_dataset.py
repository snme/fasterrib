import csv
import gzip
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

        paths = [path for path in Path(data_dir).glob("slice_pair_*.pt.gz")]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.load_data_file(idx)

        img = img[np.newaxis, :]
        label = label.permute(2, 0, 1)

        example = {"image": img, "label": label}

        return example

    @lru_cache(maxsize=10)
    def load_data_file(self, i):
        (img, label) = torch.load(gzip.GzipFile(self.paths[i], "rb"))
        return (img, label)
