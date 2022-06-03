import gzip
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transforms import MinMaxNorm, Window


class RFCDataset(Dataset):
    """RibFrac Challenge dataset."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (string): Path to the directory of preprepared data files.
        """
        self.data_dir = data_dir
        paths = [path for path in Path(data_dir).glob("*.pt.gz")]
        self.paths = paths
        self.transforms = [Window(-200, 1000), MinMaxNorm(-200, 1000)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.load_data_file(idx)

        assert label.min() >= 0
        assert label.min() <= 5

        img = self.apply_transforms(img)

        img = img[np.newaxis, :]

        example = {"image": img, "label": label}

        return example

    def load_data_file(self, i):
        (img, label) = torch.load(gzip.GzipFile(self.paths[i], "rb"))
        return (img, label)

    def apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
