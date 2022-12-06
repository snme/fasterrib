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

        # right now, labels are 2-5. let's "merge" background and unknown fractures.
        # then, fractures are from 1-4
        for i in range(len(label)):
            label[i] = max(0, label[i] - 1)

        assert label >= 0
        assert label <= 4

        img = self.apply_transforms(img)

        img = img[np.newaxis, :]

        # sclae image to [0, 1]
        img = (img - img.min()) / img.max()

        example = {"image": img, "label": label.type(torch.int64)}

        return example

    def load_data_file(self, i):
        (img, label) = torch.load(gzip.GzipFile(self.paths[i], "rb"))
        return (img, label.type(torch.LongTensor))

    def apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
