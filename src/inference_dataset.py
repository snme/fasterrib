from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.transforms import MinMaxNorm, Window


class InferenceDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (string): path to a directory of CT scans in .nii.gz format
        """
        self.data_dir = data_dir
        paths = [path for path in Path(data_dir).glob("*.nii.gz")]
        self.paths = paths
        self.transforms = [Window(-200, 1000), MinMaxNorm(-200, 1000)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.paths[idx]
        basename = path.name.strip(".nii.gz")
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        img = self.apply_transforms(img)
        img = torch.Tensor(img).permute(2, 0, 1)  # (SLICES, H, W)

        return img, basename

    def apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
