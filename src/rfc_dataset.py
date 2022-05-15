from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class RFCDataset(Dataset):
    """RibFrac Challenge dataset."""

    def __init__(self, img_dir: str, label_dir):
        """
        Args:
            img_dir (string): Path to the directory of *.nii.gz image files.
            label_dir (string): Path to the directory of *.nii.gz label files.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir

        img_paths = [path for path in Path(img_dir).glob("*.nii.gz")]
        label_paths = [path for path in Path(label_dir).glob("*.nii.gz")]

        assert len(img_paths) == len(label_paths)

        self.img_paths = img_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print("idx", idx)
        img = nib.load(self.img_paths[idx]).get_fdata()
        label = nib.load(self.label_paths[idx]).get_fdata()

        example = {"image": img, "label": label}

        return example
