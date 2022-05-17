from functools import lru_cache
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
        img_paths.sort()

        label_paths = [path for path in Path(label_dir).glob("*.nii.gz")]
        label_paths.sort()

        assert len(img_paths) == len(label_paths)

        self.img_paths = img_paths
        self.label_paths = label_paths

        # Maps slice index to the index of the image it came from.
        slice_to_img = {}

        total_slices = 0
        for i, path in enumerate(img_paths):
            num_slices = nib.load(path).shape[-1]
            for j in range(num_slices):
                slice_to_img[total_slices + j] = (i, j)

            total_slices += num_slices

        self.slice_to_img = slice_to_img

        print("total slices:", total_slices)

        self.total_slices = total_slices

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_i, img_slice_j = self.slice_to_img[idx]

        img = self.load_image_data(img_i)
        label = self.load_label_data(img_i)

        img_slice = img[:, :, img_slice_j][np.newaxis, :]
        label_slice = label[:, :, img_slice_j]

        label_slice = torch.as_tensor(label_slice, dtype=torch.long)
        label_slice = torch.nn.functional.one_hot(label_slice, num_classes=6).transpose(
            2, 0
        )  # (n_classes, 512, 512)

        example = {"image": img_slice, "label": label_slice}

        return example

    @lru_cache(maxsize=50)
    def load_image_data(self, i):
        img = nib.load(self.img_paths[i]).get_fdata().astype(np.float32)
        return img

    @lru_cache(maxsize=50)
    def load_label_data(self, i):
        label = nib.load(self.label_paths[i]).get_fdata().astype(np.int8)
        label += 1  # force labels to be 0,1,2,3,4,5 https://zenodo.org/record/3893508#.YoL-wnXMJH5
        return label
