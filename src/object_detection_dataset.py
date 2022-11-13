import gzip
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transforms import MinMaxNorm, Window


class ODDataset(Dataset):
    """Object Detection conversion of RFC Dataset. Was a
       segmentation task, now reframing as a multi-class OD task."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (string): Path to the directory of preprepared data files.
        """
        self.data_dir = data_dir
        paths = sorted([path for path in Path(data_dir).glob("*.pt.gz")])
        self.paths = paths
        self.transforms = [Window(-200, 1000), MinMaxNorm(-200, 1000)]
        print(self.__len__())

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.load_data_file(idx)

        # right now, labels are 2-5. let's "merge" background and unknown fractures.
        # then, fractures are from 1-4
        for i in range(len(target['labels'])):
            label = target['labels'][i]
            label = max(0, label - 1)
            assert label >= 0
            assert label <= 4

            target['labels'][i] = label

        img = self.apply_transforms(img)

        img = img[np.newaxis, :]

        target['image_id'] = idx
        target['labels'] = target['labels'].type(torch.int64)

        return img, target


    def load_data_file(self, i):
        img, target = torch.load(gzip.GzipFile(self.paths[i], "rb"))
        return img, target

    def apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
