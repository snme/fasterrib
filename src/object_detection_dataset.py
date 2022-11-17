import gzip
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transforms import MinMaxNorm, Window
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


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
        # return 100
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

            # bounding boxes bug change.
            b = target['boxes'][i]
            target['boxes'][i] = torch.Tensor([b[1], b[0], b[3], b[2]])

        if len(target['boxes']) == 0:
            # (N, 4) shape expected
            target['boxes'] = torch.zeros((0, 4))

        img = self.apply_transforms(img)

        # convert to [0, 1]
        img = (img - img.min()) / img.max()

        img = img[np.newaxis, :]

        target['image_id'] = idx
        target['labels'] = target['labels'].type(torch.int64)
        target['labels'] = torch.ones(target['labels'].shape)

        return img, target


    def load_data_file(self, i):
        try:
            img, target = torch.load(gzip.GzipFile(self.paths[i], "rb"))
        except Exception:
            print('------------------------------------------------------------------- PROBLEM CHILD')
            print(self.paths[i])
            import time
            with open('help.txt', 'a') as f:
                f.write(str(self.paths[i]))
            time.sleep(5)
            raise Exception(str(self.paths[i]))
            # cause error.
        return img, target

    def apply_transforms(self, img):
        for t in self.transforms:
            img = t(img)
        return img
