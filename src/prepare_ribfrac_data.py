import argparse
import csv
import gzip
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Prepares RibFrac Challenge images and labels for training and evaluation"
)
parser.add_argument(
    "--split", choices=["train", "val", "all"], help="Which split of data to prepare"
)

dirname = os.path.dirname(__file__)
train_images_dir = os.path.join(
    dirname, "../data/ribfrac-challenge/training/images/all"
)
train_labels_dir = os.path.join(
    dirname, "../data/ribfrac-challenge/training/labels/all"
)
train_info_path = os.path.join(
    dirname, "../data/ribfrac-challenge/training/ribfrac-train-info-all.csv"
)
val_images_dir = os.path.join(
    dirname, "../data/ribfrac-challenge/validation/ribfrac-val-images"
)
val_labels_dir = os.path.join(
    dirname, "../data/ribfrac-challenge/validation/ribfrac-val-labels"
)
val_info_path = os.path.join(
    dirname, "../data/ribfrac-challenge/validation/ribfrac-val-info.csv"
)

train_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/")
train_out_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/prepared/")
val_out_dir = os.path.join(dirname, "../data/ribfrac-challenge/validation/prepared/")
class_counts_path = os.path.join(train_dir, "class_counts.pt")

n_classes = 6


def prepare_data(img_dir, label_dir, info_path, out_dir):
    label_id_to_code = {}
    with open(info_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            label_id = int(row["label_id"])
            code = int(row["label_code"])
            label_id_to_code[label_id] = code

    label_ids = np.array(list(label_id_to_code.keys()))
    codes = np.array(list(label_id_to_code.values()))
    label_mapping = np.zeros(label_ids.max() + 1, dtype=codes.dtype)
    label_mapping[label_ids] = codes

    img_paths = [path for path in Path(img_dir).glob("*.nii.gz")]
    img_paths.sort()
    label_paths = [path for path in Path(label_dir).glob("*.nii.gz")]
    label_paths.sort()

    assert len(img_paths) == len(label_paths)

    class_counts = torch.zeros((n_classes,), dtype=torch.long)

    for i, (img_path, label_path) in tqdm(
        enumerate(zip(img_paths, label_paths)), total=len(img_paths)
    ):
        img = nib.load(img_path).get_fdata().astype(np.float32)

        label = nib.load(label_path).get_fdata().astype(np.int8)

        assert label.shape == img.shape

        img = img.transpose(2, 0, 1)  # (slices, W, H)
        label = label.transpose(2, 0, 1)  # (slices, W, H)

        n_slices = img.shape[0]

        img = torch.as_tensor(label, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)

        # map all label ids to codes
        label.apply_(label_id_to_code.get)
        label += 1  # Force labels to be in range [0, 5]. https://zenodo.org/record/3893508#.YoL-wnXMJH5

        assert label.min() >= 0
        assert label.max() <= 5

        # Sum class counts
        for i in range(n_classes):
            class_counts[i] += label[label == i].sum()

        label = torch.nn.functional.one_hot(label, num_classes=6)
        label = label.type(torch.int8)

        assert list(label.size()) == [n_slices, 512, 512, 6]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for s in range(n_slices):
            out_path = os.path.join(out_dir, f"slice_pair_{i}_{s}.pt.gz")
            torch.save(
                (img[s].clone(), label[s].clone()), gzip.GzipFile(out_path, "wb")
            )

    torch.save(class_counts, class_counts_path)


def prepare_train():
    print("Preparing training data")
    prepare_data(
        img_dir=train_images_dir,
        label_dir=train_labels_dir,
        info_path=train_info_path,
        out_dir=train_out_dir,
    )


def prepare_val():
    print("Preparing validation data")
    prepare_data(
        img_dir=val_images_dir,
        label_dir=val_labels_dir,
        info_path=val_info_path,
        out_dir=val_out_dir,
    )


def prepare_all():
    print("Preparing all data")
    prepare_train()
    prepare_val()


def main(args):
    if args.split == "train":
        prepare_train()
    elif args.split == "val":
        prepare_val()
    else:
        prepare_all()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
