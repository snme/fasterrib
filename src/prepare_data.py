import argparse
import csv
import gzip
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser(
    description="Prepares RibFrac Challenge images and labels for training and evaluation"
)
parser.add_argument(
    "--split", choices=["train", "val", "all"], help="Which split of data to prepare"
)

dirname = os.path.dirname(__file__)
train_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/")
val_dir = os.path.join(dirname, "../data/ribfrac-challenge/validation/")
train_images_dir = os.path.join(train_dir, "images/all")
train_labels_dir = os.path.join(train_dir, "labels/all")
train_info_path = os.path.join(train_dir, "ribfrac-train-info-all.csv")
val_images_dir = os.path.join(val_dir, "ribfrac-val-images")
val_labels_dir = os.path.join(val_dir, "ribfrac-val-labels")
val_info_path = os.path.join(val_dir, "ribfrac-val-info.csv")
train_out_dir = os.path.join(train_dir, "prepared/")
val_out_dir = os.path.join(val_dir, "prepared/")
n_classes = 6


def prepare_data(img_dir, label_dir, info_path, out_dir, split):
    pos_dir = os.path.join(out_dir, "pos")
    neg_dir = os.path.join(out_dir, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    label_map = {}
    with open(info_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            public_id = row["public_id"]
            label_id = int(row["label_id"])
            code = int(row["label_code"])

            if public_id not in label_map:
                label_map[public_id] = {}

            label_map[public_id][label_id] = code

    if split == 'train':
        assert len(label_map) == 420

    img_paths = [path for path in Path(img_dir).glob("*-image.nii.gz")]
    img_paths.sort()
    label_paths = [path for path in Path(label_dir).glob("*-label.nii.gz")]
    label_paths.sort()

    assert len(img_paths) == len(label_paths)

    def make_args():
        for i in range(len(img_paths)):
            yield (img_paths[i], label_paths[i], label_map, pos_dir, neg_dir)

    all_args = list(make_args())
    process_map(prepare_img, all_args, max_workers=24)


def prepare_img(args):
    img_path, label_path, label_map, pos_dir, neg_dir = args
    public_id = img_path.name.strip("-image.nii.gz")

    img = nib.load(img_path).get_fdata().astype(np.float32)

    label = nib.load(label_path).get_fdata().astype(np.int8)

    assert label.shape == img.shape

    img = img.transpose(2, 0, 1)  # (slices, W, H)
    label = label.transpose(2, 0, 1)  # (slices, W, H)

    n_slices = img.shape[0]

    img = torch.as_tensor(img, dtype=torch.float32)
    label = torch.as_tensor(label, dtype=torch.long)

    # map all label ids to codes
    label = label.apply_(label_map[public_id].get)
    label += 1  # Force labels to be in range [0, 5]. https://zenodo.org/record/3893508#.YoL-wnXMJH5

    assert label.min() >= 0
    assert label.max() <= 5

    assert list(label.size()) == [n_slices, 512, 512]

    for s in range(n_slices):
        if torch.any(label[s] > 1).item():
            out_path = os.path.join(pos_dir, f"{public_id}_{s}.pt.gz")
        else:
            out_path = os.path.join(neg_dir, f"{public_id}_{s}.pt.gz")
        torch.save((img[s].clone(), label[s].clone()), gzip.GzipFile(out_path, "wb"))


def prepare_train():
    print("Preparing training data")
    prepare_data(
        img_dir=train_images_dir,
        label_dir=train_labels_dir,
        info_path=train_info_path,
        out_dir=train_out_dir,
        split='train'
    )


def prepare_val():
    print("Preparing validation data")
    prepare_data(
        img_dir=val_images_dir,
        label_dir=val_labels_dir,
        info_path=val_info_path,
        out_dir=val_out_dir,
        split='val'
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
