import argparse
import csv
import gzip
import os
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import mode
import torch
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes, save_image
import matplotlib.pyplot as plt

from src.hole_detection import find_groupings
from src.config import NUM_CORES

parser = argparse.ArgumentParser(
    description="Prepares RibFrac Challenge images and labels for training and evaluation"
)
parser.add_argument(
    "--split", choices=["train", "val", "all"], help="Which split of data to prepare"
)

dirname = os.path.dirname(__file__)
train_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/")
val_dir = os.path.join(dirname, "../data/ribfrac-challenge/validation/")
train_images_dir = os.path.join(train_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")
train_info_path = os.path.join(train_dir, "ribfrac-train-info-all.csv")
val_images_dir = os.path.join(val_dir, "ribfrac-val-images")
val_labels_dir = os.path.join(val_dir, "ribfrac-val-labels")
val_info_path = os.path.join(val_dir, "ribfrac-val-info.csv")
train_out_dir = os.path.join(train_dir, "prepared/")
val_out_dir = os.path.join(val_dir, "prepared/")
n_classes = 6
area_threshold = 10


def prepare_data(img_dir, label_dir, info_path, out_dir, split):
    pos_dir = os.path.join(out_dir, "od-c-pos")
    neg_dir = os.path.join(out_dir, "od-c-neg")
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

    if split == "train":
        assert len(label_map) == 420

    img_paths = [path for path in Path(img_dir).glob("*-image.nii.gz")]
    img_paths.sort()
    label_paths = [path for path in Path(label_dir).glob("*-label.nii.gz")]
    label_paths.sort()
    # filenames_orig = set([str(path)[str(path).rfind('/') + 1:str(path).rfind('-image.nii.gz')] for path in img_paths])
    names_completed_pos = set([str(path)[str(path).rfind('/') + 1:str(path).rfind('_')] for path in Path(pos_dir).glob('*')])
    names_completed_neg = set([str(path)[str(path).rfind('/') + 1:str(path).rfind('_')] for path in Path(neg_dir).glob('*')])
    names_completed = names_completed_pos.union(names_completed_neg)

    assert len(img_paths) == len(label_paths)

    def make_args():
        for i in range(len(img_paths)):
            yield (img_paths[i], label_paths[i], label_map, pos_dir, neg_dir, names_completed)

    all_args = list(make_args())
    # print('preparing single image hahahahahahah')
    # prepare_img([img_paths[0], label_paths[0], label_map, pos_dir, neg_dir, names_completed])
    process_map(prepare_img, all_args, max_workers=NUM_CORES)


def prepare_img(args):
    img_path, label_path, label_map, pos_dir, neg_dir, names_completed = args
    public_id = img_path.name.strip("-image.nii.gz")
    if public_id in names_completed:
        print(f'{public_id} skipped since already completed')
        return

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

    for s in tqdm(range(n_slices)):
        target = {
            "boxes": [],
            "labels": [],
            "area": [],
        }
        if torch.any(label[s] > 1).item():
            _, groups = find_groupings(label[s].numpy(), 1)
            print(f'pos found, {len(groups)} fractures detected')


            # note that "0" is also to be ignored in our data processing.

            for g in groups:
                # calculate bounding box:
                g = np.array(g)

                # get class label for bounding box
                selection = label[s][tuple(g[:, 0]), tuple(g[:, 1])]
                value = int(mode(selection).mode.item())
                print(f'Fracture type detected: {value}')

                bb_row_min = g[:, 0].min()
                bb_row_max = g[:, 0].max()
                bb_col_min = g[:, 1].min()
                bb_col_max = g[:, 1].max()
                bb = [bb_row_min, bb_col_min, bb_row_max, bb_col_max]
                area = (bb_row_max - bb_row_min) * (bb_col_max - bb_col_min)

                if value != 0 and area > area_threshold:
                    target['boxes'].append(bb)
                    target['labels'].append(value)
                    target['area'].append(area)

        # if all of the fractures were labeled as having an "unknown" type, then
        # save as a negative. should not run, but here just in case.
        target['boxes'] = torch.Tensor(target['boxes'])
        target['labels'] = torch.Tensor(target['labels']).type(torch.int64)
        target['area'] = torch.Tensor(target['area'])

        if len(target['boxes']) == 0:
            out_path = os.path.join(neg_dir, f"{public_id}_{s}.pt.gz")
        else:
            out_path = os.path.join(pos_dir, f"{public_id}_{s}.pt.gz")
        torch.save((img[s].clone(), target), gzip.GzipFile(out_path, "wb"))

        """
        test = label[s].numpy()
        test2 = test.copy()
        test2[test2 == 1] = 0
        test = test + test2.T
        label[s] = torch.from_numpy(test)
        bbs = []
        bbs = np.array(bbs)
        test = label[s].type(torch.uint8)
        print('shape')
        print(test.shape)
        test = torch.unsqueeze(test, 0)
        drawn_boxes1 = draw_bounding_boxes(test.type(torch.uint8), torch.from_numpy(bbs), colors="red")
        plt.figure()
        plt.imshow(drawn_boxes1.numpy().mean(axis=0))
        plt.savefig('test-2.png', dpi=400)
        sys.exit(0)

        test = label[s].numpy()
        test2 = test.copy()
        test2[test2 == 1] = 0
        test = test + test2.T
        # although we're repurposing a hole detection algorithm, we can consider
        # fracture pixel groups to be the sort of "edges".
        print(len(groups))
        for g in groups:
            for row, col in g:
                test[row, col] = 6

        plt.figure()
        plt.imshow(test)
        plt.colorbar()
        plt.savefig('test-1.png', dpi=400)
        print('exiting')
        """

def prepare_train():
    print("Preparing training data")
    prepare_data(
        img_dir=train_images_dir,
        label_dir=train_labels_dir,
        info_path=train_info_path,
        out_dir=train_out_dir,
        split="train",
    )


def prepare_val():
    print("Preparing validation data")
    prepare_data(
        img_dir=val_images_dir,
        label_dir=val_labels_dir,
        info_path=val_info_path,
        out_dir=val_out_dir,
        split="val",
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
