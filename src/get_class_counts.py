import argparse
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.config import NUM_CORES
from src.object_detection_dataset import ODDataset

parser = argparse.ArgumentParser(
    description="Computes class weights based on counts across the whole training set"
)

dirname = os.path.dirname(__file__)
ribfrac_dir = os.path.join(dirname, "../data/ribfrac-challenge")

n_classes = 5

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

size = 512 * 512

def get_counts(prepared_dir: str, out_path: str):
    data_pos = ODDataset(data_dir=os.path.join(prepared_dir, "od-pos/"))
    data_neg = ODDataset(data_dir=os.path.join(prepared_dir, "od-neg/"))
    data = ConcatDataset([data_pos, data_neg])
    data_loader = DataLoader(data, num_workers=NUM_CORES, batch_size=1)

    class_counts = torch.zeros((n_classes,), dtype=torch.long, device=device)

    test = torch.Tensor([1, 2, 3, 4, 5])
    test = torch.unsqueeze(test, 0)

    for _, x in tqdm(data_loader):
        label = x["labels"]
        label.to(device)
        try:

            if label.numel():
                assert label.min() >= 0
                assert label.max() <= 5
        except Exception:
            print(label)

        for i in range(1, n_classes):
            class_counts[i] += int(x['area'][x['labels'] == i].sum())

        # background and unknown fractures
        class_counts[0] += int(size - x['area'].sum())

    print("class counts:", class_counts)
    torch.save(class_counts, out_path)


def get_counts_train():
    print("Getting class counts for training set")
    out_path = os.path.join(ribfrac_dir, "training/class_counts.pt")
    prepared_dir = os.path.join(ribfrac_dir, "training/prepared")
    get_counts(prepared_dir=prepared_dir, out_path=out_path)


def get_counts_val():
    print("Getting class counts for validation set")
    out_path = os.path.join(ribfrac_dir, "validation/class_counts.pt")
    prepared_dir = os.path.join(ribfrac_dir, "validation/prepared")
    get_counts(prepared_dir=prepared_dir, out_path=out_path)


if __name__ == "__main__":
    get_counts_train()
