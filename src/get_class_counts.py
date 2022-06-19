import argparse
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.config import NUM_CORES
from src.rfc_dataset import RFCDataset

parser = argparse.ArgumentParser(
    description="Computes class weights based on counts across the whole training set"
)

dirname = os.path.dirname(__file__)
ribfrac_dir = os.path.join(dirname, "../data/ribfrac-challenge")

n_classes = 6

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


def get_counts(prepared_dir: str, out_path: str):
    data_pos = RFCDataset(data_dir=os.path.join(prepared_dir, "pos/"))
    data_neg = RFCDataset(data_dir=os.path.join(prepared_dir, "neg/"))
    data = ConcatDataset([data_pos, data_neg])
    data_loader = DataLoader(data, num_workers=NUM_CORES, batch_size=16)

    class_counts = torch.zeros((n_classes,), dtype=torch.long, device=device)

    for x in tqdm(data_loader):
        label = x["label"]
        label.to(device)

        assert label.min() >= 0
        assert label.max() <= 5

        for i in range(n_classes):
            class_counts[i] += (label == i).sum()

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
