import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.rfc_dataset import RFCDataset

parser = argparse.ArgumentParser(
    description="Computes class weights based on counts across the whole training set"
)

dirname = os.path.dirname(__file__)
data_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/prepared/")
out_path = os.path.join(dirname, "../data/ribfrac-challenge/training/class_counts.pt")

n_classes = 6

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data = RFCDataset(
        data_dir="./data/ribfrac-challenge/training/prepared",
    )
    data_loader = DataLoader(data, num_workers=24, batch_size=32, shuffle=True)

    class_counts = torch.zeros((n_classes,), dtype=torch.long, device=device)

    for x in tqdm(data_loader):
        label = x["label"]
        label.to(device)
        label = torch.argmax(label, dim=1)  # one-hot -> indices

        assert label.min() >= 0
        assert label.max() <= 5

        for i in range(n_classes):
            class_counts[i] += label[label == i].sum()

    print("class counts:", class_counts)
    torch.save(class_counts, out_path)


if __name__ == "__main__":
    main()
