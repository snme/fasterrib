import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.rfc_dataset import RFCDataset
from src.unet.loss import MixedLoss
from src.unet.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(data_loader, model, optimizer, loss_fn):
    model = model.to(device)
    model.train()

    tk0 = tqdm(data_loader, total=len(data_loader))
    for b_idx, data in enumerate(tk0):
        for key, value in data.items():
            data[key] = value.to(device)

        img = data["image"]
        label = data["label"]

        out = model(img)
        loss = loss_fn(out, label)
        with torch.set_grad_enabled(True):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tk0.set_postfix(loss=loss.item(), learning_rate=optimizer.param_groups[0]["lr"])


def main():
    data = RFCDataset(
        img_dir="./data/ribfrac-challenge/training/images/all",
        label_dir="./data/ribfrac-challenge/training/labels/all",
    )
    # data = Subset(data, np.arange(10))
    data_loader = DataLoader(data)
    print(len(data))
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = MixedLoss(10.0, 2.0)
    train(data_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
