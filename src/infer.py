import argparse
import os

import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference_dataset import InferenceDataset
from src.unet.lit_unet import LitUNet
from src.unet.unet import UNet

dirname = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(
    description="Runs inference on a folder of .nii.gz files"
)
parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint file")
parser.add_argument(
    "--in-dir", type=str, help="Path to a directory of CT scan files in .nii.gz format"
)
parser.add_argument(
    "--out-dir", type=str, help="Path to a directory where results will be saved"
)

batch_size = 4


def infer_all(checkpoint, data_loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model = LitUNet(unet=UNet())
    model.load_from_checkpoint(checkpoint_path=checkpoint)
    model = model.to(device)

    for (img, basename) in tqdm(data_loader):
        preds = torch.zeros_like(img)

        for i, img_slice in enumerate(img):
            img_slice.to(device)
            probs = model(img_slice.unsqueeze(0))
            preds[i] = torch.argmax(probs, dim=1).cpu()

        img = img.permute(1, 2, 0)  # (SLICES, H, W) -> (H, W, SLICES)

        out_path = os.path.join(out_dir, f"{basename}-prediction.nii.gz")
        nib.save(img.cpu().numpy(), out_path)


def main(args):
    data = InferenceDataset(data_dir=args.in_dir)
    data_loader = DataLoader(data, batch_size=1, num_workers=24, shuffle=False)
    infer_all(checkpoint=args.checkpoint, data_loader=data_loader, out_dir=args.out_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
