import os
import typing as t

import torch
from black import Enum
from pydantic import BaseModel

dirname = os.path.dirname(__file__)

train_dir = os.path.join(dirname, "../../data/ribfrac-challenge/training/")
class_counts_path = os.path.join(train_dir, "class_counts.pt")
default_neg_dir = os.path.join(train_dir, "prepared/neg")


class ELossFunction(Enum):
    CE = "CE"
    CE_BD_MD = "CE_BD_MD"
    CE_MD = "CE_MD"
    FOCAL = "FOCAL"
    FOCAL_BD_MD = "FOCAL_BD_MD"


class HParams(BaseModel):
    enc_chs: t.List[int] = (1, 64, 128, 256, 512, 1024)
    dec_chs: t.List[int] = (1024, 512, 256, 128, 64)
    num_classes: int = 6
    class_counts: t.List[int] = None
    neg_dir: t.Optional[str]
    learning_rate: float = 1e-6
    batch_size: int = 8
    loss_fn: ELossFunction = ELossFunction.CE_BD_MD
    ce_weight: float = 1
    bd_weight: float = 10
    md_weight: float = 10
    reweight_factor: float = 0.5


def get_hparams(device: str):
    class_counts = torch.load(class_counts_path).numpy().tolist()
    # class_counts.requires_grad_(False)
    # class_counts = class_counts.to(device)
    hparams = HParams(class_counts=class_counts)
    return hparams
