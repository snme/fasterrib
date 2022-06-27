import os
import typing as t

import torch
from black import Enum
from pydantic import BaseModel

dirname = os.path.dirname(__file__)
train_dir = os.path.join(dirname, "../../data/ribfrac-challenge/training/")
class_counts_path = os.path.join(train_dir, "class_counts.pt")
default_neg_dir = os.path.join(train_dir, "prepared/neg")
default_class_counts = torch.load(class_counts_path).numpy().tolist()


class ELossFunction(Enum):
    CE = "CE"
    CE_BD = "CE_BD"
    CE_MD = "CE_MD"
    CE_BD_MD = "CE_BD_MD"
    FOCAL = "FOCAL"
    FOCAL_MD = "FOCAL_MD"
    FOCAL_BD_MD = "FOCAL_BD_MD"


class ETask(Enum):
    DETECTION = "DETECTION"
    CLASSIFICATION = "CLASSIFICATION"


class HParams(BaseModel):
    task: ETask = ETask.DETECTION
    enc_chs: t.List[int] = (1, 64, 128, 256, 512, 1024)
    dec_chs: t.List[int] = (1024, 512, 256, 128, 64)
    num_classes: int = 6
    class_counts: t.Optional[t.List[int]] = default_class_counts
    neg_dir: t.Optional[str] = default_neg_dir
    learning_rate: float = 1e-6
    batch_size: int = 12
    neg_samples: int = 4
    loss_fn: ELossFunction = ELossFunction.CE_BD
    focal_gamma: float = 2
    focal_weight: float = 1
    ce_weight: float = 1
    bd_weight: float = 2
    md_weight: float = 1
    reweight_factor: float = 0
