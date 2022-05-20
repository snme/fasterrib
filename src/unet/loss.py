import typing as t

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.flatten(start_dim=1)
    tflat = target.flatten(start_dim=1)
    intersection = (iflat * tflat).sum(dim=1)
    return (2.0 * intersection + smooth) / (
        iflat.sum(dim=1) + tflat.sum(dim=1) + smooth
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


def multiclass_dice_loss(
    input, target, num_classes: int, ignore_classes: t.Optional[t.List[int]] = None
):
    if ignore_classes is None:
        ignore_classes = []

    dices = []
    for index in range(num_classes):
        if index in ignore_classes:
            continue
        dices.append(dice_loss(input[:, index, :, :], target[:, index, :, :]))

    dice = torch.stack(dices, dim=1)
    dice = dice.sum(dim=1) / len(dices)  # taking average
    return dice


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        # ignore class=1 since this corresponds to the background class.
        dice = multiclass_dice_loss(input, target, num_classes=6, ignore_classes=[1])

        target = torch.argmax(target, dim=1)  # one-hot -> indices
        ce = self.cross_entropy(input, target)
        ce = ce.mean(dim=(1, 2))

        loss = 100 * (ce - torch.log(dice))

        return loss.mean(), dice.mean(), ce.mean()
