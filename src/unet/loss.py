import typing as t

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(input, target):
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
    input,
    target,
    num_classes: int,
    class_counts,
    ignore_classes: t.Optional[t.List[int]] = None,
):
    if ignore_classes is None:
        ignore_classes = []

    input = F.softmax(input, dim=1)

    dices = []
    for index in range(num_classes):
        if index in ignore_classes:
            continue

        gtz = class_counts[index] > 0

        dices.append(gtz * dice_loss(input[:, index], target[:, index]))

    dice = torch.stack(dices, dim=1)
    dice = dice.sum(dim=1) / (dice > 0).sum(dim=1)
    dice = torch.nan_to_num(dice, nan=0.0)
    return dice


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target, class_counts: torch.Tensor):
        # Weighted cross-entropy loss
        target_indices = torch.argmax(target, dim=1)  # one-hot -> indices

        # DICE loss
        # ignore class=1 since this corresponds to the background class.
        non_background_counts = class_counts[[0, 2, 3, 4, 5]].sum()
        if non_background_counts > 0:
            dice = multiclass_dice_loss(
                input,
                target,
                num_classes=6,
                class_counts=class_counts,
                ignore_classes=[1],
            )
        else:
            dice = None

        weights = 1 / (class_counts + 1)
        weights = weights / weights.sum()

        ce = F.cross_entropy(input, target_indices, weight=weights, reduction="none")
        ce = ce.mean(dim=(1, 2))

        if dice is None:
            loss = 100 * ce
            return loss.mean(), None, ce.mean()

        loss = torch.tensor(0.0).to(input.get_device())
        for i, d in enumerate(dice):
            if d != torch.nan and d > 0:
                loss += 100 * ce[i] - torch.log(d)
            else:
                loss += 100 * ce[i]

        return loss, dice[dice > 0].mean(), ce.mean()
