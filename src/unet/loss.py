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


def multiclass_dice_loss(
    input,
    target,
    num_classes: int,
    sample_class_counts,
    ignore_classes: t.Optional[t.List[int]] = None,
):
    if ignore_classes is None:
        ignore_classes = []

    input = F.softmax(input, dim=1)

    dices = []
    mask = []
    for index in range(num_classes):
        if index in ignore_classes:
            continue

        mask.append(sample_class_counts[:, index] > 0)
        dices.append(dice_loss(input[:, index], target[:, index]))

    # mask out classes with zero counts
    mask = torch.stack(mask, dim=1)
    dice = torch.stack(dices, dim=1)
    dice = (dice * mask).sum(dim=1) / (
        mask.sum(dim=1) + 1e-8
    )  # Average dice over classes
    return dice


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha

    def get_sample_class_counts(self, target, n_classes):
        counts = torch.zeros(
            (
                target.size(0),
                n_classes,
            ),
            dtype=torch.long,
            requires_grad=False,
        ).to(target.get_device())
        for i in range(n_classes):
            counts[:, i] += (target == i).sum(dim=(1, 2))
        return counts

    def forward(self, input, target, class_counts: torch.Tensor):
        # Weighted cross-entropy loss
        target_indices = torch.argmax(target, dim=1)  # one-hot -> indices
        weights = (class_counts + 10000) / (class_counts + 10000).sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        ce = F.cross_entropy(input, target_indices, weight=weights, reduction="none")
        ce = ce.sum(dim=(1, 2)) / weights[target_indices].sum(dim=(1, 2))

        sample_class_counts = self.get_sample_class_counts(target_indices, 6)

        # DICE loss
        # ignore class=1 since this corresponds to the background class.
        dice = multiclass_dice_loss(
            input=input,
            target=target,
            num_classes=6,
            sample_class_counts=sample_class_counts,
            ignore_classes=[1],
        )

        dice_loss = torch.where(
            dice > 0,
            -torch.log(dice + 1e-8),
            torch.tensor(
                0, dtype=torch.float, device=input.get_device(), requires_grad=False
            ),
        )

        loss = ce + 10 * dice_loss

        dice_mean = torch.sum(dice) / torch.sum(dice > 0)
        return loss.mean(), dice_mean, ce.mean()
