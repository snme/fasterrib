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
        weights = (class_counts + 1) / (class_counts + 1).sum()
        ce = F.cross_entropy(input, target_indices, weight=weights, reduction="none")
        ce = ce.mean(dim=(1, 2))

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

        loss = torch.tensor(0.0, requires_grad=True).to(input.get_device())
        dices = []
        for i, d in enumerate(dice):
            if d > 0:
                loss += 100 * (ce[i] - torch.log(d))
                dices.append(d)
            else:
                loss += 100 * ce[i]

        dice_mean = None
        if len(dices) > 0:
            dice_mean = torch.stack(dices).mean()
        return loss, dice_mean, ce.mean()
