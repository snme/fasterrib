import typing as t

import torch
import torch.nn.functional as F
from torch import nn


def dice_score(input, target):
    smooth = 1.0
    iflat = input.flatten(start_dim=1)
    tflat = target.flatten(start_dim=1)
    intersection = (iflat * tflat).sum(dim=1)
    return (2.0 * intersection + smooth) / (
        iflat.sum(dim=1) + tflat.sum(dim=1) + smooth
    )


def multiclass_dice(
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
        dices.append(dice_score(input[:, index], target[:, index]))

    # mask out classes with zero counts
    mask = torch.stack(mask, dim=1)
    dice = torch.stack(dices, dim=1)
    dice = (dice * mask).sum(dim=1) / (
        mask.sum(dim=1) + 1e-8
    )  # Average dice over classes
    return dice


class MixedLoss(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

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

    def get_class_dice_score(self, input, target_one_hot, class_, sample_class_counts):
        input = F.softmax(input, dim=1)
        mask = sample_class_counts[:, class_] > 0
        dice = dice_score(input[:, class_], target_one_hot[:, class_])
        dice = (dice * mask).sum() / mask.sum()  # can be nan
        return dice

    def get_all_dice_scores(self, input, target):
        sample_class_counts = self.get_sample_class_counts(target, 6)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        scores = []
        for i in range(self.n_classes):
            scores.append(
                self.get_class_dice_score(
                    input=input,
                    target_one_hot=target_one_hot,
                    class_=i,
                    sample_class_counts=sample_class_counts,
                )
            )
        print(scores)
        return torch.stack(scores)

    def get_binary_dice_score(
        self,
        input,
        target,
    ):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)
        input = F.softmax(input, dim=1)

        # Sum the non-background probabilities to get a 0/1 probability
        input = input.sum(dim=1) - input[:, 1]
        dice = dice_score(input, (target != 1))
        return dice

    def forward(self, input, target, class_counts: torch.Tensor):
        # Weighted cross-entropy loss
        weights = (class_counts + 1) / (class_counts + 1).sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        ce = F.cross_entropy(input, target, weight=weights, reduction="none")
        ce = ce.sum(dim=(1, 2)) / weights[target].sum(dim=(1, 2))

        # DICE loss
        # ignore class=1 since this corresponds to the background class.
        sample_class_counts = self.get_sample_class_counts(target, 6)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        multi_dice = multiclass_dice(
            input=input,
            target=target_one_hot,
            num_classes=6,
            sample_class_counts=sample_class_counts,
            ignore_classes=[0, 1],
        )

        # For reporting purposes only
        binary_dice = self.get_binary_dice_score(input, target)

        dice_loss = torch.where(
            multi_dice > 0,
            -torch.log(multi_dice + 1e-8),
            torch.tensor(
                0, dtype=torch.float, device=input.get_device(), requires_grad=False
            ),
        )

        loss = ce + 10 * dice_loss

        dice_mean = torch.sum(multi_dice) / torch.sum(multi_dice > 0)
        return loss.mean(), dice_mean, ce.mean(), binary_dice.mean()
