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
    softmax_input,
    target,
    num_classes: int,
    ignore_classes: t.Optional[t.List[int]] = None,
):
    if ignore_classes is None:
        ignore_classes = []

    dices = []
    for index in range(num_classes):
        if index in ignore_classes:
            continue

        dices.append(dice_score(softmax_input[:, index], target[:, index]))

    dice = torch.stack(dices, dim=1)
    dice = dice.sum(dim=1) / len(dices)
    return dice


class MixedLoss(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

    def get_class_dice_score(
        self, softmax_input, target_one_hot, class_, sample_class_counts
    ):
        mask = sample_class_counts[:, class_] > 0
        dice = dice_score(softmax_input[:, class_], target_one_hot[:, class_])
        dice = (dice * mask).sum() / mask.sum()  # can be nan
        return dice

    def get_all_dice_scores(self, input, target):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        softmax_input = self.get_softmax_scores(input)

        scores = []
        for i in range(self.n_classes):
            scores.append(
                self.get_class_dice_score(
                    softmax_input=softmax_input, target_one_hot=target_one_hot, class_=i
                )
            )
        return torch.stack(scores)

    def get_binary_dice_score(
        self,
        softmax_input,
        target,
    ):
        # Sum the non-background probabilities to get a 0/1 probability
        softmax_input = softmax_input.sum(dim=1) - softmax_input[:, 1]
        dice = dice_score(
            softmax_input, (target != 1).type(torch.uint8)
        )  # non-background dice score
        return dice

    def get_softmax_scores(self, input):
        # Mask out the predictions for the -1 class.
        # This causes the softmax score for it to be zero.
        softmax_input = input.clone()
        softmax_input[:, 0] = float("-inf")
        softmax_input = F.softmax(softmax_input, dim=1)
        return softmax_input

    def forward(self, input, target, class_counts: torch.Tensor):
        # Weighted cross-entropy loss
        weights = class_counts.clone()
        weights = (weights + 1) / (weights + 1).sum()
        weights = 1 / weights
        weights[0] = 0  # ignore the -1 class
        weights = weights / weights.sum()

        ce = F.cross_entropy(input, target, weight=weights, reduction="none")
        ce = ce.sum(dim=(1, 2)) / weights[target].sum(dim=(1, 2))

        # DICE loss
        # ignore class=1 since this corresponds to the background class.
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        # Mask out the predictions for the -1 class.
        # This causes the softmax score for it to be zero.
        softmax_input = self.get_softmax_scores(input)

        multi_dice = multiclass_dice(
            softmax_input=softmax_input,
            target=target_one_hot,
            num_classes=6,
            ignore_classes=[0, 1],
        )

        binary_dice = self.get_binary_dice_score(softmax_input, target)
        binary_dice_loss = -torch.log(binary_dice)

        dice_loss = torch.where(
            multi_dice > 0,
            -torch.log(multi_dice + 1e-8),
            torch.tensor(
                0, dtype=torch.float, device=input.get_device(), requires_grad=False
            ),
        )

        loss = ce + 10 * dice_loss + binary_dice_loss

        dice_mean = torch.sum(multi_dice) / torch.sum(multi_dice > 0)
        return loss.mean(), dice_mean, ce.mean(), binary_dice.mean()
