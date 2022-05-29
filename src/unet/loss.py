import typing as t
from tkinter import E

import torch
import torch.nn.functional as F
from torch import nn


def dice_score(input, target, pixel_mask=None):
    smooth = 1.0

    iflat = input.flatten(start_dim=1)
    tflat = target.flatten(start_dim=1).clone()

    if pixel_mask is not None:
        assert pixel_mask.shape == input.shape
        pixel_mask = pixel_mask.flatten(start_dim=1)
        iflat = iflat.clone()
        tflat = tflat.clone()
        iflat[pixel_mask] = 0
        tflat[pixel_mask] = 0

    intersection = (iflat * tflat).sum(dim=1)
    return (2.0 * intersection + smooth) / (
        iflat.sum(dim=1) + tflat.sum(dim=1) + smooth
    )


def multiclass_dice(
    softmax_input,
    target_one_hot,
    num_classes: int,
    ignore_classes: t.Optional[t.List[int]] = None,
):
    if ignore_classes is None:
        ignore_classes = []

    pixel_mask = target_one_hot[:, 0] == 1

    dices = []
    for index in range(num_classes):
        if index in ignore_classes:
            continue

        dices.append(
            dice_score(
                softmax_input[:, index], target_one_hot[:, index], pixel_mask=pixel_mask
            )
        )

    return torch.stack(dices, dim=1).mean(dim=1)


class MixedLoss(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

    def get_class_dice_score(self, softmax_input, target_one_hot, class_):
        pixel_mask = target_one_hot[:, 0] == 1
        dice = dice_score(
            softmax_input[:, class_], target_one_hot[:, class_], pixel_mask=pixel_mask
        )
        dice = dice.mean()
        return dice

    def get_all_dice_scores(self, input, target):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.long).permute(0, 3, 1, 2)
        softmax_input = self.get_softmax_scores(input)

        scores = []
        for i in range(self.n_classes):
            scores.append(
                self.get_class_dice_score(
                    softmax_input=softmax_input,
                    target_one_hot=target_one_hot,
                    class_=i,
                )
            )

        return torch.stack(scores)

    def get_binary_dice_score(self, softmax_input, target, target_one_hot):
        pixel_mask = target_one_hot[:, 0] == 1

        # Sum the non-background probabilities to get a 0/1 probability
        softmax_input = (
            softmax_input.sum(dim=1) - softmax_input[:, 1] - softmax_input[:, 0]
        )
        dice = dice_score(
            softmax_input, (target > 1).type(torch.uint8), pixel_mask=pixel_mask
        )
        return dice

    def get_softmax_scores(self, input):
        softmax_input = input.clone()
        softmax_input[:, 0] = float("-inf")
        softmax_input = F.softmax(softmax_input, dim=1)
        return softmax_input

    def get_sample_class_counts(self, target, n_classes=6):
        counts = torch.zeros(
            (n_classes,),
            dtype=torch.long,
            requires_grad=False,
        ).to(target.get_device())
        for i in range(n_classes):
            counts[i] += (target == i).sum()
        return counts

    def forward(self, input, target, class_counts: t.Optional[torch.Tensor] = None):
        softmax_input = self.get_softmax_scores(input)

        # reweighting
        if class_counts:
            weights = class_counts + 1
            weights = 1 / weights
            weights[0] = 0
            weights = weights / weights.sum()
        else:
            weights = None

        ce = F.cross_entropy(
            input, target, reduction="none", ignore_index=0, weight=weights
        )

        # focal_loss = (torch.pow(1 - torch.exp(-ce), 2) * ce).sum(dim=(1, 2))
        ce_loss = ce.mean(dim=(1, 2))

        # DICE loss
        # ignore class=1 since this corresponds to the background class.
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        multi_dice = multiclass_dice(
            softmax_input=softmax_input,
            target_one_hot=target_one_hot,
            num_classes=6,
            ignore_classes=[0, 1],
        )

        binary_dice = self.get_binary_dice_score(
            softmax_input, target, target_one_hot=target_one_hot
        )

        multi_dice_loss = -torch.log(multi_dice)
        binary_dice_loss = -torch.log(binary_dice)

        loss = ce_loss + binary_dice_loss + multi_dice_loss

        return loss.mean(), multi_dice.mean(), ce_loss.mean(), binary_dice.mean()
