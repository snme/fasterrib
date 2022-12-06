import math
import sys
import time

import torch
import numpy as np

import torchvision.models.detection.mask_rcnn
from torchvision.ops import nms
import torchvision.ops.boxes as bops
from src import utils
from src.coco_eval import CocoEvaluator
from src.coco_utils import get_coco_api_from_dataset
from src.unet.lit_unet import LitUNet
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    loss_history = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in (metric_logger.log_every(data_loader, print_freq, header)):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if type(v) != int else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            print(images[0].shape)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_history.append(loss_value)

        print(f"Loss is {loss_value}")
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        d = {}
        for k, v in metric_logger.meters.items():
            if k == 'lr':
                d[k] = float(str(v))
            else:
                d[k] = float(str(v).split()[0])
        wandb.log(d)

    return metric_logger, loss_history


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, to_print=False, skip_load=False, model_no=0, nms_iou_threshold=0.5):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    print('about to get coco')
    if not skip_load:
        coco = get_coco_api_from_dataset(data_loader.dataset)
        with open('full_validation_coco_no_class_large_imbalanced.pkl', 'wb') as outp:
            pickle.dump(coco, outp, pickle.HIGHEST_PROTOCOL)
    else:
        with open('full_validation_coco_no_class_large_imbalanced.pkl', 'rb') as inp:
            coco = pickle.load(inp)
    print('after getting coco')
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    pos = 0
    true_pos = 0
    false_neg = 0

    for images, targets in (metric_logger.log_every(data_loader, 100, header)):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {}
        for image, target, output in (zip(images, targets, outputs)):

            '''fig, ax = plt.subplots()
            ax.imshow(image.cpu().squeeze(0))'''

            # nms over all boxes
            nms_inds = nms(output['boxes'], output['scores'], nms_iou_threshold)
            if to_print:
                print(output['boxes'])

            # nms per class
            boxes = output['boxes'][nms_inds]
            labels = output['labels'][nms_inds]
            scores = output['scores'][nms_inds]

            '''# per class <- in previous version of the model
            for label in output['labels'].unique():
                inds = (output['labels'] == label) # & (output['scores'] > 0.4)
                nms_inds = nms(output['boxes'][inds], output['scores'][inds], 0.5)
                boxes.extend(output['boxes'][inds][nms_inds].tolist())
                labels.extend(output['labels'][inds][nms_inds].tolist())
                scores.extend(output['scores'][inds][nms_inds].tolist())'''

            boxes = torch.Tensor(boxes)
            labels = torch.Tensor(labels)
            scores = torch.Tensor(scores)
            inds = torch.argsort(scores, descending=True)
            output = {
                'boxes' : boxes[inds],

                # model does not distinguish between fractures at the prediction/training stage. characterization
                # is done with the percentage displacement calculation.
                'labels' : torch.ones(labels[inds].shape),
                'scores' : scores[inds]
            }

            # fix dimensionality
            if len(output['boxes']) == 0:
                output['boxes'] = torch.zeros((0, 4))

            res[target['image_id']] = output
            if to_print:
                print('output')
                print(output)
                print('expected')
                print(target)

            '''for box in output['boxes'].cpu():
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],  linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            for box in target['boxes'].cpu():
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],  linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

            ax.set_title(f'model number {model_no}, image {target["image_id"]}')

            fig.savefig(f"./predictions/{str(model_no)}_{str(target['image_id'])}.png", dpi=400)'''
            pos += len(target['labels'])
            for box in target['boxes']:
                for obox in output['boxes']:
                    # print((box.to(device), obox.to(device)))
                    if bops.box_iou(box.to(device).unsqueeze(0), obox.to(device).unsqueeze(0)) > 1e-5:
                        # solely for the sake of sensitivity — false positives are tracked in coco_evaluator
                        true_pos += 1
                        break

        # res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    print('pos:', pos)
    print('true_pos:', true_pos)
    print('false_neg:', false_neg)
    print('sensitivity', true_pos/pos)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # wandb.log({k : str(v) for k, v in metric_logger.meters.items()})
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
