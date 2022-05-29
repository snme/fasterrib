#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

python -m src.infer \
    --in-dir ./data/ribfrac-challenge/training/images/all \
    --out-dir ./inference-results \
    --checkpoint checkpoints-0528-1710/epoch=0-step=5400-val-bin-dice-val_bin_dice=0.63.ckpt