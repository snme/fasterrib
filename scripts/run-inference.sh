#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

python -m src.infer \
    --in-dir ./data/ribfrac-challenge/validation/ribfrac-val-images \
    --out-dir ./inference-results \
    --checkpoint checkpoints-0602-0203-ce+bd+md+rw-1/epoch=6-step=22110-val_bin_dice=0.42.ckpt