#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

python -m src.infer \
    --in-dir ./data/ribfrac-challenge/training/images/all \
    --out-dir ./inference-results \
    --checkpoint checkpoints-0527-0929/epoch=0-step=600.ckpt
