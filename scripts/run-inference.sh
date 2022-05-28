#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

python -m src.infer \
    --in-dir ./data/ribfrac-challenge/training/images/all \
    --out-dir ./inference-results \
    --checkpoint checkpoints-0527-2317/epoch=2-step=9324.ckpt