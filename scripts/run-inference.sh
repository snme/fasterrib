#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

# To be run on Sherlock
python -m src.infer \
    --in-dir ./data/ribfrac-challenge/validation/ribfrac-val-images \
    --out-dir ./inference-results \
    --checkpoint ./checkpoints-0525-1018/epoch=6-step=35598.ckpt
    