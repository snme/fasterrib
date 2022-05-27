#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

export PYTORCH_NO_CUDA_MEMORY_CACHING=1

python -m src.infer \
    --in-dir ./data/ribfrac-challenge/mini/images/ \
    --out-dir ./inference-results-mini \
    --checkpoint checkpoints-mini/best.ckpt.ckpt
