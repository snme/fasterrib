#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

# To be run on Sherlock
python -m src.gather_labeled_files \
    --search-dir ./ \
    --labels ./data/stanford/labels.csv \
    --out-dir ./data/stanford/dicoms/