#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

# To be run on Sherlock
python -m src.gather_labeled_files \
    --search-dir /scratch/groups/jdf1/ribfrac/irb58665-rit \
    --labels ./data/stanford/labels.csv \
    --out-dir ./data/stanford/dicoms/