#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/..

# To be run on Sherlock
python -m src.dcm_to_nii \
    --search-dir ./data/stanford/dicoms \
    --labels ./data/stanford/labels.csv \
    --out-dir ./data/stanford/niis/
