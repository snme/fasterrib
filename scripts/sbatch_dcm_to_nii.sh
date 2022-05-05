#!/bin/bash
#SBATCH --job-name=ribfrac-dcm-to-nii
#SBATCH --output=train-log.txt
#SBATCH --error=train-err.txt
#SBATCH --time=24:00:00
#SBATCH  -p owners
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidwb
#SBATCH --mem=64GB

set -eo pipefail

# DIR="$(cd $(dirname ${0}); pwd)"

# cd $DIR/..

# start conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ribfrac

echo "working dir:" `pwd`

# To be run on Sherlock
python -m src.dcm_to_nii \
    --search-dir ./data/stanford/dicoms \
    --labels ./data/stanford/labels.csv \
    --out-dir ./data/stanford/niis/
    