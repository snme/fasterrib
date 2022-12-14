#!/bin/sh
#SBATCH --job-name=ribfrac-get-class-counts
#SBATCH --output=get-class-counts-log.txt
#SBATCH --error=get-class-counts-err.txt
#SBATCH --time=4:00:00
#SBATCH -p owners
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidwb
#SBATCH --partition=normal
#SBATCH --mem=8GB
#SBATCH --mincpus=24

set -eo pipefail

# start conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ribfrac

echo "working dir:" `pwd`
echo "Computing class counts over training and validation sets"

python -m src.get_class_counts