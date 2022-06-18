#!/bin/sh
#SBATCH --job-name=ribfrac-prepare-data
#SBATCH --output=prepare-data-log.txt
#SBATCH --error=prepare-data-err.txt
#SBATCH --time=4:00:00
#SBATCH -p owners
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidwb
#SBATCH --partition=normal
#SBATCH --mem=64GB

set -eo pipefail

# start conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ribfrac

echo "working dir:" `pwd`
echo "Computing class counts over training and validation sets"

python -m src.get_class_counts