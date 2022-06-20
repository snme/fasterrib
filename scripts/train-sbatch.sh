#!/bin/sh
#SBATCH --job-name=ribfrac-train
#SBATCH --output=train-log.txt
#SBATCH --error=train-err.txt
#SBATCH --time=12:00:00
#SBATCH -p owners
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidwb
#SBATCH --mem=64GB
#SBATCH -C GPU_MEM:24GB
#SBATCH --gpus 1
#SBATCH --mincpus=12

set -eo pipefail

# start conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ribfrac

echo "Working dir:" `pwd`
echo "Training..."

python -m src.train