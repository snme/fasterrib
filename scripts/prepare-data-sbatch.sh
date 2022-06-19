#!/bin/sh
#SBATCH --job-name=ribfrac-prepare-data
#SBATCH --output=prepare-data-log.txt
#SBATCH --error=prepare-data-err.txt
#SBATCH --time=6:00:00
#SBATCH -p owners
#SBATCH --nodes=1
#SBATCH --mincpus=24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davidwb
#SBATCH --partition=normal
#SBATCH --mem=64GB

set -eo pipefail

# start conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ribfrac

echo "working dir:" `pwd`
echo "Preparing ribfrac data..."

python -m src.prepare_data