# Running on SLURM

## Prepare Data

```bash
sbatch ./scripts/prepare-data-sbatch.sh
```

```bash
sbatch ./scripts/get-class-counts-sbatch.sh
```

## Train

```bash
sbatch -export WANDB_API_KEY=<YOUR_API_KEY> ./scripts/train-sbatch.sh
```
