# ribfrac

# Setup

## Install Miniconda

If you don't already have `conda`, please find the install instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

## Create the conda environment

```bash
conda env create -f env.yml
```

## Download the data

Download the RibFrac challenge training and validation data, then extract and move the files to
a folder named `data/ribfrac-challenge` in the root of this repo. You will need to combine
the `Part1/` and `Part2/` images and labels into single folders named `all/`, and combine the two train info `csv` files
into a single file named `ribfrac-train-info-all.csv`.

The final folder structure should look as follows.

```
data/
  ribfrac-challenge/
    training/
      ribfrac-train-info-all.csv
      images/
        all/
          RibFrac1-image.nii.gz
          ...
      labels/
        all/
          RibFrac1-label.nii.gz
          ...
    validation/
      ribfrac-val-info.csv
      ribfrac-val-images/
          RibFrac421-image.nii.gz
          ...
      ribfrac-val-labels/
          RibFrac421-label.nii.gz
          ...
```

# 2D Model

## Training

### Prepare data for training

The next step after downloading the RFC data is to run the data preparation script:

```bash
python -m src.prepare_data --split all
```

This will prepare and save each 2d slice of every image. The script takes about 1 hour on our desktop workstation.

### Train

```bash
python -m src.train
```

## Inference

```bash
python -m src.infer --in-dir <scans-dir> --out-dir <predictions-dir> --checkpoint <checkpoint-path>
```
