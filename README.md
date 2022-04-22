# ribfrac

# Setup

## Create the conda environment

```bash
conda env create -f env.yml
```

## Download the data

Download the RibFrac challenge training and validation data, then extract and move the files to
a folder named `data/ribfrac-challenge` in the root of this repo.

The final folder structure should look as follows.

```
data/
  ribfrac-challenge/
    training/
      images/
        all/
          RibFrac1-image.nii.gz
          ...
      labels/
        all/
          RibFrac1-label.nii.gz
          ...
    validation/
      ribfrac-val-images/
          RibFrac421-image.nii.gz
          ...
      ribfrac-val-labels/
          RibFrac421-label.nii.gz
          ...
```
