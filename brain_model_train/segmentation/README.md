# RADCAD-BRAIN-MONAI-SWIN-UNETR

## Project Overview

This repository contains code for the RADCAD-BRAIN-MONAI-SWIN-UNETR project, which focuses on training MONAI SWIN UNETR models for the automatic segmentation of brain imaging data.

## Hyperparameters

- ROI (Region of Interest):
  - Height (h): 128
  - Width (w): 128
  - Depth (d): 128
- Batch Size: 4
- SWIN Batch Size: 4
- Fold: 1
- Inference Overlap: 0.5
- Maximum Epochs: 2
- Validate Every: 10 epochs
- Start Epoch: 0
- Weight Decay: 1e-5
- Learning Rate (LR): 1e-4

## Data
You can run download_data.sh to download the data into `~/data`

```bash
./download_data.sh
```

If you're on a Lambda Labs machine, you'll first need to get your filesystem mount path and then actually download the data to there via symlinks:
```bash
mounted_on=$(df -h | awk '$NF ~ /\/home/ {print $NF}')
echo "Filesystem located at $mounted_on"
mkdir -p $mounted_on/data
ln -s $mounted_on/data ~/data
./download_data.sh
```

To fix some of the affine matrix issues with xCures (the different modalities have very slightly different affine matrices), once the data is downloaded you can run the data_explorer.ipynb notebook.

## Data Paths

- MONAI Data Directory: `/home/bratsdata/output/`
  - This is where the model checkpoint is saved.

- Dataset Base Directory: `/home/bratsdata/input/brats2021/`
  - This directory contains the input dataset.

- Datalist JSON Directory: `/home/worlako/datalist_json/`
  - This directory contains JSON files related to the dataset (i.e BRATS or LUMIERE dataset).

## Usage

- Clone this repository.
- Configure the hyperparameters as needed in your code.
- Make sure the data paths are correctly set to your dataset locations.

## Requirements

To set up the required environment, run the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model
To train the model, follow these steps:

- Configure the hyperparameters in **config/config.yaml** as needed for your specific task.

- Make sure the data paths in **config/config.yaml** are correctly set to your dataset locations.

Run the training script using the following command from the directory above **segmentation/**:

```bash
python -m segmentation.main
```