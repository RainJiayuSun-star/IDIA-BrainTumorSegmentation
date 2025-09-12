# Brain Cancer Segmentation Training

## Setup
1. Clone this repository: 
```bash
git clone https://github.com/Theta-Tech-AI/brain-segmentation-training.git
cd brain-segmentation-training
```
2. Initiate submodules:
```bash
git submodule update --init --recursive
```
3. Create a python virtual environment or conda environment.
4. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Data

### Download Data
If you have the nifti files already, you can skip this step and move onto the **Prepare Data** step. Otherwise, you will have to download the data.

You will first have to get a set of Amazon keys and configure your aws cli. You can test if this is working by running:
```bash
aws sts get-caller-identity
aws s3 ls s3://wisconsin-globus-transfer/
```
and ensuring it lists some datasets. Once your Amazon credentials are configured, you can download the data into your `~/data` folder from Amazon S3 using the following script. 
```bash
./data/download_data.sh
```
This can take a long time so grab some coffee. You can also edit the shell script to change the directory the data is downloaded into.
### Prepare Data
Once the data is downloaded, you should edit the `data/datasets.yaml` file to represent the paths and glob patterns for your data, as well as the number of folds you want to create.

Here is an example of what the file should look like:
```yaml
data_root: '~/data'
folds: 5

datasets:
  CCF:
    base_path: 'CCF/Preprocessed'
    patient_pattern: '[0-9][0-9]/Baseline'
    label_pattern: '{patient}_*[Aa]ll-[Ll]abel_SRI24_brats.nii.gz'
    image_patterns:
      FLAIR: '{patient}_FLAIR_SRI24_SkullS_BiasCorrect.nii.gz'
      T1c: '{patient}_T1c_SRI24_SkullS_BiasCorrect.nii.gz'
      T1: '{patient}_T1_SRI24_SkullS_BiasCorrect.nii.gz'
      T2: '{patient}_T2_SRI24_SkullS_BiasCorrect.nii.gz'
```

In this example, the following files may exist for a given patient (patient `00`) on your machine:
```
~/data/CCF/Preprocessed/00/Baseline/00_All-Label_SRI24_brats.nii.gz
~/data/CCF/Preprocessed/00/Baseline/00_T1c_SRI24_SkullS_BiasCorrect.nii.gz
~/data/CCF/Preprocessed/00/Baseline/00_T1_SRI24_SkullS_BiasCorrect.nii.gz
~/data/CCF/Preprocessed/00/Baseline/00_T2_SRI24_SkullS_BiasCorrect.nii.gz
~/data/CCF/Preprocessed/00/Baseline/00_FLAIR_SRI24_SkullS_BiasCorrect.nii.gz
```

If you want to change which modalities are used for training, this is where you can do that. For instance, 4 modalities might have
```yaml
image_patterns:
    FLAIR: '{patient}_FLAIR_SRI24_SkullS_BiasCorrect.nii.gz'
    T1c: '{patient}_T1c_SRI24_SkullS_BiasCorrect.nii.gz'
    T1: '{patient}_T1_SRI24_SkullS_BiasCorrect.nii.gz'
    T2: '{patient}_T2_SRI24_SkullS_BiasCorrect.nii.gz'
```
while 3 modalities might have:
```yaml
image_patterns:
    FLAIR: '{patient}_FLAIR_SRI24_SkullS_BiasCorrect.nii.gz'
    T2: '{patient}_T2_SRI24_SkullS_BiasCorrect.nii.gz'
    T1: '{patient}_T1_SRI24_SkullS_BiasCorrect.nii.gz'
```

Note: The order and names of the modalities must be the same for all datasets! This order must also be preserved if you run inference using this code base and config file: https://github.com/Theta-Tech-AI/brain-inference/blob/production/config/config.yaml#L3

The `folds` parameter determines how many folds you want to create. If `folds` is 5, for instance, then each patient will be assigned to either fold 0, 1, 2, 3, or 4. It should be noted that the different datasets will not be assigned different folds. If you want to play with the folds (e.g. make one dataset entirely for validation), you can manually edit the generated `data/patients/*.json` files with a text editor and assign the folds yourself.

Once your configuration is prepared, run the preparation script:
```bash
python3 data/prepare_data.py
```
(or `python` instead of `python3` depending on your computer setup). This script will generate a `data/patients` directory with a `*.json` file for each dataset. You can open any of these files in a text editor and examine them yourself.

In addition, it will create a `data/patients/summary.yaml` file that contains statistics on the data found for each dataset, and specific patients with missing data.

## Segmentation Training
To run the training, edit the `segmentation/config.yaml` file to set the model, dataset, and other parameters. While not exhaustive, some useful config parameters are:
* `val_fold`: Which fold will be used for validation.
* `workers`: How many workers to use for data loading. With too large a number of workers, the process may crash, and with too few, the process will be slow.
* `batch_size` and `sw_batch_size` ("sliding window" batch size): The batch size for training. Aim to use as high a batch size as you can until you run out of GPU memory.
* `max_epochs`: The maximum number of epochs to train for.
* `val_check_interval`: How often to run validation. If you check too often, the training will be slow.
* `train_resize` and `val_resize`: The volumes will be resized to these dimensions before training and validation.
* `roi` and `feature_size`: These, among other architectural hyperparameters, control the region of interest the model sees and how large the model is. You may want to explore different values for your specific use case, as a small value may train faster without losing accuracy, and yet a large value may be needed for a more global understanding of the tumor boundaries.
* `wandb_enabled`: Whether to log to Weights & Biases. If this is `True`, you will need to have an account at https://wandb.ai and it will ask you to login the first time you run the script.
* `project_name`: The name of the project in Weights & Biases to log training runs.
* `num_partitions`: To make things run faster, every `num_partitions` epochs will use the same random image augmentations. This means that if the value is 3, for instance then the first 3 epochs will create and cache random augmentations of the data, but the subsequent epochs will simply use the cached data and may run faster, potentially at the cost of generalizability.

Please refer to the config file for more details on other parameters.

Once you have set up your configuration, run the training script, which will utilize up to 8 GPU's by default:
```bash
python3 segmetation/main.py
```
You can view the results in your weights and biases dashboard under the project name.
