# brain-inference

This is a script for running inference on the trained SwinUNETR model. 

See this Loom video for instructions:
https://www.loom.com/share/df695340dac248f68c01c8e23591de5e

## Requirements

- To set up the required environment, run the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

- For inference on a single scan, place all four of the **\*.nii.gz** modalities within **data/**.

## Configuration

- Open **config/config.yaml** and modify the single scan names to match the **\*.nii.gz** files placed in **data/**. 

- Open **config/example_config.yaml** and **data/multi_example.json** to view examples on how to handle multiple scans and/or compute DICE scores if a ground-truth is provided. 

## Running Inference

To run inference, open **inference.ipynb** and run all cells. The output segmentation for single scans is written to **out/segmentation_mask.nii.gz** by default. 
