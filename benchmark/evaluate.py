import nibabel as nib
import numpy as np
from monai.metrics import DiceMetric
import torch

folder_path = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Tasks/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
ground_truth = f"{folder_path}/100010/100010_time1_seg.nii.gz"

# Load prediction and ground truth (should be same shape)
pred = nib.load("segmentation.nii.gz").get_fdata() # prediction
gt = nib.load(ground_truth).get_fdata() # groundtruth

pred = np.round(pred).astype(np.uint8)
print("Unique predicted labels:", np.unique(pred))
print("Unique ground truth labes:", np.unique(gt))

# Convert to torch tensors
pred_tensor = torch.tensor(pred[None, None, ...], dtype=torch.float32)  # shape (1, 1, H, W, D)
gt_tensor = torch.tensor(gt[None, None, ...], dtype=torch.float32)

# Initialize metric
dice = DiceMetric(include_background=True, reduction="mean")
score = dice(pred_tensor, gt_tensor)
print("Dice:", score.item())
