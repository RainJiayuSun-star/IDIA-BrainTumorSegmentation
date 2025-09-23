import os
import nibabel as nib
import numpy as np

# input directory - segmentation result ran by inference
input_dir = "../brain_model_train/outputs/output_0922_time1"
output_dir = os.path.join(input_dir, "remapped")
os.makedirs(output_dir, exist_ok=True)

label_map = {
    0: 0,  # background stays background
    1: 1,  
    2: 2,  
    3: 4,  
}

def remap_segmentation(input_path, output_path, mapping):
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.uint8)

    # Apply mapping
    new_data = np.zeros_like(data, dtype=np.uint8)
    for old_label, new_label in mapping.items():
        new_data[data == old_label] = new_label

    # Save new file
    new_img = nib.Nifti1Image(new_data, affine=img.affine, header=img.header)
    nib.save(new_img, output_path)
    print(f"💾 Saved remapped file: {output_path}")

# 🔁 Process all seg.nii.gz files in the directory
for fname in os.listdir(input_dir):
    if fname.endswith("_seg.nii.gz"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace("_seg", "_seg_remap"))
        remap_segmentation(in_path, out_path, label_map)
