folder_path01 = "./all_data/UCSF-POST/Preprocessed"
folder_path02 = "./all_data/UCSF-POST/Preprocessed"
segmenter.infer_single(
    t1c=f"{folder_path01}/100010/time01/100010_time1_t1ce.nii.gz", 
    t1n=f"{folder_path01}/100010/time01/100010_time1_t1.nii.gz", 
    t2f=f"{folder_path02}/100010/time02/100010_time2_flair.nii.gz", 
    t2w=f"{folder_path02}/100010/time02/100010_time2_t2.nii.gz", 
    output_file="segmentation.nii.gz",
)
