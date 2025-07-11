from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms

segmenter = AdultGliomaPostTreatmentSegmenter(algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_1, cuda_devices="1")
# these parameters are optional, by default the winning algorithm of 2024 will be used on cuda:0
folder_path = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Tasks/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
segmenter.infer_single(
    t1c=f"{folder_path}/100010/100010_time1_t1ce.nii.gz", # t1c = t1ce [UCSF]
    t1n=f"{folder_path}/100010/100010_time1_t1.nii.gz", # t1n = t1
    t2f=f"{folder_path}/100010/100010_time2_flair.nii.gz", # t2f = t2 flair
    t2w=f"{folder_path}/100010/100010_time2_t2.nii.gz", # t2w = t2
    output_file="segmentation.nii.gz",
)