from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms
import docker
client = docker.from_env()

segmenter = AdultGliomaPostTreatmentSegmenter(algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_1, cuda_devices="1")
# these parameters are optional, by default the winning algorithm of 2024 will be used on cuda:0
folder_path01 = "./all_data/UCSF-POST/Preprocessed"
folder_path02 = "./all_data/UCSF-POST/Preprocessed"
segmenter.infer_single(
    t1c=f"{folder_path01}/100010/time01/100010_time1_t1ce.nii.gz", # t1c = t1ce [UCSF]
    t1n=f"{folder_path01}/100010/time01/100010_time1_t1.nii.gz", # t1n = t1
    t2f=f"{folder_path02}/100010/time02/100010_time2_flair.nii.gz", # t2f = t2 flair
    t2w=f"{folder_path02}/100010/time02/100010_time2_t2.nii.gz", # t2w = t2
    output_file="segmentation.nii.gz",
)
