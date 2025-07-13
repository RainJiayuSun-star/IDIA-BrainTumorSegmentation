from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms

segmenter = AdultGliomaPostTreatmentSegmenter(algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm of 2024 will be used on cuda:0
folder_path = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Tasks/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
case_num = 100001

def segInfer(case_num):
    segmenter.infer_single(
        t1c=f"{folder_path}/{case_num}/{case_num}_time1_t1ce.nii.gz", # t1c = t1ce [UCSF]
        t1n=f"{folder_path}/{case_num}/{case_num}_time1_t1.nii.gz", # t1n = t1
        t2f=f"{folder_path}/{case_num}/{case_num}_time1_flair.nii.gz", # t2f = t2 flair
        t2w=f"{folder_path}/{case_num}/{case_num}_time1_t2.nii.gz", # t2w = t2
        output_file=f"{case_num}_brats_inf_seg.nii.gz",
)
    
for i in range(0,9):
    print("Processing: ", case_num)
    segInfer(case_num)
    print("Successful inference: ", case_num)
    case_num += 1