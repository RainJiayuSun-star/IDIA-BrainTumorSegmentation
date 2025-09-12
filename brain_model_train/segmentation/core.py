
print('🧰 Importing monai...')
from monai import transforms
from monai import data
from monai.config import print_config

from submodules.brain_inference import datafold_read, UnionOfLabelsd, GetDataLoader, get_loader_val, CustomSwinUNETR

print('Printing MONAI config...')
print_config()


def get_loader_train(train_resize:list[int]|None, roi:list[int], union:bool, **kwargs):
    """
    A function to contain code unique to training loading.
    
    Args:
        train_resize: Spatially scale the data to this resolution if not None. 
        roi: Randomly crop the data to this size. Performed after train_resize. 
        union: Whether to additionally consider the union of labels. 
        kwargs: Additional arguments sent to GetDataLoader().
        
    Returns:
        A DataLoader object constructed from the input arguments. 
        
    """
    train_transform_list = [
        transforms.LoadImaged(keys=['image', 'label']),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    ]
    if union: train_transform_list.append(UnionOfLabelsd(keys='label'))
    if isinstance(train_resize, list):
        train_transform_list.append(
            transforms.Resized(keys=['image', 'label'], mode='linear', spatial_size=tuple(train_resize))
        )
    train_transform_list.extend([
        transforms.CropForegroundd(
            keys=['image', 'label'],
            source_key='image',
            k_divisible=[roi[0], roi[1], roi[2]],
        ),
        transforms.RandSpatialCropd(
            keys=['image', 'label'],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
    ])
    train_transform = transforms.Compose(train_transform_list)

    return GetDataLoader(transform=train_transform, **kwargs)


def get_loader(batch_size, val_batch_size, json_list, val_fold, roi, train_resize, val_resize, union: bool, workers, 
               val_workers, cache_dir, dataset_type, dataset_partition):
    """
    A function that retrieves both the training and validation loaders after reading the JSON data lists. 
    
    Args:
        batch_size: The batch size for the training loader.
        val_batch_size: The batch size for the validation loader. 
        data_dir: Base directory for locating data. 
        json_list: Path to a JSON file containing data file paths relative to data_dir. 
        fold: Determines which data fold should be assigned to validation.
        roi: The size of spatial crops performed in get_loader_train. 
        train_resize: Size of spatial scaling applied to training data.
        val_resize: Size of spatial scaling applied to validation data.
        workers: The number of workers for the training loader.
        val_workers: The number of workers for the validation loader.
        cache_dir: Path to where both loaders should store their caches. 
        dataset_type: The training dataset type. See GetDataLoader() documentation.
        dataset_partition: The N value for the training dataset. See GetDataLoader() documentation.
        
    Returns:
        A tuple of the training loader, validation loader, and the custom sampler used for training 
        The custom sampler is None if PartitionedPersistentDataset was not used.
         
    """
    
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, fold=val_fold, basedir="")
    train_loader, custom_sampler = get_loader_train(
        batch_size=batch_size, 
        files=train_files, 
        roi=roi, 
        union=union, 
        workers=workers, 
        cache_dir=cache_dir,
        dataset_type=dataset_type, 
        dataset_partition=dataset_partition, 
        train_resize=train_resize,
    )
    val_loader = get_loader_val(
        batch_size=val_batch_size, 
        files=validation_files, 
        val_resize=val_resize, 
        union=union, 
        workers=val_workers, 
        cache_dir=cache_dir,
    )
    return train_loader, val_loader, custom_sampler


def get_roi(cfg):
    return (int(cfg.hyperparameter.roi.h), int(cfg.hyperparameter.roi.w), int(cfg.hyperparameter.roi.d))


def create_model(cfg, num_modalities: int = 4):
    return CustomSwinUNETR(
        in_channels       = num_modalities, # one per MRI modality: T1, T2, T1-Contrast, FLAIR
        out_channels      = 4 if cfg.data.label_union else 3, # one per label: tumor core, whole tumor, enhancing tumor
        img_size          = get_roi(cfg), # e.g. 128x128x128
        feature_size      = cfg.hyperparameter.feature_size, # e.g. 48
        drop_rate         = cfg.hyperparameter.drop_rate, # e.g. 0.0
        attn_drop_rate    = cfg.hyperparameter.attn_drop_rate, # e.g. 0.0
        dropout_path_rate = cfg.hyperparameter.dropout_path_rate, # e.g. 0.0
        use_checkpoint    = True, 
        depths            = cfg.hyperparameter.depths,
        num_heads         = cfg.hyperparameter.num_heads,
        norm_name         = cfg.hyperparameter.norm_name,
        normalize         = cfg.hyperparameter.normalize,
        downsample        = cfg.hyperparameter.downsample,
        use_v2            = cfg.hyperparameter.use_v2,
        mlp_ratio         = cfg.hyperparameter.mlp_ratio,
        qkv_bias          = cfg.hyperparameter.qkv_bias,
        patch_size        = cfg.hyperparameter.patch_size,
        window_size       = cfg.hyperparameter.window_size,
    )