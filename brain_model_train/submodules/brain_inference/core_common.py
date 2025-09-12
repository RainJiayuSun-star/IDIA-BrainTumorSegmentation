import os
import json
import numpy as np

from torch import any, cat
import torch
from torch.utils.data import ConcatDataset, Sampler

from monai import transforms
from monai import data


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    print(f"Training set size: {len(tr)}")
    print(f"Validation set size: {len(val)}")
    return tr, val


class UnionOfLabelsd(transforms.Transform):
    """
    A transform that appends the union of segmentation labels to the input data. 
    
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        label = d[self.keys]
        union_of_labels = any(label > 0, dim=0, keepdim=True).bool()
        d['label'] = cat([label, union_of_labels], dim=0)
        return d
    

class RandomPersistentDataset(data.PersistentDataset):
    """
    Modified dataset that caches random trasforms in addition to deterministic ones.
    
    """
    def _pre_transform(self, item_transformed):
        item_transformed = self.transform(item_transformed, threading=True)
        if self.reset_ops_id:
            transforms.reset_ops_id(item_transformed)
        return item_transformed
    
    def _post_transform(self, item_transformed):
        return item_transformed
            

class PartitionedSampler(Sampler):
    """
    Custom sampler that caches N differently seeded random transforms. Uses the (epoch % N)th cache during training. 
    Used when PartitionedPersistentDataset is selected. 
    
    """
    def __init__(self, data_source, dataset_partition, sample_length):
        self.data_source = data_source
        self.sample_length = sample_length
        self.dataset_partition = dataset_partition
        self.selected_cache = 0
        
    def update_epoch(self, epoch):
        self.selected_cache = (epoch + 1) % self.dataset_partition

    def __iter__(self):
        indices = list(range(
            self.sample_length * self.selected_cache, 
            self.sample_length * (self.selected_cache + 1)
        ))
        return iter(indices)

    def __len__(self):
        return self.sample_length
            

def GetDataLoader(files, transform, cache_dir:str, dataset_type:str="RandomPersistentDataset", dataset_partition:int=1, batch_size:int=1, workers:int=1, 
                  cache_name:str="cache_default", dataloader_type:str="ThreadDataLoader", shuffle:bool=True):
    """
    A function to contain common code between validation loading and training loading. 
    
    Args:
        files: Training/validation files loaded from datafold_read. 
        transform: A transformation to apply to the input files.
        cache_dir: Where to store transformed data if using a cached dataset. 
        dataset_type: Which type of dataset to use:
            Dataset:                      Default dataset.
            PersistentDataset:            Only cache deterministic transforms. Generally inefficient.
            RandomPersistentDataset:      Also cache random transforms. Does not generalize as well, but very efficient.
            PartitionedPersistentDataset: Cache N (num_partitions) differently seeded random transforms. Epoch i will use cache i % N. 
        dataset_partition: The value of N if using PartitionedPersistentDataset.
        batch_size: The data size to use when loading data. 
        workers: The number of processes to run that load data in parallel. 
        cache_name: The name of the default cache so that it is distinguished from the partitioned caches. 
        dataloader_type: Which type of dataloader to use:
            DataLoader:                   Default dataloader.
            ThreadDataLoader:             Iteration is performed async. 
        shuffle: Whether to shuffle the data when iterating over the DataLoader. PartitionedPersistentDataset will not be shuffled. 
        
    Returns:
        A tuple of a DataLoader object constructed from the input arguments, and the custom sampler if PartitionedPersistentDataset was selected. 
        
    """
    dataloader_kwargs = {}
    custom_sampler = None

    match dataset_type:
        case "Dataset":
            ds = data.Dataset(
                data=files,
                transform=transform,
            )
        case "PersistentDataset":
            ds = data.PersistentDataset(
                data=files, 
                transform=transform, 
                cache_dir=os.path.join(cache_dir, cache_name),
            )
            
        case "RandomPersistentDataset":
            ds = RandomPersistentDataset(
                data=files, 
                transform=transform, 
                cache_dir=os.path.join(cache_dir, cache_name),
            )
            
        case "PartitionedPersistentDataset":
            if dataset_partition < 1:
                raise ValueError("dataset_partition should be a positive integer.")
            ds = [
                RandomPersistentDataset(
                   data=files, 
                    transform=transform, 
                    cache_dir=os.path.join(cache_dir, "cache" + str(i)),  
                ) for i in range(dataset_partition)
            ]
            sample_length = len(ds[0])
            ds = ConcatDataset(ds)
            custom_sampler = PartitionedSampler(ds, dataset_partition, sample_length)
            dataloader_kwargs["sampler"] = custom_sampler
            shuffle = False
    
    match dataloader_type:
        case "ThreadDataLoader":
            return data.ThreadDataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=True,
                collate_fn=data.utils.pad_list_data_collate,
                use_thread_workers=True,
                persistent_workers=True,
                **dataloader_kwargs,
            ), custom_sampler
        
        case "DataLoader":
            return data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=True,
                collate_fn=data.utils.pad_list_data_collate,
                **dataloader_kwargs,
            ), custom_sampler


def get_loader_val(val_resize:list[int], union:bool=True, add_label:bool=True, **kwargs):
    """
    A function to contain code unique to validation loading or inference loading.
    
    Args:
        val_resize: Spatially scale the data to this resolution. 
        union: Whether to additionally consider the union of labels. 
        add_label: Whether a label/ground truth needs to be considered. 
                   Should be set to False if inferring on data without an existing ground truth segmentation. 
        kwargs: Additional arguments sent to GetDataLoader().
        
    Returns:
        A DataLoader object constructed from the input arguments. 
        
    """
    keys = ['image', 'label'] if add_label else ['image']
    val_transform_list = [
        transforms.LoadImaged(keys=keys),
    ]
    if add_label:
        val_transform_list.append(transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'))
        if union: val_transform_list.append(UnionOfLabelsd(keys='label'))
    val_transform_list.extend([
        transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        transforms.Resized(keys=keys, mode='linear', spatial_size=val_resize),
    ])
    val_transform = transforms.Compose(val_transform_list)

    loader, _  = GetDataLoader(
        transform=val_transform, 
        dataset_partition=1,
        dataloader_type="DataLoader",
        shuffle=False,
        cache_name="cache_val",
        **kwargs,
    )
    return loader

