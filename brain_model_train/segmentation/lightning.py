import torch
import pytorch_lightning as pl
from functools import partial
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (AsDiscrete, Activations)
from monai.utils.enums import MetricReduction
import wandb
import hydra
from .core import *

import numpy as np


class BrainSegmentationModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super(BrainSegmentationModule, self).__init__()
        self.cfg = cfg
        self.model = model
        self.roi = (int(self.cfg.hyperparameter.roi.h), int(self.cfg.hyperparameter.roi.w), int(cfg.hyperparameter.roi.d))
        self.loss_func = hydra.utils.instantiate(cfg.hyperparameter.loss)
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size      = self.roi,
            sw_batch_size = self.cfg.training.val_batch_size,
            predictor     = self.model,
            overlap       = self.cfg.hyperparameter.infer_overlap,
        )
        self.accuracy = DiceMetric(
            include_background = True,
            reduction          = MetricReduction.MEAN_BATCH,
        )
        self.partitioned_sampler = None
        
    def set_partitioned_sampler(self, partitioned_sampler):
        self.partitioned_sampler = partitioned_sampler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        # Call back to the custom sampler and update the epoch if PartitionedPersistentDataset was selected. 
        if self.partitioned_sampler is not None:
            self.partitioned_sampler.update_epoch(self.current_epoch)
            
        images = batch['image']
        target = batch['label']

        logits = self.model(images)
        loss = self.loss_func(logits, target)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        images = batch['image']
        target = batch['label']
        
        logits = self.model_inferer(images)
        post_sigmoid = Activations(sigmoid=True)
        post_pred = AsDiscrete(argmax=False, threshold=0.5)
        val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in logits]
        
        acc_func = self.accuracy
        acc_func.reset()
        acc_func(y_pred=val_output_convert, y=target)
        accuracy = acc_func.aggregate().cpu().numpy()
        
        dice_tc = accuracy[0]
        dice_wt = accuracy[1]
        dice_et = accuracy[2]
        dice_mean = (dice_tc + dice_wt + dice_et) / 3
        self.log('val/dice_mean', dice_mean, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/dice_tc'  , dice_tc  , sync_dist=True, on_step=True, on_epoch=True)
        self.log('val/dice_wt'  , dice_wt  , sync_dist=True, on_step=True, on_epoch=True)
        self.log('val/dice_et'  , dice_et  , sync_dist=True, on_step=True, on_epoch=True)
        if len(accuracy) == 4:
            dice_union = accuracy[3]
            self.log('val/dice_union', dice_union, sync_dist=True, on_step=True, on_epoch=True)
            
        class_labels = {1: 'tumor core', 2: 'whole tumor', 3: 'enhancing tumor', 4: 'union'}
        labels_order = ['whole tumor', 'tumor core', 'enhancing tumor', 'union']
        label_indices = [list(class_labels.keys())[list(class_labels.values()).index(label)] - 1 for label in labels_order]
        label_indices = label_indices[:len(accuracy)]
        def get_random_segmentation_image():
            image_index    = np.random.randint(0, target.shape[0])
            slice_index    = np.random.randint(0, target.shape[4])
            modality_index = np.random.randint(0, images.shape[1])

            mask_shape = target.shape[2:4]  # Assuming target shape is [batch_size, num_labels, H, W, D]
            def prepare_mask(label_tensor):
                mask_separate = np.zeros(mask_shape)
                mask_union    = np.zeros(mask_shape)
                for label_index in label_indices:
                    label_tensor_squeezed = label_tensor[label_index, :, :, slice_index].cpu().numpy().squeeze()
                    mask = mask_separate if label_index < 3 else mask_union
                    mask[label_tensor_squeezed > 0] = label_index + 1
                return np.fliplr(mask_separate), np.flipud(mask_union)

            # Squeeze out the first dimension (image_index) from target tensor before passing
            mask_gt_separate, mask_gt_union = prepare_mask(target[image_index].squeeze(0))
            if mask_gt_separate.max() == 0:
                return

            mask_pred_separate, mask_pred_union = prepare_mask(val_output_convert[image_index])

            img_array = images[image_index, modality_index, :, :, slice_index].cpu().numpy().astype(np.float32).squeeze()
            img_array -= img_array.min()  # Make the minimum zero
            img_array /= img_array.max()  # Rescale to make the max one
            img_array = np.fliplr(img_array)  # Vertically flip the image
            
            masks = {
                'Ground Truth (subregions)': {'mask_data': mask_gt_separate.T, 'class_labels': class_labels},
                'Predictions (subregions)': {'mask_data': mask_pred_separate.T, 'class_labels': class_labels},
            }
            if len(label_indices) == 4:
                masks['Ground Truth (union)'] = {'mask_data': mask_gt_union.T, 'class_labels': class_labels}
                masks['Predictions (union)'] = {'mask_data': mask_pred_union.T, 'class_labels': class_labels}
            return wandb.Image(img_array.T, masks=masks)

        # Log the image and masks to W&B
        segmentation_masks = []
        while len(segmentation_masks) < self.cfg.training.save_images_per_batch:
            mask = get_random_segmentation_image()
            if mask is not None:
                segmentation_masks.append(mask)
        self.logger.experiment.log({'segmentation_masks': segmentation_masks})
    
    def configure_optimizers(self):
        # return optimizers and learning rate schedulers
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs
        )
        return [optimizer], [scheduler]