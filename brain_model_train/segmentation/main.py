# Import and set up the custom logger first
import sys
import os

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from submodules.theta_utils.logger import setup_logger
import logging
setup_logger(log_level='DEBUG')
logger = logging.getLogger()
logger.info('🚀 Starting segmentation process...')
logger.debug(f'🐍 Python version: {sys.version}')
logger.debug(f'📁 Current working directory: {os.getcwd()}')

logger.info('🗃️ Importing...')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import json
from glob import glob
import tempfile
from omegaconf import DictConfig, OmegaConf
import yaml

logging.getLogger('nibabel').setLevel(logging.ERROR)

logger.info('🌊 Importing hydra...')
import hydra

logger.info('🔥 Importing torch...')
import torch
torch.set_float32_matmul_precision('medium') # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision

import wandb

logger.info('⚡ Importing lightning...')
from pytorch_lightning           import Trainer
from pytorch_lightning.loggers   import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

logger.info('📦 Importing local files...')
from segmentation.core      import *
from segmentation.lightning import *

os.environ['HYDRA_FULL_ERROR'] = '1'

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_data(cfg: DictConfig, datasets_config: dict, data_root: str, datalist_json_dir: str, extension: str):
    logger.info('🔄 Preparing data...')
    datalist = {}
    datalist_file_paths = glob(os.path.join(datalist_json_dir, f'*.{extension}'))
    for datalist_path in datalist_file_paths:
        logger.debug(f'📁 Processing datalist file: {datalist_path}')

        try:
            artifact = wandb.Artifact(os.path.basename(datalist_path), type='config')
            artifact.add_file(datalist_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.warning(f'❌ Error logging artifact: {e}')
        
        dataset_name = os.path.splitext(os.path.basename(datalist_path))[0]
        logger.debug(f'📁 {dataset_name = }')
        dataset_base_path = os.path.join(data_root, datasets_config['datasets'][dataset_name]['base_path'])
        logger.debug(f'📁 {dataset_base_path = }')
        
        with open(datalist_path) as json_fn:
            dataload = json.load(json_fn)
            for key in dataload.keys():
                if key not in datalist:
                    datalist[key] = []
                for item in dataload[key]:
                    # Convert relative paths to absolute paths
                    label_path = os.path.abspath(os.path.join(dataset_base_path, item['label']))
                    logging.debug(f'🔍 Checking label path: {label_path}')
                    assert os.path.exists(label_path), f'❌ Label file does not exist: {label_path}'
                    item['label'] = label_path

                    image_paths = []
                    for img in item['image']:
                        img_path = os.path.abspath(os.path.join(dataset_base_path, img))
                        logging.debug(f'🔍 Checking image path: {img_path}')
                        assert os.path.exists(img_path), f'❌ Image file does not exist: {img_path}'
                        image_paths.append(img_path)
                    item['image'] = image_paths

                    datalist[key].append(item)
                    logging.info(f'✅ Added item to datalist: {item}')
    
    logger.debug(f'📊 Total datalist entries: {sum(len(v) for v in datalist.values())}')
    return datalist

@hydra.main(config_path=THIS_SCRIPT_DIR, config_name='config')
def main(cfg: DictConfig):
    logger.info('🚀 Starting main function...')
    logger.info('🧠 Initializing Weights & Biases...')
    
    if not cfg.wandb_enabled:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb_logger = WandbLogger(
        project = cfg.project_name,
        log_model = 'all',
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    logger.debug('🔍 Determining data directories...')
    dataconfig_dir = os.path.join(os.path.abspath(os.path.dirname(THIS_SCRIPT_DIR)), 'data')
    logger.debug(f'📁 {dataconfig_dir = }')
    datalist_json_dir = os.path.join(dataconfig_dir, 'patients')
    logger.info(f'📁 {datalist_json_dir = }')
    datasets_yaml_path = os.path.join(dataconfig_dir, 'datasets.yaml')
    logger.info(f'📁 {datasets_yaml_path = }')
    
    logger.info('📂 Loading datasets configuration...')
    with open(datasets_yaml_path, 'r') as yaml_file:
        datasets_config = yaml.safe_load(yaml_file)
    data_root = os.path.abspath(os.path.expanduser(datasets_config['data_root']))
    logger.debug(f'📁 {data_root = }')

    extension = 'json'
    with tempfile.TemporaryDirectory() as cache_dir:
        with tempfile.NamedTemporaryFile(suffix='.' + extension) as temp_fn:
            datalist_json = temp_fn.name
            
            datalist = prepare_data(cfg, datasets_config, data_root, datalist_json_dir, extension)
            num_modalities = len(datalist['training'][0]['image'])
            logger.debug(f'{num_modalities = }')

            logger.info('🏗️ Creating model...')
            model = create_model(cfg, num_modalities)
            #wandb_logger.watch(model)

            with open(datalist_json, 'w') as json_fn:
                json.dump(datalist, json_fn, indent=4)
            
            logger.info('✅ Created aggregate JSON with absolute paths.')

            logger.info('🆕 Creating new BrainSegmentationModule...')
            lightning_module = BrainSegmentationModule(cfg, model)
            
            callbacks = [LearningRateMonitor(logging_interval='step')]
            if cfg.training.save_checkpoints:
                logger.info('📌 Setting up ModelCheckpoint callback...')
                callbacks.append(ModelCheckpoint(
                    monitor    = 'val/dice_mean_epoch',
                    dirpath    = wandb_logger.save_dir,
                    filename   = 'swinunetr-{epoch:02d}',
                    save_top_k = 1,
                    mode       = 'max',
                    save_last  = True,
                ))

            logger.info('🚂 Initializing Trainer...')
            trainer = Trainer(
                max_epochs              = cfg.training.max_epochs,
                accumulate_grad_batches = cfg.training.accumulate_grad_batches,
                limit_val_batches       = cfg.training.limit_val_batches,
                val_check_interval      = cfg.training.val_check_interval,
                check_val_every_n_epoch = cfg.training.check_val_every_n_epoch,
                logger                  = wandb_logger,
                log_every_n_steps       = 1,
                callbacks               = callbacks,
                num_sanity_val_steps    = cfg.training.num_sanity_val_steps,
            )
            
            logger.info('🔄 Preparing data loaders...')
            train_loader, val_loader, custom_sampler = get_loader(
                roi                 = get_roi(cfg),
                batch_size          = cfg.training.batch_size, 
                val_batch_size      = cfg.training.val_batch_size,
                val_fold            = cfg.training.val_fold,
                workers             = cfg.data.workers,
                val_workers         = cfg.data.val_workers,
                train_resize        = cfg.training.train_resize,
                val_resize          = cfg.training.val_resize,
                union               = cfg.data.label_union,
                cache_dir           = cache_dir,
                json_list           = datalist_json,
                dataset_type        = cfg.data.dataset.type,
                dataset_partition   = cfg.data.dataset.num_partitions,
            )
            
            lightning_module.set_partitioned_sampler(custom_sampler)
                            
            logger.info('🏋️ Starting model training...')
            trainer.fit(lightning_module, train_loader, val_loader)
            
            logger.info('🔍 Running model validation...')
            trainer.validate(lightning_module, dataloaders=val_loader)
        
    logger.info('✅ Main function completed successfully.')


if __name__ == '__main__':
    logger.info('👋 Hello, world.')
    main()