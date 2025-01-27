import os
import logging

import torch
torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader import DataModule
from trainer import MusicClassifier
import yaml
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.utilities.combined_loader import CombinedLoader


log = logging.getLogger(__name__)

def get_latest_version(log_dir):
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by version number
    return version_dirs[-1] if version_dirs else None

def save_metrics_and_checkpoint(metrics, checkpoint, output_file):
    data = {
        'checkpoint': checkpoint,
        'metrics': metrics
    }
    with open(output_file, 'w') as f:
        yaml.dump(data, f)

def read_best_checkpoint_info(file_path, dataset_type=None):
    """Read the best checkpoint file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint info file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if dataset_type == "mood":
        checkpoint_line = next((line for line in lines if line.startswith("Best checkpoint (mood):")), None)
    elif dataset_type == "va":
        checkpoint_line = next((line for line in lines if line.startswith("Best checkpoint (va):")), None)
    else:
        checkpoint_line = next((line for line in lines if line.startswith("Best checkpoint:")), None)

    if not checkpoint_line:
        raise ValueError(f"No checkpoint found for dataset type '{dataset_type}' in the file.")
    
    return checkpoint_line.split(": ")[-1].strip()

@hydra.main(version_base=None, config_path="config", config_name="test_config")
def main(config: DictConfig):
    log.info("Testing starts")
    log_base_dir = 'tb_logs/train_audio_classification'
    # log_base_dir = to_absolute_path('tb_logs/train_audio_classification')

    latest_version = get_latest_version(log_base_dir)
    if not latest_version:
        raise FileNotFoundError("No version directories found in log base directory.")
    version_log_dir = os.path.join(log_base_dir, latest_version)
    output_file = os.path.join(version_log_dir, 'test_metrics.txt')

    if config.checkpoint_latest:
        if config.multitask:
            dataset_type = config.dataset_type  # Expecting 'mood' or 'va'
            best_checkpoint_file = os.path.join(version_log_dir, 'best_checkpoint.txt')
            ckpt = read_best_checkpoint_info(best_checkpoint_file, dataset_type)
        else:
            best_checkpoint_file = os.path.join(version_log_dir, 'best_checkpoint.txt')
            ckpt = read_best_checkpoint_info(best_checkpoint_file)
    else:
        ckpt = config.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    
    log.info(f"Using checkpoint: {ckpt}")
    data_module = DataModule( config )
    data_module.setup()

    testloaders = {dataset_name: loader for dataset_name, loader in zip(config.datasets, data_module.test_dataloader())}
    combined_test_loader = CombinedLoader(testloaders, mode="max_size")

    model = MusicClassifier.load_from_checkpoint(ckpt, cfg=config, output_file=output_file)
    logger = TensorBoardLogger(save_dir=log_base_dir,  
                                name="",  
                                version=latest_version)
    trainer = pl.Trainer(**config.trainer,
                        logger=logger)
    
    
    trainer.test(model, combined_test_loader)

if __name__ == '__main__':
    main()
