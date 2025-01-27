import os
import logging

import torch
torch.set_float32_matmul_precision("medium") 


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_loader import DataModule
from trainer import MusicClassifier
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.strategies import DDPStrategy

#from utilities.custom_early_stopping import MultiMetricEarlyStopping
def get_latest_version(log_dir):
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by version number
    return version_dirs[-1] if version_dirs else None

log = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path="config", config_name="train_config")
def main(config: DictConfig):

    log_base_dir = 'tb_logs/train_audio_classification'
    # log_base_dir = to_absolute_path('tb_logs/train_audio_classification')
    is_mt = False
    if "mt" in config.model.classifier:
        is_mt = True

    logger = TensorBoardLogger("tb_logs", name="train_audio_classification")
    logger.log_hyperparams(config)
    train_log_dir = logger.log_dir
    print(f"Logging to {train_log_dir}")
    log.info("Training starts")

    data_module = DataModule( config )
    data_module.setup()

    # Get the list of dataloaders for both train and validation, with dataset names
    trainloaders = {dataset_name: loader for dataset_name, loader in zip(config.datasets, data_module.train_dataloader())}
    vallowers = {dataset_name: loader for dataset_name, loader in zip(config.datasets, data_module.val_dataloader())}
    
    # Combine multiple loaders using CombinedLoader, now with dataset names
    combined_train_loader = CombinedLoader(trainloaders, mode="max_size")
    combined_val_loader = CombinedLoader(vallowers, mode="max_size")

    latest_version = get_latest_version(log_base_dir)
    next_version = int(latest_version.split('_')[-1]) + 1 if latest_version else 0
    next_version = f"version_{next_version}"

    val_epoch_file = os.path.join(log_base_dir,  latest_version, 'val_epoch.txt')

    model = MusicClassifier( config, output_file = val_epoch_file)

    if is_mt:
        checkpoint_callback_mood = ModelCheckpoint(**config.checkpoint_mood)
        checkpoint_callback_va = ModelCheckpoint(**config.checkpoint_va)
        early_stop_callback = EarlyStopping(**config.earlystopping)

        if config.model.kd == True:
            trainer = pl.Trainer(
                **config.trainer,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[checkpoint_callback_mood, checkpoint_callback_va, early_stop_callback],
                logger=logger,
                num_sanity_val_steps=0
            )
        else:
            trainer = pl.Trainer(
                **config.trainer,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback_mood, checkpoint_callback_va, early_stop_callback],
                logger=logger,
                num_sanity_val_steps=0
            )

    else:
        checkpoint_callback = ModelCheckpoint(**config.checkpoint)
        # early_stop_callback = EarlyStopping(**config.earlystopping)
        trainer = pl.Trainer(
            **config.trainer,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            num_sanity_val_steps = 0
        )
    
    trainer.fit(model, combined_train_loader, combined_val_loader)

    if trainer.global_rank == 0:
        best_checkpoint_file = os.path.join(train_log_dir, 'best_checkpoint.txt')
        with open(best_checkpoint_file, 'w') as f:
            if is_mt:
                f.write(f"Best checkpoint (mood): {checkpoint_callback_mood.best_model_path}\n")
                f.write(f"Best checkpoint (va): {checkpoint_callback_va.best_model_path}\n")
            else:
                f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
            f.write(f"Version: {logger.version}\n")

if __name__ == '__main__':
    main()
