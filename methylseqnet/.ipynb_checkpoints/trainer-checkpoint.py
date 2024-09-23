import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score
import numpy as np
import json
import gin
from datetime import datetime as dt
from pathlib import Path
import argparse
from methylseqnet.dataset import CustomH5Dataset
from methylseqnet.methylseqnn import MethylSeqNN
from collections import defaultdict
import pynvml
from lightning.pytorch import LightningDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import signal

os.environ["SLURM_JOB_NAME"] = "interactive"

@gin.configurable
class MethylSeqDataModule(LightningDataModule):
    def __init__(self, train_dataset_file, validation_dataset_file, batch_size=32):
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.validation_dataset_file = validation_dataset_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomH5Dataset(self.train_dataset_file,batch_size=self.batch_size)
        self.val_dataset = CustomH5Dataset(self.validation_dataset_file,batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, shuffle=False, num_workers=3)

def main(config,output_dir,unique_identifier,gpus):
    gin.parse_config_file(config)
    model = MethylSeqNN()
    from methylseqnet.trainer import MethylSeqDataModule
    data_module = MethylSeqDataModule()
    model_dir = Path(output_dir)/unique_identifier
    logger = WandbLogger(
        save_dir=model_dir,
        name=f"{Path(config).stem}_{unique_identifier}",  # Set your descriptive experiment name
        version=unique_identifier,
    )
    # Temporary checkpoint that overwrites every 1000 steps
    temp_checkpoint = ModelCheckpoint(
        dirpath=model_dir/'checkpoints',
        filename='temp-checkpoint',           # Fixed name for overwriting
        save_top_k=1,                         # Keep only the latest checkpoint
        every_n_train_steps=1000,             # Save every 1000 steps
        save_on_train_epoch_end=True, 
    )
    
    # Best validation checkpoint, tracked separately
    best_val_checkpoint = ModelCheckpoint(
        dirpath=model_dir/'checkpoints',
        monitor='val_loss',                   # Metric to track for "best" checkpoint
        mode='min',                           # Minimize validation loss (or 'max' if you're maximizing a metric)
        save_top_k=1,                         # Save the best checkpoint only
        filename='best-checkpoint',           # Name for the best checkpoint
        save_last=False                       # Don't save a 'last' checkpoint
    )

    temp_checkpoint_path = model_dir/'checkpoints'/'temp-checkpoint.ckpt'
    
    if os.path.isfile(temp_checkpoint_path):
        trainer=Trainer(
            resume_from_checkpoint=autorestart_path,
            callbacks = [temp_checkpoint,best_val_checkpoint],
            default_root_dir=model_dir,
            logger=logger,
            accelerator='auto', 
            devices=gpus, 
            max_epochs=100
        )
    else:
        trainer = Trainer(
            callbacks = [temp_checkpoint,best_val_checkpoint],
            default_root_dir=model_dir,
            logger=logger,
            accelerator='auto', 
            devices=gpus, 
            max_epochs=100
        )
    
    trainer.fit(model,datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MethylSeqNN model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the gin config file.')
    parser.add_argument('--output_dir', type=str, required=False, default='/clusterfs/nilah/oberon/lightning/', help='Directory to store outputs.')
    parser.add_argument('--unique_identifier', type=str, required=False, default=dt.now().strftime('%Y-%m-%d_%H-%M-%S'), help='Unique identifier for run.')
    parser.add_argument('--gpus', type=str, required=False, default='auto', help='GPU count for parallelization.')
    args = parser.parse_args()
    main(args.config,args.output_dir,args.unique_identifier,args.gpus)