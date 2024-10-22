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
from lightning.pytorch import Trainer, seed_everything
import signal

os.environ["SLURM_JOB_NAME"] = "interactive"

@gin.configurable
class MethylSeqDataModule(LightningDataModule):
    def __init__(self, train_dataset_file, validation_dataset_file, batch_size=32, transforms=[]):
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.validation_dataset_file = validation_dataset_file
        self.batch_size = batch_size
        self.transforms = transforms

    def setup(self, stage=None):
        self.train_dataset = CustomH5Dataset(
            self.train_dataset_file,
            batch_size=self.batch_size,
            transforms=self.transforms,
        )
        self.val_dataset = CustomH5Dataset(
            self.validation_dataset_file,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, shuffle=False, num_workers=3)

# def print_random_seed_and_trainer_info(trainer, model):
#     # Print the random seed being used (if set)
#     seed = torch.initial_seed()
#     print(f"PyTorch random seed: {seed}")

#     # # Get Lightning's random seed (after Trainer.seed_everything() call)
#     # print(f"Lightning random seed: {trainer.global_seed}")

#     # # Print whether the Trainer is set to deterministic mode
#     # print(f"Deterministic mode: {trainer.deterministic}")

#     # # Check model's trainer for random seed initialization
#     # print(f"Trainer's `deterministic`: {trainer._deterministic}")
    
#     # Other randomness factors (like CUDA deterministic)
#     if torch.backends.cudnn.deterministic:
#         print("CUDNN is set to deterministic.")
#     else:
#         print("CUDNN is not set to deterministic.")
    
#     if torch.backends.cudnn.benchmark:
#         print("CUDNN benchmark is enabled.")

def main(config,output_dir,unique_identifier,gpus,batch_size):
    
    gin.parse_config_file(config)
    
    model = MethylSeqNN()
    
    from methylseqnet.trainer import MethylSeqDataModule
    if batch_size>0:
        # if the script was provided with a batch_size
        data_module = MethylSeqDataModule(batch_size=batch_size)
    else:
        data_module = MethylSeqDataModule()

    # Try to retrieve the io_mappings string from the train dataset, silently skipping if missing
    # The value here lies in the fact that the task structure is dynamically created from the
    # preprocessor matches file, so having a record of what the mappings is for a given model
    # may be useful when trying different datasets, etc
    try:
        dataset = CustomH5Dataset(data_module.train_dataset_file)
        model.io_mappings_str = dataset.get_io_mappings_str()
    except AttributeError:
        print(f"No 'io_mappings' attribute found in {data_module.train_dataset_file}.")
    
    model_dir = Path(output_dir)/unique_identifier
    temp_checkpoint_path = model_dir/'checkpoints'/'temp-checkpoint.ckpt'

    logger = WandbLogger(
        save_dir=model_dir,
        name=f"{Path(config).stem}_{unique_identifier}",  # Set your descriptive experiment name
        version=unique_identifier,
    )
    # Temporary checkpoint written every epoch
    temp_checkpoint = ModelCheckpoint(
        dirpath=model_dir/'checkpoints',
        filename='temp-checkpoint',           # Fixed name for overwriting
        save_top_k=1,                         # Keep only the latest checkpoint
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
    
    trainer = Trainer(
        callbacks = [temp_checkpoint,best_val_checkpoint],
        default_root_dir=model_dir,
        logger=logger,
        accelerator='auto', 
        devices=gpus, 
        max_epochs=100
    )    

    # print_random_seed_and_trainer_info(trainer, model)
    
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=temp_checkpoint_path 
            if os.path.isfile(temp_checkpoint_path) 
            else None
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MethylSeqNN model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the gin config file.')
    parser.add_argument('--output_dir', type=str, required=False, default='/clusterfs/nilah/oberon/lightning/', help='Directory to store outputs.')
    parser.add_argument('--unique_identifier', type=str, required=False, default=dt.now().strftime('%Y-%m-%d_%H-%M-%S'), help='Unique identifier for run.')
    parser.add_argument('--gpus', type=str, required=False, default='auto', help='GPU count for parallelization.')
    parser.add_argument('--batch_size', type=int, required=False, default=-1, help='Batch size for dataloader.')
    args = parser.parse_args()
    main(args.config,args.output_dir,args.unique_identifier,args.gpus,args.batch_size)