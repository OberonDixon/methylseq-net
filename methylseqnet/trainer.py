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
from methylseqnet.dataset import *
from callbacks import GPUMemoryLogger
from methylseqnet.methylseqnn import MethylSeqNN
from collections import defaultdict
import pynvml
from lightning.pytorch import LightningDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
import signal
import pprint

os.environ["SLURM_JOB_NAME"] = "interactive"

@gin.configurable
class MethylSeqDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dataset_file=None, 
        validation_dataset_file=None, 
        predict_dataset_file=None,
        batch_size=32, 
        transforms=[], 
        dataset_class=MethylSeqDataset,
        pow=False, # temporarily restored for backwards compatibility; does nothing
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.validation_dataset_file = validation_dataset_file
        self.predict_dataset_file = predict_dataset_file
        self.batch_size = batch_size
        self.transforms = transforms
        self.dataset_class = dataset_class
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_dataset_file:
                self.train_dataset = self.dataset_class(
                    self.train_dataset_file,
                    transforms=self.transforms,
                    batch_size=None,
                )
            if self.validation_dataset_file:
                self.val_dataset = self.dataset_class(
                    self.validation_dataset_file,
                    batch_size=None,
                )

        if stage in (None, "predict") and self.predict_dataset_file:
            if self.predict_dataset_file:
                self.predict_dataset = self.dataset_class(
                    self.predict_dataset_file,
                    batch_size=None,
                )

    def train_dataloader(self):
        if self.train_dataset_file is None:
            raise ValueError("Train dataset is not set. Provide `train_dataset_file`.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.validation_dataset_file is None:
            raise ValueError("Validation dataset is not set. Provide `validation_dataset_file`.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset_file is None:
            raise ValueError("Prediction dataset is not set. Provide `predict_dataset_file`.")
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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

def get_current_epoch(ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        return checkpoint.get('epoch', 0)
    return 0

def main(
    config,
    output_dir,
    unique_identifier,
    gpus,
    batch_size,
    start_from_checkpoint,
    debug_mode = False,
):
    """
    Train a MethylSeqNN model based on a training gin config file that specifies both architecture and training plan

    Args:
        config: the path to a gin config files
        output_dir: where outputs are getting stored, i.e. best and temp checkpoints
        unique_identifier: the unique name for the folder in which the model's checkpoints will live
        gpus: how many gpus lightning gets to use, 'auto' will use all available
        batch_size: override the batch size that is in the gin config file; useful for e.g. running a config on different hardware without changing it
        start_from_checkpoint: the unique identifier for a model that you want to start from. This is assumed to be in the same output_dir. 
            best-checkpoint will be used; this can't be overridden right now.
        debug_mode: if True, no logs e.g. WandB
    
    TODO: refactor logic for resume from requeue vs starting from a possibly-differently-configured checkpoint to increase clarity and who handles what
    """
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
        dataset = data_module.dataset_class(data_module.train_dataset_file)
        model.io_mappings_str = dataset.get_io_mappings_str()
    except AttributeError:
        print(f"No 'io_mappings' attribute found in {data_module.train_dataset_file}.")
    
    model_dir = Path(output_dir)/unique_identifier
    temp_checkpoint_path = model_dir/'checkpoints'/'temp-checkpoint.ckpt'
    best_checkpoint_path = model_dir/'checkpoints'/'best-checkpoint.ckpt'
    start_checkpoint_path = Path(output_dir)/start_from_checkpoint/'checkpoints'/'best-checkpoint.ckpt' if start_from_checkpoint else None
    
    start_checkpoint_epoch = 0
    # if the temp checkpoint exists, model training has been restarted
    if os.path.isfile(temp_checkpoint_path):
        checkpoint_to_use = temp_checkpoint_path
        print(f"Starting from temp checkpoint {checkpoint_to_use}.")
    # if start_from_checkpoint was specified and we aren't already mid-training, load state_dict
    elif start_from_checkpoint:
        checkpoint_to_use = None
        try:
            checkpoint = torch.load(start_checkpoint_path, map_location=model.device)
            model.load_state_dict(checkpoint["state_dict"],strict=False)
            start_checkpoint_epoch = checkpoint.get("epoch",0)
            print(f"Starting training from state_dict for {start_checkpoint_path}; epoch will be forced to {start_checkpoint_epoch} on fit start.")
        except Exception as e:
            print(f"Failed to load checkpoint {start_checkpoint_path}: {e}. Starting from scratch instead.")
    # if there is no start point checkpoint nor temp checkpoint, we are starting from scratch
    else:
        print(f"Starting training from scratch.")
        checkpoint_to_use = None    
        
    if not debug_mode:
        logger = WandbLogger(
            save_dir=model_dir,
            name=f"{Path(config).stem}_{unique_identifier}",
            version=unique_identifier,
        )
    else:
        logger = None

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
        monitor='val/loss',                   # Metric to track for "best" checkpoint
        mode='min',                           # Minimize validation loss (or 'max' if you're maximizing a metric)
        save_top_k=1,                         # Save the best checkpoint only
        filename='best-checkpoint',           # Name for the best checkpoint
        save_last=False                       # Don't save a 'last' checkpoint
    )
    
    if model.train_stages:
        epochs_elapsed = 0
        for stage_name,stage_dict in model.train_stages.items():

            current_epoch = get_current_epoch(checkpoint_to_use) # if checkpoint to use is None, returned 0
            target_epoch = epochs_elapsed + stage_dict["epochs"]
            
            pp_stage_dict = pprint.pformat(stage_dict, indent=4, width=50)
            if current_epoch+start_checkpoint_epoch >= target_epoch:
                # if the checkpoint_to_use is already past the current stage, skip to the next stage
                print(f"""
####################################################################################################################################################
Skipping training stage {stage_name}; already complete due to start checkpoint or requeue.
Stage details: 
{pp_stage_dict}
Current epoch: {current_epoch+start_checkpoint_epoch}, target epoch: {target_epoch}
####################################################################################################################################################
                """)
            else:
                print(f"""
####################################################################################################################################################
Running training stage {stage_name}.
Stage details: 
{pp_stage_dict}
Current epoch: {current_epoch+start_checkpoint_epoch}, target epoch: {target_epoch}
####################################################################################################################################################
                """)
                # configure model for stage
                model.mode = stage_dict['mode']
                model.set_requires_grad(stage_dict['grad_dict'])
                if 'peak_subset_threshold' in stage_dict:
                    model.peak_subset_threshold = stage_dict['peak_subset_threshold']
                else:
                    model.peak_subset_threshold = 0
                if start_checkpoint_epoch > 0 and current_epoch == 0:
                    model.start_epoch = start_checkpoint_epoch
                
                trainer = Trainer(
                    callbacks = [temp_checkpoint,best_val_checkpoint,GPUMemoryLogger()],
                    default_root_dir=model_dir,
                    logger=logger,
                    accelerator='auto', 
                    devices=gpus, 
                    max_epochs=target_epoch,
                    strategy="ddp_find_unused_parameters_true",
                )  
                trainer.fit(
                    model,
                    datamodule=data_module,
                    ckpt_path=checkpoint_to_use
                )
                # once a training stage is complete, the next one should start from the best checkpoint from that stage
                checkpoint_to_use = best_checkpoint_path
                
            epochs_elapsed+=stage_dict["epochs"]
    else:
        trainer = Trainer(
            callbacks = [temp_checkpoint,best_val_checkpoint,GPUMemoryLogger()],
            default_root_dir=model_dir,
            logger=logger,
            accelerator='auto', 
            devices=gpus, 
            max_epochs=100,
            strategy="ddp_find_unused_parameters_true",
        )    
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_to_use
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MethylSeqNN model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the gin config file.')
    parser.add_argument('--output_dir', type=str, required=False, default='/clusterfs/nilah/oberon/lightning/', help='Directory to store outputs.')
    parser.add_argument('--unique_identifier', type=str, required=False, default=dt.now().strftime('%Y-%m-%d_%H-%M-%S'), help='Unique identifier for run.')
    parser.add_argument('--gpus', type=str, required=False, default='auto', help='GPU count for parallelization.')
    parser.add_argument('--batch_size', type=int, required=False, default=-1, help='Batch size for dataloader.')
    parser.add_argument('--start-from-checkpoint', type=str, required=False, default=None, help='Unique identifier for a checkpoint from which to restart. Hyperparameter mistmatch may cause errors.')
    parser.add_argument('--debug-mode', action='store_true', help='Run in debug mode: no logs, no checkpoints, no WandB.')
    args = parser.parse_args()
    main(args.config,args.output_dir,args.unique_identifier,args.gpus,args.batch_size,args.start_from_checkpoint,args.debug_mode)
