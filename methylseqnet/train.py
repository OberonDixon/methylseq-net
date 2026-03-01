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
import time
from pathlib import Path
import argparse
from methylseqnet.dataset import MultiMethylDataset, CompositeDataset
from methylseqnet.callbacks import ConditionalBestScoreReset, GPUMemoryLogger, CPUMemoryLogger, HaplotypedPredLogger, ValidationMetricsLogger, SubmodulesGradientNormLogger
from methylseqnet.model import ConditionedSeqNN
from methylseqnet.datamodule import MethylSeqDataModule
from collections import defaultdict
import pynvml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
import signal
import pprint
import logging
import warnings
import functools

os.environ["SLURM_JOB_NAME"] = "interactive"

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
    max_epochs,
    random_seed = 42,
    samples_per_step = -1,
    samples_per_log = 64,
    start_from_checkpoint = None,
    no_wandb = False,
    no_checkpoints = False,
    no_haplotype_metrics = False,
    track_gradients_for_modules = [],
):
    """
    Train a ConditionedSeqNN model based on a training gin config file that specifies both architecture and training plan

    Args:
        config: the path to a gin config files
        output_dir: where outputs are getting stored, i.e. best and temp checkpoints
        unique_identifier: the unique name for the folder in which the model's checkpoints will live
        gpus: how many gpus lightning gets to use, 'auto' will use all available
        batch_size: override the batch size that is in the gin config file; useful for e.g. running a config on different hardware without changing it
        max_epochs: maximum number of epochs to train; this is overridden if training stages are defined in the gin config file
        samples_per_step: how many samples to process per optimizer step; this is used to calculation gradient accumulation steps internally
        start_from_checkpoint: the unique identifier for a model that you want to start from. This is assumed to be in the same output_dir. 
            best-checkpoint will be used; this can't be overridden right now.
        no_wandb: if True, do not save WandB logs
        no_checkpoints: if True, do not save model checkpoints
        no_haplotype_metrics: if True, disable haplotype-specific metrics logging during training
    
    TODO: refactor logic for resume from requeue vs starting from a possibly-differently-configured checkpoint to increase clarity and who handles what
    """
    seed_everything(random_seed, workers=True)

    gin.parse_config_file(config)
    
    model = ConditionedSeqNN()

    if len(model.train_stages)>0:
        total_stage_epochs = sum([stage_dict['epochs'] for stage_name,stage_dict in model.train_stages.items()])
        if total_stage_epochs != max_epochs:
            warnings.warn(f"Training stages detected in gin config; max_epochs={max_epochs} will be ignored in favor of total stage epochs {total_stage_epochs}.")
    
    from methylseqnet.train import MethylSeqDataModule
    if batch_size>0:
        # if the script was provided with a batch_size
        data_module = MethylSeqDataModule(batch_size=batch_size)
    else:
        data_module = MethylSeqDataModule()
        batch_size = data_module.batch_size
    
    # determine accumulate_grad_batches based on samples_per_step
    batch_by_gpus = batch_size * (torch.cuda.device_count() if gpus=='auto' else int(gpus))
    if samples_per_step > 0:
        if batch_by_gpus > 0:
            if samples_per_step % batch_by_gpus != 0:
                raise ValueError(f"samples_per_step ({samples_per_step}) must be a multiple of batch_size * num_gpus ({batch_by_gpus}).")
            accumulate_grad_batches = samples_per_step // batch_by_gpus
        else:
            if samples_per_step % batch_size != 0:
                raise ValueError(f"samples_per_step ({samples_per_step}) must be a multiple of batch_size ({batch_size}).")
            accumulate_grad_batches = samples_per_step // batch_size
    else:
        accumulate_grad_batches = 1

    # Try to retrieve the io_mappings string from the train dataset
    # The value here lies in the fact that the task structure is dynamically created from the
    # preprocess matches file, so having a record of what the mappings is for a given model
    # may be useful when trying different datasets, etc
    data_module.setup(stage='fit')
    model.set_io_mappings(data_module.get_io_mappings_str())
    
    model_dir = Path(output_dir)/unique_identifier
    temp_checkpoint_path = model_dir/'checkpoints'/'temp-checkpoint.ckpt'
    start_checkpoint_path = Path(output_dir)/start_from_checkpoint/'checkpoints'/'best-checkpoint.ckpt' if start_from_checkpoint else None
    
    start_checkpoint_epoch = 0
    starting_from_best = False
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
        
    if not no_wandb:
        logger = WandbLogger(
            save_dir=model_dir,
            name=f"{Path(config).stem}_{unique_identifier}",
            version=unique_identifier,
        )
    else:
        logger = None
    
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
                model.current_stage_name = stage_name
                model.apply_current_stage()
                if start_checkpoint_epoch > 0 and current_epoch == 0:
                    model.start_epoch = start_checkpoint_epoch
                
                trainer = Trainer(
                    callbacks = create_callbacks(
                        model_dir=model_dir,
                        no_checkpoints=no_checkpoints,
                        stage_name=stage_name,
                        no_haplotype_metrics=no_haplotype_metrics,
                        starting_from_best=starting_from_best,
                        track_gradients_for_modules=track_gradients_for_modules,
                    ),
                    default_root_dir=model_dir,
                    logger=logger,
                    accelerator='auto', 
                    devices=gpus, 
                    max_epochs=target_epoch,
                    strategy="ddp_find_unused_parameters_true",
                    accumulate_grad_batches=accumulate_grad_batches,
                    log_every_n_steps=samples_per_log//accumulate_grad_batches,
                )  
                # Check if Lightning set any seed internally
                print(f"PyTorch seed after Trainer init: {torch.initial_seed()}")
                trainer.fit(
                    model,
                    datamodule=data_module,
                    ckpt_path=checkpoint_to_use
                )
                # once a training stage is complete, the next one should start from the best checkpoint from that stage
                best_checkpoint_path = model_dir/'checkpoints'/f'best-checkpoint-{stage_name}.ckpt'
                checkpoint_to_use = best_checkpoint_path
                starting_from_best = True
                
            epochs_elapsed+=stage_dict["epochs"]
    else:
        trainer = Trainer(
            callbacks = create_callbacks(
                model_dir=model_dir,
                no_checkpoints=no_checkpoints,
                stage_name=None,
                no_haplotype_metrics=no_haplotype_metrics,
                starting_from_best=starting_from_best,
                track_gradients_for_modules=track_gradients_for_modules,
            ),
            default_root_dir=model_dir,
            logger=logger,
            accelerator='auto', 
            devices=gpus, 
            max_epochs=max_epochs,
            strategy="ddp_find_unused_parameters_true",
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=samples_per_log//accumulate_grad_batches,
        )    
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_to_use
        )

    return trainer

def create_callbacks(
    model_dir,
    no_checkpoints=False,
    stage_name=None,
    no_haplotype_metrics=False,
    starting_from_best=False,
    track_gradients_for_modules=[],
) -> list:
    """
    Create a list of callbacks for the Trainer, including checkpointing and logging.
    Args:
        model_dir: directory where model checkpoints will be saved
        no_checkpoints: if True, do not create checkpoint callbacks
        stage_name: if provided, used to suffix the best-checkpoint filename
        no_haplotype_metrics: if True, disable haplotype-specific metrics logging during training
        starting_from_best: if True, reset best score tracking in ConditionalBestScoreReset callback
        track_gradients_for_modules: list of module names for which to log gradient norms
    """
    callbacks = [GPUMemoryLogger(),ValidationMetricsLogger(),CPUMemoryLogger()]
    if not no_checkpoints:
        suffix = f"-{stage_name}" if stage_name is not None else ""
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
            filename='best-checkpoint'+suffix,           # Name for the best checkpoint
            save_last=False                       # Don't save a 'last' checkpoint
        )
        # Reset best score if continuing from previous stage
        reset_best_score = ConditionalBestScoreReset(
            best_val_checkpoint,
            reset_on_train_start=starting_from_best, # only reset if starting from a best checkpoint -> then we want the callback state reset
            )
        callbacks.extend([temp_checkpoint,best_val_checkpoint,reset_best_score])
    if not no_haplotype_metrics:
        haplotyped_pred_logger_fiber = HaplotypedPredLogger(
            hp1_cpg_file = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/5mC/GM12878_WGS-pb-5mC.hap2.bw',
            hp2_cpg_file = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/5mC/GM12878_WGS-pb-5mC.hap1.bw',
            hp1_accessibility_file = "/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/FIRE/GM12878_trackHub/bw/hap2.acc.bw",
            hp2_accessibility_file = "/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/FIRE/GM12878_trackHub/bw/hap1.acc.bw",
            hp1_rna_file = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/rna_bams/GM12878.kinnex.HP1.tss.counts.bed.gz',
            hp2_rna_file = '/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/rna_bams/GM12878.kinnex.HP2.tss.counts.bed.gz',
            ref_genome_fasta = "/clusterfs/nilah/oberon/genomes/hg38.fa",
            regions = [('chrX',131789298-262_144,131789298+262_144),('chrX',149575782-262_144,149575782+262_144)],
            log_stats = True,
            upload_plots = True,    
            plot_methylation = True,   
            plot_rna = True,
        )
        callbacks.extend([haplotyped_pred_logger_fiber])
    if track_gradients_for_modules:
        callbacks.append(SubmodulesGradientNormLogger(track_gradients_for_modules))

    return callbacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ConditionedSeqNN model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the gin config file.')
    parser.add_argument('--output_dir', type=str, required=False, default='/clusterfs/nilah/oberon/lightning/', help='Directory to store outputs.')
    parser.add_argument('--unique_identifier', type=str, required=False, default=dt.now().strftime('%Y-%m-%d_%H-%M-%S'), help='Unique identifier for run.')
    parser.add_argument('--gpus', type=str, required=False, default='auto', help='GPU count for parallelization.')
    parser.add_argument('--batch_size', type=int, required=False, default=-1, help='Batch size for dataloader.')
    parser.add_argument('--max-epochs', type=int, required=False, default=100, help='Maximum number of epochs to train; this is overridden if training stages are defined in the gin config file.')
    parser.add_argument('--samples-per-step', type=int, required=False, default=32, help='How many samples to process per optimizer step; this is used to calculation gradient accumulation steps internally. If -1, no gradient accumulation is used.')
    parser.add_argument('--start-from-checkpoint', type=str, required=False, default=None, help='Unique identifier for a checkpoint from which to restart. Hyperparameter mistmatch may cause errors.')
    parser.add_argument('--no-wandb', action='store_true', help='Do not save WandB logs.')
    parser.add_argument('--no-checkpoints', action='store_true', help='Do not save model checkpoints.')
    parser.add_argument('--no-haplotype-metrics', action='store_true', help='If set, enable haplotype-specific metrics logging during training.')
    parser.add_argument('--track-gradients-for-modules', nargs='+', type=str, required=False, default=['embeddings_to_methyl_rep','embeddings_to_unconditional_seq_rep','embeddings_to_conditional_seq_rep','factorized_rep_to_output'], help='If provided, track gradients for the named module(s). Can be specified multiple times.')
    parser.add_argument('--seed', type=int, required=False, default=(int(time.time() * 1_000_000) + os.getpid()) % (2**32), help='Random seed for reproducibility.')
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Set the logging level"
    )
    args = parser.parse_args()

    level = getattr(logging, args.logging_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    main(
        config=args.config,
        output_dir=args.output_dir,
        unique_identifier=args.unique_identifier,
        gpus=args.gpus,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        samples_per_step=args.samples_per_step,
        samples_per_log=64,
        random_seed=args.seed,
        start_from_checkpoint=args.start_from_checkpoint,
        no_wandb=args.no_wandb,
        no_checkpoints=args.no_checkpoints,
        no_haplotype_metrics=args.no_haplotype_metrics,
        track_gradients_for_modules=args.track_gradients_for_modules
    )
