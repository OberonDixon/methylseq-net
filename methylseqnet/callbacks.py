import h5py
import os
from abc import ABC, abstractmethod
import tempfile
from collections import defaultdict
from io import StringIO
import psutil

import numpy as np
import matplotlib.pyplot as plt
import pysam
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
import wandb
import pandas as pd
import pyBigWig
import gin

from dimelo import load_processed
from methylseqnet import dna_io
from methylseqnet.metrics import PearsonAcrossPositions, PearsonAcrossTasks, CCCAcrossVariants
from methylseqnet.transforms import EncodingSelector

class ConditionalBestScoreReset(Callback):
    def __init__(self, checkpoint_callback, reset_on_train_start):
        self.checkpoint_callback = checkpoint_callback
        self.reset_on_train_start = reset_on_train_start
    
    def on_train_start(self, trainer, pl_module):
        if self.reset_on_train_start:
            # resets the callback as if it were freshly initialized
            self.checkpoint_callback.best_model_score = None
            self.checkpoint_callback.best_model_path = ""
            self.checkpoint_callback.current_score = None
            self.checkpoint_callback.best_k_models = {}
            self.checkpoint_callback.kth_best_model_path = ""

class BaseHDF5Writer(ABC):
    def __init__(
        self,
        output_dir=None,
    ):
        self.file_handles = {}
        self.pred_counter = 0
        if output_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = self._temp_dir_obj.name
        else:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def append_batch_to_h5(self, trainer, pl_module, predictions, specifiers, batch_indices, batch):
        inputs, targets = self._input_target_from_batch(batch, pl_module)
        # It appears that all ranks send to rank 0 and write out - but if not, then this logic currently breaks
        rank = trainer.global_rank
        if rank!=0:
            raise ValueError(f"Unexpected rank {rank}. Code in callbacks.py::HDF5PredictionWriter needs to be rewritten if ranks are not getting merged for writing, otherwise values will be missed.")
        path = os.path.join(self.output_dir, f"predictions.h5")

        pred_shape = predictions.shape[1:]
        targets_shape = targets.shape[1:]
        assert pred_shape == targets_shape, f"Predictions shape {pred_shape} does not match targets shape {targets_shape}"
        
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "w")
            self.file_handles[path].create_dataset("predictions", shape=(0, *pred_shape), maxshape=(None, *pred_shape), chunks=True)
            self.file_handles[path].create_dataset("tracks", shape=(0, *targets_shape), maxshape=(None, *targets_shape), chunks=True)
            self.file_handles[path].create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            self.file_handles[path].create_dataset("specifier", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"), chunks=True)
            self.file_handles[path].attrs['io_mappings'] = getattr(pl_module, 'io_mappings_str', '')

        f = self.file_handles[path]
        batch_indices = np.array(batch_indices)
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        curr_size = f["predictions"].shape[0]

        if batch_indices is None:
            batch_indices = np.arange(self.pred_counter, self.pred_counter + predictions_np.shape[0])
            self.pred_counter += predictions_np.shape[0]

        # Resize datasets
        f["predictions"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["predictions"][batch_indices] = predictions_np

        f["tracks"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["tracks"][batch_indices] = targets_np

        f["indices"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["indices"][batch_indices] = batch_indices

        f["specifier"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["specifier"][batch_indices] = specifiers

    def _input_target_from_batch(self, batch, pl_module):
        sequence = batch['sequence']
        targets = batch['target']
        # TODO: make this work in the case where inputs contains embeddings for pretrained
        # TODO: adjust for variants
        targets = pl_module.trim_targets(sequence,targets)
        return sequence, targets

    def __del__(self):
        self._close_all()
        if hasattr(self, '_temp_dir_obj') and self._temp_dir_obj:
            try:
                self._temp_dir_obj.cleanup()
            except Exception:
                pass

    def _close_all(self):
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.file_handles.clear()


class HDF5PredictionWriter(BasePredictionWriter, BaseHDF5Writer):
    def __init__(
        self, 
        output_dir, 
        write_interval="batch",
    ):
        BasePredictionWriter.__init__(self,write_interval)
        BaseHDF5Writer.__init__(self,output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        self.append_batch_to_h5(trainer, pl_module, prediction["predictions"], prediction["specifiers"], batch_indices, batch)

    def on_predict_end(self, trainer, pl_module):
        self._close_all()

class ValidationMetricsLogger(Callback, BaseHDF5Writer):
    def __init__(
        self,
        split_by_target_type=True,
        metrics=[PearsonAcrossPositions(), PearsonAcrossTasks(),CCCAcrossVariants()],
        in_memory=True,
        metrics_per_sample=True,
        metrics_across_dataset=False,
    ):
        Callback.__init__(self)
        self.in_memory = in_memory
        BaseHDF5Writer.__init__(self, output_dir=None)
        self.split_by_target_type = split_by_target_type
        self.metrics = metrics
        self.metrics_per_sample = metrics_per_sample
        self.metrics_across_dataset = metrics_across_dataset
        if not self.metrics_across_dataset and not self.in_memory:
            raise ValueError("if metrics_across_dataset is False, nothing gets saved between batches, so in_memory must be True.")

        self.metric_values_dict = defaultdict(dict)
        self.targets_list = []
        self.predictions_list = []

        

    def _get_channels_dict(self, pl_module):
        channels_dict = {}
        io_mappings_df = pl_module.get_io_mappings_df()
        for data_type in io_mappings_df['data_type'].unique():
            channels = io_mappings_df[io_mappings_df['data_type']==data_type]['channel'].tolist()
            channels_dict[data_type] = channels
        if len(channels_dict) == 0 and self.split_by_target_type:
            raise ValueError("split_by_target_type is True but no data types found in io_mappings.")
        return channels_dict

    def on_validation_epoch_start(self, trainer, pl_module):
        for data_type in pl_module.get_io_mappings_df()['data_type'].unique():
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                self.metric_values_dict[metric_name][data_type] = [float('nan')]  # initialize with nan to sync_dist issues
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, targets = self._input_target_from_batch(batch, pl_module)
        targets = targets.detach().cpu()
        predictions = outputs["predictions"].detach().cpu()
        batch_size = predictions.shape[0]
        io_mappings_df = pl_module.get_io_mappings_df()
        if self.metrics_per_sample:
            for i in range(batch_size):
                if self.split_by_target_type:
                    for data_type, dataset_channels in self._get_channels_dict(pl_module).items():
                        # check whether this channel is associated with the current sample's dataset_key
                        if data_type in io_mappings_df[io_mappings_df['dataset_key']==batch['dataset_key'][i]]['data_type'].values:
                            if pl_module.data_types_subset is None or data_type in pl_module.data_types_subset:
                                sample_predictions = predictions[i,:,dataset_channels,:]
                                sample_targets = targets[i,:,dataset_channels,:]
                                for metric in self.metrics:
                                    metric_name = metric.__class__.__name__
                                    metric_value = metric(sample_targets, sample_predictions)
                                    self.metric_values_dict[metric_name][data_type].append(metric_value.item())
                else:
                    sample_predictions = predictions[i]
                    sample_targets = targets[i]
                    for metric in self.metrics:
                        metric_name = metric.__class__.__name__
                        metric_value = metric(sample_targets, sample_predictions)
                        self.metric_values_dict[metric_name]['all'].append(metric_value.item())
        if self.metrics_across_dataset:
            if self.in_memory:
                self.predictions_list.extend([predictions[i] for i in range(batch_size)])
                self.targets_list.extend([targets[i] for i in range(batch_size)])
            else:
                self.append_batch_to_h5(trainer, pl_module, predictions, ["" for _ in predictions], None, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.metrics_per_sample:
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                for data_type, metric_values in self.metric_values_dict[metric_name].items():
                    # log the mean
                    if len(metric_values) > 0:
                        mean_metric_value = np.nanmean(metric_values)
                        # valid_values = [v for v in metric_values if not np.isnan(v)]
                        # if len(valid_values) == 0:
                        #     print(f"[Rank {trainer.global_rank}] WARNING: {metric_name} for {data_type} has all NaN values (n={len(metric_values)})")
                        pl_module.log(f"val/{metric_name}_mean_per_sample_{data_type}", mean_metric_value, prog_bar=True, sync_dist=True)
        if self.metrics_across_dataset:
            if self.in_memory:
                # first concatenate everything into tensors to operate upon
                predictions = torch.cat(self.predictions_list, dim=2)
                targets = torch.cat(self.targets_list, dim=2)
                # compute metrics from in-memory structure
                for metric in self.metrics:
                    metric_name = metric.__class__.__name__
                    if self.split_by_target_type:
                        for data_type, channels in self._get_channels_dict(pl_module).items():
                            if pl_module.data_types_subset is None or data_type in pl_module.data_types_subset:
                                metric_value = metric(targets[...,channels,:], predictions[...,channels,:])
                                pl_module.log(f"val/{metric_name}_across_dataset_{data_type}", metric_value.item(), prog_bar=True, sync_dist=True)
                    else:
                        metric_value = metric(targets, predictions)
                        pl_module.log(f"val/{metric_name}_across_dataset_all", metric_value.item(), prog_bar=True, sync_dist=True)
            else:
                # first close all of the file handles to flush everything to disk
                self._close_all()
                # then load from the h5 file(s) and compute metrics
                raise NotImplementedError("Metrics computation from HDF5 files not implemented yet.")
                    
        # empty the lists for next epoch
        self.predictions_list = []
        self.targets_list = []
        self.metric_values_dict = defaultdict(dict)

class GPUMemoryLogger(Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

class CPUMemoryLogger(Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval == 0:
            mem = psutil.virtual_memory()
            pl_module.log("memory/train_cpu_memory_percent", mem.percent, prog_bar=False, sync_dist=False)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            mem = psutil.virtual_memory()
            pl_module.log("memory/val_cpu_memory_percent", mem.percent, prog_bar=False, sync_dist=False)

class SubmodulesGradientNormLogger(Callback):
    def __init__(self, submodule_names: list[str]):
        super().__init__()
        self.submodule_names = submodule_names
    def on_after_backward(self, trainer, pl_module):
        grad_norms = {}
        for name, param in pl_module.named_parameters():
            for submodule_name in self.submodule_names:
                if name.startswith(submodule_name) and param.grad is not None:
                    if submodule_name not in grad_norms:
                        grad_norms[submodule_name] = 0.0
                    grad_norms[submodule_name] += param.grad.data.norm(2).item() ** 2
        for submodule_name, norm_sq in grad_norms.items():
            grad_norm = norm_sq ** 0.5
            pl_module.log(f"train/grad_norm_{submodule_name}", grad_norm, prog_bar=False)

@gin.register
@gin.configurable
class HaplotypedPredLogger(Callback):
    def __init__(
        self,
        hp1_cpg_file: str,
        hp2_cpg_file: str,
        hp1_accessibility_file: str,
        hp2_accessibility_file: str,
        ref_genome_fasta : str,
        regions: list[tuple[str, int, int]],
        hp1_rna_file: str = None,
        hp2_rna_file: str = None,
        accessibility_outputs_slice: slice | list = [0],
        rna_outputs_slice: slice | list = [-1],
        crop_for_accessibility: int = 163840,
        label_bin_size: int = 128,
        log_stats: bool = True,
        upload_plots: bool = False,
        plot_methylation: bool = False,
        plot_rna: bool = False,
        methylation_exaggeration: float = 1.0,
        ):
        super().__init__()
        self.hp1_cpg_file = hp1_cpg_file
        self.hp2_cpg_file = hp2_cpg_file
        self.hp1_accessibility_file = hp1_accessibility_file
        self.hp2_accessibility_file = hp2_accessibility_file
        self.hp1_rna_file = hp1_rna_file
        self.hp2_rna_file = hp2_rna_file
        self.genome = pysam.FastaFile(ref_genome_fasta)
        self.regions = regions
        self.accessibility_outputs_slice = accessibility_outputs_slice
        self.rna_outputs_slice = rna_outputs_slice
        self.crop_for_accessibility = crop_for_accessibility
        self.label_bin_size = label_bin_size
        self.log_stats = log_stats
        self.upload_plots = upload_plots
        self.plot_methylation = plot_methylation
        self.plot_rna = plot_rna
        self.methylation_exaggeration = methylation_exaggeration
        if not (os.path.exists(hp1_cpg_file) and os.path.exists(hp2_cpg_file) and os.path.exists(ref_genome_fasta)):
            raise ValueError("One of the provided haplotype-specific cpg bed files or reference genome fasta does not exist.")
        if (hp1_accessibility_file is not None) != (hp2_accessibility_file is not None):
            raise ValueError("Either both or neither haplotype-specific accessibility bed files must be provided.")
        if (hp1_accessibility_file is not None) and (not (os.path.exists(hp1_accessibility_file) and os.path.exists(hp2_accessibility_file))):
            raise ValueError("One of the provided haplotype-specific accessibility bed files does not exist.")

        self.input_to_methylation = nn.Sequential(
            EncodingSelector(encoding_str='interp-methyl-only'),
            nn.AvgPool1d(kernel_size=128),
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            # Get the wandb Run (works when WandbLogger is enabled)
            run = getattr(getattr(trainer, "logger", None), "experiment", None)
            epoch = getattr(trainer, "current_epoch", -1)
            if run is not None and hasattr(run, "log"):
                images, hp1_pearsons, hp2_pearsons, differential_pearsons = [], [], [], []
                for chromosome, start, end in self.regions:
                    (
                        hp1_accessibility_target,
                        hp2_accessibility_target,
                        hp1_accessibility_pred,
                        hp2_accessibility_pred,
                        hp1_methylation,
                        hp2_methylation,
                        hp1_pred_methylation,
                        hp2_pred_methylation,
                        hp1_rna_target,
                        hp2_rna_target,
                        hp1_rna_pred,
                        hp2_rna_pred,
                    ) = self._compute_haplo_pred_stats(pl_module,chromosome,start,end)
                    if hp1_pred_methylation is not None and hp2_pred_methylation is not None:
                        assert len(hp1_methylation) == len(hp2_methylation), f"methylation lengths do not match for {chromosome}:{start}-{end}"
                        pred_len = len(hp1_accessibility_pred)
                        methyl_len = len(hp1_methylation)
                        if methyl_len > pred_len:
                            crop_off_each_end = (methyl_len - pred_len) // 2
                            hp1_methylation = hp1_methylation[crop_off_each_end:crop_off_each_end+pred_len]
                            hp2_methylation = hp2_methylation[crop_off_each_end:crop_off_each_end+pred_len]
                        if len(hp1_pred_methylation) > pred_len:
                            hp1_pred_methylation = hp1_pred_methylation[crop_off_each_end:crop_off_each_end+pred_len]
                            hp2_pred_methylation = hp2_pred_methylation[crop_off_each_end:crop_off_each_end+pred_len]
                    if self.upload_plots:
                        region_str = f"{chromosome}:{start}-{end}"
                        center_coord = (start + end) // 2
                        num_bins = len(hp1_accessibility_pred)
                        start_pos = center_coord - (num_bins * self.label_bin_size) // 2
                        positions = start_pos + np.arange(num_bins) * self.label_bin_size

                        # signal_ys = [hp1_pred, -hp2_pred]
                        # signal_keys = ["Haplo 1 Prediction", "Haplo 2 Prediction"]
                        # if hp1_target is not None:
                        #     signal_ys.extend([hp1_target, -hp2_target])
                        #     signal_keys.extend(["Haplo 1 Target", "Haplo 2 Target"])

                        # run.log({
                        #     f"haplo_{region_str}/signals": wandb.plot.line_series(
                        #         xs=positions,
                        #         ys=signal_ys,
                        #         keys=signal_keys,
                        #         title=f"{region_str} - Signals",
                        #         xname="Position (binned)"
                        #     ),
                        #     "epoch": epoch
                        # })

                        # diff_ys = [hp1_pred - hp2_pred]
                        # diff_keys = ["Haplo 1 - Haplo 2 Prediction"]
                        # if hp1_target is not None:
                        #     diff_ys.append(hp1_target - hp2_target)
                        #     diff_keys.append("Haplo 1 - Haplo 2 Target")
                        
                        # run.log({
                        #     f"haplo_{region_str}/differentials": wandb.plot.line_series(
                        #         xs=positions,
                        #         ys=diff_ys,
                        #         keys=diff_keys,
                        #         title=f"{region_str} - Differentials",
                        #         xname="Position (binned)"
                        #     ),
                        #     "epoch": epoch
                        # })

                        # if self.plot_methylation:
                        #     methyl_ys = [hp1_methylation, -hp2_methylation]
                        #     methyl_keys = ["Haplo 1 Methylation", "Haplo 2 Methylation"]
                        #     if hp1_target is not None:
                        #         methyl_ys.extend([hp1_pred_methylation, -hp2_pred_methylation])
                        #         methyl_keys.extend(["Haplo 1 Imputed Methylation", "Haplo 2 Imputed Methylation"])
                            
                        #     run.log({
                        #         f"haplo_{region_str}/methylation": wandb.plot.line_series(
                        #             xs=positions,
                        #             ys=methyl_ys,
                        #             keys=methyl_keys,
                        #             title=f"{region_str} - Methylation",
                        #             xname="Position (binned)"
                        #         ),
                        #         "epoch": epoch
                        #     })
                        
                        fig, axes = plt.subplots(4, 2, figsize=(30, 15), sharex=True)
                        fig.suptitle(f"{region_str}, epoch={epoch}")
                        
                        # Row 0: Accessibility
                        if hp1_accessibility_target is not None:
                            axes[0, 0].plot(positions, hp1_accessibility_target, label="Haplo 1 Target", color='blue', alpha=0.5)
                            axes[0, 0].plot(positions, -hp2_accessibility_target, label="Haplo 2 Target", color='orange', alpha=0.5)
                            axes[0, 0].set_ylabel("hp1,2\ntarget")
                        axes[0, 0].set_title(r'$\mathbf{GROUND\ TRUTH}$' + '\n\nTrue Accessibility')
                        
                        axes[0, 1].plot(positions, hp1_accessibility_pred, label="Haplo 1 Prediction", color='blue', alpha=0.5)
                        axes[0, 1].plot(positions, -hp2_accessibility_pred, label="Haplo 2 Prediction", color='orange', alpha=0.5)
                        axes[0, 1].set_ylabel("hp1,2\npred")
                        axes[0, 1].set_title(r'$\mathbf{PREDICTIONS}$' + '\n\nPredicted Accessibility')
                        
                        # Row 1: Differential Accessibility
                        if hp1_accessibility_target is not None:
                            axes[1, 0].plot(positions, hp1_accessibility_target - hp2_accessibility_target, label="Haplo 1 - Haplo 2 Target", color='green', alpha=0.5)
                            axes[1, 0].set_ylabel("hp1-hp2\ntarget")
                            axes[1, 0].set_title("True Differential Accessibility")
                        
                        axes[1, 1].plot(positions, hp1_accessibility_pred - hp2_accessibility_pred, label="Haplo 1 - Haplo 2 Prediction", color='green', alpha=0.5)
                        axes[1, 1].set_ylabel("hp1-2\npred")
                        axes[1, 1].set_title("Predicted Differential Accessibility")
                        
                        # Row 2: Methylation (only if plot_methylation is True)
                        if self.plot_methylation:
                            axes[2, 0].plot(positions, hp1_methylation, label="Haplo 1 Methylation", color='blue', alpha=0.5)
                            axes[2, 0].plot(positions, -hp2_methylation, label="Haplo 2 Methylation", color='orange', alpha=0.5)
                            axes[2, 0].set_ylabel("hp1,2\ntru methyl")
                            axes[2, 0].set_title("True Methylation")
                            
                            axes[2, 1].plot(positions, hp1_pred_methylation, label="Haplo 1 Imputed Methylation", color='blue', alpha=0.5)
                            axes[2, 1].plot(positions, -hp2_pred_methylation, label="Haplo 2 Imputed Methylation", color='orange', alpha=0.5)
                            axes[2, 1].set_ylabel("hp1/2\nimp methyl")
                            axes[2, 1].set_title("Imputed Methylation")

                        if self.plot_rna:
                            # Row 3: RNA
                            if hp1_rna_target is not None:
                                axes[3, 0].plot(positions, hp1_rna_target, label="Haplo 1 RNA Target", color='blue', alpha=0.5)
                                axes[3, 0].plot(positions, -hp2_rna_target, label="Haplo 2 RNA Target", color='orange', alpha=0.5)
                                axes[3, 0].set_ylabel("hp1,2\nRNA target")
                                axes[3, 0].set_title("True RNA Expression")
                            
                            if hp1_rna_pred is not None:
                                axes[3, 1].plot(positions, hp1_rna_pred, label="Haplo 1 RNA Prediction", color='blue', alpha=0.5)
                                axes[3, 1].plot(positions, -hp2_rna_pred, label="Haplo 2 RNA Prediction", color='orange', alpha=0.5)
                                axes[3, 1].set_ylabel("hp1,2\nRNA pred")
                                axes[3, 1].set_title("Predicted RNA Expression")
                        
                        # Set x-label only on bottom row
                        axes[-1, 0].set_xlabel("Position (binned)")
                        axes[-1, 1].set_xlabel("Position (binned)")
                        
                        fig.canvas.draw()  # guarantee the figure is rendered NOW
                        w, h = fig.canvas.get_width_height()
                        run.log({f"haplo/phased_plots_{region_str}": wandb.Image(fig, caption=f"epoch={epoch}"), "epoch": epoch})
                        plt.close(fig)
                    if self.log_stats and (hp1_accessibility_target is not None):
                        # run.log({
                        #     f"haplo_{region_str}_haplo1_pearson": pearsonr(hp1_target, hp1_pred)[0],
                        #     f"haplo_{region_str}_haplo2_pearson": pearsonr(hp2_target, hp2_pred)[0],
                        #     f"haplo_{region_str}_haplo_differential_pearson": pearsonr(hp1_target - hp2_target, hp1_pred - hp2_pred)[0],
                        #     "epoch": epoch,
                        # })
                        hp1_pearsons.append(pearsonr(hp1_accessibility_target, hp1_accessibility_pred)[0])
                        hp2_pearsons.append(pearsonr(hp2_accessibility_target, hp2_accessibility_pred)[0])
                        differential_pearsons.append(pearsonr(hp1_accessibility_target - hp2_accessibility_target, hp1_accessibility_pred - hp2_accessibility_pred)[0])
                if self.log_stats and (hp1_accessibility_target is not None):
                    run.log({
                        "haplo/haplo1_pearson": np.mean(hp1_pearsons),
                        "haplo/haplo2_pearson": np.mean(hp2_pearsons),
                        "haplo/haplo_differential_pearson": np.mean(differential_pearsons),
                        "epoch": epoch,
                    })
            else:
                print("Wandb run not found, cannot upload.")

    def _compute_haplo_pred_stats(
        self,
        pl_module,
        chromosome: str,
        start: int,
        end: int,
    ):
        device = pl_module.device
        hp1_sequence, hp1_methylation_encoding = self._construct_input_tensor(
            genome_track_file=self.hp1_cpg_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        )
        hp2_sequence, hp2_methylation_encoding = self._construct_input_tensor(
            genome_track_file=self.hp2_cpg_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        )
        hp1_methylation = self.input_to_methylation(torch.cat([hp1_sequence, hp1_methylation_encoding],dim=1)).squeeze().cpu().numpy()
        hp2_methylation = self.input_to_methylation(torch.cat([hp2_sequence, hp2_methylation_encoding],dim=1)).squeeze().cpu().numpy()
        hp1_accessibility_target = self._construct_target_tensor(
            genome_track_file=self.hp1_accessibility_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp1_accessibility_file is not None else None
        hp2_accessibility_target = self._construct_target_tensor(
            genome_track_file=self.hp2_accessibility_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp2_accessibility_file is not None else None
        hp1_rna_target = self._construct_target_tensor(
            genome_track_file=self.hp1_rna_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp1_rna_file is not None else None
        hp2_rna_target = self._construct_target_tensor(
            genome_track_file=self.hp2_rna_file,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp2_rna_file is not None else None
        with torch.no_grad():
            training_mode = pl_module.mode
            training_true_methyl_rep_weight = pl_module.true_methyl_rep_weight
            pl_module.eval()
            if pl_module.layers:
                pl_module.mode = 'full-model'
            elif pl_module.input_to_methyl_rep:
                pl_module.mode = 'factorized-from-pretrained'
            else:
                pl_module.mode = 'pretrained-only'
            pl_module.true_methyl_rep_weight = 1.0
            hp1_output = pl_module(hp1_sequence,hp1_methylation_encoding.unsqueeze(1))
            hp1_accessibility_pred = hp1_output[:, self.accessibility_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp1_rna_pred = hp1_output[:, self.rna_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp1_pred_methylation = (
                pl_module.hooked_activations[id(pl_module.capture_imputed_methyl_rep)].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
                if id(pl_module.capture_imputed_methyl_rep) in pl_module.hooked_activations
                else np.ones_like(hp1_methylation)
            )
            hp2_output = pl_module(hp2_sequence,hp2_methylation_encoding.unsqueeze(1))
            hp2_accessibility_pred = hp2_output[:, self.accessibility_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp2_rna_pred = hp2_output[:, self.rna_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp2_pred_methylation = (
                pl_module.hooked_activations[id(pl_module.capture_imputed_methyl_rep)].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
                if id(pl_module.capture_imputed_methyl_rep) in pl_module.hooked_activations
                else np.zeros_like(hp2_methylation)
            )
            pl_module.mode = training_mode
            pl_module.true_methyl_rep_weight = training_true_methyl_rep_weight
        return (
            hp1_accessibility_target,
            hp2_accessibility_target,
            hp1_accessibility_pred,
            hp2_accessibility_pred,
            hp1_methylation,
            hp2_methylation,
            hp1_pred_methylation,
            hp2_pred_methylation,
            hp1_rna_target,
            hp2_rna_target,
            hp1_rna_pred,
            hp2_rna_pred,
        )


    def _construct_input_tensor(
        self,
        genome_track_file,
        chromosome,
        start,
        end,
        device,
    ) -> torch.Tensor:
        cpg_ratio, non_zero_mask = self._load_track_from_file(
            genome_track_file=genome_track_file,
            motif="CG,0",
            chromosome=chromosome,
            start=start,
            end=end,
            bin_size=1,
            crop=0,
        )
        exp_cpg_ratio = self._exaggerate_methylation(cpg_ratio, non_zero_mask)
        x_methylseq = torch.permute(
            torch.tensor(
                dna_io.one_hot_encode_dna(
                    dna_strand=self.genome.fetch(chromosome,start,end), 
                    cpg_methylation=exp_cpg_ratio, 
                    valid_cpgs=non_zero_mask,
                ),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            (0,2,1),
        )
        sequence = x_methylseq[:, :4, :]
        methylation = x_methylseq[:, 4:, :]
        return sequence, methylation
    
    def _exaggerate_methylation(self, cpg_ratio, non_zero_mask, eps=1e-7) -> np.ndarray:
        if self.methylation_exaggeration==1.0:
            return cpg_ratio
        else:
            result = cpg_ratio.copy()
            
            # Clip to avoid numerical issues, then convert to logits, scale, convert back
            clipped = np.clip(result[non_zero_mask], eps, 1 - eps)
            logits = np.log(clipped / (1 - clipped))
            result[non_zero_mask] = 1 / (1 + np.exp(-self.methylation_exaggeration * logits))
            
            return result

    def _construct_target_tensor(
        self,
        genome_track_file,
        chromosome,
        start,
        end,
        device,
    ) -> torch.Tensor:
        accessibility_ratio, _ = self._load_track_from_file(
            genome_track_file=genome_track_file,
            motif="A,0",
            chromosome=chromosome,
            start=start,
            end=end,
            bin_size=self.label_bin_size,
            crop=self.crop_for_accessibility,
        )
        return torch.tensor(
            accessibility_ratio,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).unsqueeze(0)

    def _load_track_from_file(
        self,
        genome_track_file,
        motif,
        chromosome,
        start,
        end,
        bin_size=1,
        crop=0,
    ) -> np.ndarray:
        if genome_track_file.endswith(".bed.gz"):
            mod_vector, val_vector = load_processed.pileup_vectors_from_bedmethyl(
                bedmethyl_file = genome_track_file,
                motif = motif,
                regions = f'{chromosome}:{start+crop}-{end-crop}',
                quiet=True,
            )
            mod_vector_binned = mod_vector.reshape(-1, bin_size).sum(axis=1)
            val_vector_binned = val_vector.reshape(-1, bin_size).sum(axis=1)
            non_zero_mask = val_vector_binned != 0
            methylated_ratio = np.zeros_like(mod_vector_binned, dtype=float)
            methylated_ratio[non_zero_mask] = mod_vector_binned[non_zero_mask] / val_vector_binned[non_zero_mask]
            return methylated_ratio, non_zero_mask
        elif genome_track_file.endswith(".bw") or genome_track_file.endswith(".bigwig"):
            bw = pyBigWig.open(genome_track_file)
            values = np.array(bw.values(chromosome, start+crop, end-crop, numpy=True))
            bw.close()
            values[np.isnan(values)] = 0.0
            values_binned = values.reshape(-1, bin_size).mean(axis=1)
            non_zero_mask = values_binned != 0
            return values_binned, non_zero_mask
        elif genome_track_file.endswith(".bam"):
            bam = pysam.AlignmentFile(genome_track_file, "rb")
            coverage = np.array([sum(x) for x in zip(*(bam.count_coverage(chromosome,start+crop,end-crop)))])
            bam.close()
            coverage_binned = coverage.reshape(-1, bin_size).mean(axis=1)
            non_zero_mask = coverage_binned != 0
            return coverage_binned, non_zero_mask
        else:
            raise ValueError(f"Unsupported genome track file format: {genome_track_file}")
