import h5py
import os
import tempfile
from collections import defaultdict
from io import StringIO
import psutil

import numpy as np
import matplotlib.pyplot as plt
import pysam
from lightning.pytorch.callbacks import Callback
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
import wandb
import pandas as pd
import pyBigWig
import gin

from methylseqnet import encoding
from methylseqnet.metrics import GenomicTensorMetric, PearsonAcrossPositions, PearsonAcrossTasks, CCCAcrossVariants
from methylseqnet.transforms import EncodingSelector
from methylseqnet.readers import load_sequence, load_track, load_masked_track
from methylseqnet.tensor_ops import gather_to_rank0
from methylseqnet.writers import HDF5PredictionWriter
from methylseqnet.predict import Predictor

class ConditionalBestScoreReset(Callback):
    def __init__(self, checkpoint_callback, reset_on_train_start):
        self.checkpoint_callback = checkpoint_callback
        self.reset_on_train_start = reset_on_train_start
    
    def on_train_start(self, train, pl_module):
        if self.reset_on_train_start:
            # resets the callback as if it were freshly initialized
            self.checkpoint_callback.best_model_score = None
            self.checkpoint_callback.best_model_path = ""
            self.checkpoint_callback.current_score = None
            self.checkpoint_callback.best_k_models = {}
            self.checkpoint_callback.kth_best_model_path = ""

class ValidationMetricsLogger(Callback):
    def __init__(
        self,
        split_by_target_type=True,
        metric_classes=[PearsonAcrossPositions, PearsonAcrossTasks],
        metrics_per_sample=False,
        metrics_across_dataset=True,
    ):
        Callback.__init__(self)
        self.split_by_target_type = split_by_target_type
        self.metric_classes = metric_classes
        self.metrics_per_sample = metrics_per_sample
        self.metrics_across_dataset = metrics_across_dataset

        self.metric_values_dict = defaultdict(dict)

        # metric_instances[(metric_name, dataset_key, data_type)] -> GenomicTensorMetric
        # one independent instance per (metric, dataset_key, data_type) combination
        # so that accumulation is never mixed across keys
        self.metric_instances: dict[tuple, GenomicTensorMetric] = {}

    def _get_channels_dict(self, pl_module):
        channels_dict = {}
        io_mappings_df = pl_module.get_io_mappings_df()
        for data_type in io_mappings_df['data_type'].unique():
            channels = io_mappings_df[io_mappings_df['data_type']==data_type]['channel'].tolist()
            channels_dict[data_type] = channels
        if len(channels_dict) == 0 and self.split_by_target_type:
            raise ValueError("split_by_target_type is True but no data types found in io_mappings.")
        return channels_dict

    def _get_or_create_instance(self, cls, dataset_key, data_type=None) -> GenomicTensorMetric:
        key = (cls.__name__, dataset_key, data_type)
        if key not in self.metric_instances:
            self.metric_instances[key] = cls()  # gin injects parameters at construction time
        return self.metric_instances[key]

    def on_validation_epoch_start(self, train, pl_module):
        for instance in self.metric_instances.values():
            instance.reset()
        io_mappings_df = pl_module.get_io_mappings_df()
        for dataset_key in io_mappings_df['dataset_key'].unique():
            if self.split_by_target_type:
                for data_type in io_mappings_df[io_mappings_df['dataset_key']==dataset_key]['data_type'].unique():
                    for cls in self.metric_classes:
                        metric_name = cls.__name__
                        self.metric_values_dict[metric_name][f"{dataset_key}_{data_type}"] = [float('nan')]  # initialize with nan to sync_dist issues
            else:
                for cls in self.metric_classes:
                    metric_name = cls.__name__
                    self.metric_values_dict[metric_name][dataset_key] = [float('nan')]
    
    def on_validation_batch_end(self, train, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        targets = pl_module.targets_from_batch(batch)
        targets = targets.detach().cpu()
        dataset_key = batch['dataset_key'][0]
        predictions = outputs["predictions"].detach().cpu()
        batch_size = predictions.shape[0]
        io_mappings_df = pl_module.get_io_mappings_df()
        if self.metrics_per_sample:
            for i in range(batch_size):
                if self.split_by_target_type:
                    for data_type, dataset_channels in self._get_channels_dict(pl_module).items():
                        # check whether this channel is associated with the current sample's dataset_key
                        if data_type in io_mappings_df[io_mappings_df['dataset_key']==dataset_key]['data_type'].values:
                            if pl_module.data_types_subset is None or data_type in pl_module.data_types_subset:
                                sample_predictions = predictions[i,:,dataset_channels,:]
                                sample_targets = targets[i,:,dataset_channels,:]
                                for cls in self.metric_classes:
                                    metric_name = cls.__name__
                                    metric_value = cls()(sample_targets, sample_predictions)
                                    self.metric_values_dict[metric_name][f"{dataset_key}_{data_type}"].append(metric_value.item())
                else:
                    sample_predictions = predictions[i]
                    sample_targets = targets[i]
                    for cls in self.metric_classes:
                        metric_name = cls.__name__
                        metric_value = cls()(sample_targets, sample_predictions)
                        self.metric_values_dict[metric_name][dataset_key].append(metric_value.item())
        if self.metrics_across_dataset:
            if self.split_by_target_type:
                for data_type, channels in self._get_channels_dict(pl_module).items():
                    if data_type not in io_mappings_df[io_mappings_df['dataset_key'] == dataset_key]['data_type'].values:
                        continue
                    if pl_module.data_types_subset is not None and data_type not in pl_module.data_types_subset:
                        continue
                    t = targets[..., channels, :]
                    p = predictions[..., channels, :]
                    for cls in self.metric_classes:
                        self._get_or_create_instance(cls, dataset_key, data_type).update(t, p)
            else:
                for cls in self.metric_classes:
                    self._get_or_create_instance(cls, dataset_key).update(targets, predictions)

    def on_validation_epoch_end(self, train, pl_module):
        if self.metrics_per_sample:
            for cls in self.metric_classes:
                metric_name = cls.__name__
                for data_description, metric_values in self.metric_values_dict[metric_name].items():
                    # log the mean
                    if len(metric_values) > 0:
                        mean_metric_value = np.nanmean(metric_values)
                        # valid_values = [v for v in metric_values if not np.isnan(v)]
                        # if len(valid_values) == 0:
                        #     print(f"[Rank {train.global_rank}] WARNING: {metric_name} for {data_type} has all NaN values (n={len(metric_values)})")
                        pl_module.log(f"val/{metric_name}_mean_per_sample_{data_description}", mean_metric_value, prog_bar=True, sync_dist=True)
        if self.metrics_across_dataset:
            for (metric_name, dataset_key, data_type), instance in self.metric_instances.items():
                result = instance.compute()
                if data_type is not None:
                    log_key = f"val/{metric_name}_across_dataset_{dataset_key}_{data_type}"
                else:
                    log_key = f"val/{metric_name}_across_dataset_{dataset_key}_all"
                pl_module.log(log_key, result.item(), prog_bar=True, sync_dist=True)

        # clear per-sample accumulation for next epoch; metric_instances are reset
        # at epoch_start rather than here so state is inspectable after training ends
        self.metric_values_dict = defaultdict(dict)

class GPUMemoryLogger(Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
    def on_train_batch_start(self, train, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, train, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

    def on_validation_batch_start(self, train, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_batch_end(self, train, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

    def on_test_batch_start(self, train, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_test_batch_end(self, train, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            if batch_idx % self.log_interval == 0:
                pl_module.log("memory/train_gpu_peak_MB", peak_mem, prog_bar=False)

class CPUMemoryLogger(Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
        
    def on_train_batch_end(self, train, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval == 0:
            mem = psutil.virtual_memory()
            pl_module.log("memory/train_cpu_memory_percent", mem.percent, prog_bar=False, sync_dist=False)
    
    def on_validation_batch_end(self, train, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            mem = psutil.virtual_memory()
            pl_module.log("memory/val_cpu_memory_percent", mem.percent, prog_bar=False, sync_dist=False)

class SubmodulesGradientNormLogger(Callback):
    def __init__(self, submodule_names: list[str]):
        super().__init__()
        self.submodule_names = submodule_names
    def on_after_backward(self, train, pl_module):
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
        hp1_genome_fasta: str | None = None,
        hp2_genome_fasta: str | None = None,
        accessibility_outputs_slice: slice | list = [0],
        rna_outputs_slice: slice | list = [-1],
        log_stats: bool = True,
        upload_plots: bool = False,
        plot_methylation: bool = False,
        plot_rna: bool = False,
        methylation_exaggeration: float = 1.0,
        crop_for_accessibility: int = 0, # dummy, to support older gin configs for now
        label_bin_size: int = 128, # dummy, to support older gin configs for now
        ):
        super().__init__()
        self.hp1_cpg_file = hp1_cpg_file
        self.hp2_cpg_file = hp2_cpg_file
        self.hp1_accessibility_file = hp1_accessibility_file
        self.hp2_accessibility_file = hp2_accessibility_file
        self.hp1_rna_file = hp1_rna_file
        self.hp2_rna_file = hp2_rna_file
        self.genome = ref_genome_fasta
        if hp1_genome_fasta and hp2_genome_fasta:
            self.hp1_genome = hp1_genome_fasta
            self.hp2_genome = hp2_genome_fasta
        else:
            self.hp1_genome = self.genome
            self.hp2_genome = self.genome
        self.regions = regions
        self.accessibility_outputs_slice = accessibility_outputs_slice
        self.rna_outputs_slice = rna_outputs_slice
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

    def on_validation_epoch_end(self, train, pl_module) -> None:
        self.total_stride = pl_module.total_stride
        if train.is_global_zero:
            # Get the wandb Run (works when WandbLogger is enabled)
            run = getattr(getattr(train, "logger", None), "experiment", None)
            epoch = getattr(train, "current_epoch", -1)
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
                        start_pos = center_coord - (num_bins * pl_module.total_stride) // 2
                        positions = start_pos + np.arange(num_bins) * pl_module.total_stride

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
        hp1_accessibility_target = pl_module.crop_targets(
                self._construct_target_tensor(
                genome_track_file=self.hp1_accessibility_file,
                chromosome=chromosome,
                start=start,
                end=end,
                device=device,
            )
        ).squeeze().cpu().numpy() if self.hp1_accessibility_file is not None else None
        hp2_accessibility_target = pl_module.crop_targets(
                self._construct_target_tensor(
                genome_track_file=self.hp2_accessibility_file,
                chromosome=chromosome,
                start=start,
                end=end,
                device=device,
            )
        ).squeeze().cpu().numpy() if self.hp2_accessibility_file is not None else None
        hp1_rna_target = pl_module.crop_targets(
                self._construct_target_tensor(
                genome_track_file=self.hp1_rna_file,
                chromosome=chromosome,
                start=start,
                end=end,
                device=device,
            )
        ).squeeze().cpu().numpy() if self.hp1_rna_file is not None else None
        hp2_rna_target = pl_module.crop_targets(
                self._construct_target_tensor(
                genome_track_file=self.hp2_rna_file,
                chromosome=chromosome,
                start=start,
                end=end,
                device=device,
            )
        ).squeeze().cpu().numpy() if self.hp2_rna_file is not None else None
        with Predictor(pl_module,supplemental_outputs = {"imputed_conditioning_state_rep","true_conditioning_state_rep"}) as predictor:
            hp1_output = predictor.predict_locus(
                chromosome,
                start,
                end,
                sequence_path=self.hp1_genome,
                methylation_paths=self.hp1_cpg_file,
                channel_subset = None,
                methylation_load_kwargs = {
                    'extend_cpg_sites':True,
                    'cpg_values_rescale':0.01,
                },
            )
            hp1_accessibility_pred = hp1_output["predictions"][:, self.accessibility_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp1_rna_pred = hp1_output["predictions"][:, self.rna_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp1_pred_methylation = hp1_output["imputed_conditioning_state_rep"].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            if hp1_pred_methylation.shape!=hp1_accessibility_pred.shape:
                hp1_pred_methylation = np.ones_like(hp1_accessibility_pred)
            hp1_methylation = hp1_output["true_conditioning_state_rep"].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            if hp1_methylation.shape!=hp1_accessibility_pred.shape:
                hp1_methylation = np.ones_like(hp1_accessibility_pred)
            hp2_output = predictor.predict_locus(
                chromosome,
                start,
                end,
                sequence_path=self.hp2_genome,
                methylation_paths=self.hp2_cpg_file,
                channel_subset = None,
                methylation_load_kwargs = {
                    'extend_cpg_sites':True,
                    'cpg_values_rescale':0.01,
                },
            )
            hp2_accessibility_pred = hp2_output["predictions"][:, self.accessibility_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp2_rna_pred = hp2_output["predictions"][:, self.rna_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp2_pred_methylation = hp2_output["imputed_conditioning_state_rep"].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            if hp2_pred_methylation.shape!=hp2_accessibility_pred.shape:
                hp2_pred_methylation = np.ones_like(hp2_accessibility_pred)
            hp2_methylation = hp2_output["true_conditioning_state_rep"].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            if hp2_methylation.shape!=hp2_accessibility_pred.shape:
                hp2_methylation = np.ones_like(hp2_accessibility_pred)
        if hp1_methylation is not None and hp2_methylation is not None and hp1_methylation.ndim>1 and hp2_methylation.ndim>1:
            hp1_methylation = hp1_methylation.mean(axis=0)
            hp2_methylation = hp2_methylation.mean(axis=0)
        if hp1_pred_methylation is not None and hp2_pred_methylation is not None and hp1_pred_methylation.ndim>1 and hp2_pred_methylation.ndim>1:
            hp1_pred_methylation = hp1_pred_methylation.mean(axis=0)
            hp2_pred_methylation = hp2_pred_methylation.mean(axis=0)
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
            bin_size=self.total_stride,
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
        values = load_track(file_path=genome_track_file, contig=chromosome, start=start+crop, end=end-crop, nan_to_zero=True, bin_size=bin_size, motif=motif)
        return values, values != 0
