import h5py
import os
from abc import ABC, abstractmethod
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import pysam
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from scipy.stats import pearsonr, spearmanr
import torch
import wandb

from dimelo import load_processed
from methylseqnet import dna_io

class BaseHDF5Writer(ABC):
    def __init__(
        self,
        output_dir=None,
        io_mappings_str="",
    ):
        self.io_mappings_str = io_mappings_str
        self.file_handles = {}
        self.pred_counter = 0
        if output_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = self._temp_dir_obj.name
        else:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def append_batch_to_h5(self, trainer, pl_module, predictions, specifiers, batch_indices, batch):
        inputs = batch[0]
        targets = batch[1]
        # TODO: make this work in the case where inputs contains embeddings for pretrained
        targets = pl_module.trim_targets(inputs,targets)
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
            self.file_handles[path].attrs['io_mappings'] = self.io_mappings_str

        f = self.file_handles[path]
        batch_indices = np.array(batch_indices)
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        curr_size = f["predictions"].shape[0]

        # Resize datasets
        f["predictions"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["predictions"][batch_indices] = predictions_np

        f["tracks"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["tracks"][batch_indices] = targets_np

        f["indices"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["indices"][batch_indices] = batch_indices

        f["specifier"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["specifier"][batch_indices] = specifiers

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
        io_mappings_str="",
    ):
        BasePredictionWriter.__init__(self,write_interval)
        BaseHDF5Writer.__init__(self,output_dir,io_mappings_str)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        self.append_batch_to_h5(trainer, pl_module, prediction["predictions"], prediction["specifiers"], batch_indices, batch)

    def on_predict_end(self, trainer, pl_module):
        self._close_all()

class ValidationMetricsLogger(Callback, BaseHDF5Writer):
    def __init__(
        self,
        io_mappings_str="",
        split_by_target_type=True,
        metrics=[],
        in_memory=True,
    ):
        Callback.__init__(self)
        self.in_memory = in_memory
        BaseHDF5Writer.__init__(self, output_dir=None, io_mappings_str=io_mappings_str)
        self.split_by_target_type = split_by_target_type
        self.metrics = metrics

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.in_memory:
            pass
            # append to in-memory structure
        else:
            pass
            # call self.append_batch_to_h5

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.in_memory:
            # compute metrics from in-memory structure
            pass
        else:
            # computer metrics from h5 files
            pass


class GPUMemoryLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
            pl_module.log("train/gpu_peak_MB", peak_mem, prog_bar=False)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            pl_module.log("val/gpu_peak_MB", peak_mem, prog_bar=False)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            pl_module.log("test/gpu_peak_MB", peak_mem, prog_bar=False)

class HaplotypedPredLogger(Callback):
    def __init__(
        self,
        hp1_cpg_bedgz: str,
        hp2_cpg_bedgz: str,
        hp1_accessibility_bedgz: str,
        hp2_accessibility_bedgz: str,
        ref_genome_fasta : str,
        regions: list[tuple[str, int, int]],
        model_outputs_slice: slice = slice(None),
        crop_for_accessibility: int = 163840,
        label_bin_size: int = 128,
        log_stats: bool = True,
        upload_plots: bool = False,
        ):
        super().__init__()
        self.hp1_cpg_bedgz = hp1_cpg_bedgz
        self.hp2_cpg_bedgz = hp2_cpg_bedgz
        self.hp1_accessibility_bedgz = hp1_accessibility_bedgz
        self.hp2_accessibility_bedgz = hp2_accessibility_bedgz
        self.genome = pysam.FastaFile(ref_genome_fasta)
        self.regions = regions
        self.model_outputs_slice = model_outputs_slice
        self.crop_for_accessibility = crop_for_accessibility
        self.label_bin_size = label_bin_size
        self.log_stats = log_stats
        self.upload_plots = upload_plots
        if not (os.path.exists(hp1_cpg_bedgz) and os.path.exists(hp2_cpg_bedgz) and os.path.exists(ref_genome_fasta)):
            raise ValueError("One of the provided haplotype-specific cpg bed files or reference genome fasta does not exist.")
        if (hp1_accessibility_bedgz is not None) != (hp2_accessibility_bedgz is not None):
            raise ValueError("Either both or neither haplotype-specific accessibility bed files must be provided.")
        if (hp1_accessibility_bedgz is not None) and (not (os.path.exists(hp1_accessibility_bedgz) and os.path.exists(hp2_accessibility_bedgz))):
            raise ValueError("One of the provided haplotype-specific accessibility bed files does not exist.")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            # Get the wandb Run (works when WandbLogger is enabled)
            run = getattr(getattr(trainer, "logger", None), "experiment", None)
            epoch = getattr(trainer, "current_epoch", -1)
            if run is not None:
                images, hp1_pearsons, hp2_pearsons, differential_pearsons = [], [], [], []
                for chromosome, start, end in self.regions:
                    hp1_target, hp2_target, hp1_pred, hp2_pred = self._compute_haplo_pred_stats(pl_module,chromosome,start,end)
                    if self.upload_plots:
                        region_str = f"{chromosome}:{start}-{end}"
                        fig, axes = plt.subplots(4,1,figsize=(20,10), sharex=True)
                        fig.suptitle(f"{region_str}, epoch={epoch}")
                        axes[0].plot(hp1_pred, label="Haplo 1 Prediction", color='blue', alpha=0.5)
                        axes[0].plot(hp2_pred, label="Haplo 2 Prediction", color='orange', alpha=0.5)
                        axes[0].set_ylabel("Haplo 1/2 Prediction")
                        axes[1].plot(hp1_pred - hp2_pred, label="Haplo 1 - Haplo 2 Prediction", color='green', alpha=0.5)
                        axes[1].set_ylabel("Haplo 1 minus Haplo 2 Prediction")
                        if hp1_target is not None:
                            axes[2].plot(hp1_target, label="Haplo 1 Target", color='blue', alpha=0.5)
                            axes[2].plot(hp2_target, label="Haplo 2 Target", color='orange', alpha=0.5)
                            axes[2].set_ylabel("Haplo 1/2 Target")
                            axes[3].plot(hp1_target - hp2_target, label="Haplo 1 - Haplo 2 Target", color='green', alpha=0.5)
                            axes[3].set_ylabel("Haplo 1 minus Haplo 2 Target")
                        fig.canvas.draw()  # guarantee the figure is rendered NOW
                        w, h = fig.canvas.get_width_height()
                        run.log({f"haplo/phased_plots_{region_str}": wandb.Image(fig, caption=f"epoch={epoch}"), "epoch": epoch})
                        plt.close(fig)
                    if self.log_stats and (hp1_target is not None):
                        hp1_pearsons.append(pearsonr(hp1_target, hp1_pred)[0])
                        hp2_pearsons.append(pearsonr(hp2_target, hp2_pred)[0])
                        differential_pearsons.append(pearsonr(hp1_target - hp2_target, hp1_pred - hp2_pred)[0])
                if self.log_stats and (hp1_target is not None):
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
        hp1_input = self._construct_input_tensor(
            pileup_file=self.hp1_cpg_bedgz,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        )
        hp2_input = self._construct_input_tensor(
            pileup_file=self.hp2_cpg_bedgz,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        )
        hp1_target = self._construct_target_tensor(
            pileup_file=self.hp1_accessibility_bedgz,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp1_accessibility_bedgz is not None else None
        hp2_target = self._construct_target_tensor(
            pileup_file=self.hp2_accessibility_bedgz,
            chromosome=chromosome,
            start=start,
            end=end,
            device=device,
        ).squeeze().cpu().numpy() if self.hp2_accessibility_bedgz is not None else None
        with torch.no_grad():
            training_mode = pl_module.mode
            pl_module.eval()
            pl_module.mode = 'full-model'
            hp1_pred = pl_module(hp1_input)[:, self.model_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            hp2_pred = pl_module(hp2_input)[:, self.model_outputs_slice, :].mean(dim=1, keepdim=True).squeeze().cpu().numpy()
            pl_module.mode = training_mode
        return hp1_target, hp2_target, hp1_pred, hp2_pred


    def _construct_input_tensor(
        self,
        pileup_file,
        chromosome,
        start,
        end,
        device,
    ) -> torch.Tensor:
        cpg_ratio, non_zero_mask = self._load_methyl_ratio_from_bedgz(
            pileup_file=pileup_file,
            motif="CG,0",
            chromosome=chromosome,
            start=start,
            end=end,
            bin_size=1,
            crop=0,
        )
        return torch.permute(
            torch.tensor(
                dna_io.one_hot_encode_dna(
                    dna_strand=self.genome.fetch(chromosome,start,end), 
                    cpg_methylation=cpg_ratio, 
                    valid_cpgs=non_zero_mask,
                ),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            (0,2,1),
        )

    def _construct_target_tensor(
        self,
        pileup_file,
        chromosome,
        start,
        end,
        device,
    ) -> torch.Tensor:
        accessibility_ratio, non_zero_mask = self._load_methyl_ratio_from_bedgz(
            pileup_file=pileup_file,
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

    def _load_methyl_ratio_from_bedgz(
        self,
        pileup_file,
        motif,
        chromosome,
        start,
        end,
        bin_size=1,
        crop=0,
    ) -> np.ndarray:
        mod_vector, val_vector = load_processed.pileup_vectors_from_bedmethyl(
            bedmethyl_file = pileup_file,
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
