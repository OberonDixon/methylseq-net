import argparse
import json
from pathlib import Path
import gin
from collections import defaultdict
import re
import os
from io import StringIO
import ast
import re
from multiprocessing import Pool
import warnings
from typing import Type, Set
from functools import partial

import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from captum.attr import Attribution, IntegratedGradients
from lightning import Trainer

from methylseqnet.model import ConditionedSeqNN
from methylseqnet.encoding import one_hot_encode_dna
from methylseqnet.dataset import MultiMethylDataset,BaseHDF5Dataset
from methylseqnet.writers import HDF5PredictionWriter
from methylseqnet.readers import load_track
from methylseqnet.datamodule import MethylSeqDataModule
from methylseqnet.builders import SingleFastaHandler, MultiFileCpGHandler
from methylseqnet.transforms import LoaderTransform, DinucShuffleSyntheticCpG, InsertSyntheticCpG
from methylseqnet.peaks import selected_peaks_from_target

class Predictor:
    def __init__(
        self,
        model: str | Path | nn.Module,
        true_conditioning_state_weight: float | None = None,
        device: str = 'auto',
        supplemental_outputs: set = set(),
        remove_crop_for_variable_input_length: bool = False,
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if isinstance(model, nn.Module):
            self.model = model
            if remove_crop_for_variable_input_length:
                warnings.warn("Model provided directly as nn.Module; it may be unsafe to change crop settings so this will be skipped.")
        else:
            from methylseqnet.callbacks import ValidationMetricsLogger, GPUMemoryLogger, CPUMemoryLogger, HaplotypedPredLogger
            self.model = ConditionedSeqNN.load_from_checkpoint(model, map_location = self.device)
            if remove_crop_for_variable_input_length:
                self.model.crop_off_output = 0
                self.model.crop_off_conditioning_input = 0
                for module in self.model.sequence_encoder.modules():
                    if hasattr(module, 'crop'):
                        module.crop = nn.Identity()
                    if hasattr(module, 'cropping'):
                        module.cropping = nn.Identity()
        self.model.eval()
        self.supplemental_predict_outputs_at_load_time = self.model.supplemental_predict_outputs
        self.true_conditioning_state_weight_at_load_time = self.model.true_conditioning_state_weight
        self.model.supplemental_predict_outputs = supplemental_outputs
        if true_conditioning_state_weight is not None:
            self.model.true_conditioning_state_weight = true_conditioning_state_weight
        self.model.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def cuda(self, device=None):
        self.device = torch.device('cuda' if device is None else device)
        return self.to(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        return self.to(self.device)
    
    def predict_dataset(
        self,
        dataset_path: str | Path | dict[str,str | Path],
        output_path: str | Path,
        gpus: int = 1,
        num_workers: int = 4,
        no_targets: bool = False,
        **kwargs,
    ):
        data_module = MethylSeqDataModule(
            predict_dataset_file = dataset_path,
            batch_size = 1,
            **kwargs,
        )
        data_module.setup(stage="predict")

        pred_writer = HDF5PredictionWriter(output_dir=output_path, write_interval="batch", no_targets=no_targets)

        train = Trainer(
            accelerator="auto",
            devices=gpus,
            strategy="auto",
            callbacks=[pred_writer],
            logger=False,
        )

        train.predict(
            model=self.model,
            dataloaders=data_module,
            return_predictions=False,
        )

    def predict_locus(
        self,
        chromosome,
        start,
        end,
        sequence_path,
        methylation_paths,
        target_paths: tuple[tuple[str | Path] | str | Path,...] | None = None, # match with channel subset
        channel_subset: tuple[int] or None = None, # match with target_paths
        methylation_load_kwargs = {
            'extend_cpg_sites':False,
            'cpg_values_rescale':1.0,
        },
        **kwargs
    ) -> dict[str,torch.Tensor]:
        if target_paths is not None and channel_subset is not None:
            if len(target_paths) != len(channel_subset):
                warnings.warn(f"If target_paths and channel_subset are both provided, you may want them both to have the same length. Got {len(target_paths)} target paths and channel subset {channel_subset} of length {len(channel_subset)}.")
        cpg_handler = MultiFileCpGHandler(
            cpg_files=methylation_paths if isinstance(methylation_paths,list) else [methylation_paths],
            **methylation_load_kwargs,
        )
        fasta_handler = SingleFastaHandler(
            ref_genome=sequence_path,
        )
        cpg_ratio, cpg_valid = cpg_handler.load_cpg(source=chromosome,start=start,end=end)
        sequence = fasta_handler.load_sequences(source=chromosome,start=start,end=end)
        prediction_dict = self.predict_from_sequence(
            sequence = sequence,
            methylation = cpg_ratio,
            valid_cpgs = cpg_valid,
            channel_subset = channel_subset,
            **kwargs,
        )
        if target_paths is not None:
            targets = self.load_targets(
                chromosome=chromosome,
                start=start,
                end=end,
                target_paths=target_paths,
            )
            prediction_dict["targets"] = targets
        prediction_dict["specifier"] = f"{chromosome}:{start}-{end}|{channel_subset}"
        prediction_dict["output_coordinates"] = self.model.crop_targets(
            torch.arange(start, end, step=self.model.total_stride, device='cpu')
        ) + self.model.total_stride // 2  # coordinates of the center of each output bin
        INPUT_SHAPED = {"sequence", "conditioning_state", "sequence_attributions", "conditioning_state_attributions"}
        if any(key in prediction_dict for key in INPUT_SHAPED):
            prediction_dict["input_coordinates"] = torch.arange(start, end, device='cpu')


        return prediction_dict

    def predict_tiled_locus(
        self,
        chromosome,
        start,
        end,
        step,
        chunk_length,
        sequence_path,
        methylation_paths,
        target_paths = None,
        channel_subset = None,
        methylation_load_kwargs = {
            'extend_cpg_sites':False,
            'cpg_values_rescale':1.0,
        },        
    ) -> dict[str,torch.Tensor]:
        if (end - start - chunk_length) % step != 0:
            raise ValueError(
                f"Step size {step} with chunk length {chunk_length} does not evenly divide the locus length {end - start}."
                +f"Remainder (end - start - chunk_length) % step = {(end - start - chunk_length) % step} != 0.")
        predictions_dict_list = []    
        for chunk_start in tqdm(range(start, end - chunk_length + 1, step), leave=False, desc=f"Predicting {chromosome}:{start}-{end} in tiles"):
            chunk_end = chunk_start + chunk_length
            prediction_chunk_dict = self.predict_locus(
                chromosome=chromosome,
                start=chunk_start,
                end=chunk_end,
                sequence_path=sequence_path,
                methylation_paths=methylation_paths,
                target_paths=target_paths,
                channel_subset=channel_subset,
                methylation_load_kwargs=methylation_load_kwargs,
            )
            predictions_dict_list.append(prediction_chunk_dict)
        prediction_dict = {
            key: torch.cat([d[key] for d in predictions_dict_list], dim=-1) for key in predictions_dict_list[0].keys() if key != "specifier"
        }
        prediction_dict["specifier"] = f"{chromosome}:{start}-{end}|{channel_subset}"

        return prediction_dict

    def predict_from_sequence(
        self,
        sequence: str,
        methylation: np.ndarray,
        valid_cpgs: np.ndarray,
        channel_subset = None,
        **kwargs,
    ) -> dict[str,torch.Tensor]:
        x_methylseq = torch.permute(
            torch.tensor(
                one_hot_encode_dna(
                    dna_strand=sequence, 
                    cpg_methylation=methylation, 
                    valid_cpgs=valid_cpgs,
                ),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            (0,2,1),
        )
        sequence = x_methylseq[:, :4, :]
        methylation = x_methylseq[:, 4:, :]
        return self.predict_from_tensors(
            sequence_tensor=sequence,
            methylation_tensor=methylation,
            channel_subset=channel_subset,
            **kwargs,
        )

    def predict_from_tensors(
        self,
        sequence_tensor,
        methylation_tensor,
        channel_subset = None,
        capture_attributions: bool = False,
        attribution_class: Type[Attribution] = IntegratedGradients,
        attribution_peak_kwargs={"peak_threshold": 10},
        attribution_constructor_kwargs: dict = {},
        attribution_baseline_kwargs: dict = {"attribution_baselines_per_sample": 5},
        attribution_kwargs: dict = {},
    ) -> dict[str,torch.Tensor]:
        if methylation_tensor.ndim == 3:
            methylation_tensor = methylation_tensor.unsqueeze(1)
        with torch.no_grad():
            output = self.model(sequence_tensor.to(self.device), methylation_tensor.to(self.device))
            if channel_subset is not None:
                output = output[:,channel_subset,:]
        prediction_dict = {
            "predictions": output.cpu(),
            **{key: value.cpu() for key, value in self.model.hooked_supplemental_outputs.items()},
        }
        self.model.hooked_supplemental_outputs.clear()
        if capture_attributions:
            peak_positions, peak_weights = self._find_peaks(
                predictions=output,
                **attribution_peak_kwargs,
            )
            sequence_baselines, conditioning_baselines = self._build_attribution_baselines(
                sequence=sequence_tensor,
                conditioning_state=methylation_tensor,
                **attribution_baseline_kwargs,
            )
            attribution_dict = self.compute_attributions(
                sequence=sequence_tensor.to(self.device),
                conditioning_state=methylation_tensor.to(self.device),
                sequence_baselines=[sequence_baseline.to(self.device) for sequence_baseline in sequence_baselines],
                conditioning_baselines=[conditioning_baseline.to(self.device) for conditioning_baseline in conditioning_baselines],
                channel_subset=channel_subset,
                positions=peak_positions.to(self.device),
                weights=peak_weights.to(self.device),
                attribution_class=attribution_class,
                attribution_constructor_kwargs=attribution_constructor_kwargs,
                attribution_kwargs=attribution_kwargs,
            )
            prediction_dict.update(attribution_dict)
        return prediction_dict

    def load_targets(
        self,
        chromosome,
        start,
        end,
        target_paths,
        **kwargs,
    ):
        # TODO: builders::MultiFileLabelHandler should exist and be used here
        targets_list_by_channel = []
        for target_path_for_channel in target_paths:
            targets_list = [
                load_track(
                    file_path=target_path,
                    contig=chromosome,
                    start=start,
                    end=end,
                    nan_to_zero=True,
                    bin_size=self.model.total_stride,
                    **kwargs
                ) for target_path in (target_path_for_channel if isinstance(target_path_for_channel,list) else [target_path_for_channel])]
            targets_list_by_channel.append(
                self.model.crop_targets(
                    torch.tensor(
                        np.stack(targets_list, axis=0), dtype=torch.float32
                    ).mean(dim=0, keepdim=True)
                ).unsqueeze(1)  # (1, 1, L)
            )
        targets = torch.cat(
            targets_list_by_channel,
            dim=1,
        ) # (1, n_targets, L)
        return targets

    def compute_attributions(
        self,
        sequence: torch.Tensor,           # (B, C, L)
        conditioning_state: torch.Tensor, # (B, states, C, L)
        sequence_baselines: list[torch.Tensor], # list of (B, C, L) tensors
        conditioning_baselines: list[torch.Tensor], # list of (B, states, C, L) tensors
        channel_subset: Set[int] | list[int] | None, # if not None, subset of output channels to attribute over (e.g., {0,2} to attribute over channels 0 and 2 and ignore channel 1)
        positions: torch.Tensor | None = None,  # (n_positions,) — integer indices, indexes L
        weights: torch.Tensor | None = None,    # (n_positions,) — same length as positions
        attribution_class: Type[Attribution] = IntegratedGradients,
        attribution_constructor_kwargs: dict = {},
        attribution_kwargs: dict = {},
    ) -> dict[str, np.ndarray]:
        """
        Returns dict with 'sequence_attributions'      (B, C, L)
                    and 'conditioning_state_attributions' (B, states, C, L)
        """
        if weights is not None and sum(weights) == 0:
            warnings.warn("All attribution weights are zero, skipping attribution computation and returning zero attributions.")
            return {
                'sequence_attributions': torch.zeros_like(sequence).detach().cpu(),
                'conditioning_state_attributions': torch.zeros_like(conditioning_state).detach().cpu(),
            }           
        
        if len(sequence_baselines) == 0 or len(conditioning_baselines) == 0:
            raise ValueError("Attribution baselines must be provided for both sequence and conditioning state.")

        sequence = sequence.requires_grad_(True)
        conditioning_state = conditioning_state.requires_grad_(True)

        saved_supplemental = self.model.supplemental_predict_outputs
        self.model.supplemental_predict_outputs = set()
        try:
            wrapper = self._make_attribution_wrapper(channel_subset, positions, weights)
            dl = attribution_class(wrapper,**attribution_constructor_kwargs)
            all_seq_attrs = []
            all_cond_attrs = []
            for sequence_baseline, conditioning_baseline in zip(sequence_baselines, conditioning_baselines):
                seq_attr, cond_attr = dl.attribute(
                    inputs=(sequence, conditioning_state),
                    baselines=(sequence_baseline, conditioning_baseline),
                    **attribution_kwargs,
                )
                all_seq_attrs.append(seq_attr)
                all_cond_attrs.append(cond_attr)
            seq_attr = torch.stack(all_seq_attrs).mean(0)
            cond_attr = torch.stack(all_cond_attrs).mean(0)
        finally:
            self.model.supplemental_predict_outputs = saved_supplemental

        return {
            'sequence_attributions': seq_attr.detach().cpu(),
            'conditioning_state_attributions': cond_attr.detach().cpu(),
        }

    def _find_peaks(
        self,
        predictions,
        peak_threshold: float | None,
        min_peak_distance_bins: int = 128,
        weight_peaks_by_magnitude: bool = False,
    ):
        if peak_threshold is None:
            peak_positions = torch.arange(predictions.shape[-1], dtype=torch.int)
        else:
            peak_positions = torch.tensor(
                selected_peaks_from_target(
                    target = predictions.mean(dim=1).squeeze(0).cpu().numpy(),
                    peak_threshold = peak_threshold,
                    min_peak_distance_bins = 128,
                ),
                dtype=torch.long,
            )
        if weight_peaks_by_magnitude:
            peak_weights = predictions.mean(dim=1).squeeze(0)[peak_positions]
        else:
            peak_weights = torch.ones_like(peak_positions, dtype=torch.float32)
        return peak_positions, peak_weights     
    
    def _build_attribution_baselines(
        self,
        sequence: torch.Tensor,           # (B, C, L)
        conditioning_state: torch.Tensor, # (B, states, C, L)
        attribution_baselines_per_sample: int = 1,
        attribution_baseline_transform: LoaderTransform | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if attribution_baseline_transform is not None:
            baseline_transform = attribution_baseline_transform
        else:
            baseline_transform = DinucShuffleSyntheticCpG(
                cpg_frac = 0.95,
            )
        sequence_baselines, conditioning_baselines = zip(*[
            baseline_transform(sequence, conditioning_state, torch.zeros(0), torch.zeros(0))[:2]
            for _ in range(attribution_baselines_per_sample)
        ])
        return sequence_baselines, conditioning_baselines

    def _make_attribution_wrapper(
        self,
        channel_subset: Set[int] | None = None,
        positions: torch.Tensor | None = None,  # (n_positions,) — integer indices
        weights: torch.Tensor | None = None,    # (n_positions,) — same length as positions
    ) -> nn.Module:
        model = self.model

        class _Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, sequence, conditioning_state):
                preds = model(sequence, conditioning_state)
                if channel_subset is not None:
                    track = preds[:, channel_subset, :].mean(dim=1)  # (B, L)
                else:
                    track = preds.mean(dim=1)  # (B, L)
                if positions is not None:
                    track = track[:, positions]  # (B, n_positions)
                if weights is not None:
                    # weights should be (n_positions,), broadcasts to (B, n_positions)
                    return (track * weights).sum(dim=-1)  # (B,)
                return track.sum(dim=-1)  # (B,)
        
        return _Wrapper(self.model).eval()

    def _restore_model_state(self):
        self.model.supplemental_predict_outputs = self.supplemental_predict_outputs_at_load_time
        self.model.true_conditioning_state_weight = self.true_conditioning_state_weight_at_load_time

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_model_state()
        return False

    def __del__(self):
        self._restore_model_state()

def main():
    DEFAULT_SUPPLEMENTAL_OUTPUTS = [
        "conditional_seq_rep",
        "unconditional_seq_rep",
        "true_conditioning_state_rep",
        "imputed_conditioning_state_rep",
        "cpg_density",
    ]
    parser = argparse.ArgumentParser(description="Run predictions with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--true-conditioning-state-weight", type=float, default=None, help="If set, override the model's true_conditioning_state_weight with this value for prediction.")
    parser.add_argument("--no-targets", action='store_true', help="If set, do not include target tracks in the output H5 files.")
    parser.add_argument("--dataset-keys", nargs='+', required=True, help="Dataset keys (e.g., atlas, longread) corresponding to the datasets being predicted on (must match length of --dataset-files)")
    parser.add_argument("--dataset-files", nargs='+', required=True, help="Paths to dataset H5 files (must match length of --dataset-keys)")
    parser.add_argument("--supplemental-outputs", nargs="*", required=False, default=DEFAULT_SUPPLEMENTAL_OUTPUTS, help=f"Supplemental outputs to include in predictions. Default: {DEFAULT_SUPPLEMENTAL_OUTPUTS}")
    parser.add_argument("--synthetic-cpg", action='store_true', help="If set, add synthetic CpG data.")
    parser.add_argument("--variable-input-length", action='store_true', help="If set, sequence length can be any integer multiple of 128 that is >=16384.")
    parser.add_argument("--center-methyl-frac", type=float, default=0.05, help="Fraction of CpGs methylated in the center window.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers")
    
    args = parser.parse_args()    

    if len(args.dataset_keys) != len(args.dataset_files):
        parser.error(f"--dataset-keys ({len(args.dataset_keys)}) and --dataset-files ({len(args.dataset_files)}) must have the same number of arguments")

    dataset_paths = [
        {dataset_key:dataset_file} for dataset_key, dataset_file in zip(args.dataset_keys, args.dataset_files)
    ]

    if args.synthetic_cpg:
        transforms = (
            partial(
                InsertSyntheticCpG,
                # modify parameters if needed
                center_window_size = 500,
                flank_width = 1000,
                center_cpg_frac = args.center_methyl_frac,
                flanking_cpg_frac = (args.center_methyl_frac + 0.95)/2,
                background_cpg_frac = 0.95,
                offset = 0,
            ),
        )
    else:
        transforms = ()

    for dataset_path in dataset_paths:
        dataset_name = Path(list(dataset_path.values())[0]).stem
        dataset_dir = Path(list(dataset_path.values())[0]).parent
        print(f"Running through {dataset_name}.")
        if args.synthetic_cpg:
            output_path = dataset_dir / args.model_identifier / f"{dataset_name}_synthetic_{args.center_methyl_frac}"
        elif args.true_conditioning_state_weight is not None:
            output_path = dataset_dir / args.model_identifier / f"{dataset_name}_truecondweight_{args.true_conditioning_state_weight}"
        else:
            output_path = dataset_dir / args.model_identifier / dataset_name
        best_ckpt = max(
            Path(f"/clusterfs/nilah/oberon/lightning/{args.model_identifier}/checkpoints/").glob('best*.ckpt'),
            key=lambda p: p.stat().st_mtime
        )   
        temp_ckpt = f'/clusterfs/nilah/oberon/lightning/{args.model_identifier}/checkpoints/temp-checkpoint.ckpt'

        predictor = Predictor(
            model=best_ckpt,
            true_conditioning_state_weight=args.true_conditioning_state_weight,
            supplemental_outputs = set(args.supplemental_outputs),
            remove_crop_for_variable_input_length = args.variable_input_length,
        )
        predictor.predict_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            gpus = args.gpus,
            num_workers = args.num_workers,
            no_targets = args.no_targets,
            transforms = transforms,
        )

if __name__ == "__main__":
    main()