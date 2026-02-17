import argparse
import torch
from torch import nn
import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import gin
from collections import defaultdict
import re
import methylseqnet
import os
from lightning import Trainer
import pandas as pd
from io import StringIO
import ast
import re
from multiprocessing import Pool
import warnings

from lightning import Trainer

from methylseqnet.model import ConditionedSeqNN
from methylseqnet.encoding import one_hot_encode_dna
from methylseqnet.dataset import MultiMethylDataset,BaseHDF5Dataset
from methylseqnet.writers import HDF5PredictionWriter
from methylseqnet.datamodule import MethylSeqDataModule
from methylseqnet.builders import SingleFastaHandler, MultiFileCpGHandler
from methylseqnet.transforms import InsertSyntheticCpG

class Predictor:
    def __init__(
        self,
        model: str | Path | nn.Module,
        device: str = 'auto',
        supplemental_outputs: set = set(),
        remove_crop_for_variable_input_length: bool = False,
    ):
        if isinstance(model, nn.Module):
            self.model = model
            if remove_crop_for_variable_input_length:
                warnings.warn("Model provided directly as nn.Module; it may be unsafe to change crop settings so this will be skipped.")
        else:
            self.model = ConditionedSeqNN.load_from_checkpoint(model)
            if remove_crop_for_variable_input_length:
                self.model.crop_off_output = 0
                self.model.crop_off_conditioning_input = 0
                try:
                    self.model.pretrained_seq_model.crop = nn.Identity()
                except:
                    pass
        self.model.eval()
        self.supplemental_predict_outputs_at_load_time = self.model.supplemental_predict_outputs
        self.model.supplemental_predict_outputs = supplemental_outputs
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

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
        channel_subset = None,
        methylation_load_kwargs = {
            'extend_cpg_sites':False,
            'cpg_values_rescale':1.0,
        },
    ) -> dict[str,torch.Tensor]:
        cpg_handler = MultiFileCpGHandler(
            cpg_files=methylation_paths if isinstance(methylation_paths,list) else [methylation_paths],
            **methylation_load_kwargs,
        )
        fasta_handler = SingleFastaHandler(
            ref_genome=sequence_path,
        )
        cpg_ratio, cpg_valid = cpg_handler.load_cpg(source=chromosome,start=start,end=end)
        sequence = fasta_handler.load_sequence(source=chromosome,start=start,end=end)
        prediction_dict = self.predict_from_sequence(sequence, cpg_ratio, cpg_valid, channel_subset)
        prediction_dict["specifier"] = f"{chromosome}:{start}-{end}|{channel_subset}"

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
        channel_subset = None,
        methylation_load_kwargs = {
            'extend_cpg_sites':False,
            'cpg_values_rescale':1.0,
        },        
    ) -> dict[str,torch.Tensor]:
        if (end - start) % step != (chunk_length - step):
            raise ValueError(f"Step size {step} with chunk length {chunk_length} does not evenly divide the locus length {end - start}.")
        predictions_dict_list = []    
        for chunk_start in range(start, end, step):
            chunk_end = chunk_start + chunk_length
            prediction_chunk_dict = self.predict_locus(
                chromosome=chromosome,
                start=chunk_start,
                end=chunk_end,
                sequence_path=sequence_path,
                methylation_paths=methylation_paths,
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
    ) -> dict[str,torch.Tensor]:
        x_methylseq = torch.permute(
            torch.tensor(
                encoding.one_hot_encode_dna(
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
        return self.predict_from_tensors(sequence, methylation, channel_subset)

    def predict_from_tensors(
        self,
        sequence_tensor,
        methylation_tensor,
        channel_subset = None,
    ) -> dict[str,torch.Tensor]:
        if methylation_tensor.ndim == 3:
            methylation_tensor = methylation_tensor.unsqueeze(1)
        with torch.no_grad():
            output = self.model(sequence_tensor.to(self.device), methylation_tensor.to(self.device))
            if channel_subset is not None:
                output = output[:,channel_subset,:]
        prediction_dict = {
            "predictions": output.cpu(),
            **self.model.hooked_supplemental_outputs,
        }
        self.model.hooked_supplemental_outputs.clear()
        return prediction_dict

    def load_targets(
        self,
        chromosome,
        start,
        end,
        label_paths,
    ):
        pass

    def _restore_supplemental_outputs(self):
        self.model.supplemental_predict_outputs = self.supplemental_predict_outputs_at_load_time

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_supplemental_outputs()
        return False

    def __del__(self):
        self._restore_supplemental_outputs()

def main():
    parser = argparse.ArgumentParser(description="Run predictions with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--no-targets", action='store_true', help="If set, do not include target tracks in the output H5 files.")
    parser.add_argument("--dataset-keys", nargs='+', required=True, help="Dataset keys (e.g., atlas, longread) corresponding to the datasets being predicted on (must match length of --dataset-files)")
    parser.add_argument("--dataset-files", nargs='+', required=True, help="Paths to dataset H5 files (must match length of --dataset-keys)")
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
            output_path = dataset_dir / args.model_identifier / f"{dataset_name}_synthetic_{center_methyl_frac}"
        else:
            output_path = dataset_dir / args.model_identifier / dataset_name
        best_ckpt = max(
            Path(f"/clusterfs/nilah/oberon/lightning/{args.model_identifier}/checkpoints/").glob('best*.ckpt'),
            key=lambda p: p.stat().st_mtime
        )   
        temp_ckpt = f'/clusterfs/nilah/oberon/lightning/{args.model_identifier}/checkpoints/temp-checkpoint.ckpt'

        predictor = Predictor(
            model=best_ckpt,
            supplemental_outputs = {"conditional_seq_rep","unconditional_seq_rep","true_conditioning_state_rep","imputed_conditioning_state_rep","cpg_density"},
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