import argparse
from methylseqnet.inference import run_dataset_save_h5
from methylseqnet.transforms import InsertSyntheticCpG
from pathlib import Path
from functools import partial
import h5py
import numpy as np

def main(model_identifier, gpus, mode='factorized-from-pretrained'):
    dataset_paths = [
        {
            "atlas":f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold{fold}.h5",
        } for fold in range(8)
    ] + [
        {
            "longread":f"/global/scratch/projects/vector_streetslab/oberon/datasets/vollger_mendelian/preprocessed_datasets/borzoi_splits_pb_cpg/fold{fold}.h5",
        } for fold in range(8)
    ]
    no_targets = False
    supplemental_predict_outputs = {"methyl_dep_seq_rep","methyl_indep_seq_rep","true_methyl_rep","imputed_methyl_rep"}
    for dataset_path in dataset_paths:
        dataset_name = Path(list(dataset_path.values())[0]).stem
        dataset_dir = Path(list(dataset_path.values())[0]).parent
        print(f"Running through {dataset_name}, saving to {dataset_dir / model_identifier / dataset_name}.")
        run_dataset_save_h5(
            model_path=f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/temp-checkpoint.ckpt',
            mode=mode,
            dataset_path=dataset_path,
            dataset_type='multimethyl',
            output_path=dataset_dir / model_identifier / dataset_name,
            gpus = gpus,
            num_workers = 4,
            no_targets = no_targets,
            transforms = (),
            supplemental_predict_outputs=supplemental_predict_outputs,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for inference.")
    parser.add_argument("--mode", choices=['factorized-from-pretrained', 'pretrained-only'], default='factorized-from-pretrained', help="Mode of inference.")
    args = parser.parse_args()
    main(args.model_identifier, args.gpus, args.mode)