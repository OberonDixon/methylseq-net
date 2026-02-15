import argparse
from methylseqnet.predict import run_dataset_save_h5
from methylseqnet.transforms import InsertSyntheticCpG
from pathlib import Path
from functools import partial
import h5py
import numpy as np

def main(model_identifier, gpus, mode='factorized-from-pretrained', checkpoint_type='best'):
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
    supplemental_predict_outputs = {"conditional_seq_rep","unconditional_seq_rep","true_conditioning_state_rep","imputed_conditioning_state_rep","cpg_density","sequence_embedding"}
    for dataset_path in dataset_paths:
        dataset_name = Path(list(dataset_path.values())[0]).stem
        dataset_dir = Path(list(dataset_path.values())[0]).parent
        output_path = dataset_dir / model_identifier / f'{dataset_name}_{checkpoint_type}'
        print(f"Running through {dataset_name}, saving to {output_path}.")
        if checkpoint_type == 'best':
            ckpt_path = max(
                Path(f"/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/").glob('best*.ckpt'),
                key=lambda p: p.stat().st_mtime
            ) 
        elif checkpoint_type == 'temp':
            ckpt_path = f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/temp-checkpoint.ckpt'
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
        run_dataset_save_h5(
            model_path=ckpt_path,
            mode=mode,
            dataset_path=dataset_path,
            dataset_type='multimethyl',
            output_path=output_path,
            gpus = gpus,
            num_workers = 4,
            no_targets = no_targets,
            transforms = (),
            supplemental_predict_outputs=supplemental_predict_outputs,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predict with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for predict.")
    parser.add_argument("--mode", choices=['factorized-from-pretrained', 'pretrained-only'], default='factorized-from-pretrained', help="Mode of predict.")
    parser.add_argument("--checkpoint-type", choices=['best', 'temp'], default='best', help="Type of checkpoint to use.")
    args = parser.parse_args()
    main(args.model_identifier, args.gpus, args.mode, args.checkpoint_type)