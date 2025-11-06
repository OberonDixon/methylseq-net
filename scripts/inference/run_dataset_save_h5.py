import argparse
from methylseqnet.inference import run_dataset_save_h5
from methylseqnet.transforms import InsertSyntheticCpG
from pathlib import Path
from functools import partial

def main(model_identifier, no_targets, dataset_type='atlas', synthetic_cpg=True):
    if dataset_type == 'atlas':
        dataset_paths = [
            {
                "atlas":f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold{fold}.h5",
            } for fold in range(8)
        ]
    elif dataset_type == 'synthetic':
        dataset_paths = [
            {
                "all":"/clusterfs/nilah/oberon/datasets/motif_insertion_test/pred.h5",
            }
        ]
    if synthetic_cpg:
        transforms = (
            partial(
                InsertSyntheticCpG,
                # modify parameters if needed
                center_window_size = 500,
                flank_width = 1000,
                center_cpg_frac = 0.05,
                flanking_cpg_frac = 0.5,
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
        run_dataset_save_h5(
            model_path=f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/temp-checkpoint.ckpt',
            mode='factorized-from-pretrained',
            dataset_path=dataset_path,
            dataset_type='multimethyl',
            output_path=dataset_dir / model_identifier / dataset_name,
            gpus = 1,
            num_workers = 4,
            no_targets = no_targets,
            transforms = transforms,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--no-targets", action='store_true', help="If set, do not include target tracks in the output H5 files.")
    parser.add_argument("--dataset-type", choices=['atlas', 'synthetic'], default='atlas', help="Type of dataset to run inference on.")
    parser.add_argument("--synthetic-cpg", action='store_true', help="If set, add synthetic CpG data.")
    args = parser.parse_args()
    main(args.model_identifier, args.no_targets, args.dataset_type, args.synthetic_cpg)