import argparse
from methylseqnet.inference import run_dataset_save_h5
from methylseqnet.transforms import InsertSyntheticCpG
from pathlib import Path
from functools import partial

def main(model_identifier, no_targets, dataset_type='atlas', dataset_path=None, synthetic_cpg=True, center_methyl_frac=0.05, variable_input_length=False):
    if dataset_type == 'atlas':
        if dataset_path is not None:
            dataset_paths = [{"atlas":dataset_path}]
        else:
            dataset_paths = [
                {
                    "atlas":f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold{fold}.h5",
                } for fold in range(8)
            ]
    elif dataset_type == 'synthetic':
        if dataset_path is not None:
            dataset_paths = [{"all":dataset_path}]
        else:
            dataset_paths = [
                {
                    "all":"/clusterfs/nilah/oberon/datasets/motif_insertion_test/motif_insertions.h5",
                }
            ]
    if synthetic_cpg:
        transforms = (
            partial(
                InsertSyntheticCpG,
                # modify parameters if needed
                center_window_size = 500,
                flank_width = 1000,
                center_cpg_frac = center_methyl_frac,
                flanking_cpg_frac = (center_methyl_frac + 0.95)/2,
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
        if synthetic_cpg:
            output_path = dataset_dir / model_identifier / f"{dataset_name}_synthetic_{center_methyl_frac}"
        else:
            output_path = dataset_dir / model_identifier / dataset_name
        best_ckpt = max(
            Path(f"/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/").glob('best*.ckpt'),
            key=lambda p: p.stat().st_mtime
        )   
        temp_ckpt = f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/temp-checkpoint.ckpt'

        print(f"Using model checkpoint: {best_ckpt}")
        print(f"Saving outputs to: {output_path}")

        run_dataset_save_h5(
            model_path=best_ckpt,
            mode='factorized-from-pretrained',
            dataset_path=dataset_path,
            dataset_type='multimethyl',
            output_path=output_path,
            gpus = 1,
            num_workers = 8,
            no_targets = no_targets,
            transforms = transforms,
            variable_input_length=variable_input_length,
            supplemental_predict_outputs = {"conditional_seq_rep","unconditional_seq_rep","true_conditioning_state_rep","imputed_conditioning_state_rep","cpg_density"},
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    parser.add_argument("--no-targets", action='store_true', help="If set, do not include target tracks in the output H5 files.")
    parser.add_argument("--dataset-type", choices=['atlas', 'synthetic'], default='atlas', help="Type of dataset to run inference on.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset H5 file. If not provided, defaults will be used based on dataset type.")
    parser.add_argument("--synthetic-cpg", action='store_true', help="If set, add synthetic CpG data.")
    parser.add_argument("--variable-input-length", action='store_true', help="If set, sequence length can be any integer multiple of 128 that is >=16384.")
    parser.add_argument("--center-methyl-frac", type=float, default=0.05, help="Fraction of CpGs methylated in the center window.")
    args = parser.parse_args()
    main(args.model_identifier, args.no_targets, args.dataset_type, args.dataset_path, args.synthetic_cpg, args.center_methyl_frac, args.variable_input_length)