import argparse
from methylseqnet.inference import run_dataset_save_h5

def main(model_identifier):
    for fold in range(8):
        print(f"Running through fold{fold}.")
        run_dataset_save_h5(
            model_path=f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/best-checkpoint-train-residual-only.ckpt',
            mode='residual-w/-pretrained-embeddings',
            dataset_path=(
                f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold{fold}.h5",
                f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-embeddings/fold{fold}.h5",
            ),
            dataset_type='multimethyl-and-embeddings',
            output_path=f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/inference/{model_identifier}/fold{fold}",
            gpus = 1,
            num_workers = 4,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    args = parser.parse_args()
    main(args.model_identifier)