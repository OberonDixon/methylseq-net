import argparse
from methylseqnet.predict import run_dataset_save_h5

def main(model_identifier):
    for hp in ["HP1","HP2"]:
        print(f"Running {hp}.")
        run_dataset_save_h5(
            model_path=f'/clusterfs/nilah/oberon/lightning/{model_identifier}/checkpoints/best-checkpoint.ckpt',
            mode='full-model',
            dataset_path=f"/global/scratch/users/dixonluinenburg/atlas_datasets/dimelo-Xinactivation/{hp}.h5",
            dataset_type='methylseq',
            output_path=f"/global/scratch/users/dixonluinenburg/atlas_datasets/dimelo-Xinactivation/predict/{model_identifier}/{hp}",
            gpus = 1,
            num_workers = 4,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predict with a specified model.")
    parser.add_argument("--model-identifier", required=True, help="e.g. slurm24807693task2; will reference to /clusterfs/nilah/oberon/lightning/")
    args = parser.parse_args()
    main(args.model_identifier)