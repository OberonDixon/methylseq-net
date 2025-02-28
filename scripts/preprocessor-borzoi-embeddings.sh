#!/bin/bash
#SBATCH --job-name=train_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.err
#SBATCH --array=7-7 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:
# Define an array of config files
FOLDS=(
    "fold0"
    "fold1"
    "fold2"
    "fold3"
    "fold4"
    "fold5"
    "fold6"
    "fold7"
    )
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}
CONFIG_FILE="../configs/residual/borzoi_pretrain_head_only.gin"
INPUT_DIR="/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128-multimethyl-bisulfite-atac-cage/"
OUTPUT_DIR="/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-embeddings-rerun/"

source activate basenji2-pytorch
python ../methylseqnet/preprocessor.py pretrained_model_embeddings --config $CONFIG_FILE --embeddings-shape 1920 6144 --input-datasets-directory $INPUT_DIR --output-datasets-directory $OUTPUT_DIR --subset $FOLD --batch-size 2 --write-batch-size 16