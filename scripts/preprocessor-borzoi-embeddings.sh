#!/bin/bash
#SBATCH --job-name=preprocess-borzoi-embeddings
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.err
#SBATCH --array=0-7 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
#SBATCH --exclude=n0174.savio3,n0175.savio3,n0176.savio3

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
OUTPUT_DIR="/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-embeddings-lzf/"
# Check if this is a requeued job by examining SLURM_RESTART_COUNT
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    EXTRA_ARGS="--append-to-existing"
else
    EXTRA_ARGS=""
fi

source activate basenji2-pytorch
python ../methylseqnet/preprocessor.py pretrained_model_embeddings --config $CONFIG_FILE --embeddings-shape 1920 6144 --input-datasets-directory $INPUT_DIR --output-datasets-directory $OUTPUT_DIR --subset $FOLD --batch-size 2 --write-batch-size 16 $EXTA_ARGS