#!/bin/bash
#SBATCH --job-name=preprocess_methylseqnet
#SBATCH --account=fc_streets
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.err
#SBATCH --array=0-7 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:
# Define an array of config files
DATASET_SPLITS=(
    "fold0"
    "fold1"
    "fold2"
    "fold3"
    "fold4"
    "fold5"
    "fold6"
    "fold7"
    "fold0"
    "fold1"
    "fold2"
    "fold3"
    "fold4"
    "fold5"
    "fold6"
    "fold7"
)

CONFIG_FILES=(
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi128_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
    "../configs/preprocessor/preprocessor_borzoi_multimethyl.gin"
) 
# Get the config for this array task
DATASET_SPLIT=${DATASET_SPLITS[$SLURM_ARRAY_TASK_ID]}
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

source activate basenji2-pytorch
python ../methylseqnet/preprocessor.py --config $CONFIG_FILE --subset $DATASET_SPLIT --mode parallel