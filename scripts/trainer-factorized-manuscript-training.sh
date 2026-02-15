#!/bin/bash
#SBATCH --job-name=train_methylseqnet_factorized
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --exclude=n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
#SBATCH --array=0-7 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/manuscript_training/borzoi-rep0_factorized_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized_atlas+longread_imputed.gin"
    "../configs/manuscript_training/borzoi-rep0_concat_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_FiLM_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_probe_atlas+longread.gin"
    "../configs/manuscript_training/basenji2-reinit_factorized_atlas+longread_true.gin"
    "../configs/manuscript_training/basenji2-reinit_factorized_atlas+longread_imputed.gin"
    "../configs/manuscript_training/borzoi-rep1_factorized_atlas+longread_true.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet
python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1