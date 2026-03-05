#!/bin/bash
#SBATCH --job-name=train_methylseqnet_factorized
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --exclude=n0217.savio3,n0215.savio3,n0130.savio4,n0132.savio4,n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
#SBATCH --array=1,11,12,13,16,20,21,22,27,28,33 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
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
    "../configs/manuscript_training/borzoi-rep0_factorized_atlas_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized_atlas_imputed.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized1k_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized1k_atlas+longread_imputed.gin"
    "../configs/manuscript_training/basenji2_factorized_atlas+longread_true.gin"
    "../configs/manuscript_training/basenji2_factorized_atlas+longread_imputed.gin"
    "../configs/manuscript_training/borzoi-rep2_factorized_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep3_factorized_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized128_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized256_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-transformer4x256_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-transformer8x128_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-dilate1_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-dilate3_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-dilate7_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-cond1to15_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-cond1to3_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-cond3to1_atlas+longread_true.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized-cond15to1_atlas+longread_true.gin"
    "../configs/manuscript_training/bassetlike_methylseq-only_atlas+longread_onehot.gin"
    "../configs/manuscript_training/bassetlike_methylseq-only_atlas+longread_interp.gin"
    "../configs/manuscript_training/bassetlike_methylseq-only_atlas+longread_smoothed.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized_longread_imputed.gin"
    "../configs/manuscript_training/borzoi-rep0_factorized_longread_true.gin"
    "../configs/manuscript_training/bassetlike_methylseq-only_atlas+longread_smoothed1k.gin"
    "../configs/manuscript_training/bassetlike_methylseq-only_atlas+longread_smoothed8k.gin"
    "../configs/manuscript_training/bassetlike_seq-only_atlas+longread.gin"
    "../configs/manuscript_training/bassetlike_methyl-only_atlas+longread_smoothed.gin"
)

UNIQUE_IDENTIFIERS=(
    "slurm32260895task0"
    "slurm32260895task1"
    "slurm32260895task2"
    "slurm32260895task3"
    "slurm32273742task4"
    "slurm32260895task5"
    "slurm32260895task6"
    "slurm32260895task7"
    "slurm32260895task8"
    "slurm32260895task9"
    "slurm32260895task10"
    "slurm32260895task11"
    "slurm32260926task12"
    "slurm32260926task13"
    "slurm32260895task14"
    "slurm32260895task15"
    "slurm32260895task16"
    "slurm32260895task17"
    "slurm32260895task18"
    "slurm32260895task19"
    "slurm32260895task20"
    "slurm32260895task21"
    "slurm32260895task22"
    "slurm32260895task23"
    "slurm32260895task24"
    "slurm32260895task25"
    "slurm32260895task26"
    "slurm32263821task27"
    "slurm32263821task28"
    "slurm32263821task29"
    "slurm32263821task30"
    "slurm32263821task31"
    "slurm32292528task32"
    "slurm32292528task33"
    "slurm32292528task34"
    "slurm32293065task35"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER=${UNIQUE_IDENTIFIERS[$SLURM_ARRAY_TASK_ID]}
source activate methylseqnet-prod
python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1
