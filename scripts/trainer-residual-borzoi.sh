#!/bin/bash
#SBATCH --job-name=train_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --requeue
#SBATCH --time=7:15:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --array=0-3 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/residual/borzoi_pretrain_head_only.gin"
    "../configs/residual/borzoi_pretrain_methylseq-residual_multiply.gin"
    "../configs/residual/borzoi_pretrain_seq-residual_multiply.gin"
    "../configs/residual/borzoi_pretrain_smoothed-methyl-residual_multiply.gin")

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/trainer.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1