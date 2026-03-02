#!/bin/bash
#SBATCH --job-name=train_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --array=0-1 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:

# Define an array of config files
CONFIG_FILES=(
    # "../configs/train_seq-drop-cpg_lightning.gin"
    # "../configs/train_seq-cpg_lightning.gin"
#    "../configs/train_seq-cpg-weight-decay_lightning.gin"
   "../configs/train_seq-cpg-288-filters_lightning.gin"
   "../configs/train_seq-cpg-288-filters_lightning.gin"
#    "../configs/train_seq-cpg-5-channels_lightning.gin"
    # "../configs/train_seq-only_lightning.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet
NCCL_P2P_DISABLE=1 python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 8