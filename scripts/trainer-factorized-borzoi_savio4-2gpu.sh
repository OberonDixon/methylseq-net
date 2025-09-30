#!/bin/bash
#SBATCH --job-name=train_methylseqnet_factorized_2gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --array=3-3 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/factorized/borzoi_factorized_true-methyl-0.5-at-output.gin"
    "../configs/factorized/borzoi_factorized_true-methyl-0.5-at-rep.gin"
    "../configs/factorized/borzoi_factorized_true-methyl-1.0,0.5-at-output.gin"
    "../configs/factorized/borzoi_factorized_true-methyl-1.0-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_true-methyl-1.0.gin"
    "../configs/factorized/borzoi_factorized_true-methyl-0.0.gin"
    "../configs/factorized/borzoi_residual-methyl-local-only.gin"
    "../configs/factorized/borzoi_residual-methylseq-basset.gin"
    "../configs/factorized/borzoi_residual-methylseq-basset_concat-emb-penult.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/trainer.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1 --logging-level DEBUG
