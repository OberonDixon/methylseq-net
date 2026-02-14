#!/bin/bash
#SBATCH --job-name=train_methylseqnet_multinomial_2gpu
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
#SBATCH --array=0-6 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji-multi.01.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji-multi.05.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji-multi-zero-weight.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji-poisson.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji-multi.05-nonspatial.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji_concat-emb-final-multi.gin"
    "../configs/residual/multinomial/borzoi_residual-methylseq-basenji_concat-emb-final-poisson.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1 --logging-level DEBUG