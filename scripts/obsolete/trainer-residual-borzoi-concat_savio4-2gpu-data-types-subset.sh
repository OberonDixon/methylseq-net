#!/bin/bash
#SBATCH --job-name=train_methylseqnet_concat_2gpu
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
#SBATCH --array=0-7 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# excluded node 386 temporarily because it has a hardware issue
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methyl-local-only-CAGE-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset-CAGE-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset_concat-emb-final-CAGE-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset_concat-emb-penult-CAGE-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methyl-local-only-ATAC-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset-ATAC-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset_concat-emb-final-ATAC-seq.gin"
    "../configs/residual/concat_pretrained_embeddings/data_types_subset/borzoi_residual-methylseq-basset_concat-emb-penult-ATAC-seq.gin"
)

# START_CHECKPOINT="slurm28108438task1"

# echo $START_CHECKPOINT

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1 --logging-level DEBUG #--start-from-checkpoint $START_CHECKPOINT
