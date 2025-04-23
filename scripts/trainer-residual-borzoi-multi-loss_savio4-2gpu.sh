#!/bin/bash
#SBATCH --job-name=train_methylseqnet_multi-loss_2gpu
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
#SBATCH --array=10-11 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
#SBATCH --exclude=n0386.savio4
# exclude node 386 temporarily because it has a hardware issue
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_5-5-10-seq-loss.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_5-5-10.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_co-train-20-seq-loss.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_co-train-20-seq-loss-0.5.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_co-train-20.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basenji_co-train-20-seq-loss.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basenji_co-train-20-seq-loss-0.5.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basenji_co-train-20.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_co-train-20-seq-loss-concat-final.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basenji_co-train-20-seq-loss-concat-final.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basset_co-train-20-seq-loss-concat-final-mx+b.gin"
    "../configs/residual/multi-loss/borzoi_residual-methylseq-basenji_co-train-20-seq-loss-concat-final-mx+b.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/trainer.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1