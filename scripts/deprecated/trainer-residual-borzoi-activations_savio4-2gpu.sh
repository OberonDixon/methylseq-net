#!/bin/bash
#SBATCH --job-name=train_methylseqnet_activations_2gpu
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
#SBATCH --array=16-19 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
#SBATCH --exclude=n0386.savio4
# exclude node 386 temporarily because it has a hardware issue
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/residual/activations/borzoi_residual-methylseq-basset_concat-final-crelu.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_concat-final-crelu.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_crelu.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_crelu.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_concat-final-logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_concat-final-logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_softplus-mult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_tanh-logmult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_logmult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_tanh-logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_softplus-mult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_tanh-logmult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_logmult_init-unity.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_tanh-logmult.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_concat-final-mx+b.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basset_mx+b.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_concat-final-mx+b.gin"
    "../configs/residual/activations/borzoi_residual-methylseq-basenji_mx+b.gin"
)

START_CHECKPOINT="slurm23966228task0"

echo $START_CHECKPOINT

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet-prod
NCCL_P2P_DISABLE=1 python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1 --start-from-checkpoint $START_CHECKPOINT